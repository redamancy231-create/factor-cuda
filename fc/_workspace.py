# factor-cuda -- fc adapter workspace auto-cache (P3, 2026-08-08).
#
# Transparent per-shape device-buffer reuse. The low-level bindings
# (factor_cuda_pybind) accept an optional workspace handle (CsRankWorkspace /
# RollingIcWorkspace) that keeps device buffers cached across calls with the
# same shape, removing per-call cudaMalloc/cudaFree. This module owns that
# cache at the adapter layer, fully invisible to fc.* users:
#
#   - key = (T, N) per operation. Mask presence does NOT enter the key: the
#     C++ workspaces lazily grow their mask buffer on the first masked call and
#     keep it (review F02 pattern), so mask on/off alternation reuses the
#     workspace. Device changes are handled INSIDE the C++ workspace (its own
#     shape key includes the device ordinal; a mismatch clears + re-allocates),
#     so the Python key stays (T, N) -- a device switch degrades to the
#     uncached path for one call, which is correct.
#   - One threading.Lock per key: C++ workspaces are NOT thread-safe, so calls
#     sharing a workspace must be serialized. Different keys/ops run in
#     parallel. clear() also takes each key's lock so a workspace is never
#     released while a call is in flight.
#   - enabled flag (default True) lets benchmarks measure cached vs uncached on
#     the same panel; when disabled the cache returns None (binding allocates
#     and frees every call, the pre-P3 behavior).
#
# Users never touch workspaces directly. fc.clear_workspaces() releases all
# cached device buffers. ASCII-only comments (Windows GBK-safe).
import threading
from contextlib import contextmanager

_CACHES: list = []


def _register(cache):
    _CACHES.append(cache)
    return cache


def _clear_all():
    for c in _CACHES:
        c.clear()


class _WorkspaceCache:
    """Shape-keyed workspace cache for one operation.

    factory: zero-arg callable returning a fresh binding workspace handle
    (e.g. lambda: u.fcb().CsRankWorkspace()). The cache holds the handles
    forever (process lifetime) unless clear() is called; the C++ workspace
    types have trivial destructors, so dropped handles would leak device
    buffers -- that is why every entry stays referenced here.
    """

    def __init__(self, factory):
        self._factory = factory
        self._items = {}
        self._guard = threading.Lock()  # guards _items dict only
        self._enabled = True
        _register(self)

    def set_enabled(self, enabled):
        self._enabled = bool(enabled)

    @property
    def enabled(self):
        return self._enabled

    def _get(self, key):
        """Return (workspace | None, lock | None) with the per-key lock ALREADY
        held when workspace is non-None (use() releases it).

        The acquire happens INSIDE the _guard critical section so clear() can
        never free the workspace between _get returning and the caller's GPU
        call starting: clear() must wait on _guard, by which time this thread
        already holds the per-key lock, so clear() blocks until the call ends.
        (Acquiring outside _guard would leave a window where clear() deletes
        and frees the entry first -> dangling buffers.)"""
        if not self._enabled:
            return None, None
        with self._guard:
            item = self._items.get(key)
            if item is None:
                try:
                    ws = self._factory()
                except Exception:
                    # The binding does not expose a workspace handle (an older
                    # binding, or a test double). Fail-safe: disable THIS cache
                    # and fall back to the uncached path -- the cache is a
                    # performance optimization and must never break a call.
                    self._enabled = False
                    return None, None
                item = (ws, threading.Lock())
                self._items[key] = item
            ws, lock = item
            lock.acquire()
        return ws, lock

    @contextmanager
    def use(self, key):
        """Yield the cached workspace (or None when disabled) holding the per-key
        lock for the whole body. Serializes same-key calls so the shared
        (thread-unsafe) workspace is never used concurrently, and a concurrent
        clear_workspaces() cannot free buffers mid-call. Exceptions in the body
        release the lock (and the C++ ws survives a runtime failure -- buffers
        stay cached for a retry, per the kernel's fail-path contract)."""
        ws, lock = self._get(key)
        if lock is None:
            yield None
            return
        try:
            yield ws
        finally:
            lock.release()

    def clear(self):
        """Release all cached device buffers and drop the entries. Takes each
        key's lock first so a workspace is never freed while a call using it is
        still in flight. Idempotent; safe to call between workloads. Also
        re-enables the cache: a fail-safe disable (a one-off factory failure)
        is recoverable through this public path, honoring fc.clear_workspaces()'
        "the next call re-creates the caches" contract."""
        with self._guard:
            items = list(self._items.items())
            self._items = {}
            self._enabled = True
        for _, (ws, lock) in items:
            with lock:
                ws.clear()
