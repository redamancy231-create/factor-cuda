# factor-cuda -- P3 adapter workspace auto-cache tests (2026-08-08).
#
# Guards the transparent per-shape device-buffer cache in fc/_workspace.py and
# the workspace parameters added to the low-level bindings:
#   1. cached vs uncached bitwise identity -- the P3 core correctness property
#      (cache must never change results).
#   2. same-shape reuse / shape-switch new workspace / mask on-off alternation
#      does not discard the workspace (C++ lazy mask capacity).
#   3. fc.clear_workspaces() releases entries and the next call re-creates.
#   4. a failed call leaves the workspace usable (kernel fail-path contract).
#   5. concurrent same-key calls are serialized (C++ ws is not thread-safe)
#      and produce bitwise-identical results.
#   6. binding-level workspace handles (CsRankWorkspace / RollingIcWorkspace)
#      exist, are idempotent-clearable, and yield identical results.
#   7. fc package version bumped + clear_workspaces exported.
#
# Run: PYTHONIOENCODING=utf-8 python -m pytest tests/test_ws_cache_v1.py -v
# ASCII-only comments (Windows GBK-safe).
import concurrent.futures

import numpy as np
import pytest

import fc
from fc import _workspace as _ws

try:
    import torch
    HAS_CUDA = bool(torch.cuda.is_available())
except Exception:  # pragma: no cover - CPU-only environments
    torch = None
    HAS_CUDA = False

NEED_CUDA = pytest.mark.skipif(not HAS_CUDA, reason="CUDA unavailable")

_T, _N = 64, 32


def _panel(dtype="float64", seed=7, T=_T, N=_N):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, N)).astype(dtype)


def _mask(T=_T, N=_N, frac=0.9, seed=3):
    rng = np.random.default_rng(seed)
    m = rng.random((T, N)) < frac
    return m


@pytest.fixture(autouse=True)
def _clean_caches():
    """Isolate process-level cache state between tests: re-enable, then drop."""
    yield
    for c in _ws._CACHES:
        c.set_enabled(True)
    fc.clear_workspaces()


# ---- 1. cached vs uncached bitwise identity ----------------------------------

@NEED_CUDA
def test_cs_rank_cached_matches_uncached_bitwise():
    x = _panel("float32")
    x[::7, ::5] = np.nan  # inject invalid cells -> NaN payload on both paths
    cached = fc.cross_sectional_rank(x)
    _disable_all()
    uncached = fc.cross_sectional_rank(x)
    assert np.array_equal(cached, uncached, equal_nan=True)
    # NaN payload preserved (quiet NaN 0x7fc00000) on both paths
    bad = ~np.isfinite(cached)
    assert bad.any()
    np.testing.assert_array_equal(
        cached.view(np.uint32)[bad].astype(np.uint32),
        uncached.view(np.uint32)[bad].astype(np.uint32))


@NEED_CUDA
def test_rolling_ic_cached_matches_uncached_bitwise():
    f = _panel("float64", seed=11)
    r = _panel("float64", seed=12)
    fm = _mask()
    rm = _mask(seed=9)
    cached = fc.rolling_ic(f, r, fm, rm, min_valid=16, device="cuda")
    _disable_all()
    uncached = fc.rolling_ic(f, r, fm, rm, min_valid=16, device="cuda")
    assert np.array_equal(cached, uncached, equal_nan=True)


@NEED_CUDA
def test_parameter_scan_cached_matches_uncached_bitwise():
    x = _panel("float32", seed=21)
    m = _mask()
    axes = [("direction", ["ascending", "descending"]),
            ("mask_mode", ["masked", "unmasked"])]
    cached = fc.parameter_scan(axes, x, m)
    _disable_all()
    uncached = fc.parameter_scan(axes, x, m)
    for gc, gu in zip(cached["groups"], uncached["groups"]):
        assert np.array_equal(gc["result"], gu["result"], equal_nan=True)


def _disable_all():
    for c in _ws._CACHES:
        c.set_enabled(False)


# ---- 2. reuse / shape switch / mask alternation ------------------------------

@NEED_CUDA
def test_same_shape_reuses_single_workspace():
    x = _panel("float32", seed=1)
    n0 = sum(len(c._items) for c in _ws._CACHES)
    for _ in range(3):
        fc.cross_sectional_rank(x)
    n1 = sum(len(c._items) for c in _ws._CACHES)
    # rank cache created exactly one entry; subsequent same-shape calls reuse it
    assert n1 == n0 + 1


@NEED_CUDA
def test_shape_switch_creates_new_workspace():
    x = _panel("float32", seed=1, T=_T, N=_N)
    x2 = _panel("float32", seed=2, T=_T, N=_N * 2)
    fc.cross_sectional_rank(x)
    n_after_1 = sum(len(c._items) for c in _ws._CACHES)
    fc.cross_sectional_rank(x2)
    n_after_2 = sum(len(c._items) for c in _ws._CACHES)
    assert n_after_2 == n_after_1 + 1  # different shape -> separate workspace
    # back to shape 1 reuses the original entry (no third allocation)
    fc.cross_sectional_rank(x)
    n_after_3 = sum(len(c._items) for c in _ws._CACHES)
    assert n_after_3 == n_after_2


@NEED_CUDA
def test_mask_on_off_alternation_reuses_workspace():
    x = _panel("float32", seed=5)
    m = _mask()
    n0 = sum(len(c._items) for c in _ws._CACHES)
    fc.cross_sectional_rank(x, m)          # masked call
    fc.cross_sectional_rank(x)             # unmasked -> lazy mask capacity kept
    fc.cross_sectional_rank(x, m)          # masked again
    n1 = sum(len(c._items) for c in _ws._CACHES)
    assert n1 == n0 + 1  # no new workspace across mask toggling


# ---- 3. clear_workspaces ------------------------------------------------------

@NEED_CUDA
def test_clear_workspaces_drops_and_rebuilds():
    fc.cross_sectional_rank(_panel("float32"))
    assert any(len(c._items) for c in _ws._CACHES)
    fc.clear_workspaces()
    assert all(not len(c._items) for c in _ws._CACHES)
    fc.clear_workspaces()  # idempotent
    # next call rebuilds and is correct
    out = fc.cross_sectional_rank(_panel("float32", seed=6))
    assert np.isfinite(out).all()


# ---- 4. failed call leaves the workspace usable ------------------------------

@NEED_CUDA
def test_error_then_retry_uses_same_workspace():
    x = _panel("float32")
    good = fc.cross_sectional_rank(x)
    with pytest.raises(ValueError):
        fc.cross_sectional_rank(x.ravel())  # ndim=1 -> adapter ValueError
    n_after_error = sum(len(c._items) for c in _ws._CACHES)
    retry = fc.cross_sectional_rank(x)
    assert np.array_equal(good, retry, equal_nan=True)
    assert n_after_error >= 1  # error did not destroy the cache entry


# ---- 5. concurrent same-key calls are serialized + bitwise correct -----------

@NEED_CUDA
def test_concurrent_same_shape_calls_bitwise_identical():
    x = _panel("float32", seed=9)
    expected = fc.cross_sectional_rank(x)

    def _call(_):
        return fc.cross_sectional_rank(x)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(_call, range(8)))
    for r in results:
        assert np.array_equal(r, expected, equal_nan=True)


@NEED_CUDA
def test_concurrent_rolling_ic_calls_bitwise_identical():
    f = _panel("float64", seed=31)
    r = _panel("float64", seed=32)
    expected = fc.rolling_ic(f, r, min_valid=16, device="cuda")

    def _call(_):
        return fc.rolling_ic(f, r, min_valid=16, device="cuda")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(_call, range(8)))
    for out in results:
        assert np.array_equal(out, expected, equal_nan=True)


@NEED_CUDA
def test_concurrent_clear_does_not_race_inflight_calls():
    """clear_workspaces() during in-flight same-shape calls must not free a
    workspace between _get and the GPU call (dangling-buffer race): every call
    either holds its per-key lock before clear can free, or rebuilds fresh, and
    results stay bitwise-identical."""
    x = _panel("float32", seed=13)
    expected = fc.cross_sectional_rank(x)

    def _call(_):
        for _ in range(4):
            out = fc.cross_sectional_rank(x)
            if not np.array_equal(out, expected, equal_nan=True):
                return False
        return True

    def _clear():
        for _ in range(4):
            fc.clear_workspaces()

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        call_futures = [ex.submit(_call, i) for i in range(4)]
        clear_future = ex.submit(_clear)
        results = [f.result() for f in call_futures]
        clear_future.result()  # _clear returns None; only call results are asserted
    assert all(r is True for r in results)


# ---- 6. binding-level workspace handles ---------------------------------------

@NEED_CUDA
def test_binding_workspace_handles_exist_and_clear():
    from fc import _util as u
    fb = u.fcb()
    cw = fb.CsRankWorkspace()
    rw = fb.RollingIcWorkspace()
    cw.clear(); cw.clear()  # idempotent
    rw.clear()
    x = _panel("float32")
    r0 = fb.cs_rank_f32(x, None, False)
    r1 = fb.cs_rank_f32(x, None, False, cw)
    r2 = fb.cs_rank_f32(x, None, False, cw)
    assert np.array_equal(r0, r1, equal_nan=True)
    assert np.array_equal(r1, r2, equal_nan=True)


@NEED_CUDA
def test_binding_workspace_none_default_keeps_old_behavior():
    from fc import _util as u
    fb = u.fcb()
    x = _panel("float32")
    r_default = fb.cs_rank_f32(x)                      # no workspace arg
    r_none = fb.cs_rank_f32(x, None, False, None)      # explicit None
    assert np.array_equal(r_default, r_none, equal_nan=True)


# ---- 7. fail-closed evidence validation (synthetic negatives) ----------------

def _mk_op(**overrides):
    base = {
        "op": "fake", "uncached_ms": 50.0, "cached_ms": 30.0,
        "speedup_x": 50.0 / 30.0, "bitwise_identical": True,
        "cache_reused": True, "cache_entries_after_first": 1,
        "cache_entries_after_second": 1, "output_kind": "ic",
        "output_sha256": "X", "uncached_raw_ms": [], "cached_raw_ms": [],
    }
    base.update(overrides)
    return base


def _validate():
    from benchmarks import ws_py_cache_v1 as ev
    return ev._validate


def test_validate_healthy_passes():
    assert _validate()([_mk_op(), _mk_op(op="fake2")]) == []


def test_validate_rejects_bitwise_mismatch():
    problems = _validate()([_mk_op(bitwise_identical=False)])
    assert any("bitwise" in p for p in problems)


def test_validate_rejects_cache_not_reused():
    problems = _validate()(
        [_mk_op(cache_reused=False, cache_entries_after_second=2)])
    assert any("reuse" in p for p in problems)


def test_validate_rejects_wrong_direction():
    problems = _validate()([_mk_op(uncached_ms=30.0, cached_ms=50.0,
                                   speedup_x=0.6)])
    assert any("direction" in p for p in problems)


def test_validate_rejects_below_threshold():
    problems = _validate()([_mk_op(speedup_x=1.1)])
    assert any("1.2x" in p for p in problems)


# ---- 8. package API -----------------------------------------------------------

def test_version_and_clear_api():
    assert fc.__version__ == "1.1.0"
    # release helper: module attribute, NOT part of the locked contract __all__
    assert callable(fc.clear_workspaces)
    assert "clear_workspaces" not in fc.__all__
