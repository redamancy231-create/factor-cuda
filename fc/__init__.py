# factor-cuda -- fc.* contract adapter package (Phase 1).
# Public API re-exports only; does NOT import the GPU bindings eagerly (they
# are lazily loaded on first GPU call, so importing fc is cheap and works on
# CPU-only machines for the cpu-backend ops).
from .cross_sectional_rank import cross_sectional_rank, factor_plane
from .correlation import factor_corr, stock_corr
from .rolling_ic import rolling_ic
from .parameter_scan import parameter_scan
from ._workspace import _clear_all as _clear_workspaces


def clear_workspaces():
    """Release all cached GPU device buffers held by the adapter (idempotent).

    The adapter transparently auto-caches a per-shape device workspace for
    cross_sectional_rank / rolling_ic / parameter_scan (P3, 2026-08-08) so
    repeated calls with the same (T,N) panel reuse device buffers instead of
    allocating/freeing every call. Call this to free that device memory, e.g.
    before a leak check or after finishing a large workload. The next call
    re-creates the caches; it is never required for correctness."""
    _clear_workspaces()


__version__ = "1.1.1"
# clear_workspaces stays a module attribute (callable as fc.clear_workspaces)
# but is intentionally NOT in __all__: the F01 signature snapshot locks the
# contract public API exactly, and the release helper is not contract.
__all__ = ["cross_sectional_rank", "factor_plane", "factor_corr", "stock_corr",
           "rolling_ic", "parameter_scan", "__version__"]
