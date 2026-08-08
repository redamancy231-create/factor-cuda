# factor-cuda -- fc.* contract adapter package (Phase 1).
# Public API re-exports only; does NOT import the GPU bindings eagerly (they
# are lazily loaded on first GPU call, so importing fc is cheap and works on
# CPU-only machines for the cpu-backend ops).
from .cross_sectional_rank import cross_sectional_rank, factor_plane
from .correlation import factor_corr, stock_corr
from .rolling_ic import rolling_ic
from .parameter_scan import parameter_scan

__version__ = "1.0.0"
__all__ = ["cross_sectional_rank", "factor_plane", "factor_corr", "stock_corr",
           "rolling_ic", "parameter_scan", "__version__"]
