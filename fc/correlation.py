# factor-cuda -- fc.factor_corr + fc.stock_corr (Phase 1 adapter).
# Contract: CLAUDE.md L0 Spec section 2. backend='cpu' is the oracle backend
# (always numpy output); backend='cuda' executes on GPU and mirrors the input
# device (torch out for torch-in, torch-CPU out for numpy-in). Correlation
# domain precondition (max|x|<=1e150, min nonzero |x|>=1e-150 over valid cells)
# is adapter-owned for the GPU path (factor_corr_gpu has no domain check; the
# binding maps any rc!=0 to RuntimeError) and is a ValueError with the frozen
# _exc_kw substrings. CPU core delegates to fc._cpu_core (single source).
# ASCII-only comments. Phase 1.
import numpy as np

from . import _util as u
from ._cpu_core import np_factor_corr, np_stock_corr


def _gpu_domain_check(arr, mask):
    """Domain precondition over the pooled valid subset (mask & finite). Handles
    both 3-D factor_corr panels (mask broadcast per column) and 2-D stock_corr
    panels. The ValueError message must contain '数值域外' / '1e150' / '1e-150'
    (frozen parity _exc_kw). Returns nothing; raises ValueError out of domain.

    f32 inputs skip the check entirely: the finite-nonzero f32 value range
    [1.4e-45, 3.4e38] lies strictly inside the corr domain [1e-150, 1e150], so
    max|x|<=1e150 and min-nonzero-|x|>=1e-150 are trivially satisfied for any
    f32 panel (inf/nan are pooled out by the CUDA kernel's own isfinite
    validity mask -- the same set the check would pool; the pybind11 binding
    only upcasts f32->f64 at the argument-conversion stage, it does not mask).
    Skipping the Python-side f64 astype + isfinite + abs + min/max pass removes
    2.09 s/call on a 1218x5000x12 f32 panel (same-panel before/after, measured
    2026-08-07). f64 inputs keep the full check (zero-copy view, no astype)."""
    if arr.dtype == np.float32:
        return
    arr64 = np.asarray(arr, dtype=np.float64)
    if arr64.ndim == 3:
        T, N, F = arr64.shape
        flat = arr64.reshape(T * N, F)
        valid = np.isfinite(flat)
        if mask is not None:
            valid &= np.asarray(mask, dtype=bool).reshape(T * N, 1)
    else:
        flat = arr64.reshape(-1)
        valid = np.isfinite(flat)
        if mask is not None:
            valid &= np.asarray(mask, dtype=bool).reshape(-1)
    # in_corr_domain RETURNS bool (it does not raise); the adapter must turn a
    # domain violation into the contract ValueError with the frozen _exc_kw
    # substrings (real bug found in parity gpu 2026-08-06: no check -> no raise).
    if not u.in_corr_domain(flat, valid):
        raise ValueError("corr 数值域外：max|x|>1e150 或 min 非零|x|<1e-150")


def factor_corr(data, mask=None, names=None, backend="cpu"):
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend must be 'cpu' or 'cuda', got "
                         f"{backend!r}")
    kind, device, arr = u.to_numpy(data, name="data", ndim=3, dtypes="f32f64")
    T, N, F = arr.shape
    if names is not None and len(names) != F:
        raise ValueError(f"names length {len(names)} must equal F={F}")
    if backend == "cuda":
        u.require_cuda("factor_corr backend='cuda'")
        mask = u.mask_must_match_device(mask, device, "mask")
        m = u.check_mask(mask, name="mask", T=T, N=N)
        if F > 128:
            raise ValueError("F > 128 (factor pair grid cap, HG-2 frozen)")
        _gpu_domain_check(arr, m)
        u.sync_entry(device)
        res = u.fcp().factor_corr_f64(arr, m)
        return u.make_output(res, kind=kind, device=device, mode="corr_cuda")
    # cpu backend: always numpy output; np_factor_corr does its own domain check
    m = u.check_mask(mask, name="mask", T=T, N=N)
    return np_factor_corr(arr, m)


def stock_corr(data, mask=None, backend="cpu"):
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend must be 'cpu' or 'cuda', got "
                         f"{backend!r}")
    kind, device, arr = u.to_numpy(data, name="data", ndim=2, dtypes="f32f64")
    T, N = arr.shape
    if backend == "cuda":
        u.require_cuda("stock_corr backend='cuda'")
        mask = u.mask_must_match_device(mask, device, "mask")
        m = u.check_mask(mask, name="mask", T=T, N=N)
        if int(N) * N > (1 << 31) - 1:
            raise ValueError("N*N exceeds INT32_MAX (output grid cap, HG-2 "
                             "frozen; N <= 46340)")
        _gpu_domain_check(arr, m)
        u.sync_entry(device)
        res = u.fcb().stock_corr_f64(arr, m)
        return u.make_output(res, kind=kind, device=device, mode="corr_cuda")
    m = u.check_mask(mask, name="mask", T=T, N=N)
    return np_stock_corr(arr, m)
