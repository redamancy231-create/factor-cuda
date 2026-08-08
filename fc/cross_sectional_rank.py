# factor-cuda -- fc.cross_sectional_rank + fc.factor_plane (Phase 1 adapter).
# Contract: CLAUDE.md L0 Spec section 1. Signature (derived from the frozen
# clause set): values (T,N), mask=None, descending=False -> (T,N) float32.
# Stable ordinal rank 1..K on the float32 canonical tensor (f64 downcast is the
# adapter's job, IEEE round-to-nearest-even); invalid cells -> quiet NaN payload
# 0x7fc00000 preserved bitwise. GPU-only (no device param, no CPU fallback).
# ASCII-only comments. Phase 1.
import numpy as np

from . import _util as u


def cross_sectional_rank(values, mask=None, descending=False):
    if type(descending) is not bool:
        raise ValueError("descending must be a bool (type(x) is bool), got "
                         f"{type(descending).__name__}")
    kind, device, x = u.to_numpy(values, name="values", ndim=2,
                                 dtypes="f32f64", downcast_to="float32")
    T, N = x.shape
    u.require_cuda("cross_sectional_rank")
    mask = u.mask_must_match_device(mask, device, "mask")
    m = u.check_mask(mask, name="mask", T=T, N=N)
    u.sync_entry(device)
    res = u.fcb().cs_rank_f32(x, m, descending)
    return u.make_output(res, kind=kind, device=device, mode="mirror")


def factor_plane(factors, f):
    """Contract helper: validate 0<=f<F and return factor f as a (T,N)
    C-contiguous copy (no memory sharing with the input). ndim=2 requires
    f==0. Output container mirrors the input (numpy->numpy, torch->torch)."""
    if not isinstance(f, int) or isinstance(f, bool):
        raise ValueError(f"f must be an integer, got {type(f).__name__}")
    kind, device, arr = u.to_numpy(factors, name="factors", ndim=(2, 3),
                                   dtypes="f32f64")
    if arr.ndim == 2:
        if f != 0:
            raise ValueError("ndim=2 input requires f == 0")
        plane = np.ascontiguousarray(arr)  # fresh copy, no sharing
    else:
        if not (0 <= f < arr.shape[2]):
            raise ValueError(f"f={f} out of range [0,F={arr.shape[2]})")
        plane = np.ascontiguousarray(arr[..., f])  # N-stride=F -> always a copy
    return u.make_output(plane, kind=kind, device=device, mode="mirror")
