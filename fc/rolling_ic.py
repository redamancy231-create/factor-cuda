# factor-cuda -- fc.rolling_ic (Phase 1 adapter).
# Contract: CLAUDE.md L0 Spec section 3. Contract names factor/forward_returns
# (not F/R). device=None -> current_device if CUDA else auto-CPU (a documented
# exception to the global device rule). Output: numpy iff all-numpy inputs AND
# CPU execution; else torch mirroring the factor device. min_valid STRICT int
# (bool rejected). ASCII-only comments. Phase 1.
from . import _util as u
from ._cpu_core import np_rolling_ic


def rolling_ic(factor, forward_returns, factor_mask=None, fwd_mask=None,
               min_valid=30, device=None):
    if device not in (None, "cpu", "cuda"):
        raise ValueError("device must be None, 'cpu', or 'cuda', got "
                         f"{device!r}")
    if not isinstance(min_valid, int) or isinstance(min_valid, bool):
        raise ValueError("min_valid must be an integer (bool rejected)")
    if min_valid < 2:
        raise ValueError("min_valid must be >= 2")
    kf, df, f = u.to_numpy(factor, name="factor", ndim=2, dtypes="f32f64")
    kr, dr, r = u.to_numpy(forward_returns, name="forward_returns", ndim=2,
                           dtypes="f32f64")
    T, N = f.shape
    if r.shape != (T, N):
        raise ValueError("factor and forward_returns shapes must match (T,N), "
                         f"got {f.shape} vs {r.shape}")
    factor_mask = u.mask_must_match_device(factor_mask, df, "factor_mask")
    fwd_mask = u.mask_must_match_device(fwd_mask, df, "fwd_mask")
    fm = u.check_mask(factor_mask, name="factor_mask", T=T, N=N)
    rm = u.check_mask(fwd_mask, name="fwd_mask", T=T, N=N)

    exec_cuda = {"cpu": False,
                 None: (u._torch is not None and u._torch.cuda.is_available()),
                 "cuda": True}[device]
    if exec_cuda:
        if u._torch is None or not u._torch.cuda.is_available():
            raise RuntimeError("device='cuda' but CUDA unavailable")
        exec_dev = df if kf == "torch" else (dr if kr == "torch" else None)
        if exec_dev:
            u.sync_entry(exec_dev)
        res = u.fcb().rolling_ic_f64(f, r, fm, rm, min_valid)
        return u.make_output(res, kind="torch" if (kf == "torch" or kr == "torch")
                             else "numpy", device=df, mode="roll")
    # CPU execution (device=None auto-CPU, or device='cpu')
    res = np_rolling_ic(f, r, fm, rm, min_valid)
    if kf == "numpy" and kr == "numpy":
        return res
    return u._torch.as_tensor(res, device=u._torch_device(df))
