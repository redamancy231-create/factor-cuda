# factor-cuda -- fc adapter shared helpers (Phase 1).
#
# Container detection (numpy -> torch -> DLPack), dtype/mask gates, device
# resolution, error remapping, lazy binding loader, output-container mapping
# and the correlation-domain check (delegated to fc._cpu_core, the single CPU
# oracle source). Implements CLAUDE.md L0 Spec section 0 conventions.
#
# Container semantics (contract): accept numpy array / torch.Tensor / DLPack
# capsule; dtype whitelist {float32,float64} (else ValueError); mask STRICT bool
# (uint8 -> ValueError, contract is strict-bool); non-contiguous -> copy;
# non-container object -> TypeError. Result device mirrors the input factor
# device. Zero-copy applies only to host-resident inputs (HG-2 2026-08-06);
# torch-CUDA inputs transfer D2H->H2D->D2H->H2D (semantically correct).
#
# ASCII-only comments (Windows GBK-safe). Phase 1.
from __future__ import annotations

import functools
import importlib
import pathlib
import sys

import numpy as np

try:
    import torch as _torch
except Exception:  # torch optional at import; ops requiring it raise clearly
    _torch = None

from ._cpu_core import in_corr_domain as _in_corr_domain  # single oracle source

_F32 = np.float32
_F64 = np.float64
_WHITELIST = (_F32, _F64)

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _torch_device(dev: str):
    if _torch is None:
        raise RuntimeError("torch is required for torch/DLPack inputs (not installed)")
    if dev == "cpu":
        return _torch.device("cpu")
    idx = int(dev.split(":")[1]) if ":" in dev else _torch.cuda.current_device()
    return _torch.device(f"cuda:{idx}")


def require_cuda(context: str = ""):
    """Contract: CUDA unavailable -> RuntimeError (correlation supports
    backend='cpu', rolling_ic supports device=None auto-CPU)."""
    if _torch is None or not _torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA unavailable{f': {context}' if context else ''}; "
            "correlation supports backend='cpu', rolling_ic supports "
            "device=None (auto-CPU)")


def _dtype_ok(dt, dtypes):
    """Whitelist check accepting numpy AND torch dtypes.

    The DLPack branch yields a torch tensor (from_dlpack), and torch.float32 is
    NOT equal to np.float32, so a numpy-only comparison wrongly rejected valid
    float32/float64 DLPack capsules (real bug found by tests/test_adapter_v1.py
    F07, 2026-08-06). Torch dtypes route against _torch dtypes; numpy dtypes
    against the _WHITELIST."""
    if _torch is not None and isinstance(dt, _torch.dtype):
        if dtypes == "f32":
            return dt == _torch.float32
        if dtypes == "f64":
            return dt == _torch.float64
        return dt in (_torch.float32, _torch.float64)
    if dtypes == "f32":
        return dt == _F32
    if dtypes == "f64":
        return dt == _F64
    return dt in _WHITELIST


def to_numpy(x, *, name, ndim, dtypes="f32f64", downcast_to=None):
    """Normalize input to a C-contiguous numpy array.
    Returns (kind, device, np_arr) where kind in {'numpy','torch','dlpack'}
    and device in {'cpu','cuda:N'} (the input's device, for output mirroring).
    Raises TypeError (non-container), ValueError (dtype/ndim)."""
    kind, device = "numpy", "cpu"
    if isinstance(x, np.ndarray):
        if not _dtype_ok(x.dtype, dtypes):
            raise ValueError(
                f"{name} dtype {x.dtype} not in whitelist {{float32,float64}}")
        arr = np.ascontiguousarray(x)
    elif _torch is not None and isinstance(x, _torch.Tensor):
        kind = "torch"
        device = "cpu" if not x.is_cuda else f"cuda:{x.device.index}"
        x = x.detach()  # read-only contract: never autograd
        dt = x.dtype
        if not _dtype_ok(dt, dtypes):  # single whitelist authority (_dtype_ok)
            raise ValueError(
                f"{name} dtype {dt} not in whitelist {{float32,float64}}")
        if not x.is_contiguous():
            x = x.contiguous()
        arr = x.cpu().numpy()
    else:
        # DLPack (torch.utils.dlpack.from_dlpack). F06: only a raw PyCapsule is
        # a consumed-capsule ValueError; any other conversion failure is a
        # TypeError for a non-container object.
        if _torch is None:
            raise TypeError(f"{name} must be a numpy array, torch.Tensor, or "
                            "DLPack capsule (torch not installed)")
        try:
            t = _torch.utils.dlpack.from_dlpack(x)
        except RuntimeError:
            if type(x).__name__ == "PyCapsule":
                raise ValueError(
                    f"{name}: DLPack capsule already consumed") from None
            raise TypeError(
                f"{name} must be a numpy array, torch.Tensor, or DLPack capsule"
            ) from None
        kind = "dlpack"
        device = "cpu" if not t.is_cuda else f"cuda:{t.device.index}"
        t = t.detach()
        if not _dtype_ok(t.dtype, dtypes):
            raise ValueError(
                f"{name} dtype {t.dtype} not in whitelist {{float32,float64}} "
                "(capsule consumed)")
        arr = t.cpu().numpy() if not t.is_contiguous() else t.detach().cpu().numpy()
        if not t.is_contiguous():
            arr = np.ascontiguousarray(arr)

    if isinstance(ndim, int):
        ndim_ok = arr.ndim == ndim
    else:
        ndim_ok = arr.ndim in ndim
    if not ndim_ok:
        raise ValueError(f"{name} must be {ndim}-D, got ndim={arr.ndim}")
    if downcast_to == "float32" and arr.dtype == _F64:
        arr = arr.astype(_F32)  # IEEE round-to-nearest-even; ties on f32 tensor
    return kind, device, np.ascontiguousarray(arr)


def check_mask(mask, *, name, T, N, allow_none=True):
    """Strict-bool mask -> numpy bool (T,N) or None. Non-bool (incl. uint8) ->
    ValueError (contract strict-bool; bindings accept uint8 but the adapter
    must not). Shape mismatch -> ValueError."""
    if mask is None:
        if allow_none:
            return None
        raise ValueError(f"{name} is required")
    if isinstance(mask, np.ndarray):
        if mask.dtype != np.bool_:
            raise ValueError(f"{name} must be bool, got {mask.dtype}")
        arr = np.ascontiguousarray(mask)
    elif _torch is not None and isinstance(mask, _torch.Tensor):
        if mask.dtype != _torch.bool:
            raise ValueError(f"{name} must be bool, got {mask.dtype}")
        arr = mask.detach().cpu().numpy()
    elif _torch is not None:
        try:
            t = _torch.utils.dlpack.from_dlpack(mask)
        except RuntimeError:
            raise TypeError(f"{name} must be a bool array or DLPack capsule") from None
        if t.dtype != _torch.bool:
            raise ValueError(f"{name} must be bool, got {t.dtype}")
        arr = t.detach().cpu().numpy()
    else:
        raise TypeError(f"{name} must be a bool numpy array or torch.Tensor")
    if arr.shape != (T, N):
        raise ValueError(f"{name} shape must be (T,N) matching input, got {arr.shape}")
    return arr


def mask_must_match_device(mask, device, name):
    """Global rule: mask must be on the same device as values (no auto-migrate;
    correlation cpu backend is the exception, handled by the caller). numpy /
    torch masks carry a known device. DLPack capsules are consumed here to
    resolve their REAL device (a CUDA capsule with CUDA values must NOT be
    rejected as host -- GPT-5.6-Sol review 2026-08-06 finding #9). Returns the
    resolved mask (a consumed DLPack capsule becomes its torch tensor) so
    check_mask reuses it instead of double-consuming."""
    if mask is None:
        return None
    if isinstance(mask, np.ndarray):
        md = "cpu"
    elif _torch is not None and isinstance(mask, _torch.Tensor):
        md = "cpu" if not mask.is_cuda else f"cuda:{mask.device.index}"
    else:
        if _torch is not None:
            try:
                t = _torch.utils.dlpack.from_dlpack(mask)
            except RuntimeError:
                return mask  # already consumed -> let check_mask raise
            mask = t  # reuse the consumed tensor in check_mask
            md = "cpu" if not t.is_cuda else f"cuda:{t.device.index}"
        else:
            md = "cpu"
    if md != device:
        raise ValueError(
            f"{name} must be on the same device as values ({device}), got {md}")
    return mask


@functools.lru_cache(maxsize=1)
def _load_binding(name: str):
    build = ROOT / "build"
    if str(build) not in sys.path:
        sys.path.insert(0, str(build))
    try:
        return importlib.import_module(name)
    except ImportError as e:
        raise RuntimeError(
            f"CUDA binding {name} not found in {build}/; run dev-build.bat") from e


@functools.lru_cache(maxsize=1)
def fcb():
    return _load_binding("factor_cuda_pybind")


@functools.lru_cache(maxsize=1)
def fcp():
    return _load_binding("factor_corr_pybind")


def sync_entry(device: str):
    """Entry sync: wait for the input device's current stream (contract: the op
    is synchronous on entry; a producer stream must be drained first)."""
    if _torch is not None and _torch.cuda.is_available() and device.startswith("cuda"):
        idx = int(device.split(":")[1])
        _torch.cuda.current_stream(idx).synchronize()


def sync_return():
    """Return sync: results are materialized before return. The low-level
    bindings are host-blocking (cudaDeviceSynchronize inside), so a full device
    sync here is a cheap guarantee for torch-CUDA outputs."""
    if _torch is not None and _torch.cuda.is_available():
        _torch.cuda.synchronize()


def in_corr_domain(arr, mask=None):
    """Correlation domain precondition (ValueError message must contain the
    frozen parity _exc_kw substrings). Delegates to the single CPU oracle."""
    return _in_corr_domain(np.asarray(arr, dtype=np.float64), mask)


def make_output(np_result, *, kind, device, mode):
    """Map a numpy result to the contract output container.
    mode: 'mirror' (rank/scan: numpy->numpy, torch->torch on input device)
          'corr_cpu' (always numpy) / 'corr_cuda' (torch mirror input device)
          'roll' (numpy iff all-numpy AND CPU exec, else torch mirror factor)."""
    if mode == "corr_cpu":
        return np_result
    if mode == "roll":
        if kind == "numpy" and device == "cpu":
            return np_result
        return _torch.as_tensor(np_result, device=_torch_device(device))
    # mirror / corr_cuda
    if kind == "numpy":
        return np_result
    return _torch.as_tensor(np_result, device=_torch_device(device))
