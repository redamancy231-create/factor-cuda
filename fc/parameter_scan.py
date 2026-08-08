# factor-cuda -- fc.parameter_scan (Phase 1 adapter).
# Contract: CLAUDE.md L0 Spec section 4. axes = list of (axis_name,
# values_list); effective spec is canonical direction->mask_mode with defaults;
# G = cartesian product, dict order (direction slowest). GPU-only, no device
# param, results always CPU. Mask override: only validated/required when any
# effective combo is masked (all-unmasked -> user mask IGNORED entirely).
# Subset scans use the binding's active_groups selector (only effective G
# groups are launched/timed/failure-isolated). elapsed_ms authority = this
# function's wall-clock (binding _elapsed_ms_diag is diagnostic only).
# ASCII-only comments. Phase 1.
import time

import numpy as np

from . import _util as u

_AXES = ("direction", "mask_mode")
_DIR_VALUES = ("ascending", "descending")
_MASK_VALUES = ("masked", "unmasked")
# Binding group order (fixed): asc-masked=0, asc-unmasked=1, desc-masked=2,
# desc-unmasked=3.
_BINDING_INDEX = {"ascending": {"masked": 0, "unmasked": 1},
                  "descending": {"masked": 2, "unmasked": 3}}


def _normalize_axes(axes):
    if not isinstance(axes, (list, tuple)) or not all(
            isinstance(a, (list, tuple)) and len(a) == 2 for a in axes):
        raise ValueError("axes must be a list of (axis_name, values_list) tuples")
    seen = set()
    user = {}
    for axis_name, vals in axes:
        if axis_name not in _AXES:
            raise ValueError(f"unknown scan axis {axis_name!r}; allowed: "
                             f"direction, mask_mode")
        if axis_name in seen:
            raise ValueError(f"duplicate axis {axis_name!r}")
        seen.add(axis_name)
        if not isinstance(vals, (list, tuple)) or len(vals) == 0:
            raise ValueError(f"axis {axis_name!r} values must be non-empty")
        if len(set(vals)) != len(vals):
            raise ValueError(f"axis {axis_name!r} contains duplicate values")
        if axis_name == "direction" and not set(vals) <= set(_DIR_VALUES):
            raise ValueError(f"direction values must be in {_DIR_VALUES}")
        if axis_name == "mask_mode" and not set(vals) <= set(_MASK_VALUES):
            raise ValueError(f"mask_mode values must be in {_MASK_VALUES}")
        user[axis_name] = list(vals)
    # Effective spec: canonical axis order direction -> mask_mode, defaults filled.
    effective = {"direction": user.get("direction", ["ascending"]),
                 "mask_mode": user.get("mask_mode", ["masked"])}
    combos = [(d, m) for d in effective["direction"] for m in effective["mask_mode"]]
    return effective, combos


def parameter_scan(axes, X, mask=None):
    effective, combos = _normalize_axes(axes)
    G = len(combos)

    kind, device, x = u.to_numpy(X, name="X", ndim=2, dtypes="f32f64",
                                 downcast_to="float32")
    T, N = x.shape
    u.require_cuda("parameter_scan")
    if N > (1 << 24):
        raise ValueError("N > 2^24 (rank precision cap, HG-2 frozen)")

    # Mask override (contract): only validated/required when any effective
    # combo is masked; all-unmasked -> user mask ignored entirely (synthesize
    # an all-true mask for the binding, which rejects None).
    any_masked = any(mm == "masked" for _, mm in combos)
    if any_masked:
        m = u.check_mask(mask, name="mask", T=T, N=N, allow_none=False)
    else:
        m = np.ones((T, N), dtype=np.bool_)

    active = [0, 0, 0, 0]
    for d, mm in combos:
        active[_BINDING_INDEX[d][mm]] = 1

    u.sync_entry(device)
    t_start = time.perf_counter()
    r = u.fcb().parameter_scan_f32(x, m, return_timing=True, active_groups=active)
    elapsed_ms = (time.perf_counter() - t_start) * 1e3  # authority (review F12)

    groups = []
    for gi, (d, mm) in enumerate(combos):
        g = _BINDING_INDEX[d][mm]
        st = r["group_status"][g]
        if st == 0:
            groups.append({
                "group_index": gi,
                "axis_values": {"direction": d, "mask_mode": mm},
                "result": r["groups"][g],  # (T,N) f32 numpy, always CPU
                "status": "ok", "error": None, "error_stage": None,
                "time_ms": r["time_ms"][g], "time_gpu_ms": r["time_gpu_ms"][g],
            })
        elif st in (9, 701):  # whitelist launch failures -> group failed
            groups.append({
                "group_index": gi,
                "axis_values": {"direction": d, "mask_mode": mm},
                "result": None, "status": "failed",
                "error": f"group-level CUDA error {st}", "error_stage": "launch",
                "time_ms": 0.0, "time_gpu_ms": 0.0,  # failed timing = 0.0
            })
        else:  # non-whitelist -> scan-level RuntimeError (contract §4 failure)
            raise RuntimeError(
                f"non-whitelist CUDA status {st} for group {g} "
                "(must be scan-level, not group-failed)")
    n_failed = sum(1 for gr in groups if gr["status"] == "failed")
    ok = [gr for gr in groups if gr["status"] == "ok"]
    return {
        "spec": effective,
        "groups": groups,
        "summary": {
            "total_groups": G, "n_failed": n_failed,
            "total_time_ms": sum(gr["time_ms"] for gr in ok),
            "total_time_gpu_ms": sum(gr["time_gpu_ms"] for gr in ok),
            "elapsed_ms": elapsed_ms,
        },
    }
