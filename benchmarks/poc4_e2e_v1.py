# factor-cuda -- PoC 4 end-to-end minimal parameter-scan pipeline measurement.
# Verifies the PoC4 criterion (CLAUDE.md stop condition 4 / PLAN.md prereg):
#   end-to-end speedup (incl. transfer/merge, not a single op) >= 2x (target 5x)
#   vs the BEST free alternative, same data / same mask / same semantics.
#
# Pipeline (covers all 5 kernels + parameter_scan merge):
#   per factor: parameter_scan (4 groups) -> per rank: rolling_ic -> factor_corr
#   + IC stack merge (4F,T) -- timed INSIDE each arm (review 2026-08-06: the
#   prereg amend names H2D + merge; the merge must be in end_to_end_s).
#   + stock_corr branch at N in {500, 2000} full-T (N=5000 GPU-only demo,
#     excluded from the main verdict -- separate branch, reported honestly).
#
# Arms:
#   gpu   : factor_cuda_pybind / factor_corr_pybind bindings (Python entry)
#   numpy : benchmarks/backends.py np_* (CPU)
#   cupy  : benchmarks/backends.py cp_* (GPU, decision-required baseline)
#   qg    : registered ONLY when qgplearn is importable (availability detection,
#           execution arm and result field stay in sync -- review 2026-08-06).
# best_free_total = min over the STRICT baseline whitelist {numpy, cupy, qg}
# (never the gpu arm; any future diagnostic arm is excluded).
#
# Data scale preregistered (design review MINOR-22): full corpus 1218x5000x12
# (full-workload) + F in {4,8,12} sequence as scan-scaling evidence.
# Mask fixed (INFO-25): fmask = mask, rmask = None. min_valid = 30.
#
# Measurement (review 2026-08-06 closure): symmetric reps (3) for every arm,
# median + min/max + per-sample recorded; stock_corr branch also 3 reps.
# Evidence envelope: runtime ISO timestamp (no hardcoded date), git HEAD,
# command line, corpus data_sha256.
#
# Per-op decomposition is wall-clock (each op incl. its H2D/D2H through the
# binding); the transfer/compute split inside a binding call is not exposed by
# the low-level bindings -- end-to-end wall-clock is the honest "incl. transfer"
# measure. cudaEvent-level per-kernel timing already lives in the PoC3 perf
# programs (C++ side), reported separately.
#
# Run: PYTHONIOENCODING=utf-8 python benchmarks/poc4_e2e_v1.py
# ASCII-only (Windows GBK-safe). PoC 4.
import datetime
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "build"))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
sys.path.insert(0, os.path.join(ROOT, "benchmark_corpus"))

import factor_cuda_pybind as fcb  # noqa: E402
import factor_corr_pybind as fcp  # noqa: E402
import backends  # noqa: E402
from corpus_loader_v1 import load  # noqa: E402

MIN_VALID = 30
F_SEQUENCE = [4, 8, 12]
STOCK_CORR_N = [500, 2000]
BASELINE_WHITELIST = {"numpy", "cupy", "qg"}  # strict: only these may be a baseline


def _best_free(total_map: dict) -> float:
    """Strict baseline whitelist: min over {numpy, cupy, qg} only. The gpu arm
    is the candidate and is never part of its own baseline (review BLOCKER-2);
    any future diagnostic arm is likewise excluded. At least one baseline must
    have run."""
    vals = [v for k, v in total_map.items() if k in BASELINE_WHITELIST and v is not None]
    if not vals:
        raise RuntimeError("_best_free: no baseline arm available (need numpy/cupy/qg)")
    return min(vals)


# ---------------------------------------------------------------------------
# Arms (each constructs the (4F,T) IC stack INSIDE its timed region)
# ---------------------------------------------------------------------------

def gpu_arm(factors, fwd, mask):
    """GPU pipeline via pybind bindings. factors: (T,N,F) f32 slice."""
    F = factors.shape[2]
    times = {}
    ranks = []
    t0 = time.perf_counter()
    for f in range(F):
        res = fcb.parameter_scan_f32(factors[:, :, f], mask)
        ranks.extend(res["groups"])
    times["parameter_scan"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    ics = [fcb.rolling_ic_f64(rk, fwd, fmask=mask, rmask=None, min_valid=MIN_VALID,
                              return_ranks=False) for rk in ranks]
    times["rolling_ic"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    corr_f = fcp.factor_corr_f64(factors, mask)
    times["factor_corr"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    ic_stack = np.stack(ics)  # (4F, T) merge -- prereg amend element, timed
    times["ic_stack"] = time.perf_counter() - t0
    return {"ranks": ranks, "ics": ics, "ic_stack": ic_stack, "corr_f": corr_f}, times


def numpy_arm(factors, fwd, mask):
    F = factors.shape[2]
    times = {}
    ranks = []
    t0 = time.perf_counter()
    for f in range(F):
        ranks.extend(backends.np_parameter_scan(factors[:, :, f], mask))
    times["parameter_scan"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    ics = [backends.np_rolling_ic(rk, fwd, mask, None, MIN_VALID) for rk in ranks]
    times["rolling_ic"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    corr_f = backends.np_factor_corr(factors, mask)
    times["factor_corr"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    ic_stack = np.stack(ics)
    times["ic_stack"] = time.perf_counter() - t0
    return {"ranks": ranks, "ics": ics, "ic_stack": ic_stack, "corr_f": corr_f}, times


def cupy_arm(factors, fwd, mask):
    F = factors.shape[2]
    times = {}
    ranks = []
    t0 = time.perf_counter()
    for f in range(F):
        ranks.extend(backends.cp_parameter_scan(factors[:, :, f], mask))
    times["parameter_scan"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    ics = [backends.cp_rolling_ic(rk, fwd, mask, None, MIN_VALID) for rk in ranks]
    times["rolling_ic"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    corr_f = backends.cp_factor_corr(factors, mask)
    times["factor_corr"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    ic_stack = np.stack(ics)
    times["ic_stack"] = time.perf_counter() - t0
    return {"ranks": ranks, "ics": ics, "ic_stack": ic_stack, "corr_f": corr_f}, times


def qg_arm(factors, fwd, mask):
    """QuantGplearn-Torch pipeline (registered ONLY when qgplearn imports).
    QG has no factor_corr arm (COMPETITOR_ANALYSIS: N/A) so factor_corr falls
    back to the numpy reference -- recorded in the per-op breakdown, not
    hidden. Untested here (qgplearn not installed at this session's runtime)."""
    F = factors.shape[2]
    times = {}
    ranks = []
    t0 = time.perf_counter()
    for f in range(F):
        ranks.extend(backends.qg_parameter_scan(factors[:, :, f], mask))
    times["parameter_scan"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    ics = [backends.qg_rolling_ic(rk, fwd, mask, None, MIN_VALID) for rk in ranks]
    times["rolling_ic"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    corr_f = backends.np_factor_corr(factors, mask)  # no QG factor_corr
    times["factor_corr"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    ic_stack = np.stack(ics)
    times["ic_stack"] = time.perf_counter() - t0
    return {"ranks": ranks, "ics": ics, "ic_stack": ic_stack, "corr_f": corr_f}, times


def _qg_available() -> bool:
    try:
        import qgplearn  # noqa: F401
        return True
    except Exception:
        return False


ARMS = {"gpu": gpu_arm, "numpy": numpy_arm, "cupy": cupy_arm}
REPS = {"gpu": 3, "numpy": 3, "cupy": 3}  # symmetric reps (review 2026-08-06)
if _qg_available():
    ARMS["qg"] = qg_arm
    REPS["qg"] = 3


def run_arm(arm_name, factors, fwd, mask):
    fn = ARMS[arm_name]
    reps = REPS[arm_name]
    e2e = []
    per_op_runs = []
    for _ in range(reps):
        t_start = time.perf_counter()
        _, times = fn(factors, fwd, mask)
        e2e.append(time.perf_counter() - t_start)
        per_op_runs.append(times)
    ops = list(per_op_runs[0].keys())
    return {
        "end_to_end_s": statistics.median(e2e),
        "end_to_end_min_s": min(e2e),
        "end_to_end_max_s": max(e2e),
        "per_op_s": {k: statistics.median([r[k] for r in per_op_runs]) for k in ops},
        "per_op_all_s": {k: [r[k] for r in per_op_runs] for k in ops},
        "reps": reps,
        "end_to_end_all_s": e2e,
    }


def run_stock_corr_branch(returns, mask, reps=3):
    """stock_corr branch at N in STOCK_CORR_N (full T), 3 reps median. This is
    a SEPARATE branch -- it never enters the main pipeline end_to_end_s nor the
    main verdict (review 2026-08-06: the GO scope is the minimal parameter-scan
    pipeline only; sub-2x branch values are reported honestly, not hidden)."""
    out = {}
    for n in STOCK_CORR_N:
        Xn = returns[:, :n]
        Mn = mask[:, :n]
        row = {}
        for arm, call in (("gpu", lambda: fcb.stock_corr_f64(Xn, Mn, False)),
                          ("numpy", lambda: backends.np_stock_corr(Xn, Mn)),
                          ("cupy", lambda: backends.cp_stock_corr(Xn, Mn))):
            samples = []
            for _ in range(reps):
                t0 = time.perf_counter()
                call()
                samples.append(time.perf_counter() - t0)
            row[f"{arm}_s"] = statistics.median(samples)
            row[f"{arm}_all_s"] = samples
        row["best_free_s"] = min(row["numpy_s"], row["cupy_s"])
        row["speedup_vs_best"] = row["best_free_s"] / row["gpu_s"]
        row["reps"] = reps
        out[str(n)] = row
    return out


def env_info():
    info = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "cupy": None,
        "torch_available": None,
        "qgplearn_available": _qg_available(),
    }
    try:
        import cupy
        info["cupy"] = cupy.__version__
    except Exception:
        pass
    try:
        import torch
        info["torch_available"] = f"{torch.__version__} cuda={torch.cuda.is_available()}"
    except Exception:
        pass
    return info


def evidence_info(manifest):
    """Runtime evidence envelope: no hardcoded date (review 2026-08-06)."""
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        head = "unknown"
    return {
        "date_iso": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_head": head,
        "cmdline": " ".join(sys.argv),
        "corpus_data_sha256": manifest.get("hash", {}).get("data_sha256"),
    }


def main():
    data, manifest = load("corpus_synth_v1")
    d = data
    full_t, full_n, full_f = d["factors"].shape
    fwd = np.ascontiguousarray(d["forward_returns"], dtype=np.float64)
    mask = np.ascontiguousarray(d["mask"], dtype=bool)  # (T,N) bool
    returns = np.ascontiguousarray(d["returns"], dtype=np.float32)

    out = {
        "name": "poc4_e2e_v1",
        "evidence": evidence_info(manifest),
        "env": env_info(),
        "data_scale": {"full_T": full_t, "full_N": full_n, "full_F": full_f,
                       "min_valid": MIN_VALID, "corpus": "corpus_synth_v1"},
        "factors_sequence": F_SEQUENCE,
        "arms": list(ARMS.keys()),
        "results": {},
        "stock_corr_branch": {},
        "criterion": "end-to-end speedup (incl. transfer/merge incl. the (4F,T) "
                     "IC stack) vs best free alternative (strict whitelist "
                     "min numpy/cupy/qg), same data/mask/semantics >= 2x (target 5x)",
        "verdict_scope": "PoC 4 minimal parameter-scan pipeline "
                         "(parameter_scan->rolling_ic->factor_corr + IC merge); "
                         "the stock_corr branch is separate and NOT part of the "
                         "main verdict (reported honestly)",
        "qg_note": ("qgplearn " + ("available; qg arm registered" if "qg" in ARMS
                                   else "NOT installed at runtime; qg arm not "
                                        "registered, best_free_total = min(numpy, "
                                        "cupy)")),
    }

    for f in F_SEQUENCE:
        factors = np.ascontiguousarray(d["factors"][:, :, :f])  # (T,N,f) f32
        print(f"== F={f} ==", flush=True)
        totals = {}
        for arm in ARMS:
            print(f"  running {arm} ({REPS[arm]} reps) ...", flush=True)
            res = run_arm(arm, factors, fwd, mask)
            totals[arm] = res["end_to_end_s"]
            out["results"].setdefault(str(f), {})[arm] = res
        bft = _best_free(totals)
        out["results"][str(f)]["best_free_total_s"] = bft
        out["results"][str(f)]["gpu_total_s"] = totals["gpu"]
        out["results"][str(f)]["speedup_vs_best"] = bft / totals["gpu"]
        ops = ("parameter_scan", "rolling_ic", "factor_corr", "ic_stack")
        per_op_best = {}
        for op in ops:
            per_op_best[op] = min(out["results"][str(f)][a]["per_op_s"][op]
                                  for a in ARMS if a in BASELINE_WHITELIST)
        out["results"][str(f)]["per_op_best_s"] = per_op_best
        out["results"][str(f)]["per_op_speedup_gpu_vs_best"] = {
            op: per_op_best[op] / out["results"][str(f)]["gpu"]["per_op_s"][op]
            for op in ops
        }
        sp = out["results"][str(f)]["speedup_vs_best"]
        print(f"  F={f}: gpu={totals['gpu']:.3f}s numpy={totals['numpy']:.3f}s "
              f"cupy={totals['cupy']:.3f}s best={bft:.3f}s speedup={sp:.2f}x", flush=True)

    print("== stock_corr branch ==", flush=True)
    out["stock_corr_branch"] = run_stock_corr_branch(returns, mask)
    for n, row in out["stock_corr_branch"].items():
        print(f"  N={n}: gpu={row['gpu_s']*1000:.1f}ms best={row['best_free_s']*1000:.1f}ms "
              f"speedup={row['speedup_vs_best']:.2f}x", flush=True)

    # verdict on the FULL-corpus (F=12) main criterion (pipeline only)
    f12 = out["results"]["12"]
    verdict = "PASS" if f12["speedup_vs_best"] >= 2.0 else "FAIL"
    out["verdict_main_F12"] = verdict
    out["verdict"] = verdict
    out["below_5x_note"] = (
        f"F=12 speedup {f12['speedup_vs_best']:.2f}x meets the >=2x criterion but "
        "is below the 5x target line -- recorded, not claimed as 'optimal'")

    os.makedirs(os.path.join(ROOT, "benchmarks", "results"), exist_ok=True)
    out_path = os.path.join(ROOT, "benchmarks", "results", "poc4_e2e_v1.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n== VERDICT (F=12 main criterion, pipeline only) ==")
    print(f"  speedup = {f12['speedup_vs_best']:.2f}x  -> {verdict}")
    print(f"  saved: {out_path}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
