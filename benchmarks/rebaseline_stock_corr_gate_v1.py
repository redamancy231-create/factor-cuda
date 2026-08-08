# -*- coding: utf-8 -*-
"""stock_corr v2 gate re-baseline (2026-08-05).

Rationale (user decision): the stock_corr v2 FAST path (de-mean Gram, 1
accumulator) only applies to fully-valid panels (every column count == T), and
the realistic stock_corr use case is clean, all-finite returns. The corpus
gate panel is NOT fully valid (23/5000 columns), so the fast path cannot hit
the corpus gate. Re-baseline measures BOTH the CuPy reference and (via the C++
perf harness) the GPU fast path on the SAME fully-valid returns panel
(benchmark_corpus/stock_corr_panel_v1_5000.bin, generator committed:
benchmark_corpus/generate_stock_corr_panel_v1.py).

Gate = exact_half = median/2 (PoC 2 convention: 2x speedup over the best free
alternative). Evidence + the v2 gate values are saved to
runs/stock_corr_v2_rebaseline_20260805/gate.json. docs/gate_config_v1.json is
INTENTIONALLY NOT modified: it is the formal corpus-panel gate (checked by
generate_gate_config_v1.py check) and the v2 fast-path gate is a separate
measurement on a different panel.

Usage: python benchmarks/rebaseline_stock_corr_gate_v1.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
BENCH = pathlib.Path(__file__).resolve().parent
RUN_DIR = BENCH / "results" / "runs" / "stock_corr_v2_rebaseline_20260805"
GATE_PATH = ROOT / "docs" / "gate_config_v1.json"
PANEL_BIN = ROOT / "benchmark_corpus" / "stock_corr_panel_v1_5000.bin"
T = 1218
N_COLS = 5000
SIZES = [500, 2000, 5000]
REPS = {500: 11, 2000: 7, 5000: 5}

sys.path.insert(0, str(BENCH))
from backends import cp_stock_corr  # noqa: E402


def floor2(v: float) -> float:
    return math.floor(v * 100.0) / 100.0


def canonical_payload(config: dict) -> bytes:
    return json.dumps(config, ensure_ascii=False, indent=1,
                      separators=(",", ": ")).encode("utf-8")


def main() -> int:
    import cupy as cp
    X = np.fromfile(PANEL_BIN, dtype=np.float64).reshape(T, N_COLS)
    cp.cuda.get_current_stream().synchronize()

    evidence = {"schema_version": "1.1.0", "run_id": "stock_corr_v2_rebaseline_20260805",
                "panel": str(PANEL_BIN.relative_to(ROOT)), "T": T, "N_COLS": N_COLS,
                "generator": "benchmark_corpus/generate_stock_corr_panel_v1.py",
                "backend": "cupy", "operations": {}}
    for N in SIZES:
        sub = X[:, :N]
        cp_stock_corr(sub, None)  # warmup
        cp.cuda.get_current_stream().synchronize()
        times = []
        for _ in range(REPS[N]):
            t0 = time.perf_counter()
            cp_stock_corr(sub, None)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        median = sorted(times)[len(times) // 2]
        evidence["operations"][f"stock_corr(N={N})"] = {
            "median_ms": median, "raw_wall_ms": median, "reps_ms": times,
        }
        print(f"N={N}: median {median:.6f} ms -> exact_half {median / 2:.6f} ms")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "cupy.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"evidence written: {RUN_DIR / 'cupy.json'}")

    # v2 fast-path gates (CuPy exact_half on the fully-valid panel). Kept
    # separate from docs/gate_config_v1.json (the formal corpus-panel gate).
    gates = {"schema_version": "1.1.0",
             "run_id": "stock_corr_v2_rebaseline_20260805",
             "panel": str(PANEL_BIN.relative_to(ROOT)),
             "scope": "v2 fast-path same-panel gate (fully-valid returns panel)",
             "gates": {}}
    for N in SIZES:
        key = f"stock_corr(N={N})"
        raw = evidence["operations"][key]["median_ms"]
        half = raw / 2.0
        gates["gates"][key] = {"raw_wall_ms": raw, "exact_half": half,
                               "display": floor2(half)}
    (RUN_DIR / "gate.json").write_text(
        json.dumps(gates, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"v2 gates written: {RUN_DIR / 'gate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
