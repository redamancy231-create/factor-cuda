# -*- coding: utf-8 -*-
"""stock_corr general 路径同面板 gate 重基线（2026-08-06）。

背景（Phase 2-3 验收 spec §2 P0-2）：`poc/poc3_stock_corr_perf.cu` 的 general
路径用 `make_panel("returns")` 合成面板（~6% NaN + ~6% 整数 + ~6% 精确 0，即
~18% 无效格 + 非连续分布），对 corpus 面板 CuPy gate 报 52.19ms NOT beat ——
这是跨数据比较（合成面板 ≠ corpus 面板），违反 FAIR_BASELINE 同数据同 mask 纪律。
同项目自证 `benchmarks/results/poc4_e2e_v1.json` stock_corr_branch：同一 general
kernel 在 **corpus returns+mask 面板**上 N=500 gpu=27.05ms vs cupy=53.09ms =
1.96x、N=2000 = 1.99x。本脚本在 corpus 同面板上做对称 reps 专项重测，产出独立
general gate（CuPy exact_half），据此裁决 general 路径加速比。

不改 docs/gate_config_v1.json（正式 corpus gate 保持冻结）；general gate 独立
文档化于 runs/stock_corr_general_gate_20260806/。

Usage: PYTHONIOENCODING=utf-8 python benchmarks/rebaseline_stock_corr_general_gate_v1.py
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import statistics
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
BENCH = pathlib.Path(__file__).resolve().parent
RUN_DIR = BENCH / "results" / "runs" / "stock_corr_general_gate_20260806"
T = 1218
SIZES = [500, 2000]
REPS = {500: 11, 2000: 7}  # 对称 reps（对齐 v2 rebaseline）
SYNTHESIS_NOTE = (
    "corpus_synth_v1 returns[:, :N] + mask[:, :N] 同面板；GPU general 走绑定层 "
    "fcb.stock_corr_f64（含 H2D/D2H），CuPy 走 backends.cp_stock_corr（masked-GEMM "
    "+ 抵消检测回退）。对称 reps，median。"
)

for p in (str(ROOT / "build"), str(BENCH), str(ROOT / "benchmark_corpus")):
    sys.path.insert(0, p)

import factor_cuda_pybind as fcb  # noqa: E402
import backends  # noqa: E402
from corpus_loader_v1 import load  # noqa: E402


def floor2(v: float) -> float:
    return math.floor(v * 100.0) / 100.0


def _median_ms(samples: list) -> float:
    # samples 已是毫秒（收集处 ×1000）；此处只取中位数，不再 ×1000（曾双乘致 1000× 单位错误）
    return statistics.median(samples)


def main() -> int:
    data, manifest = load("corpus_synth_v1")
    returns = np.ascontiguousarray(data["returns"], dtype=np.float32)
    mask = np.ascontiguousarray(data["mask"], dtype=bool)
    data_sha256 = manifest["hash"]["data_sha256"]

    evidence = {
        "schema_version": "1.1.0",
        "run_id": "stock_corr_general_gate_20260806",
        "panel": "corpus_synth_v1 returns[:, :N] + mask[:, :N]",
        "corpus_data_sha256": data_sha256,
        "T": T,
        "sizes": SIZES,
        "reps": REPS,
        "synthesis_note": SYNTHESIS_NOTE,
        "backend": "cupy + gpu(fcb binding)",
        "operations": {},
    }

    for N in SIZES:
        Xn = np.ascontiguousarray(returns[:, :N])
        Mn = np.ascontiguousarray(mask[:, :N])
        # 两端 warmup
        fcb.stock_corr_f64(Xn, Mn, False)
        backends.cp_stock_corr(Xn, Mn)
        # 交替批次（GPT-5.6-Sol 审查高11：降低固定顺序/GPU 温漂对边界裁决的偏差）
        gpu_ms, cp_ms = [], []
        for _ in range(REPS[N]):
            t0 = time.perf_counter()
            fcb.stock_corr_f64(Xn, Mn, False)
            gpu_ms.append((time.perf_counter() - t0) * 1000.0)
            t0 = time.perf_counter()
            backends.cp_stock_corr(Xn, Mn)
            cp_ms.append((time.perf_counter() - t0) * 1000.0)
        # 两端数值 parity（同 run 断言，审查高11：确认同面板两端语义一致）
        g_out = np.asarray(fcb.stock_corr_f64(Xn, Mn, False))
        c_out = np.asarray(backends.cp_stock_corr(Xn, Mn))
        finite = np.isfinite(g_out) & np.isfinite(c_out)
        max_abs_dr = float(np.max(np.abs(g_out[finite] - c_out[finite]))) \
            if finite.any() else 0.0
        nan_parity = bool(np.array_equal(np.isnan(g_out), np.isnan(c_out)))
        parity_ok = max_abs_dr <= 1e-12 and nan_parity

        gpu_med = _median_ms(gpu_ms)
        cp_med = _median_ms(cp_ms)
        speedup = cp_med / gpu_med
        evidence["operations"][f"stock_corr_general(N={N})"] = {
            "gpu_median_ms": gpu_med, "gpu_reps_ms": gpu_ms,
            "gpu_min_ms": min(gpu_ms), "gpu_max_ms": max(gpu_ms),
            "cupy_median_ms": cp_med, "cupy_reps_ms": cp_ms,
            "cupy_min_ms": min(cp_ms), "cupy_max_ms": max(cp_ms),
            "speedup_gpu_vs_cupy": speedup,
            "parity_ok": parity_ok, "max_abs_dr": max_abs_dr, "nan_parity": nan_parity,
            "verdict_2x": "PASS" if speedup >= 2.0 else "NRR_negative",
        }
        print(f"N={N}: gpu median {gpu_med:.4f} ms, cupy median {cp_med:.4f} ms, "
              f"speedup {speedup:.3f}x parity_ok={parity_ok} "
              f"-> {evidence['operations'][f'stock_corr_general(N={N})']['verdict_2x']}",
              flush=True)
        if not parity_ok:
            print(f"  WARN parity FAIL: max|dr|={max_abs_dr:.3e} "
                  f"nan_parity={nan_parity}", flush=True)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=1), encoding="utf-8")

    gates = {
        "schema_version": "1.1.0",
        "run_id": "stock_corr_general_gate_20260806",
        "panel": "corpus_synth_v1 returns[:, :N] + mask[:, :N]",
        "corpus_data_sha256": data_sha256,
        "scope": "stock_corr general-path same-panel gate (corpus returns+mask, "
                 "partial-validity panels)",
        "gates": {},
    }
    for N in SIZES:
        key = f"stock_corr_general(N={N})"
        raw = evidence["operations"][key]["cupy_median_ms"]
        gates["gates"][key] = {"raw_wall_ms": raw, "exact_half": raw / 2.0,
                               "display": floor2(raw / 2.0)}
    (RUN_DIR / "gate.json").write_text(
        json.dumps(gates, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"general gates written: {RUN_DIR / 'gate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
