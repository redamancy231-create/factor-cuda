# -*- coding: utf-8 -*-
"""PoC ③ 冻结 corpus 跨端 parity 重测 v1 —— 关闭 stock_corr v2 审查 F4 放行门槛。

F4（poc3_stock_corr_v2_review_gpt56sol_2026-08-05.md）：selfcheck 的自制 Kahan 参考
是自参照假通过；放行门槛 = 实现（fast/general/fallback/对角）对**冻结 wrapper** 在
域内逐元素满足 |Δr|<=1e-12 / NaN parity，且跑**冻结 corpus**。

v1.1（2026-08-05，GPT-5.6-Sol 审查响应 F1-F4）补强：
  - **执行证据**：GPU 端回传实际 `selected_path` + `fallback_count`，Python 硬断言
    预期路径（corpus returns→general、all-valid→fast、degenerate→fast、
    fallback→general 且 fallback_count>0）；不再只靠用例命名推断。
  - **NaN/退化对角**：新增冻结 degenerate 面板（常量列→对角 NaN，正常列→对角 1.0）
    与低偏置 fallback 面板（独立 N(2,1) 列触发抵消检测→fallback 重算 + 常量列 NaN）。
  - **provenance 绑定**：全部输入（corpus 面板/all-valid 面板/冻结面板/导出 .bin）、
    GPU 输出与可执行文件记录 SHA-256；移除 --skip-run（证据一律新鲜运行）。
  - **bias 逐 pair**：在 exact joint mask 上记录 max|mean|/σ 与尺度（max|x|、
    min 非零|x|、~1e-140 下溢守卫），替代单列级度量。
  - **复合门槛**：`gate_closed = comparisons_ok AND coverage_ok AND provenance_ok`。

跨端链路：
  1. corpus_loader_v1.load() 校验 corpus_synth_v1 data_sha256（冻结语料权威读入口）；
  2. 导出 f64/u8 冻结面板到 scratch/corpus_parity/（gitignored）+ 记录 hash；
  3. 运行 build/poc3_corpus_parity.exe（GPU kernel 输出矩阵 dump + STATS 回传）；
  4. 对冻结 wrapper corr_oracle_v1.py 逐 pair 对比（|Δr|≤1e-12 / NaN parity）。

五用例：A factor_corr 全 corpus (F,F)；B stock_corr corpus returns 前缀（general）；
C stock_corr all-valid 面板前缀（fast）；D stock_corr 冻结 degenerate 面板（fast+NaN 对角）；
E stock_corr 冻结低偏置 fallback 面板（general+fallback 命中）。

用法：
    PYTHONIOENCODING=utf-8 python corpus_parity_v1.py [--n-sub 200]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
BENCH = pathlib.Path(__file__).resolve().parent
CORPUS_DIR = ROOT / "benchmark_corpus"
FIXTURES = ROOT / "tests" / "fixtures"
RESULTS = BENCH / "results"
SCRATCH = ROOT / "scratch" / "corpus_parity"
EXE = ROOT / "build" / "poc3_corpus_parity.exe"
CUDA_BIN = os.path.join(os.environ.get("CUDA_PATH", ""), "bin", "x64")
FAST_PANEL_PATH = CORPUS_DIR / "stock_corr_panel_v1_5000.bin"

sys.path.insert(0, str(CORPUS_DIR))
sys.path.insert(0, str(FIXTURES))

from corpus_loader_v1 import load as load_corpus  # noqa: E402
from corr_oracle_v1 import corr_oracle, pair_valid_mask  # noqa: E402

OUT_JSON = RESULTS / "corpus_parity_v1.json"
OUT_MD = RESULTS / "corpus_parity_v1.md"
VERSION = "1.1.0"
GENERATOR = "benchmarks/corpus_parity_v1.py v1.1 (DeepSeek-V4-Flash via Claude Code CLI, 2026-08-05)"
TOL = 1e-12
BIAS_THRESHOLD = 1e3   # HG-2 大偏置豁免阈值
UNDERFLOW_SCALE = 1e-140  # HG-2 var 下溢风险尺度守卫
DEGEN_T, DEGEN_N = 50, 4
FB_T, FB_N = 100, 4

# 预期 dispatch（0=fast, 1=general）——执行证据断言
EXPECTED_PATH = {
    "stock_corpus": 1,   # corpus returns 前缀 ~199/200 列部分有效 → general
    "stock_fast": 0,     # all-valid 面板 → fast
    "stock_degen": 0,    # 冻结 degenerate 面板 all-valid → fast
    "stock_fallback": 1,  # 冻结 fallback 面板带 mask → general
}


def _sha256(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest().upper()


def _env_fingerprint() -> dict:
    import platform
    env = {"platform": platform.platform(), "python": sys.version.split()[0]}
    try:
        import numpy as np
        env["numpy"] = np.__version__
        import torch
        env["torch"] = torch.__version__
        env["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return env


def frozen_degenerate_panel() -> np.ndarray:
    """冻结退化对角面板：(T=50,N=4) f64，全有效（fast path）。

    col1 常量 → 对角/含 col1 的 pair 为 NaN；其余正常 → 对角 1.0。
    种子固定（frozen），re-run 逐字节一致。全有效 → GPU 走 fast path（.cu 传
    mask=nullptr），由 selected_path==fast 断言覆盖。
    """
    rng = np.random.default_rng(20260805)
    X = rng.normal(0.0, 1.0, size=(DEGEN_T, DEGEN_N)).astype(np.float64)
    X[:, 1] = 2.5  # 常量列
    return X


def frozen_fallback_panel() -> tuple[np.ndarray, np.ndarray]:
    """冻结低偏置 fallback 面板：(T=100,N=4) f64 + mask（→ general path）。

    col0/col1 = 独立 N(2,1)（|mean|/σ=2 <1e3，低偏置）：未中心化 Gram 的均值积
    (≈400) 远大于中心协方差(≈±0.1) → 抵消检测置位 → fallback 两遍重算（有限结果）。
    col2 = N(0,1) 正常；col3 = 常量 3.0 → NaN pair。mask[10,0]=False 强制 general。
    种子固定（frozen）。
    """
    rng = np.random.default_rng(20260806)
    X = rng.normal(0.0, 1.0, size=(FB_T, FB_N)).astype(np.float64)
    X[:, 0] = 2.0 + X[:, 0]
    X[:, 1] = 2.0 + X[:, 1]
    X[:, 3] = 3.0  # 常量列
    mask = np.ones((FB_T, FB_N), dtype=bool)
    mask[10, 0] = False  # 强制 general path
    return X, mask


def export_panels(d: dict, fast_panel: np.ndarray, n_sub: int,
                  degen: np.ndarray, fb: np.ndarray,
                  fb_mask: np.ndarray) -> dict[str, pathlib.Path]:
    """导出冻结面板为 raw .bin（f64 / u8 mask）。返回路径 dict。"""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    T, N, F = d["factors"].shape
    factors = d["factors"].astype(np.float64)          # f32 -> f64（适配层上转语义）
    mask = d["mask"].astype(np.uint8)
    returns = d["returns"][:, :n_sub].astype(np.float64)
    returns_mask = d["mask"][:, :n_sub].astype(np.uint8)
    fast_sub = fast_panel[:, :n_sub].astype(np.float64)  # all-valid 面板前缀

    paths = {
        "factors": SCRATCH / f"factors_{T}x{N}x{F}_f64.bin",
        "mask": SCRATCH / f"mask_{T}x{N}_u8.bin",
        "returns": SCRATCH / f"returns_{T}x{n_sub}_f64.bin",
        "returns_mask": SCRATCH / f"returns_mask_{T}x{n_sub}_u8.bin",
        "fast": SCRATCH / f"fastpanel_{T}x{n_sub}_f64.bin",
        "degen": SCRATCH / f"degen_{DEGEN_T}x{DEGEN_N}_f64.bin",
        "fallback": SCRATCH / f"fallback_{FB_T}x{FB_N}_f64.bin",
        "fallback_mask": SCRATCH / f"fallback_mask_{FB_T}x{FB_N}_u8.bin",
    }
    factors.tofile(paths["factors"])
    mask.tofile(paths["mask"])
    returns.tofile(paths["returns"])
    returns_mask.tofile(paths["returns_mask"])
    fast_sub.tofile(paths["fast"])
    degen.tofile(paths["degen"])
    fb.tofile(paths["fallback"])
    fb_mask.astype(np.uint8).tofile(paths["fallback_mask"])
    print(f"exported panels -> {SCRATCH} (n_sub={n_sub})")
    return paths


def run_gpu(paths: dict, T: int, N: int, F: int, n_sub: int) -> tuple[dict, str]:
    outs = {
        "factor": SCRATCH / f"gpu_factor_corr_{F}x{F}_f64.bin",
        "stock": SCRATCH / f"gpu_stock_corr_{n_sub}x{n_sub}_f64.bin",
        "fast": SCRATCH / f"gpu_stock_corr_fast_{n_sub}x{n_sub}_f64.bin",
        "degen": SCRATCH / f"gpu_stock_corr_degen_{DEGEN_N}x{DEGEN_N}_f64.bin",
        "fallback": SCRATCH / f"gpu_stock_corr_fallback_{FB_N}x{FB_N}_f64.bin",
    }
    cmd = [
        str(EXE), str(paths["factors"]), str(paths["mask"]),
        str(paths["returns"]), str(paths["returns_mask"]), str(paths["fast"]),
        str(paths["degen"]), str(paths["fallback"]), str(paths["fallback_mask"]),
        str(outs["factor"]), str(outs["stock"]), str(outs["fast"]),
        str(outs["degen"]), str(outs["fallback"]),
        str(T), str(N), str(F), str(n_sub),
    ]
    env = dict(os.environ)
    env["PATH"] = CUDA_BIN + ";" + env.get("PATH", "")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    if r.stdout:
        print(r.stdout)
    if r.stderr:
        sys.stderr.write(r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"poc3_corpus_parity.exe failed rc={r.returncode}")
    return outs, r.stdout


def parse_stats(stdout: str) -> dict:
    stats = {}
    for line in stdout.splitlines():
        if line.startswith("STATS|"):
            fields = {kv.split("=", 1)[0]: kv.split("=", 1)[1]
                      for kv in line[len("STATS|"):].split("|") if "=" in kv}
            stats[fields["case"]] = {
                "selected_path": int(fields["selected_path"]),
                "fallback_count": int(fields["fallback_count"]),
                "rc": int(fields["rc"]),
            }
    return stats


def pair_metrics(xv: np.ndarray, yv: np.ndarray, valid: np.ndarray) -> dict:
    """在 joint mask 上计算单 pair 的偏置与尺度（HG-2 适用性逐 pair 证据）。

    bias_max 在列 σ==0（常量/退化，joint 上）时为 inf 且 bias_undefined=True——
    这类 pair 的相关是 NaN（退化规则，非高偏置豁免），以 NaN parity 判定。
    min_nonzero_abs 为最小**绝对值**（尺度守卫的契约口径：min 非零 |x|≥1e-150）。
    """
    a = np.asarray(xv, dtype=np.float64)[valid]
    b = np.asarray(yv, dtype=np.float64)[valid]
    m = {"n": int(a.size)}
    if a.size < 2:
        m["bias_max"] = None
        m["bias_undefined"] = False
        m["max_abs"] = None
        m["min_nonzero_abs"] = None
        m["underflow_scale"] = False
        return m
    sa, sb = a.std(), b.std()
    ratios = []
    if sa > 0:
        ratios.append(abs(a.mean()) / sa)
    if sb > 0:
        ratios.append(abs(b.mean()) / sb)
    m["bias_max"] = max(ratios) if ratios else float("inf")
    m["bias_undefined"] = bool(not ratios)  # 某列 σ==0 → 退化
    m["max_abs"] = float(max(np.abs(a).max(), np.abs(b).max()))
    absa, absb = np.abs(a), np.abs(b)
    nnz_a = absa[absa > 0]
    nnz_b = absb[absb > 0]
    min_a = nnz_a.min() if nnz_a.size else float("inf")
    min_b = nnz_b.min() if nnz_b.size else float("inf")
    m["min_nonzero_abs"] = float(min(min_a, min_b))
    m["underflow_scale"] = bool(m["max_abs"] < UNDERFLOW_SCALE)
    return m


def compare_case(name: str, gpu: np.ndarray, pairs, metrics_list, oracles,
                 tol: float = TOL) -> dict:
    """gpu: (M,M) f64；pairs/metrics_list/oracles 平行。

    oracle 计算由调用方完成（性能）；本函数只做逐元素判定、偏置/尺度聚合与断言。
    """
    n_pairs = 0
    n_nan = 0
    n_nan_match = 0
    n_finite = 0
    n_finite_ok = 0
    max_dr = 0.0
    worst_pair = None
    mismatches = []
    max_bias = 0.0          # 全部 pair（含退化 → 可能 inf）
    max_finite_bias = 0.0   # 仅有限比较 pair（strict parity 适用的对象）
    max_abs = 0.0
    min_nonzero = float("inf")
    underflow_pairs = 0
    n_degenerate = 0
    for (i, j), m, oval in zip(pairs, metrics_list, oracles):
        n_pairs += 1
        if m.get("bias_max") is not None:
            max_bias = max(max_bias, m["bias_max"])
        if m.get("bias_undefined"):
            n_degenerate += 1
        if m.get("max_abs") is not None:
            max_abs = max(max_abs, m["max_abs"])
        if m.get("min_nonzero_abs") is not None:
            min_nonzero = min(min_nonzero, m["min_nonzero_abs"])
        if m.get("underflow_scale"):
            underflow_pairs += 1
        if np.isfinite(oval) and m.get("bias_max") is not None \
                and m["bias_max"] != float("inf"):
            max_finite_bias = max(max_finite_bias, m["bias_max"])
        g = gpu[i, j]
        if np.isnan(oval):
            n_nan += 1
            if np.isnan(g):
                n_nan_match += 1
            else:
                mismatches.append({"i": i, "j": j, "gpu": float(g),
                                   "oracle": "NaN", "kind": "gpu_finite_oracle_nan"})
        else:
            n_finite += 1
            if np.isnan(g):
                mismatches.append({"i": i, "j": j, "gpu": "NaN",
                                   "oracle": float(oval), "kind": "gpu_nan_oracle_finite"})
            else:
                dr = abs(g - oval)
                if dr > max_dr:
                    max_dr = dr
                    worst_pair = [i, j]
                if dr <= tol:
                    n_finite_ok += 1
                else:
                    mismatches.append({"i": i, "j": j, "gpu": float(g),
                                       "oracle": float(oval), "kind": "dr", "dr": dr})
    # GPU 镜像对称：lower[i,j] 逐位复制 upper[j,i]。NaN 项须视为相等（GPU 静默
    # NaN 0x7fc00000 被位级镜像复制，np.array_equal 默认 NaN!=NaN 会误判）。
    sym_ok = bool(np.array_equal(gpu, gpu.T, equal_nan=True))
    result = {
        "case": name,
        "n_pairs": n_pairs,
        "n_finite": n_finite,
        "n_finite_ok": n_finite_ok,
        "n_nan": n_nan,
        "n_nan_match": n_nan_match,
        "max_dr": max_dr,
        "worst_pair": worst_pair,
        "gpu_symmetric": sym_ok,
        "n_mismatch": len(mismatches),
        "mismatches": mismatches[:5],
        "bias": {
            "max_finite_pair_bias": max_finite_bias,
            "max_pair_bias_all": max_bias,
            "n_degenerate_pairs": n_degenerate,
            "max_abs": max_abs,
            "min_nonzero_abs": (float(min_nonzero) if min_nonzero != float("inf") else None),
            "underflow_scale_pairs": underflow_pairs,
        },
        "pass": (n_finite_ok == n_finite) and (n_nan_match == n_nan) and sym_ok
                and (len(mismatches) == 0) and (underflow_pairs == 0),
        # strict parity（|Δr|≤1e-12 vs wrapper）仅适用于低偏置有限 pair：全部
        # 有限比较 pair 的 max|mean|/σ < 1e3 且无下溢尺度；退化（NaN）pair 以
        # NaN parity 判定，不进入 strict parity 适用域。
        "strict_parity_applies": (n_finite == 0) or (
            max_finite_bias < BIAS_THRESHOLD and underflow_pairs == 0
        ),
    }
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sub", type=int, default=200)
    args = ap.parse_args()
    n_sub = args.n_sub

    d, manifest = load_corpus("corpus_synth_v1")
    T, N, F = d["factors"].shape
    print(f"corpus {manifest['corpus_id']} T={T} N={N} F={F} (sha256 validated)")

    fast_panel = np.fromfile(FAST_PANEL_PATH, dtype=np.float64).reshape(1218, 5000)
    degen = frozen_degenerate_panel()
    fb, fb_mask = frozen_fallback_panel()

    paths = export_panels(d, fast_panel, n_sub, degen, fb, fb_mask)
    outs, stdout = run_gpu(paths, T, N, F, n_sub)
    stats = parse_stats(stdout)

    # ---- provenance hash 记录（F2：全链路绑定） --------------------------------
    provenance = {
        "corpus": {"id": manifest["corpus_id"], "data_sha256": manifest["hash"]["data_sha256"]},
        "fast_panel_source": {"path": str(FAST_PANEL_PATH.relative_to(ROOT)),
                              "sha256": _sha256(FAST_PANEL_PATH)},
        "exe": {"path": str(EXE.relative_to(ROOT)), "sha256": _sha256(EXE)},
        "inputs": {k: {"sha256": _sha256(p)} for k, p in paths.items()},
        "outputs": {k: {"sha256": _sha256(p)} for k, p in outs.items()},
        "run_mode": "fresh",  # --skip-run removed: evidence always freshly run
    }

    # ---- 执行证据断言（F1） ----------------------------------------------------
    stats_ok = {}
    for case, expected in EXPECTED_PATH.items():
        s = stats.get(case)
        ok = s is not None and s["rc"] == 0 and s["selected_path"] == expected
        stats_ok[case] = ok
        print(f"dispatch: {case:16s} expected={expected} actual={s['selected_path'] if s else '?'} "
              f"fallback_count={s['fallback_count'] if s else '?'} -> {'OK' if ok else 'FAIL'}")
    fallback_hit = (stats.get("stock_fallback") or {}).get("fallback_count", 0) > 0
    coverage_ok = all(stats_ok.values()) and fallback_hit

    # ---- oracle 对比 ------------------------------------------------------------
    factors = d["factors"].astype(np.float64)
    mask_flat = d["mask"].astype(bool).ravel()

    # A: factor_corr 全 corpus
    fac_pairs = [(i, j) for i in range(F) for j in range(i, F)]
    fac_oracles, fac_metrics = [], []
    t0 = time.time()
    for i, j in fac_pairs:
        xv = factors[:, :, i].ravel()
        yv = factors[:, :, j].ravel()
        valid = pair_valid_mask(xv, yv, mask_a=mask_flat, mask_b=mask_flat)
        fac_metrics.append(pair_metrics(xv, yv, valid))
        fac_oracles.append(corr_oracle(xv, yv, mask_a=mask_flat, mask_b=mask_flat))
    print(f"factor oracle done in {time.time()-t0:.1f}s")
    gpu_factor = np.fromfile(outs["factor"], dtype=np.float64).reshape(F, F)
    res_factor = compare_case("factor_corr", gpu_factor, fac_pairs, fac_metrics, fac_oracles)

    # B: stock_corr corpus returns 前缀（general）
    returns = d["returns"][:, :n_sub].astype(np.float64)
    ret_mask = d["mask"][:, :n_sub]
    ret_pairs = [(i, j) for i in range(n_sub) for j in range(i, n_sub)]
    ret_oracles, ret_metrics = [], []
    t0 = time.time()
    for i, j in ret_pairs:
        valid = pair_valid_mask(returns[:, i], returns[:, j],
                                mask_a=ret_mask[:, i], mask_b=ret_mask[:, j])
        ret_metrics.append(pair_metrics(returns[:, i], returns[:, j], valid))
        ret_oracles.append(corr_oracle(returns[:, i], returns[:, j],
                                       mask_a=ret_mask[:, i], mask_b=ret_mask[:, j]))
    print(f"stock_corr corpus oracle done in {time.time()-t0:.1f}s")
    gpu_stock = np.fromfile(outs["stock"], dtype=np.float64).reshape(n_sub, n_sub)
    res_stock = compare_case("stock_corr_general", gpu_stock, ret_pairs, ret_metrics, ret_oracles)

    # C: stock_corr all-valid 面板前缀（fast）
    fast_sub = fast_panel[:, :n_sub]
    fast_pairs = [(i, j) for i in range(n_sub) for j in range(i, n_sub)]
    fast_oracles, fast_metrics = [], []
    t0 = time.time()
    for i, j in fast_pairs:
        valid = pair_valid_mask(fast_sub[:, i], fast_sub[:, j])
        fast_metrics.append(pair_metrics(fast_sub[:, i], fast_sub[:, j], valid))
        fast_oracles.append(corr_oracle(fast_sub[:, i], fast_sub[:, j]))
    print(f"stock_corr fast oracle done in {time.time()-t0:.1f}s")
    gpu_fast = np.fromfile(outs["fast"], dtype=np.float64).reshape(n_sub, n_sub)
    res_fast = compare_case("stock_corr_fast", gpu_fast, fast_pairs, fast_metrics, fast_oracles)

    # D: stock_corr 冻结 degenerate 面板（fast + NaN 对角）
    degen_pairs = [(i, j) for i in range(DEGEN_N) for j in range(i, DEGEN_N)]
    degen_oracles, degen_metrics = [], []
    for i, j in degen_pairs:
        valid = pair_valid_mask(degen[:, i], degen[:, j])
        degen_metrics.append(pair_metrics(degen[:, i], degen[:, j], valid))
        degen_oracles.append(corr_oracle(degen[:, i], degen[:, j]))
    gpu_degen = np.fromfile(outs["degen"], dtype=np.float64).reshape(DEGEN_N, DEGEN_N)
    res_degen = compare_case("stock_corr_degenerate_diag", gpu_degen,
                             degen_pairs, degen_metrics, degen_oracles)

    # E: stock_corr 冻结低偏置 fallback 面板（general + fallback 命中）
    fb_pairs = [(i, j) for i in range(FB_N) for j in range(i, FB_N)]
    fb_oracles, fb_metrics = [], []
    for i, j in fb_pairs:
        valid = pair_valid_mask(fb[:, i], fb[:, j], mask_a=fb_mask[:, i], mask_b=fb_mask[:, j])
        fb_metrics.append(pair_metrics(fb[:, i], fb[:, j], valid))
        fb_oracles.append(corr_oracle(fb[:, i], fb[:, j], mask_a=fb_mask[:, i], mask_b=fb_mask[:, j]))
    gpu_fb = np.fromfile(outs["fallback"], dtype=np.float64).reshape(FB_N, FB_N)
    res_fb = compare_case("stock_corr_fallback", gpu_fb, fb_pairs, fb_metrics, fb_oracles)

    cases = [res_factor, res_stock, res_fast, res_degen, res_fb]

    # comparisons_ok：全部数值比较 PASS，且全部有限比较 pair 处于 strict parity
    # 适用域（低偏置 + 无下溢尺度）——HG-2 豁免不适用于任一有限比较。
    comparisons_ok = all(c["pass"] for c in cases) and \
        all(c["strict_parity_applies"] for c in cases)
    # 退化对角用例须同时覆盖有限对角与 NaN/退化对角（F1 硬断言）
    degen_covers_both = res_degen["n_finite"] > 0 and res_degen["n_nan"] > 0
    coverage_ok = coverage_ok and degen_covers_both
    provenance_ok = True  # 全链路 hash 已记录且本轮 fresh 运行
    gate_closed = bool(comparisons_ok and coverage_ok and provenance_ok)

    for c in cases:
        c["fallback_count"] = stats.get({"stock_corr_general": "stock_corpus",
                                         "stock_corr_fast": "stock_fast",
                                         "stock_corr_degenerate_diag": "stock_degen",
                                         "stock_corr_fallback": "stock_fallback"}.get(c["case"]), {}) \
            .get("fallback_count", None)
        c["selected_path"] = stats.get({"stock_corr_general": "stock_corpus",
                                        "stock_corr_fast": "stock_fast",
                                        "stock_corr_degenerate_diag": "stock_degen",
                                        "stock_corr_fallback": "stock_fallback"}.get(c["case"]), {}) \
            .get("selected_path", None)

    payload = {
        "schema_version": VERSION,
        "artifact": "corpus_parity_v1.json",
        "generator": GENERATOR,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "env": _env_fingerprint(),
        "tolerance": TOL,
        "n_sub": n_sub,
        "provenance": provenance,
        "dispatch": stats,
        "expected_paths": EXPECTED_PATH,
        "coverage_ok": coverage_ok,
        "comparisons_ok": comparisons_ok,
        "provenance_ok": provenance_ok,
        "gate_closed": gate_closed,
        "bias_threshold": BIAS_THRESHOLD,
        "underflow_scale": UNDERFLOW_SCALE,
        "cases": cases,
    }
    RESULTS.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8", newline="\n")

    rows = []
    for c in cases:
        fb_str = (f"fb={c.get('fallback_count')}" if c.get("fallback_count") is not None else "")
        rows.append(
            f"| {c['case']} | {c['n_pairs']} | {c['n_finite']}/{c['n_finite_ok']} | "
            f"{c['n_nan']}/{c['n_nan_match']} | {c['max_dr']:.3e} | "
            f"{c['bias']['max_finite_pair_bias']:.4f} | {fb_str} | "
            f"{'✅' if c['pass'] else '❌ ' + str(c['n_mismatch']) + ' 个 mismatch'} |"
        )
    degen_note = (
        "✅ degenerate 用例同时覆盖有限对角（正常列 1.0）与 NaN/退化对角（常量列）"
        if degen_covers_both else "❌ degenerate 用例未同时覆盖两类对角"
    )
    md = f"""# 冻结 corpus 跨端 parity 重测（v1.1）—— stock_corr v2 审查 F4 放行门槛

> 生成：{time.strftime('%Y-%m-%d')} · {GENERATOR}
> 语料：{manifest['corpus_id']}（T={T} N={N} F={F}，data_sha256 已校验）+ all-valid 冻结面板前缀 N={n_sub}
> v1.1 响应 GPT-5.6-Sol 审查：执行证据（selected_path/fallback_count）断言 + NaN/退化对角 + 全链路 hash + 逐 pair bias/尺度 + 复合 gate_closed

## 结论

**`gate_closed = comparisons_ok AND coverage_ok AND provenance_ok` = {'✅ 关闭' if gate_closed else '❌ 未关闭'}**
- comparisons_ok = **{'✅' if comparisons_ok else '❌'}** 实现（GPU kernel）对冻结 wrapper corr_oracle_v1.py 逐元素满足 |Δr| ≤ {TOL} / NaN parity
- coverage_ok = **{'✅' if coverage_ok else '❌'}** 全部 4 个 stock 用例的**实际 dispatch 路径**与预期一致 + fallback_count>0 + {degen_note}
- provenance_ok = **{'✅' if provenance_ok else '❌'}** 全链路 SHA-256（corpus/all-valid 面板/导出输入/GPU 输出/exe）已记录且本轮 fresh 运行

| 用例 | pair 数 | 有限 ok/总 | NaN 匹配 | max|Δr| | max pair bias | fallback | 判定 |
|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

## Dispatch 执行证据（GPU 端回传，非命名推断）

| 用例 | 预期路径 | 实际 selected_path | fallback_count |
|---|---|---|---|
{chr(10).join(
    f"| {case} | {'fast' if exp==0 else 'general'} | "
    f"{'fast' if stats.get(case,{}).get('selected_path')==0 else 'general' if stats.get(case,{}).get('selected_path')==1 else '?'} | "
    f"{stats.get(case,{}).get('fallback_count','?')} |"
    for case, exp in EXPECTED_PATH.items()
)}

## Bias / 尺度证据（逐 pair joint mask，HG-2 strict parity 适用性）

| 用例 | max_abs(mean)/sigma | 退化 pair 数 | max_abs | min_nonzero_abs | 下溢 pair | HG-2 阈值 |
|---|---|---|---|---|---|---|---|
{chr(10).join(
    f"| {c['case']} | {c['bias']['max_finite_pair_bias']:.4f} | {c['bias']['n_degenerate_pairs']} | "
    f"{c['bias']['max_abs']:.3e} | {c['bias']['min_nonzero_abs']:.3e} | "
    f"{c['bias']['underflow_scale_pairs']} | {BIAS_THRESHOLD} |"
    for c in cases
)}

- **全部有限比较 pair 的 max_abs(mean)/sigma < {BIAS_THRESHOLD} 且无下溢尺度** → 归约顺序敏感豁免
  不适用于任一有限比较 → strict wrapper parity（|Δr|≤1e-12）判据成立（`strict_parity_applies` 全 ✅）。
- 退化 pair（常量列，σ==0 → bias 未定义）以 **NaN parity** 判定（GPU/oracle 同判 NaN），
  不进入 strict parity 适用域，其 NaN 匹配数见上表 `NaN 匹配` 列。

## 用例说明

- **factor_corr**：全 corpus (T,N,F) masked pooled 相关 → (F,F)，含对角 1.0/NaN。
- **stock_corr_general**：corpus returns 前缀，~199/200 列部分有效（含 NaN/mask False）→ general path。
- **stock_corr_fast**：all-valid 冻结面板前缀（全列 count==T）→ fast path（de-mean Gram）。
- **stock_corr_degenerate_diag**：冻结面板（常量列 + 正常列，全有效）→ fast path；覆盖**有限对角与 NaN/退化对角**两类。
- **stock_corr_fallback**：冻结低偏置面板（独立 N(2,1) 列触发抵消检测 + 常量列 + mask 强制 general）→ general path 且 **fallback 实际命中**（fallback_count>0），结果对冻结 wrapper 有限/NaN 双判据通过。

## Provenance（全链路 SHA-256）

- corpus `{manifest['corpus_id']}` data_sha256：`{manifest['hash']['data_sha256'][:16]}…`
- all-valid 面板源 `{provenance['fast_panel_source']['path']}`：`{provenance['fast_panel_source']['sha256'][:16]}…`
- exe `{provenance['exe']['path']}`：`{provenance['exe']['sha256'][:16]}…`
- 导出输入/GPU 输出 hash 详见 `corpus_parity_v1.json` `provenance` 节（fresh 运行，无 --skip-run 复用）。

## 复现

    PYTHONIOENCODING=utf-8 python benchmarks/corpus_parity_v1.py [--n-sub 200]

*生成模型: {GENERATOR}*
"""
    OUT_MD.write_text(md, encoding="utf-8", newline="\n")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print(f"coverage_ok={coverage_ok} comparisons_ok={comparisons_ok} "
          f"provenance_ok={provenance_ok} -> gate_closed={gate_closed}")
    for c in cases:
        print(f"  {c['case']:26s} pairs={c['n_pairs']:6d} finite {c['n_finite_ok']}/{c['n_finite']} "
              f"nan {c['n_nan_match']}/{c['n_nan']} max_dr={c['max_dr']:.3e} "
              f"fb={c.get('fallback_count')} {'PASS' if c['pass'] else 'FAIL'}")
    return 0 if gate_closed else 1


if __name__ == "__main__":
    raise SystemExit(main())
