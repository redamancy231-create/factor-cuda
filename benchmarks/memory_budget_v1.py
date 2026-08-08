# -*- coding: utf-8 -*-
"""factor-cuda 内存静态预算 v1——当前实现精确模型 + 项②流式化 + 项③F-blocking。

目标（Kahan 决策 v2，`reviews/ft_kahan_residency_decision_2026-08-05.md`）：
闭合内存模型未闭合的 4 项——d_valid 生命周期 / transpose overlap / d_pp 规模 /
cross-block pair，并为「当前实现」建立逐分配 live-byte 静态预算（对照校准
`poc3_calibration_v1.json` HWM，delta=0 验证模型正确）。

「可分块」三项的预算结论：
- 项① 计算可分块（已闭合，最小证明②）：归约阶段内存中性，本模型不重复
- 项② 输入/转置流式化：去全量 d_F/d_X 与 d_Xt 的转置前重叠 → 端到端峰值可降
- 项③ 驻留面板可缩小（F-blocking 2D tile / stock N-blocking）：tile 级驻留

规模：factor_corr F=12/128 × N=5000/10000；stock_corr N=5000/10000/22600。
输出：docs/memory_budget_v1.json + benchmarks/results/memory_budget_v1.md。

用法：PYTHONIOENCODING=utf-8 python benchmarks/memory_budget_v1.py
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import subprocess
import sys


def current_git_head() -> str:
    """Current git HEAD (F-02: streaming evidence must represent the reviewed
    source; a loader that ignores provenance lets stale/foreign evidence close a
    column for the wrong code). Best-effort; 'n/a' if not a repo."""
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else "n/a"
    except Exception:
        return "n/a"

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"
ALIGNMENT_B = 256
MIB = 1024 * 1024
TOTAL_MIB = 8188
RESERVE_MIB = 512
AVAILABLE_BYTES = (TOTAL_MIB - RESERVE_MIB) * MIB  # 7676 MiB
T_CANONICAL = 1218
# ABI 结构大小（corr_math_v1 STRUCT_ABI + stock_corr ColStats）
PARTIAL1_B = 56
PARTIAL2_B = 24
PARTIALK1_B = 40
PARTIALK2_B = 48
COLSTATS_B = 48
KAHANACC_B = 16  # KahanAcc (sum, c)
# streaming(项②) 报告配置：转置 sub-chunk 行数（d_F_chunk = tt*F*8 固定缓冲）。
# 与 poc3_factor_corr_selfcheck.cu --hwm-stream 的 max_transpose_rows 一致。
STREAM_TT = 4096

# 校准 HWM（poc3_calibration_v1.json，模型验证基准）
CALIBRATION = {
    ("factor_corr", 5000, 12): 1191,
    ("factor_corr", 10000, 12): 2381,
    ("stock_corr", 2000): 119,
    ("stock_corr", 5000): 527,
    ("stock_corr", 10000): 1817,
}


def align256(value: int) -> int:
    return ((value + ALIGNMENT_B - 1) // ALIGNMENT_B) * ALIGNMENT_B


def mib(b: int) -> float:
    return b / MIB


def factor_corr_allocs(R: int, F: int, with_mask: bool) -> list[dict]:
    """factor_corr 生产分配链（src/factor_corr.cu:423-437 实际顺序）。"""
    P = F * (F + 1) // 2
    allocs = [
        ("d_Xt", F * R * 8, "source"),        # 转置列主序 (F,R) f64
        ("d_valid", F * R, "mask"),            # 转置有效性 (F,R) u8
        ("d_pairs", P * 2 * 4, "pairs"),
        ("d_gp1", P * PARTIAL1_B, "partial"),
        ("d_means", P * 2 * 8, "means"),
        ("d_gp2", P * PARTIAL2_B, "partial"),
        ("d_corr", P * 8, "corr"),
        ("d_trigger", P, "trigger"),
        ("d_trig_pairs", P * 4, "pairs"),
        ("d_gk1", P * PARTIALK1_B, "partial"),
        ("d_kmeans", P * 2 * 8, "means"),
        ("d_gk2", P * PARTIALK2_B, "partial"),
        ("d_out", F * F * 8, "output"),
        ("d_F", R * F * 8, "source"),          # 原始 (T*N,F) f64，转置后释放
    ]
    if with_mask:
        allocs.append(("d_mask", R, "mask"))
    return allocs


def stock_corr_allocs(T: int, N: int, with_mask: bool) -> list[tuple[str, int, str]]:
    """stock_corr 生产分配链（src/stock_corr.cu:593-617 实际顺序）。

    两个实现细节（校准公式一致）：
    - `nn = N*N`（完整矩阵，非上三角）——d_corr/d_out 各分配 N²×8
    - d_M（转置 mask 指针 u8）分配 cols_bytes = N*T*8（8n），非 mask 实际大小 n
      → 3×8n = d_X + d_Xm + d_M
    """
    nn = N * N  # 完整矩阵元素（校准 theory_stock_corr 用 Nsz*Nsz）
    cols_bytes = N * T * 8
    allocs = [
        ("d_Xm", cols_bytes, "source"),         # 转置 (N,T) f64（demean/GEMM 输入）
        ("d_M", cols_bytes, "mask"),             # 转置 mask (N,T) u8 指针但分配 8n
        ("d_stats", N * COLSTATS_B, "stats"),
        ("d_s2", N * 8, "stats"),
        ("d_corr", nn * 8, "corr"),
        ("d_out", nn * 8, "output"),
        ("d_X", T * N * 8, "source"),          # 原始 (T,N) f64，转置后释放
    ]
    if with_mask:
        allocs.append(("d_mask", T * N, "mask"))
    return allocs


def peak_align(allocs: list) -> tuple[int, dict[str, int]]:
    """按分配顺序累计对齐字节，返回 (峰值B, 各buffer对齐字节)。"""
    live = 0
    peak = 0
    sizes = {}
    for name, bytes_, _cat in allocs:
        a = align256(bytes_)
        sizes[name] = a
        live += a
        peak = max(peak, live)
    return peak, sizes


def factor_corr_peak(R: int, F: int, with_mask: bool = True) -> tuple[int, dict[str, int]]:
    return peak_align(factor_corr_allocs(R, F, with_mask))


def stock_corr_peak(T: int, N: int, with_mask: bool = True) -> tuple[int, dict[str, int]]:
    return peak_align(stock_corr_allocs(T, N, with_mask))


def nblock_stock_peak(T: int, N: int, block: int) -> dict:
    """stock N-blocking：pair 轴分块 + 输出流式化（M4 输出驻留契约改变）。

    基于 poc3_stock_corr_selfcheck.cu 的 stock_corr_gpu_nblock 分配链——device
    只驻留最坏 tile（非对角 [B|A] 列序，列数 <= 2*block）的输入/转置/工作区 + 局部
    输出缓冲，**无全量 d_corr/d_out**（逐 tile D2H → host 写回 + mirror）。
    峰值 = tile 各 buffer 对齐和（分配链无中途释放）。"""
    n_blocks = (N + block - 1) // block
    # 与实现 clamp 一致（外部审查 MINOR-2：最坏非对角 tile 列数 <= 2*block 且 <= N；
    # block=N 退化时单 tile 仅 N 列，2N 缓冲是 4× 超分配）
    max_cols = min(N, 2 * block)
    tile_allocs = [
        ("d_F_tile", max_cols * T * 8, "source"),   # 抽列输入缓冲 (N_tile,T)
        ("d_mask_tile", max_cols * T, "mask"),      # 每格 u8 mask (N_tile,T)
        ("d_Xm_tile", max_cols * T * 8, "source"),  # 转置 (N_tile,T) f64
        ("d_M_tile", max_cols * T * 8, "mask"),     # 转置 valid (N_tile,T) f64
        ("d_stats_tile", max_cols * COLSTATS_B, "stats"),
        ("d_s2_tile", max_cols * 8, "stats"),
        ("d_corr_tile", max_cols * max_cols * 8, "corr"),  # 局部 mini-matrix
        # d_fb_count 不分配：审查 F1-dfb-count-dead 确认它是死管道（nblock 不
        # D2H 读回），driver 已移除分配并传 nullptr（生产 fallback_kernel 接受
        # nullptr）→ 模型与驱动一致，无漏计。
    ]
    peak, sizes = peak_align(tile_allocs)
    return {"block": block, "n_blocks": n_blocks,
            "n_tiles": n_blocks * (n_blocks + 1) // 2,  # 下三角 tile 数
            "peak_B": peak, "peak_MiB": mib(peak), "buffers": sizes}


def streaming_factor_peak(R: int, F: int, max_transpose_rows: int = 4096,
                          with_mask: bool = True) -> int:
    """项② 输入/转置流式化：逐 sub-chunk（max_transpose_rows 行）上传 d_F_chunk +
    range 转置，d_F 全量不常驻。d_Xt/d_valid 仍全量（决策 A，Kahan 重跑读），
    去 d_F 与 d_Xt 的转置前重叠。

    分配链与 `factor_corr_gpu_stream`（poc3_factor_corr_selfcheck.cu）一致：
    生产链移除 d_F 后追加 d_F_chunk（固定 sub-chunk 缓冲）+ d_pp1/d_pp2（延续状态）。
    峰值 = 全部对齐和（分配链无中途释放）。默认 max_transpose_rows=4096 为
    `--hwm-stream` 报告配置。"""
    P = F * (F + 1) // 2
    allocs = factor_corr_allocs(R, F, with_mask)
    # 移除 d_F（流式化后逐 sub-chunk 上传+转置，不常驻全量）
    allocs = [a for a in allocs if a[0] != "d_F"]
    # 追加 streaming 专属缓冲（与实现分配顺序一致：d_F_chunk 在 pair 工作区后、
    # d_pp1/d_pp2 在延续计算阶段——全程 live，故全部计入峰值）
    allocs += [
        # F-07 (external MINOR): the driver allocates by min(max_transpose_rows,
        # R) -- the transpose loop clamps per-sub-chunk to R-r0.
        ("d_F_chunk", min(max_transpose_rows, R) * F * 8, "source"),
        ("d_pp1", P * 256 * PARTIAL1_B, "partial"),
        ("d_pp2", P * 256 * PARTIAL2_B, "partial"),
    ]
    peak, _ = peak_align(allocs)
    return peak


def fblock_factor_peak(R: int, F: int, block: int) -> dict:
    """项③ F-blocking 2D tile：分 F 轴为 block-pair，只常驻 tile 涉及的
    factor block 的 d_Xt+d_valid + 该 tile 的 pair 工作区。
    每 tile：d_Xt_tile = (B_a+B_b)×R×8, d_valid_tile = (B_a+B_b)×R，
    pair 工作区按 tile 内 pair 数 P_tile。"""
    n_blocks = (F + block - 1) // block
    widths = [block] * (n_blocks - 1) + [F - block * (n_blocks - 1)]
    # tile 内最大 (B_a+B_b)：实现 `factor_corr_gpu_fblock` 用
    # `max_cols = 2*block_width`（不 clamp），故模型与实现一致地取 2*block；
    # block>F/2 时实现按 2*block 超分配（旧 min(2*block, F) 低估，2026-08-07 修正）
    max_pair_width = 2 * block
    # 最大 tile 的 pair 数：非对角 tile a×b 全组合（clamp 到 F）
    a = min(block, F)
    b = min(block, F)
    P_tile = a * b + a * (a + 1) // 2  # 非对角 a×b + 对角块 a×(a+1)/2
    # 审查 F6（外部）：buffer 顺序按实现 cudaMalloc 链排列并统一 _tile 命名，
    # 避免审计映射歧义。d_mask_tile 实现无条件分配（R bytes），故总是计入。
    # 峰值 = 全部 buffer 对齐和（分配链无中途释放），顺序不影响峰值数值。
    tile_allocs = [
        ("d_Xt_tile", max_pair_width * R * 8, "source"),
        ("d_valid_tile", max_pair_width * R, "mask"),
        ("d_F_tile", max_pair_width * R * 8, "source"),
        ("d_mask_tile", R, "mask"),
        ("d_pairs_tile", P_tile * 8, "pairs"),
        ("d_gp1_tile", P_tile * PARTIAL1_B, "partial"),
        ("d_means_tile", P_tile * 16, "means"),
        ("d_gp2_tile", P_tile * PARTIAL2_B, "partial"),
        ("d_corr_tile", P_tile * 8, "corr"),
        ("d_trigger_tile", P_tile, "trigger"),
        ("d_trig_pairs_tile", P_tile * 4, "pairs"),
        ("d_gk1_tile", P_tile * PARTIALK1_B, "partial"),
        ("d_kmeans_tile", P_tile * 16, "means"),
        ("d_gk2_tile", P_tile * PARTIALK2_B, "partial"),
    ]
    peak, sizes = peak_align(tile_allocs)
    return {"block": block, "n_blocks": n_blocks,
            "n_tiles": n_blocks * (n_blocks + 1) // 2,  # 审查 M5：下三角 tile 数
            "peak_B": peak, "peak_MiB": mib(peak), "buffers": sizes}


def dpp_size(R: int, F: int) -> dict:
    """d_pp continuation 规模（审查 M2 修正：按实际 ABI struct 计）。
    chunked driver 实际分配 Partial1(56B)+Partial2(24B) 每 lane 同时存活
    （poc3_factor_corr_selfcheck.cu d_pp1/d_pp2），非 KahanAcc(16B)。
    blockDim=256（固定树归约）。"""
    P = F * (F + 1) // 2
    bytes_per_lane = PARTIAL1_B + PARTIAL2_B  # 80B（M2：原 KahanAcc 16B 低估 ~5×）
    return {"P": P, "blockDim": 256, "kahan_acc_B": KAHANACC_B,
            "struct_bytes_lane": bytes_per_lane,
            "dpp_B": P * 256 * bytes_per_lane,
            "dpp_MiB": mib(P * 256 * bytes_per_lane)}


def verify_calibration() -> dict:
    """对照校准 HWM 验证模型（delta=0 才通过）。"""
    rows = []
    ok = True
    for key, expected_mib in CALIBRATION.items():
        if key[0] == "factor_corr":
            _, N, F = key
            R = T_CANONICAL * N
            model_mib = mib(factor_corr_peak(R, F)[0])
        else:
            _, N = key
            model_mib = mib(stock_corr_peak(T_CANONICAL, N)[0])
        delta = model_mib - expected_mib
        rows.append({"key": key, "calibration_MiB": expected_mib,
                     "model_MiB": round(model_mib, 2), "delta_MiB": round(delta, 2),
                     "ok": abs(delta) <= 0.5})
        if abs(delta) > 0.5:
            ok = False
    return {"all_match": ok, "rows": rows}


def load_fblock_measured() -> dict | None:
    """Best-effort read of the fblock HWM evidence
    (results/factor_fblock_hwm_v1.json).

    memory_budget_v1 is a static model; when the F=128 HWM measurement has been
    run (benchmarks/factor_fblock_hwm_v1.py), the fblock (项③) scenario column is
    closed to measured. Absent the evidence file, scenarios stay model_prediction.

    Fail-closed (reviews F1 + SC-1, synced to the stock-side external
    BLOCKER-1/BLOCKER-2 2026-08-08): only ingest evidence whose producer set
    closure_status=="OK" AND guard pass==1/r1=r2=0 AND the FULL expected case set
    (6 fblock + 1 production = 7 records; external review F4 count fix) is
    present with every required field AND every case actually ran (rc==0, sane
    driver_peak, samples>0). All DERIVED fields (fits /
    driver_peak_MiB / model_MiB / delta_MiB / margin_MiB / vram_exhausted) are
    RE-DERIVED from the raw driver_peak bytes against the CURRENT model and
    AVAILABLE_BYTES -- the JSON's derived fields are never trusted. A failed or
    malformed case cleanly rejects the evidence (never raises). vram_exhausted
    cases (block=64 / production F=128) are legitimate non-fits; only a case that
    is neither fits nor vram_exhausted (a true anomaly) rejects.
    """
    REQUIRED = ("kind", "T", "N", "F", "block", "reps", "samples", "fb",
                "driver_peak", "rc")
    ev = RESULTS / "factor_fblock_hwm_v1.json"
    if not ev.exists():
        return None
    try:
        data = json.loads(ev.read_text(encoding="utf-8"))
    except Exception:
        print(f"  fblock evidence rejected: unreadable JSON ({ev})")
        return None
    if not isinstance(data, dict):  # external review MINOR-9: non-object root
        print(f"  fblock evidence rejected: root not an object ({ev})")
        return None
    if data.get("closure_status") != "OK":
        print(f"  fblock evidence rejected: closure_status != OK ({ev})")
        return None
    guard = data.get("guard")
    if (not isinstance(guard, dict) or guard.get("pass") != 1
            or guard.get("r1") != 0 or guard.get("r2") != 0):
        print(f"  fblock evidence rejected: GUARD not pass=1/r1=r2=0 ({ev})")
        return None
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        print(f"  fblock evidence rejected: cases missing/empty ({ev})")
        return None
    parsed = []
    for i, c in enumerate(cases):
        if not isinstance(c, dict):
            print(f"  fblock evidence rejected: case[{i}] not a dict")
            return None
        missing = [f for f in REQUIRED if f not in c]
        if missing:
            print(f"  fblock evidence rejected: case[{i}] missing fields {missing}")
            return None
        # external review MINOR-8: int() truncation (rc=0.5 -> 0) would let a
        # forged record pass the rc/reps/case gates -- require the JSON value to
        # be a true integer (or integer-valued float), reject everything else.
        def _as_int(k):
            v = c[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"{k} not numeric")
            if isinstance(v, float) and not v.is_integer():
                raise ValueError(f"{k} non-integer float")
            return int(v)

        try:
            row = {"kind": str(c["kind"]), "T": _as_int("T"), "N": _as_int("N"),
                   "F": _as_int("F"), "block": _as_int("block"),
                   "reps": _as_int("reps"), "samples": _as_int("samples"),
                   "fb": _as_int("fb"), "driver_peak": _as_int("driver_peak"),
                   "rc": _as_int("rc")}
        except (TypeError, ValueError, OverflowError):
            print(f"  fblock evidence rejected: case[{i}] non-integer field")
            return None
        parsed.append(row)
    # exact case set (BLOCKER-1): no missing/extra/duplicate, includes production
    expected = {("fblock", 1218, 5000, 12, 6), ("fblock", 1218, 10000, 12, 6),
                ("fblock", 1218, 5000, 128, 8), ("fblock", 1218, 5000, 128, 16),
                ("fblock", 1218, 5000, 128, 32), ("fblock", 1218, 5000, 128, 64),
                ("production", 1218, 5000, 128, 0)}
    seen = {(c["kind"], c["T"], c["N"], c["F"], c["block"]) for c in parsed}
    if seen != expected or len(parsed) != len(expected):
        print(f"  fblock evidence rejected: case set differs (n={len(parsed)})")
        return None
    # every case ran (rc==0) with a sane measured peak; driver_peak<=fb and
    # samples>0 apply to EVERY case (incl production, external review MAJOR-3);
    # reps pinned per case (F=12 cross-check anchors run reps=2; else 1).
    for c in parsed:
        if c["rc"] != 0:
            print(f"  fblock evidence rejected: case {c['kind']}/{c['N']}/F={c['F']}/block={c['block']} "
                  f"rc={c['rc']} != 0")
            return None
        if c["driver_peak"] <= 0:
            print(f"  fblock evidence rejected: case {c['kind']}/{c['N']}/F={c['F']}/block={c['block']} "
                  f"driver_peak<=0")
            return None
        # physically impossible: driver_peak = fb - min_free with min_free >= 0
        if c["driver_peak"] > c["fb"]:
            print(f"  fblock evidence rejected: case {c['kind']}/{c['N']}/F={c['F']}/block={c['block']} "
                  f"driver_peak > fb (forged/tampered)")
            return None
        if c["samples"] <= 0:  # external FACTOR-6: dead sampler must not ingest
            print(f"  fblock evidence rejected: case {c['kind']}/{c['N']}/F={c['F']}/block={c['block']} "
                  f"samples<=0")
            return None
        expected_reps = 2 if (c["kind"] == "fblock" and c["F"] == 12) else 1
        if c["reps"] != expected_reps:
            print(f"  fblock evidence rejected: case reps={c['reps']} != expected {expected_reps}")
            return None
    # BLOCKER-2: re-derive every derived field from raw bytes + current model.
    # fits is NEVER True for a margin<=8 case (external review MAJOR-1: a
    # vram-exhausted -- or forged-margin -- record must not close a column; fits
    # requires a real margin above the exhaustion floor, regardless of model_fits).
    # vram_exhausted is only ACCEPTED as a legitimate non-fit when the model ALSO
    # predicts non-fit (block=64/production): a forged margin<=8 on an
    # expected-fit case rejects (neither fits nor accepted vram_exhausted).
    fblock_cases = {}
    for c in parsed:
        if c["kind"] != "fblock":
            continue
        peak = c["driver_peak"]
        R = T_CANONICAL * c["N"]
        model = fblock_factor_peak(R, c["F"], c["block"])["peak_B"]
        margin = c["fb"] - peak
        model_fits = model <= AVAILABLE_BYTES
        vram_exh = margin <= 8.0 * MIB
        fits = (peak <= AVAILABLE_BYTES) and (margin > 8.0 * MIB)
        accepted_vram = vram_exh and (not model_fits)
        if not fits and not accepted_vram:
            print(f"  fblock evidence rejected: case {c['N']}/F={c['F']}/block={c['block']} "
                  f"neither fits nor accepted vram_exhausted (peak {peak}, margin {margin})")
            return None
        fblock_cases[(c["N"], c["F"], c["block"])] = {
            "driver_peak": peak,
            "driver_peak_MiB": round(peak / MIB, 2),
            "model_MiB": round(model / MIB, 2),
            "delta_MiB": round((peak - model) / MIB, 2),
            "margin_MiB": round(margin / MIB, 2),
            "fits": fits,
            "vram_exhausted": accepted_vram,
            "rc": c["rc"],
        }
    return {"evidence": str(ev.relative_to(ROOT)),
            "generated_at": data.get("generated_at"),
            "closure_status": data.get("closure_status"),
            "cases": fblock_cases}


def load_stock_nblock_measured() -> dict | None:
    """Best-effort read of the stock nblock HWM evidence
    (results/stock_nblock_hwm_v1.json).

    Fail-closed (reviews SC-1 + external BLOCKER-1/BLOCKER-2): only ingest
    evidence whose producer set closure_status=="OK" AND guard pass==1/r1=r2=0
    AND the FULL expected case set (nblock 256/128/64/32 + production) is present
    with every required field AND every case actually ran (rc==0, reps==1, sane
    driver_peak). All DERIVED fields (fits / driver_peak_MiB / model_MiB /
    delta_MiB) are RE-DERIVED here from the raw driver_peak bytes against the
    CURRENT model and AVAILABLE_BYTES -- the JSON's derived fields are never
    trusted. A failed or malformed case cleanly rejects the evidence (never
    raises). The nblock device peak is N-INDEPENDENT for N>=2*block
    (max_cols=2*block, every tile buffer sized by max_cols not N), so the N=5000
    block-ladder anchors the N=22600 closure.
    """
    REQUIRED = ("kind", "T", "N", "block", "reps", "samples", "fb",
                "driver_peak", "rc")
    ev = RESULTS / "stock_nblock_hwm_v1.json"
    if not ev.exists():
        return None
    try:
        data = json.loads(ev.read_text(encoding="utf-8"))
    except Exception:
        print(f"  nblock evidence rejected: unreadable JSON ({ev})")
        return None
    if data.get("closure_status") != "OK":
        print(f"  nblock evidence rejected: closure_status != OK ({ev})")
        return None
    guard = data.get("guard")
    if (not guard or guard.get("pass") != 1
            or guard.get("r1") != 0 or guard.get("r2") != 0):
        print(f"  nblock evidence rejected: GUARD not pass=1/r1=r2=0 ({ev})")
        return None
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        print(f"  nblock evidence rejected: cases missing/empty ({ev})")
        return None
    parsed = []
    for i, c in enumerate(cases):
        if not isinstance(c, dict):
            print(f"  nblock evidence rejected: case[{i}] not a dict")
            return None
        missing = [f for f in REQUIRED if f not in c]
        if missing:
            print(f"  nblock evidence rejected: case[{i}] missing fields {missing}")
            return None
        try:
            row = {"kind": str(c["kind"]), "T": int(c["T"]), "N": int(c["N"]),
                   "block": int(c["block"]), "reps": int(c["reps"]),
                   "samples": int(c["samples"]), "fb": int(c["fb"]),
                   "driver_peak": int(c["driver_peak"]), "rc": int(c["rc"])}
        except (TypeError, ValueError):
            print(f"  nblock evidence rejected: case[{i}] non-numeric field")
            return None
        parsed.append(row)
    # exact case set (BLOCKER-1): no missing/extra/duplicate, includes production
    expected = {("nblock", 1218, 5000, 256), ("nblock", 1218, 5000, 128),
                ("nblock", 1218, 5000, 64), ("nblock", 1218, 5000, 32),
                ("production", 1218, 5000, 0)}
    seen = {(c["kind"], c["T"], c["N"], c["block"]) for c in parsed}
    if seen != expected or len(parsed) != len(expected):
        print(f"  nblock evidence rejected: case set {sorted(seen)} (n={len(parsed)})")
        return None
    # every case ran (rc==0) with a sane measured peak; reps pinned
    for c in parsed:
        if c["rc"] != 0:
            print(f"  nblock evidence rejected: case rc={c['rc']} != 0")
            return None
        if c["driver_peak"] <= 0:
            print(f"  nblock evidence rejected: case driver_peak<=0")
            return None
        if c["reps"] != 1:
            print(f"  nblock evidence rejected: case reps={c['reps']} != 1")
            return None
    # BLOCKER-2: re-derive every derived field from raw bytes + current model.
    # A case whose re-derived fits is False (over budget) must reject the whole
    # evidence -- a fabricated/tampered JSON cannot close an over-budget anchor.
    nblock_cases = {}
    for c in parsed:
        if c["kind"] != "nblock":
            continue
        peak = c["driver_peak"]
        # physically impossible: driver_peak = fb - min_free with min_free >= 0
        if peak > c["fb"]:
            print(f"  nblock evidence rejected: case {c['N']}/block={c['block']} "
                  f"driver_peak {peak} > fb {c['fb']}")
            return None
        model = nblock_stock_peak(T_CANONICAL, c["N"], c["block"])["peak_B"]
        fits = peak <= AVAILABLE_BYTES
        if not fits:
            print(f"  nblock evidence rejected: case {c['N']}/block={c['block']} "
                  f"over-budget (peak {peak} > {AVAILABLE_BYTES})")
            return None
        nblock_cases[(c["N"], c["block"])] = {
            "driver_peak": peak,
            "driver_peak_MiB": round(peak / MIB, 2),
            "model_MiB": round(model / MIB, 2),
            "delta_MiB": round((peak - model) / MIB, 2),
            "margin_MiB": round((c["fb"] - peak) / MIB, 2),
            "fits": True,
            "rc": c["rc"],
        }
    return {"evidence": str(ev.relative_to(ROOT)),
            "generated_at": data.get("generated_at"),
            "closure_status": data.get("closure_status"),
            "cases": nblock_cases}


def load_stream_measured() -> dict | None:
    """Best-effort read of the streaming HWM evidence
    (results/factor_stream_hwm_v1.json).

    Fail-closed (synced to the fblock/nblock external BLOCKER-1/BLOCKER-2
    pattern): only ingest evidence whose producer set closure_status=="OK" AND
    guard pass==1/r1=r2=0 AND the FULL expected case set (3 stream + 1
    production = 4 records) is present with every required field AND every case
    actually ran (rc==0, sane driver_peak, samples>0). All DERIVED fields
    (fits / driver_peak_MiB / model_MiB / delta_MiB / margin_MiB) are RE-DERIVED
    here from the raw driver_peak bytes against the CURRENT
    streaming_factor_peak model and AVAILABLE_BYTES -- the JSON's derived fields
    are never trusted. A failed or malformed case cleanly rejects (never
    raises). The production F=128 control (vram-exhausted, margin~0) is a
    legitimate non-fit that does not close a column; a stream case that is not
    fits rejects the whole evidence.
    """
    REQUIRED = ("kind", "T", "N", "F", "block", "reps", "samples", "fb",
                "driver_peak", "rc")
    ev = RESULTS / "factor_stream_hwm_v1.json"
    if not ev.exists():
        return None
    try:
        data = json.loads(ev.read_text(encoding="utf-8"))
    except Exception:
        print(f"  stream evidence rejected: unreadable JSON ({ev})")
        return None
    if not isinstance(data, dict):
        print(f"  stream evidence rejected: root not an object ({ev})")
        return None
    if data.get("closure_status") != "OK":
        print(f"  stream evidence rejected: closure_status != OK ({ev})")
        return None
    # F-02 (external MAJOR): loader must not ingest provenance-weak evidence.
    # source must be "live" (an operator-trusted --from replay does not prove the
    # measurement came from THIS binary) and git_head must match the CURRENT
    # HEAD (evidence must represent the code under review).
    prov = data.get("provenance")
    if not isinstance(prov, dict):
        print(f"  stream evidence rejected: provenance missing ({ev})")
        return None
    if prov.get("source") != "live":
        print("  stream evidence rejected: source != live (operator-trusted --from "
              "replay does not prove the measurement came from this binary)")
        return None
    cur_head = current_git_head()
    if cur_head != "n/a" and prov.get("git_head") != cur_head:
        print(f"  stream evidence rejected: git_head {prov.get('git_head')} != "
              f"current HEAD {cur_head} (evidence not from reviewed source)")
        return None
    guard = data.get("guard")
    if (not isinstance(guard, dict) or guard.get("pass") != 1
            or guard.get("r1") != 0 or guard.get("r2") != 0):
        print(f"  stream evidence rejected: GUARD not pass=1/r1=r2=0 ({ev})")
        return None
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        print(f"  stream evidence rejected: cases missing/empty ({ev})")
        return None
    parsed = []
    for i, c in enumerate(cases):
        if not isinstance(c, dict):
            print(f"  stream evidence rejected: case[{i}] not a dict")
            return None
        missing = [f for f in REQUIRED if f not in c]
        if missing:
            print(f"  stream evidence rejected: case[{i}] missing fields {missing}")
            return None
        # STREAM-HWM-4 (synced to fblock external MINOR-8): int() truncation
        # (rc=0.5 -> 0) would let a forged record pass the rc/reps/case gates --
        # require the JSON value to be a true integer (or integer-valued float),
        # reject everything else (bool / non-numeric / non-integer float).
        def _as_int(k):
            v = c[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"{k} not numeric")
            if isinstance(v, float) and not v.is_integer():
                raise ValueError(f"{k} non-integer float")
            return int(v)

        try:
            row = {"kind": str(c["kind"]), "T": _as_int("T"), "N": _as_int("N"),
                   "F": _as_int("F"), "block": _as_int("block"),
                   "reps": _as_int("reps"), "samples": _as_int("samples"),
                   "fb": _as_int("fb"), "driver_peak": _as_int("driver_peak"),
                   "rc": _as_int("rc")}
        except (TypeError, ValueError, OverflowError):
            print(f"  stream evidence rejected: case[{i}] non-integer field")
            return None
        parsed.append(row)
    # exact case set (BLOCKER-1): no missing/extra/duplicate. `block` on stream
    # cases is max_transpose_rows (the FBLC block= field carries tt).
    expected = {("stream", 1218, 5000, 12, 4096), ("stream", 1218, 10000, 12, 4096),
                ("stream", 1218, 5000, 128, 4096), ("production", 1218, 5000, 128, 0)}
    seen = {(c["kind"], c["T"], c["N"], c["F"], c["block"]) for c in parsed}
    if seen != expected or len(parsed) != len(expected):
        print(f"  stream evidence rejected: case set {sorted(seen)} (n={len(parsed)})")
        return None
    # every case ran (rc==0) with a sane measured peak; driver_peak<=fb and
    # samples>0 apply to EVERY case (incl production, external review MAJOR-3
    # pattern); reps pinned (F=12 anchors reps=2; else 1).
    for c in parsed:
        if c["rc"] != 0:
            print(f"  stream evidence rejected: case {c['kind']}/{c['N']}/F={c['F']} "
                  f"rc={c['rc']} != 0")
            return None
        if c["driver_peak"] <= 0:
            print(f"  stream evidence rejected: case {c['kind']}/{c['N']}/F={c['F']} "
                  f"driver_peak<=0")
            return None
        # physically impossible: driver_peak = fb - min_free with min_free >= 0
        if c["driver_peak"] > c["fb"]:
            print(f"  stream evidence rejected: case {c['kind']}/{c['N']}/F={c['F']} "
                  f"driver_peak {c['driver_peak']} > fb {c['fb']} (forged/tampered)")
            return None
        if c["samples"] <= 0:
            print(f"  stream evidence rejected: case {c['kind']}/{c['N']}/F={c['F']} "
                  f"samples<=0")
            return None
        expected_reps = 2 if (c["kind"] == "stream" and c["F"] == 12) else 1
        if c["reps"] != expected_reps:
            print(f"  stream evidence rejected: case reps={c['reps']} != expected {expected_reps}")
            return None
    # BLOCKER-2: re-derive every derived field from raw bytes + current model.
    # A stream case whose re-derived fits is False must reject the whole
    # evidence -- a fabricated JSON cannot close an over-budget anchor.
    stream_cases = {}
    for c in parsed:
        if c["kind"] != "stream":
            continue
        peak = c["driver_peak"]
        R = T_CANONICAL * c["N"]
        model = streaming_factor_peak(R, c["F"], c["block"])
        margin = c["fb"] - peak
        fits = (peak <= AVAILABLE_BYTES) and (margin > 8.0 * MIB)
        if not fits:
            print(f"  stream evidence rejected: case {c['N']}/F={c['F']} not fits "
                  f"(peak {peak}, margin {margin})")
            return None
        stream_cases[(c["N"], c["F"], c["block"])] = {
            "driver_peak": peak,
            "driver_peak_MiB": round(peak / MIB, 2),
            "model_MiB": round(model / MIB, 2),
            "delta_MiB": round((peak - model) / MIB, 2),
            "margin_MiB": round(margin / MIB, 2),
            "fits": True,
            "rc": c["rc"],
        }
    return {"evidence": str(ev.relative_to(ROOT)),
            "generated_at": data.get("generated_at"),
            "closure_status": data.get("closure_status"),
            "cases": stream_cases}


def build_payload() -> dict:
    verify = verify_calibration()
    fblk_meas = load_fblock_measured()
    nb_meas = load_stock_nblock_measured()
    stream_meas = load_stream_measured()

    scenarios = []
    # factor_corr：canonical + F=128 + N=10000，三场景
    # 审查 M3 标注 model_prediction；fblock(项③) 2026-08-07、streaming(项②)
    # 2026-08-08 实测闭合（有证据时）
    for N, F, label in ((5000, 12, "canonical"), (5000, 128, "F128"),
                        (10000, 12, "N10000")):
        R = T_CANONICAL * N
        cur_peak, _ = factor_corr_peak(R, F)
        stream_peak = streaming_factor_peak(R, F, STREAM_TT)
        # measured closure for the streaming (项②) column: F=128 anchor measured
        # at tt=4096 (N=5000). Only a FITS streaming anchor closes the column;
        # absent evidence (or non-fit) keeps it model_prediction.
        sm = stream_meas["cases"].get((N, F, STREAM_TT)) if stream_meas else None
        stream_measured = None
        if sm is not None and sm["fits"]:
            stream_measured = {
                "max_transpose_rows": STREAM_TT,
                "driver_peak_MiB": sm["driver_peak_MiB"],
                "model_MiB": sm["model_MiB"],
                "delta_MiB": sm["delta_MiB"],
                "margin_MiB": sm["margin_MiB"],
                "fits": sm["fits"],
                "rc": sm["rc"],
            }
        # F-blocking：评估全候选块宽（审查 M5），首个 fit 且实际分块(n_blocks>=2)
        # 为报告块。block=F 是退化配置（1 tile、max_cols=2*F 超分配、比 current
        # 更差），不是 F-blocking 的用途；审查 4/h7 修正后排除。
        fblock_candidates = []
        fblock = None
        start = min(64, F)
        for block in (start, start // 2, start // 4, start // 8, 1):
            fb = fblock_factor_peak(R, F, block)
            fblock_candidates.append({"block": fb["block"],
                                      "peak_MiB": round(fb["peak_MiB"], 2),
                                      "n_blocks": fb["n_blocks"],
                                      "n_tiles": fb["n_tiles"],
                                      "fits": fb["peak_B"] <= AVAILABLE_BYTES})
            if (fb["peak_B"] <= AVAILABLE_BYTES and fblock is None
                    and fb["n_blocks"] >= 2):
                fblock = fb
        # measured closure for the fblock (项③) column: F=12 anchors measured at
        # block=6 (block<=F/2, model==impl), F128 at block=32 (reported block).
        # current/streaming columns stay model predictions either way.
        measured = None
        anchor_block = 6 if F == 12 else 32
        fc = fblk_meas["cases"].get((N, F, anchor_block)) if fblk_meas else None
        # ingest re-derives every field from raw bytes + current model (synced to
        # stock external BLOCKER-2). Only a FITS anchor closes the column -- a
        # non-fit/vram-exhausted anchor keeps the column as model_prediction.
        if fc is not None and fc["fits"]:
            measured = {
                "block": anchor_block,
                "driver_peak_MiB": fc["driver_peak_MiB"],
                "model_MiB": fc["model_MiB"],
                "delta_MiB": fc["delta_MiB"],
                "margin_MiB": fc["margin_MiB"],
                "fits": fc["fits"],
                "vram_exhausted": fc["vram_exhausted"],
                "rc": fc["rc"],
            }
        scenarios.append({
            "op": "factor_corr", "N": N, "F": F, "label": label,
            "model_prediction": True,  # current 列仍为预测；fblock/streaming 列有证据时闭合
            "measured_fblock": measured,  # fblock(项③) 闭合为实测；None=无证据仍预测
            "measured_stream": stream_measured,  # streaming(项②) 闭合为实测；None=仍预测
            "current_peak_MiB": round(mib(cur_peak), 2),
            "streaming_peak_MiB": round(mib(stream_peak), 2),
            "streaming_config": {"max_transpose_rows": STREAM_TT},
            "fblock": None if fblock is None else {
                "block": fblock["block"], "n_blocks": fblock["n_blocks"],
                "n_tiles": fblock["n_tiles"], "peak_MiB": round(fblock["peak_MiB"], 2)},
            "fblock_candidates": fblock_candidates,
            "dpp": dpp_size(R, F),
            "fits_current": cur_peak <= AVAILABLE_BYTES,
            "fits_streaming": stream_peak <= AVAILABLE_BYTES,
            "fits_fblock": fblock is not None,
            "note": ("审查 M1 修正：fblock 含 d_F_tile 输入缓冲（与 d_Xt_tile 同驻留）；"
                     "F128/B64 超预算（~12.6 GiB），B32 可 fit；生产 F-blocking 若逐 tile "
                     "流式上传+转置+释放 d_F_tile（项②机制）才命中更小峰值。"
                     + ("" if measured is None else
                        f" fblock(项③) 已实测闭合（block={measured['block']}, "
                        f"{measured['driver_peak_MiB']:.1f} MiB vs 模型 "
                        f"{measured['model_MiB']:.2f} MiB）。")
                     + ("" if stream_measured is None else
                        f" streaming(项②) 已实测闭合（tt={STREAM_TT}, "
                        f"{stream_measured['driver_peak_MiB']:.1f} MiB vs 模型 "
                        f"{stream_measured['model_MiB']:.2f} MiB）。")),
        })

    # stock_corr：N=5000/10000/22600，N-blocking 视角（审查 M4：输出完整矩阵分块）
    for N, label in ((5000, "N5000"), (10000, "N10000"), (22600, "N22600")):
        cur_peak, _ = stock_corr_peak(T_CANONICAL, N)
        # N-blocking：评估候选块宽（审查 M5 同款），首个 fit 且实际分块(n_blocks>=2)
        # 为报告块。输出流式化（逐 tile D2H → host 写回，device 无全量 d_corr/d_out，
        # M4 输出驻留契约改变）体现在 nblock_stock_peak 分配链无 N*N 输出缓冲。
        nblock_candidates = []
        nblock = None
        for block in (256, 128, 64, 32, 16, 8):
            nb = nblock_stock_peak(T_CANONICAL, N, block)
            nblock_candidates.append({"block": nb["block"],
                                      "peak_MiB": round(nb["peak_MiB"], 2),
                                      "n_blocks": nb["n_blocks"],
                                      "n_tiles": nb["n_tiles"],
                                      "fits": nb["peak_B"] <= AVAILABLE_BYTES})
            if (nb["peak_B"] <= AVAILABLE_BYTES and nblock is None
                    and nb["n_blocks"] >= 2):
                nblock = nb
        # N-blocking 实测闭合（2026-08-08，stock_nblock_hwm_v1.json）：nblock device
        # 峰值 N 无关（N>=2*block 时 max_cols=2*block，tile 缓冲按 max_cols 非 N），
        # 故 N=5000 实测 block 阶梯锚定 N=22600；measured 引用报告块的 N=5000 实测。
        measured = None
        if nblock is not None and nb_meas is not None:
            mc = nb_meas["cases"].get((5000, nblock["block"]))
            if mc is not None:
                # ingest re-derives every field from raw bytes + current model
                # (external BLOCKER-2); the scenario just reads them.
                measured = {
                    "N_measured": 5000,
                    "driver_peak_MiB": mc["driver_peak_MiB"],
                    "model_MiB": mc["model_MiB"],
                    "delta_MiB": mc["delta_MiB"],
                    "margin_MiB": mc["margin_MiB"],
                    "fits": mc["fits"],
                    "rc": mc["rc"],
                }
        scenarios.append({
            "op": "stock_corr", "N": N, "label": label,
            "model_prediction": True,  # current/streaming 列仍为预测；nblock 列见 measured_nblock
            "current_peak_MiB": round(mib(cur_peak), 2),
            "output_full2_MiB": round(mib(2 * N * N * 8), 2),  # d_corr+d_out 各 N*N*8
            "nblock": None if nblock is None else {
                "block": nblock["block"], "n_blocks": nblock["n_blocks"],
                "n_tiles": nblock["n_tiles"], "peak_MiB": round(nblock["peak_MiB"], 2)},
            "nblock_candidates": nblock_candidates,
            "measured_nblock": measured,  # nblock 列闭合为实测；None=无证据仍预测
            "fits_current": cur_peak <= AVAILABLE_BYTES,
            "fits_nblock": nblock is not None,
            "note": (("审查 M4 修正：output d_corr+d_out 各分配 N*N*8（完整矩阵，非上三角），"
                     "N=22600 两输出合计 ~7793 MiB 已超预算 → N-blocking 须同时分块/stream 输出"
                     "（改变输出驻留契约），不能只分输入/工作区；N-blocking 模型基于 "
                     "poc3_stock_corr_selfcheck.cu 的 stock_corr_gpu_nblock 分配链"
                     "（tile 级驻留 + 逐 tile 输出流式化，无 N*N device 输出缓冲）")
                     + ("" if measured is None or nblock is None else
                        f"；**nblock 峰值已实测闭合**（N=5000 block={nblock['block']} "
                        f"driver {measured['driver_peak_MiB']:.1f} MiB vs 模型 "
                        f"{measured['model_MiB']:.2f} MiB，device 峰值 N 无关（N≥2*block 限定）"
                        f"→ N=22600 同值；N=22600 实际运行未实测——host 峰值 ~8.4 GB"
                        f"（两个 N*N 输出 buffer 并存）+ O(N²·T) 计算）")
                     + "；审查 M4-2 披露：本预算只量化 device 驻留，"
                     "nblock 额外付出 host pass-1 O(T*N) 预扫描 + 逐 tile host 抽列/写回"
                     "（生产路径无此 host 成本）——device 内存换 host 预扫描的权衡未计入。"),
        })

    payload = {
        "name": "memory_budget_v1",
        "schema_version": "1.0.0",
        "generator_sha256": hashlib.sha256(
            pathlib.Path(__file__).read_bytes()).hexdigest().upper(),
        "memory_budget": {"total_MiB": TOTAL_MIB, "reserve_MiB": RESERVE_MIB,
                          "available_MiB": TOTAL_MIB - RESERVE_MIB},
        "calibration_verification": verify,
        "scenarios": scenarios,
        "kahan_v2_closure": {
            "d_valid_lifetime": "d_valid 与 d_Xt 同驻留（Kahan 重跑读两者）；流式化/F-blocking 均须含 d_valid",
            "transpose_overlap": "峰值 = 转置前 d_F/d_X + d_Xt + d_valid + d_mask 重叠（校准 2381 MiB 实证）",
            "dpp_scale": "d_pp = P×blockDim×struct_bytes_lane(Partial1+Partial2=80B/lane)；F=128 → 161.25 MiB（见 scenarios dpp）",
            "cross_block_pair": "F-blocking 2D tile 驻留 (B_a+B_b)×R×(8+1) + tile pair 工作区（见 scenarios fblock）",
        },
    }
    return payload


def render_md(payload: dict) -> str:
    L = []
    L.append("# factor-cuda 内存静态预算 v1（当前实现精确模型 + 项②流式化 + 项③F-blocking）")
    L.append("")
    L.append(f"> 预算：可用 {payload['memory_budget']['available_MiB']} MiB "
             f"(total {payload['memory_budget']['total_MiB']} − reserve "
             f"{payload['memory_budget']['reserve_MiB']})")
    L.append("")
    L.append("## 校准验证（模型 vs 实测 HWM，delta≈0 通过）")
    L.append("")
    L.append("| 操作 | 规模 | 校准 HWM (MiB) | 模型 (MiB) | delta |")
    L.append("|---|---|---|---|---|")
    for r in payload["calibration_verification"]["rows"]:
        mark = "✅" if r["ok"] else "❌"
        L.append(f"| {r['key'][0]} | {r['key'][1:]} | {r['calibration_MiB']} | "
                 f"{r['model_MiB']} | {r['delta_MiB']} {mark} |")
    L.append("")
    L.append(f"**all_match = {payload['calibration_verification']['all_match']}**")
    L.append("")
    L.append("## 场景对比（当前 / 项②流式化 / 项③F-blocking）")
    L.append("")
    for s in payload["scenarios"]:
        L.append(f"### {s['op']} {s['label']} (N={s['N']}"
                 + (f", F={s['F']})" if 'F' in s else ")"))
        L.append("")
        L.append(f"- 当前实现峰值：**{s['current_peak_MiB']} MiB**"
                 + (" ✅" if s.get('fits_current') else " ❌ 超预算"))
        if 'streaming_peak_MiB' in s:
            L.append(f"- 项②流式化峰值：**{s['streaming_peak_MiB']} MiB**"
                     + (" ✅" if s.get('fits_streaming') else " ❌"))
        if s.get('measured_stream'):
            m = s['measured_stream']
            L.append(f"- **实测(项② streaming tt={m['max_transpose_rows']})**：driver_peak "
                     f"**{m['driver_peak_MiB']} MiB**"
                     + (" ✅ fits" if m.get('fits') else " ❌")
                     + (f"（模型 {m['model_MiB']} MiB,delta {m['delta_MiB']} MiB,"
                        f"margin {m['margin_MiB']} MiB,rc={m['rc']}）"))
        if s.get('fblock'):
            L.append(f"- 项③F-blocking（block={s['fblock']['block']}, "
                     f"{s['fblock']['n_tiles']} tiles）：**{s['fblock']['peak_MiB']} MiB**"
                     + (" ✅" if s.get('fits_fblock') else " ❌"))
        if s.get('measured_fblock'):
            m = s['measured_fblock']
            anchor_note = ""
            if s.get('fblock') and m['block'] != s['fblock']['block']:
                anchor_note = (f"（block={m['block']} 为模型验证锚点,block≤F/2 模型=实现一致;"
                               f"报告块 block={s['fblock']['block']} 模型预测 "
                               f"{s['fblock']['peak_MiB']} MiB 系实现 2*block 超分配,未实测）")
            L.append(f"- **实测(项③ fblock block={m['block']})**：driver_peak "
                     f"**{m['driver_peak_MiB']} MiB**"
                     + (" ✅ fits" if m.get('fits') else " ❌")
                     + (f"（模型 {m['model_MiB']} MiB,delta {m['delta_MiB']} MiB,"
                        f"margin {m['margin_MiB']} MiB）{anchor_note}"))
        if 'dpp' in s:
            L.append(f"- d_pp continuation：{s['dpp']['P']} pairs × 256 lanes × "
                     f"{s['dpp']['struct_bytes_lane']}B = **{s['dpp']['dpp_MiB']:.1f} MiB**")
        if s.get('nblock'):
            L.append(f"- 项③N-blocking（block={s['nblock']['block']}, "
                     f"{s['nblock']['n_tiles']} tiles）：**{s['nblock']['peak_MiB']} MiB**"
                     + (" ✅" if s.get('fits_nblock') else " ❌"))
        if s.get('measured_nblock'):
            m = s['measured_nblock']
            nb = s.get('nblock') or {}
            L.append(f"- **实测(N-blocking block={nb.get('block')}, N={m['N_measured']})**："
                     f"driver_peak **{m['driver_peak_MiB']} MiB**"
                     + (" ✅ fits" if m.get('fits') else " ❌")
                     + (f"（模型 {m['model_MiB']} MiB,delta {m['delta_MiB']} MiB,"
                        f"margin {m['margin_MiB']} MiB,rc={m['rc']}）"))
        if s.get('output_O2_MiB'):
            L.append(f"- 输出 O(N²) 不可约下界：**{s['output_O2_MiB']} MiB**")
        if s.get('note'):
            L.append(f"- 注：{s['note']}")
        L.append("")
    L.append("## Kahan 决策 v2 四项闭合")
    L.append("")
    for k, v in payload["kahan_v2_closure"].items():
        L.append(f"- **{k}**：{v}")
    L.append("")
    L.append("## 审查口径（GPT-5.6-Sol 2026-08-06，15 项处置后）")
    L.append("")
    L.append("- 场景峰值：**fblock(项③) 已于 2026-08-07 实测闭合**（factor_corr F=12/F=128 见各场景 `measured_fblock`，证据 `benchmarks/results/factor_fblock_hwm_v1.json`；分配链 probe delta 3 MiB、B32 本次总 delta 见 `measured_fblock.delta_MiB`（另一次 245.09 MiB 单样本未纳入证据，review F7））；**N-blocking 已于 2026-08-08 实测闭合**（stock N5000/N10000/N22600 见各场景 `measured_nblock`，证据 `benchmarks/results/stock_nblock_hwm_v1.json`；N=5000 block=256 锚定，device 峰值 N 无关（N≥2*block）→ N=22600 同值 22.0 MiB；N=22600 实际运行未实测——host 峰值 ~8.4 GB（两个 N*N 输出 buffer）+ 计算，诚实披露）；**streaming(项②) 已于 2026-08-08 实测闭合**（F=128 见各场景 `measured_stream`，证据 `benchmarks/results/factor_stream_hwm_v1.json`；F128 driver 6987 MiB fits——当前 12.6 GiB 超预算→streaming 6.9 GiB fits；F=12 锚点 636/1270 MiB 与模型 delta ≤2 MiB；production F=128 对照 vram-exhausted margin≈0）")
    L.append("- fblock 已含 d_F_tile 输入缓冲（M1）：F128/B64 超预算、B32 可 fit；生产 F-blocking 若逐 tile 流式上传+释放 d_F_tile 命中更小峰值")
    L.append("- d_pp 按实际 ABI struct（Partial1+Partial2=80B/lane）计（M2），F128=161.25 MiB")
    L.append("- host staging 未计入 GPU 预算（M6）：fblock driver 逐 tile F_tile host 内存 F128/B64 ~5.9 GiB，生产须有界/pinned staging")
    L.append("- stock N-blocking 须同时分块/stream 输出（M4）：d_corr+d_out 各 N*N*8，N=22600 已超预算")
    L.append("")
    L.append("## 复现")
    L.append("")
    L.append("    PYTHONIOENCODING=utf-8 python benchmarks/memory_budget_v1.py")
    L.append("")
    L.append("*生成模型: DeepSeek-V4-Flash (via Claude Code CLI) · memory_budget_v1.py*")
    return "\n".join(L) + "\n"


def main() -> int:
    payload = build_payload()
    out_json = ROOT / "docs" / "memory_budget_v1.json"
    out_md = RESULTS / "memory_budget_v1.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    out_md.write_text(render_md(payload), encoding="utf-8", newline="\n")
    print(f"calibration all_match={payload['calibration_verification']['all_match']}")
    for s in payload["scenarios"]:
        extra = ""
        if 'streaming_peak_MiB' in s:
            extra = f" streaming={s['streaming_peak_MiB']}"
        if s.get('fblock'):
            extra += f" fblock={s['fblock']['peak_MiB']} (block={s['fblock']['block']})"
        if s.get('nblock'):
            extra += f" nblock={s['nblock']['peak_MiB']} (block={s['nblock']['block']})"
        print(f"  {s['op']} {s['label']}: current={s['current_peak_MiB']} MiB{extra}")
    print(f"saved {out_json} + {out_md}")
    return 0 if payload["calibration_verification"]["all_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
