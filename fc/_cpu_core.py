# -*- coding: utf-8 -*-
"""factor-cuda shared CPU oracle core (Phase 1, F19).

Single source of truth for the CPU numeric implementations (np_cs_rank /
np_factor_corr / np_stock_corr / np_rolling_ic / np_parameter_scan + helpers),
extracted mechanically from benchmarks/backends.py (2026-08-06) so fc and the
benchmark arm depend on the SAME implementation (no semantic drift). backends.py
now imports from here; fc/_cpu.py delegates to here. Importable without the
benchmark layer.
"""
from __future__ import annotations

import numpy as np

def in_corr_domain(x: np.ndarray, mask: np.ndarray | None = None) -> bool:
    """仅对 mask 交集 + finite 的有效子集校验 max|x|≤1e150 且 min 非零|x|≥1e-150。

    mask 可为 numpy 或 CuPy（GPU resident 口径）。S6：CuPy 改为可选导入——
    纯 NumPy 基线环境（无 CUDA/CuPy）也能独立运行。
    """
    try:
        import cupy as _cp
    except ImportError:
        _cp = None
    if _cp is not None and isinstance(mask, _cp.ndarray):
        mask = _cp.asnumpy(mask)
    x = np.asarray(x, dtype=np.float64)
    valid = np.isfinite(x)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    xv = x[valid]
    if xv.size == 0:
        return True
    amax = np.max(np.abs(xv))
    nonzero = xv[xv != 0.0]
    amin = np.min(np.abs(nonzero)) if nonzero.size else None
    if amax > 1e150:
        return False
    if amin is not None and amin < 1e-150:
        return False
    return True

def _ordinal_rank_1d(x: np.ndarray) -> np.ndarray:
    """stable ordinal 整数秩 1..K（非有限不参与，输出 NaN）。保留输入精度。"""
    x = np.asarray(x)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        return out
    xs = x[finite]
    order = np.argsort(xs, kind="stable")
    r = np.empty(len(xs), dtype=np.float64)
    r[order] = np.arange(1, len(xs) + 1, dtype=np.float64)
    out[finite] = r
    return out

def np_cs_rank(X: np.ndarray, mask: np.ndarray | None, descending: bool) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"cs_rank 输入须 (T,N) 2D，got ndim={X.ndim}")
    if X.shape[1] > 2**24:
        raise ValueError(f"N={X.shape[1]} > 2^24，秩无法精确表示")
    Xw = -X if descending else X
    out = np.full(X.shape, np.nan, dtype=np.float32)
    part = np.isfinite(Xw)
    if mask is not None:
        part = part & np.asarray(mask, dtype=bool)
    T = X.shape[0]
    for t in range(T):
        p = part[t]
        if not p.any():
            continue
        xs = Xw[t][p]
        order = np.argsort(xs, kind="stable")
        r = np.empty(len(xs), dtype=np.float32)
        r[order] = np.arange(1, len(xs) + 1, dtype=np.float32)
        out[t][p] = r
    return out

def _kahan_mean(x: np.ndarray) -> float:
    """Serial Kahan-compensated mean (HG-2 high-precision reference semantics).

    O(n) Python loop; used only on reduction-sensitive (large-bias) inputs via
    the _bias_sensitive guard in _two_pass_corr, so the common low-bias path
    stays fully vectorized. Found by tests/test_adapter_v1.py F10 2026-08-06:
    numpy mean on ~1e15-bias data loses ~9e-3 in the correlation, violating the
    HG-2 clause (all backends must match the serial-Kahan reference <=1e-12)."""
    x = np.asarray(x, dtype=np.float64)
    s = 0.0
    c = 0.0
    for v in x:
        y = v - c
        t = s + y
        c = (t - s) - y
        s = t
    return s / x.size


def _bias_sensitive(x: np.ndarray) -> bool:
    """HG-2 reduction-sensitive guard: |mean| > 1e3*sigma -> serial-Kahan mean.

    For |mean| <= 1e3*sigma the numpy-mean error contributes <~2e-13 to the
    correlation (well inside the 1e-12 wrapper parity), so the vectorized numpy
    mean is kept; above the threshold the mean error is amplified (bias_1e15:
    ~9e-3) and the serial Kahan mean is required."""
    x = np.asarray(x, dtype=np.float64)
    sd = float(x.std())  # ddof=0; accurate even when numpy mean itself is not
    return float(abs(x.mean())) > 1e3 * sd


def _two_pass_corr(xa: np.ndarray, xb: np.ndarray) -> float:
    """两遍中心化 Pearson（ddof=1），低偏置典型输入与 corr_oracle（np.corrcoef）一致。

    输入已切共同有效子集（xa/xb 等长且均有限）。用于大偏置对抗锚点（pair-specific）。
    顺序除法 (sxy/sqrt(sxx))/sqrt(syy) 与 safe_pearson / np.corrcoef 一致（2026-08-05
    异后端审查发现）：原乘积型分母 sqrt(sxx*syy) 在正次正规方差（S>0 且 S*S 下溢为 0）
    上会把有限自相关错判成 NaN。
    HG-2（2026-08-05）：归约顺序敏感输入（|mean|>1e3·σ）改用串行 Kahan 均值，与高精度
    参考 ≤1e-12（numpy 均值在 1e15 偏置上丢 ~9e-3，2026-08-06 测试套件实证）；低偏置
    典型输入保持 numpy 均值（与 wrapper 一致，向量化）。
    """
    a = np.asarray(xa, dtype=np.float64)
    b = np.asarray(xb, dtype=np.float64)
    if a.size < 2:
        return float("nan")
    if _bias_sensitive(a) or _bias_sensitive(b):
        am = _kahan_mean(a); bm = _kahan_mean(b)
    else:
        am = a.mean(); bm = b.mean()
    ac = a - am; bc = b - bm
    sxx = float((ac * ac).sum())
    syy = float((bc * bc).sum())
    sxy = float((ac * bc).sum())
    # 正值 + finite 守卫（同 safe_pearson）：零方差 / 平方和溢出 → NaN
    if not (sxx > 0.0 and syy > 0.0) or not (np.isfinite(sxx) and np.isfinite(syy)):
        return float("nan")
    return float((sxy / np.sqrt(sxx)) / np.sqrt(syy))

def _masked_gemm_stats(X: np.ndarray, mask: np.ndarray | None) -> tuple:
    """一遍式 masked-GEMM 相关矩阵统计（float64 + 安全置零）。返回 8 元组。

    返回 (corr, valid, n, sumx, sumy, sumxy, sumx2, sumy2)。
    corr 已置 n<2 → NaN；调用方负责抵消检测回退 + 对角 + 逐位镜像。
    数学上等价于 pairwise-complete 两遍中心化（canonical 偏置小时无抵消，实测 ≤1e-12）；
    大偏置对抗输入须由 _gemm_cancel_mask 检测并回退 pair-specific（审查 R1/H5）。
    """
    X64 = X.astype(np.float64, copy=False)
    _T, _N = X64.shape
    valid = np.isfinite(X64)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    M = valid.astype(np.float64)
    Xm = np.where(valid, X64, 0.0)  # 安全置零（修复 NaN×0，H1）
    n = M.T @ M
    sumx = Xm.T @ M
    sumy = M.T @ Xm
    sumxy = Xm.T @ Xm
    Xsq = Xm * Xm  # 复用一次（省一次 584MB 构造）
    sumx2 = Xsq.T @ M
    sumy2 = M.T @ Xsq
    n_safe = np.maximum(n, 1.0)
    den = n_safe - 1.0
    with np.errstate(all="ignore"):
        cov = (sumxy - sumx * sumy / n_safe) / den
        varx = (sumx2 - sumx * sumx / n_safe) / den
        vary = (sumy2 - sumy * sumy / n_safe) / den
        corr = cov / np.sqrt(varx * vary)
    corr[n < 2] = np.nan
    return corr, valid, n, sumx, sumy, sumxy, sumx2, sumy2

_CANCEL_RATIO_THRESHOLD = 3.0

def _gemm_cancel_mask(sumx, sumy, sumxy, sumx2, sumy2, n) -> np.ndarray:
    """GEMM 一遍式抵消检测（审查 R1/H-01）：返回需回退 pair-specific 的布尔矩阵。

    对 pair (i,j)，若 |sumx·sumy/n|、|sumx²/n|、|sumy²/n| 任一相对差分项
    （sumxy−sumx·sumy/n 等）超过 `_CANCEL_RATIO_THRESHOLD` 倍 → GEMM 该格不可信 → 回退。
    """
    n_safe = np.maximum(n, 1.0)
    cov_u = sumxy - sumx * sumy / n_safe
    varx_u = sumx2 - sumx * sumx / n_safe
    vary_u = sumy2 - sumy * sumy / n_safe
    # F7 (GPT-5.6-Sol stock_corr v1 review): NO 1e-300 additive. The comparison
    # is already cross-multiplied (no divide-by-zero), and the additive swallows
    # real cancellation near the contract's low-scale bound. Synced with
    # src/stock_corr.cu finalize_cell.
    fix = (
        (np.abs(sumx * sumy / n_safe) > _CANCEL_RATIO_THRESHOLD * np.abs(cov_u))
        | (np.abs(sumx * sumx / n_safe) > _CANCEL_RATIO_THRESHOLD * np.abs(varx_u))
        | (np.abs(sumy * sumy / n_safe) > _CANCEL_RATIO_THRESHOLD * np.abs(vary_u))
    )
    return fix | fix.T

def np_stock_corr(X: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """stock_corr：masked-GEMM 快路径 + 抵消检测回退 pair-specific（R6 支撑）。

    输入 = returns（协议，GPT H1 修复）。canonical（偏置小，均值≈0）全向量化 GEMM
    （快，协议 §6.2 冻结语义：N=500 ~36ms）；大偏置对抗输入（H5/R1）由
    _gemm_cancel_mask 逐格回退 _two_pass_corr（正确性 |Δr|≤1e-12）。
    退化对角 NaN，三角逐位镜像。
    """
    if not in_corr_domain(X, mask):
        raise ValueError("corr 数值域外")
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[1]
    corr, valid, n, sumx, sumy, sumxy, sumx2, sumy2 = _masked_gemm_stats(X, mask)
    fix = _gemm_cancel_mask(sumx, sumy, sumxy, sumx2, sumy2, n)
    fix |= np.abs(corr) > 1.0  # H-01 合理性二次判定：|r|>1 说明 GEMM 抵消失真（不可靠阈值兜底）
    fix = fix | fix.T
    # 回退 pair-specific（仅 fix 格；canonical 集合近空 → 几乎零开销，全 GEMM）
    for i, j in zip(*np.where(fix)):
        if i >= j:
            continue
        o = valid[:, i] & valid[:, j]
        if o.sum() < 2:
            v = np.nan
        else:
            v = _two_pass_corr(X[o, i], X[o, j])
        corr[i, j] = corr[j, i] = v
    for i in range(N):
        o = valid[:, i]
        # F1 (review, 2026-08-05): diagonal follows the computed self-correlation
        # -- 1.0 iff finite (positive centered variance), else NaN (var
        # underflow / constant / n<2). A blind ptp()!=0 test would wrongly force
        # 1.0 on a var-underflow column (e.g. tiny-adjacent 1e-150 values);
        # _two_pass_corr(x,x) returns NaN there, matching the frozen oracle.
        v = _two_pass_corr(X[o, i], X[o, i])
        corr[i, i] = 1.0 if np.isfinite(v) else np.nan
    # 逐位镜像（保上三角，下三角 = 上三角转置）
    return np.triu(corr) + np.triu(corr, 1).T

def np_factor_corr(F3: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """factor_corr：masked-GEMM 快路径 + 抵消检测回退 pair-specific（R6 支撑，同 stock_corr）。

    F3 (T,N,F)：reshape (T*N, F) 后列=12 factor、行=全部样本 → GEMM 计算量 12×12×TN
    （毫秒级，替代 H5 后 pair-specific 逐对 22s 的性能倒退）。大偏置对抗输入由
    _gemm_cancel_mask 逐格回退 _two_pass_corr（|Δr|≤1e-12）。R5：域校验只查参与元素。
    """
    F3 = np.asarray(F3)
    if F3.ndim != 3:
        raise ValueError(f"factor_corr 输入须 (T,N,F) 3D，got ndim={F3.ndim}")
    T, N, F = F3.shape
    X64 = F3.astype(np.float64).reshape(T * N, F)
    valid = np.isfinite(X64)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool).reshape(-1, 1)
    # R5 + M-01：域校验用逐元素参与元素（valid 全展平），非整行（避免 masked-out 跨列极端值误拒）
    if not in_corr_domain(X64, valid):
        raise ValueError("corr 数值域外：max|x|>1e150 或 min 非零|x|<1e-150")
    corr, valid_, n, sumx, sumy, sumxy, sumx2, sumy2 = _masked_gemm_stats(X64, valid)
    fix = _gemm_cancel_mask(sumx, sumy, sumxy, sumx2, sumy2, n)
    fix |= np.abs(corr) > 1.0  # H-01 合理性二次判定
    fix = fix | fix.T
    for i, j in zip(*np.where(fix)):
        if i >= j:
            continue
        o = valid_[:, i] & valid_[:, j]
        if o.sum() < 2:
            v = np.nan
        else:
            v = _two_pass_corr(X64[o, i], X64[o, j])
        corr[i, j] = corr[j, i] = v
    for i in range(F):
        o = valid_[:, i]
        # F1 (review, 2026-08-05): diagonal follows the computed self-correlation
        # (see np_stock_corr); _two_pass_corr(x,x) is NaN on var-underflow.
        v = _two_pass_corr(X64[o, i], X64[o, i])
        corr[i, i] = 1.0 if np.isfinite(v) else np.nan
    return np.triu(corr) + np.triu(corr, 1).T

def np_rolling_ic(f: np.ndarray, r: np.ndarray,
                  factor_mask: np.ndarray | None, fwd_mask: np.ndarray | None,
                  min_valid: int) -> np.ndarray:
    """rolling_ic：双侧 finite ∧ 双侧 mask 交集，ordinal Spearman（float64）。"""
    f = np.asarray(f, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    T = f.shape[0]
    ok = np.isfinite(f) & np.isfinite(r)
    if factor_mask is not None:
        ok &= np.asarray(factor_mask, dtype=bool)
    if fwd_mask is not None:
        ok &= np.asarray(fwd_mask, dtype=bool)
    out = np.full(T, np.nan)
    for t in range(T):
        o = ok[t]
        if o.sum() < min_valid:
            continue
        if np.ptp(f[t][o]) == 0 or np.ptp(r[t][o]) == 0:
            continue
        ra = _ordinal_rank_1d(f[t][o])
        rb = _ordinal_rank_1d(r[t][o])
        out[t] = _two_pass_corr(ra, rb)
    return out

def np_parameter_scan(X: np.ndarray, mask: np.ndarray | None) -> list:
    res = []
    for desc in (False, True):
        for use_mask in (True, False):
            m = mask if use_mask else None
            res.append(np_cs_rank(X, m, desc))
    return res
