# -*- coding: utf-8 -*-
"""PoC ② 公平基线——三臂共享算子层 v1（parity 与 perf 唯一实现入口）。

设计依据（GPT-5.6-Sol 审查 H4 修复 + 冻结契约）：
- parity 与 perf 必须调用同一算子实现，禁止两套语义代码（H4）
- 每操作正确输入 + mask 语义（H1/H2）
- 稳健中心化算法（两遍、pair-specific）+ 退化对角 + 三角镜像（H5/M1）
- rolling_ic 支持 factor_mask/fwd_mask 独立输入（H2）

本模块是三臂（numpy/CuPy/QuantGplearn-Torch）实现契约操作语义的唯一来源。
parity_check_v1.py 与 perf_bench_v1.py 均 import 本模块。

接口约定（统一 panel 语义）：
- cs_rank(X(T,N), mask, descending) -> (T,N) float32 ordinal 秩
- factor_corr(F3(T,N,F), mask) -> (F,F) float64
- stock_corr(X(T,N), mask) -> (N,N) float64（pairwise-complete，稳健中心化）
- rolling_ic(f(T,N), r(T,N), factor_mask, fwd_mask, min_valid) -> (T,) float64
- parameter_scan(X(T,N), mask) -> list of 4 (T,N) float32
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

# Phase 1 (F19): the CPU numeric core now lives in fc/_cpu_core.py (single
# source of truth). backends depends on fc (NOT the reverse); the repo root is
# added to sys.path so 'fc' is importable from this benchmark module.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fc._cpu_core import (  # noqa: E402
    in_corr_domain, _ordinal_rank_1d, np_cs_rank, _two_pass_corr,
    _masked_gemm_stats, _CANCEL_RATIO_THRESHOLD, _gemm_cancel_mask,
    np_stock_corr, np_factor_corr, np_rolling_ic, np_parameter_scan,
)



# ---------------------------------------------------------------------------
# CuPy 臂（GPU）
# ---------------------------------------------------------------------------

def _as_cupy(x, cp=None, dtype=None):
    """接受 numpy 或 CuPy 数组，返回 CuPy（已上传则零拷贝，支持 resident 口径）。

    S4：同 dtype CuPy 数组直接返回原对象（原 `astype()` 默认复制，指针不同，零拷贝不成立）。
    """
    if cp is None:
        import cupy as cp
    if isinstance(x, cp.ndarray):
        if dtype is None or x.dtype == dtype:
            return x
        return x.astype(dtype)
    return cp.asarray(x, dtype=dtype, blocking=True)



def cp_cs_rank(X, mask: np.ndarray | None, descending: bool) -> np.ndarray:
    """CuPy cs_rank（host 入口）。S3：N guard fail-fast——上传前从输入元数据读 shape。"""
    import cupy as cp
    n_in = X.shape[1] if isinstance(X, cp.ndarray) else np.asarray(X).shape[1]
    if n_in > 2**24:
        raise ValueError(f"N={n_in} > 2^24，秩无法精确表示")
    outg = cp_cs_rank_gpu(
        _as_cupy(X, cp, dtype=cp.float32),
        None if mask is None else _as_cupy(mask, cp, dtype=cp.bool_),
        descending,
    )
    return cp.asnumpy(outg)


def cp_cs_rank_gpu(Xg, Mg, descending: bool):
    """纯设备 cs_rank：输入 GPU、输出 GPU（R4 resident 口径用）。

    Xg: CuPy float32/float64 (T,N)；Mg: CuPy bool (T,N) 或 None；descending: bool。
    返回 CuPy float32 (T,N) ordinal 秩（无效格 NaN）。不触发 D2H。
    """
    import cupy as cp
    if Xg.shape[1] > 2**24:
        raise ValueError(f"N={Xg.shape[1]} > 2^24，秩无法精确表示")
    Xg = Xg.astype(cp.float32, copy=False) if Xg.dtype != cp.float32 else Xg
    if descending:
        Xg = -Xg
    part = cp.isfinite(Xg)
    if Mg is not None:
        part = part & Mg
    order = cp.argsort(cp.where(part, Xg, cp.inf), axis=1, kind="stable")
    ranks = cp.argsort(order, axis=1, kind="stable").astype(cp.float32) + 1.0
    return cp.where(part, ranks, cp.nan)


def cp_stock_corr(X, mask: np.ndarray | None) -> np.ndarray:
    """CuPy stock_corr：GPU masked-GEMM 快路径 + 抵消检测回退（R6 支撑）。

    接受 numpy 或已上传 CuPy，返回 numpy (N,N)。GPU 侧 GEMM 全向量化（快）；
    抵消格（_gemm_cancel_mask 语义）回退 CPU _two_pass_corr（pair-specific，|Δr|≤1e-12）。
    """
    import cupy as cp
    Xg = _as_cupy(X, cp, dtype=cp.float64)
    _T, N = Xg.shape
    X_cpu = cp.asnumpy(Xg)
    valid = np.isfinite(X_cpu)
    if mask is not None:
        m_cpu = mask.get() if isinstance(mask, cp.ndarray) else np.asarray(mask)
        valid &= np.asarray(m_cpu, dtype=bool)
    # R5 + M-01：域校验用逐元素参与元素（valid），非整行（避免 masked-out 跨列极端值误拒）
    if not in_corr_domain(X_cpu, valid):
        raise ValueError("corr 数值域外")
    # GPU masked-GEMM
    valid_g = cp.asarray(valid, dtype=cp.bool_, blocking=True)
    M = valid_g.astype(cp.float64)
    Xm = cp.where(valid_g, Xg, 0.0)
    n = M.T @ M
    sumx = Xm.T @ M
    sumy = M.T @ Xm
    sumxy = Xm.T @ Xm
    sumx2 = (Xm * Xm).T @ M
    sumy2 = M.T @ (Xm * Xm)
    n_safe = cp.maximum(n, 1.0)
    den = n_safe - 1.0
    # CuPy 除零返回 nan/inf（不抛错），无需 errstate
    cov = (sumxy - sumx * sumy / n_safe) / den
    varx = (sumx2 - sumx * sumx / n_safe) / den
    vary = (sumy2 - sumy * sumy / n_safe) / den
    corr = cov / cp.sqrt(varx * vary)
    corr[n < 2] = cp.nan
    # 抵消检测（GPU 计算 → 回拷判定；F7：无 1e-300 加数，与 np/_gemm_cancel_mask 同步）
    cov_u = sumxy - sumx * sumy / n_safe
    varx_u = sumx2 - sumx * sumx / n_safe
    vary_u = sumy2 - sumy * sumy / n_safe
    fix = ((cp.abs(sumx * sumy / n_safe) > _CANCEL_RATIO_THRESHOLD * cp.abs(cov_u))
           | (cp.abs(sumx * sumx / n_safe) > _CANCEL_RATIO_THRESHOLD * cp.abs(varx_u))
           | (cp.abs(sumy * sumy / n_safe) > _CANCEL_RATIO_THRESHOLD * cp.abs(vary_u)))
    fix |= cp.abs(corr) > 1.0  # H-01 合理性二次判定
    fix = (fix | fix.T)
    corr_host = corr.get()
    fix_cpu = fix.get()
    # 回退 + 对角（CPU，pair-specific）
    for i, j in zip(*np.where(fix_cpu)):
        if i >= j:
            continue
        o = valid[:, i] & valid[:, j]
        if o.sum() < 2:
            v = np.nan
        else:
            v = _two_pass_corr(X_cpu[o, i], X_cpu[o, j])
        corr_host[i, j] = corr_host[j, i] = v
    for i in range(N):
        o = valid[:, i]
        # F1 (review, 2026-08-05): diagonal follows the computed self-correlation
        # (see np_stock_corr); _two_pass_corr(x,x) is NaN on var-underflow.
        v = _two_pass_corr(X_cpu[o, i], X_cpu[o, i])
        corr_host[i, i] = 1.0 if np.isfinite(v) else np.nan
    cp.cuda.get_current_stream().synchronize()
    return np.triu(corr_host) + np.triu(corr_host, 1).T


def cp_factor_corr(F3, mask: np.ndarray | None) -> np.ndarray:
    """CuPy factor_corr：GPU masked-GEMM 快路径 + 抵消检测回退（R6 支撑，同 cp_stock_corr）。

    接受 numpy 或已上传 CuPy。R5：域校验用参与元素。GEMM 12×12 小矩阵（毫秒级），
    替代 H5 后 pair-specific 回拷逐对（canonical ~22s）的性能倒退。
    """
    import cupy as cp
    if isinstance(F3, cp.ndarray):
        if F3.ndim != 3:
            raise ValueError("factor_corr 输入须 (T,N,F) 3D")
        F3_cpu = cp.asnumpy(F3)
    else:
        F3 = np.asarray(F3)
        if F3.ndim != 3:
            raise ValueError("factor_corr 输入须 (T,N,F) 3D")
        F3_cpu = F3
    T, N, F = F3_cpu.shape
    X64 = F3_cpu.astype(np.float64).reshape(T * N, F)
    valid = np.isfinite(X64)
    if mask is not None:
        m_cpu = mask.get() if isinstance(mask, cp.ndarray) else np.asarray(mask)
        valid &= np.asarray(m_cpu, dtype=bool).reshape(-1, 1)
    # R5 + M-01：域校验用逐元素参与元素（valid），非整行（避免 masked-out 跨列极端值误拒）
    if not in_corr_domain(X64, valid):
        raise ValueError("corr 数值域外：max|x|>1e150 或 min 非零|x|<1e-150")
    # GPU masked-GEMM
    Xg = cp.asarray(X64, blocking=True)
    valid_g = cp.asarray(valid, dtype=cp.bool_, blocking=True)
    M = valid_g.astype(cp.float64)
    Xm = cp.where(valid_g, Xg, 0.0)
    n = M.T @ M
    sumx = Xm.T @ M
    sumy = M.T @ Xm
    sumxy = Xm.T @ Xm
    sumx2 = (Xm * Xm).T @ M
    sumy2 = M.T @ (Xm * Xm)
    n_safe = cp.maximum(n, 1.0)
    den = n_safe - 1.0
    cov = (sumxy - sumx * sumy / n_safe) / den
    varx = (sumx2 - sumx * sumx / n_safe) / den
    vary = (sumy2 - sumy * sumy / n_safe) / den
    corr = cov / cp.sqrt(varx * vary)
    corr[n < 2] = cp.nan
    cov_u = sumxy - sumx * sumy / n_safe
    varx_u = sumx2 - sumx * sumx / n_safe
    vary_u = sumy2 - sumy * sumy / n_safe
    # F7：无 1e-300 加数，与 np/_gemm_cancel_mask 同步
    fix = ((cp.abs(sumx * sumy / n_safe) > _CANCEL_RATIO_THRESHOLD * cp.abs(cov_u))
           | (cp.abs(sumx * sumx / n_safe) > _CANCEL_RATIO_THRESHOLD * cp.abs(varx_u))
           | (cp.abs(sumy * sumy / n_safe) > _CANCEL_RATIO_THRESHOLD * cp.abs(vary_u)))
    fix |= cp.abs(corr) > 1.0  # H-01 合理性二次判定
    fix = (fix | fix.T)
    corr_host = corr.get()
    fix_cpu = fix.get()
    for i, j in zip(*np.where(fix_cpu)):
        if i >= j:
            continue
        o = valid[:, i] & valid[:, j]
        if o.sum() < 2:
            v = np.nan
        else:
            v = _two_pass_corr(X64[o, i], X64[o, j])
        corr_host[i, j] = corr_host[j, i] = v
    for i in range(F):
        o = valid[:, i]
        # F1 (review, 2026-08-05): diagonal follows the computed self-correlation
        # (see np_stock_corr); _two_pass_corr(x,x) is NaN on var-underflow.
        v = _two_pass_corr(X64[o, i], X64[o, i])
        corr_host[i, i] = 1.0 if np.isfinite(v) else np.nan
    cp.cuda.get_current_stream().synchronize()
    return np.triu(corr_host) + np.triu(corr_host, 1).T


def cp_rolling_ic(f, r,
                  factor_mask: np.ndarray | None, fwd_mask: np.ndarray | None,
                  min_valid: int) -> np.ndarray:
    """CuPy rolling_ic（host 入口）。面板向量化 ordinal rank + 两遍中心化（带 mask）。"""
    import cupy as cp
    outg = cp_rolling_ic_gpu(
        _as_cupy(f, cp, dtype=cp.float64),
        _as_cupy(r, cp, dtype=cp.float64),
        None if factor_mask is None else _as_cupy(factor_mask, cp, dtype=cp.bool_),
        None if fwd_mask is None else _as_cupy(fwd_mask, cp, dtype=cp.bool_),
        min_valid,
    )
    out = cp.asnumpy(outg)
    cp.cuda.Stream.null.synchronize()
    return out


def cp_rolling_ic_gpu(fg, rg, fmg, rmg, min_valid: int):
    """纯设备 rolling_ic：输入 GPU、输出 GPU（R4 resident 口径用）。

    fg/rg: CuPy float64 (T,N)；fmg/rmg: CuPy bool (T,N) 或 None；min_valid: int。
    ok（finite ∧ 双侧 mask）全程在 GPU 计算，不触发 D2H。返回 CuPy float64 (T,)。
    """
    import cupy as cp
    fg = fg.astype(cp.float64, copy=False) if fg.dtype != cp.float64 else fg
    rg = rg.astype(cp.float64, copy=False) if rg.dtype != cp.float64 else rg
    ok = cp.isfinite(fg) & cp.isfinite(rg)
    if fmg is not None:
        ok = ok & fmg
    if rmg is not None:
        ok = ok & rmg
    # 整面板一次 ordinal rank（fill inf 后 argsort，跨行独立）
    ff = cp.where(ok, fg, cp.inf)
    rf = cp.where(ok, rg, cp.inf)
    of = cp.argsort(ff, axis=1, kind="stable")
    rf_args = cp.argsort(rf, axis=1, kind="stable")
    ra = cp.argsort(of, axis=1, kind="stable").astype(cp.float64) + 1.0
    rb = cp.argsort(rf_args, axis=1, kind="stable").astype(cp.float64) + 1.0
    ra = cp.where(ok, ra, cp.nan)
    rb = cp.where(ok, rb, cp.nan)
    n = ok.sum(axis=1, keepdims=True)
    # 面板两遍中心化（NaN 感知）
    ma = cp.nansum(cp.where(ok, ra, 0), axis=1, keepdims=True) / n
    mb = cp.nansum(cp.where(ok, rb, 0), axis=1, keepdims=True) / n
    ac = cp.where(ok, ra - ma, 0)
    bc = cp.where(ok, rb - mb, 0)
    denom = cp.sqrt(cp.nansum(ac * ac, axis=1) * cp.nansum(bc * bc, axis=1))
    ic = cp.nansum(ac * bc, axis=1) / denom
    ic = cp.where(n.ravel() >= min_valid, ic, cp.nan)
    # 常量截面 → NaN（契约 constant_all_invalid）
    fmin = cp.nanmin(cp.where(ok, fg, cp.nan), axis=1)
    fmax = cp.nanmax(cp.where(ok, fg, cp.nan), axis=1)
    rmin = cp.nanmin(cp.where(ok, rg, cp.nan), axis=1)
    rmax = cp.nanmax(cp.where(ok, rg, cp.nan), axis=1)
    const_row = (fmin == fmax) | (rmin == rmax)
    return cp.where(const_row, cp.nan, ic)


def _cp_ord(cp, g):
    order = cp.argsort(g, kind="stable")
    rr = cp.empty(len(g), dtype=cp.float64)
    rr[order] = cp.arange(1, len(g) + 1, dtype=cp.float64)
    return rr


def cp_parameter_scan(X: np.ndarray, mask: np.ndarray | None) -> list:
    res = []
    for desc in (False, True):
        for use_mask in (True, False):
            m = mask if use_mask else None
            res.append(cp_cs_rank(X, m, desc))
    return res


# ---------------------------------------------------------------------------
# QuantGplearn-Torch 臂（GPU）
# ---------------------------------------------------------------------------

# Local development baseline dependency (PoC 2 fair baseline). QuantGplearn is an
# OPTIONAL third backend (not shipped with the package). The repo path is
# resolved in priority order: FACTOR_CUDA_QG_REPO env var, else a sibling
# `reference/QuantGplearn` under the projects dir (local dev clone -- resolved
# RELATIVELY via pathlib, no hardcoded absolute path). When absent, qg_* raises
# a clear error (parity/perf callers already try/except-degrade, so this is a
# soft failure for a non-core validation backend).
import os as _os
import pathlib as _pathlib
_QG_REPO = (_os.environ.get("FACTOR_CUDA_QG_REPO", "") or str(
    # benchmarks/ -> factor-cuda/ -> projects/ -> reference/QuantGplearn
    _pathlib.Path(__file__).resolve().parent.parent.parent / "reference" / "QuantGplearn"))


def _qg_imports():
    import sys
    if not _os.path.isdir(_QG_REPO):
        raise RuntimeError(
            "QuantGplearn baseline not found (set FACTOR_CUDA_QG_REPO); "
            "QG is an optional fair-baseline backend for development/validation, "
            "not part of the shipped package")
    sys.path.insert(0, _QG_REPO)
    from QuantGplearn.tensor_fitness import rank_2d, batch_spearmanr  # type: ignore
    import torch
    return rank_2d, batch_spearmanr, torch


def _as_torch(x, torch, dtype, device="cuda"):
    """接受 numpy 或 torch tensor，返回 CUDA tensor（已上传则零拷贝，支持 resident）。"""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


def qg_cs_rank(X, mask: np.ndarray | None, descending: bool) -> np.ndarray:
    _rank_2d, _, torch = _qg_imports()
    # S3：上传前从 host/device 元数据读 shape 校验（torch.Tensor.shape 不触发同步）
    n_in = X.shape[1] if isinstance(X, torch.Tensor) else np.asarray(X).shape[1]
    if n_in > 2**24:
        raise ValueError(f"N={n_in} > 2^24，秩无法精确表示")
    outg = qg_cs_rank_device(
        _as_torch(X, torch, torch.float32),
        None if mask is None else _as_torch(mask, torch, torch.bool),
        descending,
    )
    res = outg.cpu().numpy().astype(np.float32)
    torch.cuda.synchronize()
    return res


def qg_cs_rank_device(Xg, Mg, descending: bool):
    """纯设备 cs_rank：输入/输出 CUDA tensor（R4 resident 口径用）。"""
    rank_2d, _, torch = _qg_imports()
    if Xg.shape[1] > 2**24:
        raise ValueError(f"N={Xg.shape[1]} > 2^24，秩无法精确表示")
    if descending:
        Xg = -Xg
    return rank_2d(Xg, mask=Mg, dim=1)


def qg_rolling_ic(f: np.ndarray, r: np.ndarray,
                  factor_mask: np.ndarray | None, fwd_mask: np.ndarray | None,
                  min_valid: int) -> np.ndarray:
    """QG rolling_ic：原生 float32 batch_spearmanr——**known-deviation**（不具同语义资格）。

    披露：QG 原生是 float32 秩+Pearson，与契约 float64 偏差 ~1e-7。
    parity 应标 N/A-same-semantics；不纳入同语义最佳替代。
    """
    _rank_2d, _batch_spearmanr, torch = _qg_imports()
    outg = qg_rolling_ic_device(
        _as_torch(f, torch, torch.float32),
        _as_torch(r, torch, torch.float32),
        None if factor_mask is None else _as_torch(factor_mask, torch, torch.bool),
        None if fwd_mask is None else _as_torch(fwd_mask, torch, torch.bool),
        min_valid,
    )
    res = outg.cpu().numpy().astype(np.float64)
    torch.cuda.synchronize()
    return res


def qg_rolling_ic_device(fg, rg, fmg, rmg, min_valid: int):
    """纯设备 rolling_ic：输入/输出 CUDA tensor（R4 resident 口径用）。known-deviation（float32）。

    ok（finite ∧ 双侧 mask）、min_valid、常量截面判定全程在 device，不触发 D2H。
    """
    _, batch_spearmanr, torch = _qg_imports()
    ok = torch.isfinite(fg) & torch.isfinite(rg)
    if fmg is not None:
        ok = ok & fmg
    if rmg is not None:
        ok = ok & rmg
    out = batch_spearmanr(fg, rg, mask=ok)  # (T,) float32
    nan = torch.full_like(out, torch.nan)
    n = ok.sum(dim=1)
    out = torch.where(n >= min_valid, out, nan)
    # 常量截面 → NaN（masked 填 ±inf 后 min/max——有效行 ok 保证无 inf，替代 torch.nanmin）
    pos_inf = torch.tensor(float("inf"), device=fg.device)
    neg_inf = torch.tensor(float("-inf"), device=fg.device)
    fmin = torch.min(torch.where(ok, fg, pos_inf), dim=1).values
    fmax = torch.max(torch.where(ok, fg, neg_inf), dim=1).values
    rmin = torch.min(torch.where(ok, rg, pos_inf), dim=1).values
    rmax = torch.max(torch.where(ok, rg, neg_inf), dim=1).values
    const = (fmin == fmax) | (rmin == rmax)
    return torch.where(const, nan, out)


def qg_parameter_scan(X: np.ndarray, mask: np.ndarray | None) -> list:
    res = []
    for desc in (False, True):
        for use_mask in (True, False):
            m = mask if use_mask else None
            res.append(qg_cs_rank(X, m, desc))
    return res
