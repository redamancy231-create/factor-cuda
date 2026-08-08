# -*- coding: utf-8 -*-
"""factor-cuda benchmark corpus 分布统计脚本 v1 — 设计 §4.1。

输出 manifest.stats 字段。IC 口径：stable ordinal 秩 + Pearson（Spearman），min_valid=30，
mask&finite 交集，显式不用 scipy.average（契约 §3 显式偏离）。正确性以
tests/fixtures/rolling_ic_labels_v1 fixture 交叉校验。
"""
from __future__ import annotations

import json
import pathlib
from typing import Dict

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
MIN_VALID = 30
H, LAG = 5, 1
W = 21


def _corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """对共同有效子集调用 numpy corrcoef（同 corr_oracle_v1 语义，含 mask）。

    GPT-5.6-Sol #2 修复：必须纳入 mask——mask=False 的有限存储值不得参与，
    否则 masked correlation 与冻结 oracle 不一致（实证：mask 外有限值使 -0.9999 vs 1.0）。
    """
    valid = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    a, b = a[valid], b[valid]
    if a.size < 2:
        return float("nan")
    with np.errstate(all="ignore"):
        return float(np.corrcoef(np.stack([a, b]))[0, 1])


def _ordinal_rank(x: np.ndarray) -> np.ndarray:
    """stable ordinal 秩（契约 §1）：1-based，并列按索引。"""
    order = np.argsort(x, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    return ranks


def _spearman_ic(factor: np.ndarray, fwd: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """逐行截面 Spearman（stable ordinal），min_valid=30，输出 (T,) float64。

    GPT-5.6-Sol #1 修复：定秩前判断任一侧是否常量（有效因子或收益全等）→ NaN。
    契约 §3 constant_all_invalid：常量截面输出 NaN（ordinal 下虽可得互异秩，
    但无信息截面的 Spearman 语义未定义）。实证：T=28,N=100 双常量 day21 原得 0.9999。
    """
    t = factor.shape[0]
    out = np.full(t, np.nan, dtype=np.float64)
    for day in range(W, t - (H + LAG)):
        valid = mask[day] & np.isfinite(factor[day]) & np.isfinite(fwd[day])
        if valid.sum() < MIN_VALID:
            continue
        # 常量截面前置分支（契约 constant_all_invalid）
        if np.ptp(factor[day, valid]) == 0.0 or np.ptp(fwd[day, valid]) == 0.0:
            out[day] = np.nan
            continue
        rf = _ordinal_rank(factor[day, valid])
        rr = _ordinal_rank(fwd[day, valid])
        out[day] = _corr(rf, rr)
    return out


def compute_stats(d: dict) -> Dict[str, object]:
    t, n, f = d["factors"].shape
    factors, mask, fwd, price = d["factors"], d["mask"], d["forward_returns"], d["price"]
    stats: Dict[str, object] = {}

    # ---- mask/停牌/缺失 ----
    stats["mask_coverage"] = float(mask.mean())
    valid_count = mask.sum(axis=1)
    stats["valid_count_by_day"] = {
        k: float(v) for k, v in zip(
            ("min", "p25", "p50", "p75", "max"),
            (valid_count.min(), np.percentile(valid_count, 25), np.median(valid_count),
             np.percentile(valid_count, 75), valid_count.max()),
        )
    }
    # days_below_min_valid_30（半开区间 [W, T-6)）
    base_rows = mask[W:t - (H + LAG)]
    stats["days_below_min_valid_30"] = int((base_rows.sum(axis=1) < MIN_VALID).sum())
    stats["nan_total_rate"] = float(1 - np.isfinite(price).mean())
    stats["nan_in_tradable_rate"] = float(
        1 - np.isfinite(price[mask]).mean() if mask.any() else 0.0
    )

    # ---- 因子相关结构（pooled pairwise-complete）----
    cm = np.full((f, f), np.nan, dtype=np.float64)
    for i in range(f):
        for j in range(f):
            cm[i, j] = _corr(factors[:, :, i].ravel(), factors[:, :, j].ravel(), mask=mask.ravel())
    stats["factor_corr_matrix"] = cm.tolist()
    off = np.abs(cm[~np.eye(f, dtype=bool)])
    stats["mean_abs_offdiag"] = float(np.nanmean(off)) if off.size else None
    stats["max_abs_offdiag"] = float(np.nanmax(off)) if off.size else None
    stats["pct_abs_offdiag_gt_05"] = float((off > 0.5).mean()) if off.size else None

    # ---- IC 分布（stable ordinal Spearman）----
    ic_cols = []
    for k in range(f):
        ic = _spearman_ic(factors[:, :, k], fwd, mask)
        ic_cols.append(ic)
        valid_ic = ic[np.isfinite(ic)]
        if valid_ic.size:
            stats[f"ic_f{k}"] = {
                "mean": float(valid_ic.mean()),
                "std": float(valid_ic.std()),
                "icir": float(valid_ic.mean() / valid_ic.std()) if valid_ic.std() > 0 else None,
                "p05": float(np.percentile(valid_ic, 5)),
                "p50": float(np.median(valid_ic)),
                "p95": float(np.percentile(valid_ic, 95)),
                "pct_gt_0.01": float((valid_ic > 0.01).mean()),
                "pct_lt_neg0.01": float((valid_ic < -0.01).mean()),
            }
    stats["ic_mean_abs"] = float(np.nanmean(np.abs(ic_cols)))

    # ---- 厚尾/tie/值域 ----
    # GPT-5.6-Sol #3 修复：tie 率 = 逐日参与格中属于重复组的单元比例
    #   (counts[counts>1].sum() / counts.sum())，逐日独立后汇总中位数；
    #   全并列时该比例 = 1.0（旧口径 1-(counts==1).mean() 在全体并列时误返回 0.0）。
    stats["tie_rate_by_factor"] = []
    for k in range(f):
        x = factors[:, :, k]
        daily_rates = []
        for day in range(x.shape[0]):
            valid = mask[day] & np.isfinite(x[day])
            vals = x[day, valid]
            if vals.size == 0:
                continue
            _, counts = np.unique(vals, return_counts=True)
            if counts.size <= 1:
                daily_rates.append(1.0 if counts.size == 1 else 0.0)
            else:
                daily_rates.append(float(counts[counts > 1].sum() / counts.sum()))
        stats["tie_rate_by_factor"].append(float(np.median(daily_rates)) if daily_rates else 0.0)
    finite_price = price[np.isfinite(price)]
    stats["in_corr_domain"] = bool(
        finite_price.size == 0 or (
            np.abs(finite_price).max() <= 1e150
            and np.min(np.abs(finite_price[finite_price != 0])) >= 1e-150
        )
    )
    return stats


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(HERE))
    from corpus_loader_v1 import load

    d, manifest = load("corpus_synth_smoke_v1" if pathlib.Path(HERE / "corpus_synth_smoke_v1.npz").exists()
                       else "corpus_synth_v1")
    stats = compute_stats(d)
    print(json.dumps(stats, ensure_ascii=False, indent=2, default=str))
