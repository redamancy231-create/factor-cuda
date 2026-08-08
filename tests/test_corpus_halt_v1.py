# -*- coding: utf-8 -*-
"""real corpus 停牌（halted）标记逻辑测试——2026-08-07 Workflow 审查处置。

守护对象（build_corpus 的 halted 参数 + fetch 的 baostock 行解析）：
1. halted 必须同时作用于 mask（停牌日 mask=False，但 price 保留填价）
   与 forward_returns（CORPUS_DESIGN §2.5：入场或出场停牌 → fwd NaN，
   防填价伪收益泄漏进 rolling_ic）。
2. mask 回退 all-True 或 fwd 污染回退（停牌窗口非 NaN）均须 FAIL——
   这两个方向是"停牌标记"修正的静默回归方向。
3. fetch 解析（parse_close_volume）的 volume/close 缺失边界。
"""
from __future__ import annotations

import numpy as np
import pytest

import fc
from benchmark_corpus.fetch_real_corpus_v1 import (derive_halted,
                                                   parse_close_volume)
from benchmark_corpus.generate_real_corpus_v1 import build_corpus

_H = 5
_LAG = 1


def _mk(T: int = 10, N: int = 3, F: int = 2):
    price = np.full((T, N), 100.0)
    factors = np.ones((T, N, F), dtype=np.float32)
    names = ["a", "b"]
    dates = [f"2024-01-{i + 1:02d}" for i in range(T)]
    return price, factors, names, dates


def test_halted_marks_mask_false_keeps_price():
    T, N = 10, 3
    price, factors, names, dates = _mk(T, N)
    halted = np.zeros((T, N), dtype=bool)
    halted[2, 1] = True
    d = build_corpus(price=price, factors=factors, names=names,
                     trade_dates=dates, halted=halted)
    assert not d["mask"][2, 1]
    assert bool(d["mask"][0, 1])
    # 停牌日 price 保留填价（mask=False 但有限，锻炼算子路径）——检查 d["price"]
    assert np.isfinite(d["price"][2, 1])


def test_mask_never_all_true_when_halted_present():
    """守护 mask 回退 all-True 回归：halted 含 True 时 mask 必须非全 True。"""
    T, N = 10, 3
    price, factors, names, dates = _mk(T, N)
    halted = np.zeros((T, N), dtype=bool)
    halted[1, 2] = True
    d = build_corpus(price=price, factors=factors, names=names,
                     trade_dates=dates, halted=halted)
    assert not d["mask"].all()
    assert d["mask"].sum() == T * N - 1


def test_fwd_nan_when_entry_or_exit_halted():
    """§2.5：入场日（t+1）或出场日（t+1+h）停牌 → fwd NaN。"""
    T, N = 12, 3
    price, factors, names, dates = _mk(T, N)
    # 入场日停牌：fwd[t=2] entry=t+1=3 停牌
    halted = np.zeros((T, N), dtype=bool)
    halted[3, 0] = True
    d = build_corpus(price=price, factors=factors, names=names,
                     trade_dates=dates, halted=halted)
    assert np.isnan(d["forward_returns"][2, 0])
    assert np.isfinite(d["forward_returns"][3, 0])
    # 出场日停牌：fwd[t=0] exit=t+1+5=6 停牌
    halted2 = np.zeros((T, N), dtype=bool)
    halted2[6, 1] = True
    d2 = build_corpus(price=price, factors=factors, names=names,
                      trade_dates=dates, halted=halted2)
    assert np.isnan(d2["forward_returns"][0, 1])
    assert np.isfinite(d2["forward_returns"][1, 1])


def test_fwd_mid_window_halt_ok():
    """窗口中间日停牌（非入场/出场）不影响持有收益（fwd 有限）。"""
    T, N = 12, 3
    price, factors, names, dates = _mk(T, N)
    halted = np.zeros((T, N), dtype=bool)
    halted[5, 1] = True  # t=0 窗口 [1,6] 的中间日
    d = build_corpus(price=price, factors=factors, names=names,
                     trade_dates=dates, halted=halted)
    assert np.isfinite(d["forward_returns"][0, 1])


def test_halted_none_unchanged():
    T, N = 10, 3
    price, factors, names, dates = _mk(T, N)
    d = build_corpus(price=price, factors=factors, names=names,
                     trade_dates=dates)
    assert bool((d["mask"] == np.isfinite(price)).all())
    # 无停牌：前 T-(h+lag) 行 fwd 全有限，末 6 行 NaN
    fwd = d["forward_returns"]
    assert np.isfinite(fwd[:T - (_H + _LAG)]).all()
    assert np.isnan(fwd[T - (_H + _LAG):]).all()


def test_halted_shape_mismatch_raises():
    T, N = 10, 3
    price, factors, names, dates = _mk(T, N)
    with pytest.raises(ValueError):
        build_corpus(price=price, factors=factors, names=names,
                     trade_dates=dates, halted=np.zeros((T - 1, N), dtype=bool))


def test_parse_close_volume_boundaries():
    """baostock 行解析：close/volume 缺失、短行、零值。"""
    assert parse_close_volume(["2024-01-01", "8.79", "46604260"]) == (8.79, 46604260.0)
    c, v = parse_close_volume(["2024-01-01", "", ""])
    assert np.isnan(c) and v == 0.0
    c, v = parse_close_volume(["2024-01-01", "None", "None"])
    assert np.isnan(c) and v == 0.0
    c, v = parse_close_volume(["2024-01-01", "8.79"])  # 短行（无 volume）
    assert c == 8.79 and v == 0.0
    c, v = parse_close_volume(["2024-01-01", "8.79", "0"])
    assert v == 0.0
    c, v = parse_close_volume(["2024-01-01", "8.79", "0.0"])
    assert v == 0.0


def test_parse_close_volume_more_boundaries():
    """审查 F4：nan/inf/空白/非法 token/负值 边界不抛异常、保守判停牌候选。"""
    c, v = parse_close_volume(["2024-01-01", "8.79", "nan"])  # "nan" → 0.0
    assert c == 8.79 and v == 0.0
    c, v = parse_close_volume(["2024-01-01", "8.79", "inf"])  # inf → 0.0
    assert v == 0.0
    c, v = parse_close_volume(["2024-01-01", "8.79", " 5000 "])  # 空白 strip
    assert v == 5000.0
    c, v = parse_close_volume(["2024-01-01", "8.79", "abc"])  # 非法 → 0.0
    assert v == 0.0
    c, v = parse_close_volume(["2024-01-01", "8.79", "-5"])  # 负 → 0.0
    assert v == 0.0
    c, v = parse_close_volume(["2024-01-01", "NaN", "5000"])  # close NaN
    assert np.isnan(c) and v == 5000.0
    c, v = parse_close_volume(["2024-01-01", "INF", "5000"])  # close inf → NaN
    assert np.isnan(c) and v == 5000.0


def test_derive_halted():
    """审查 F4：volume<=0（含缺失→0）→ halted True。"""
    h = derive_halted([100.0, 0.0, -1.0, 5.0])
    assert list(h) == [False, True, True, False]


def test_fetch_pipeline_halted_wiring():
    """审查 F7：fetch 接线端到端（parse → derive_halted → build_corpus）——
    守护 fetch 删 halted=halted / 改坏 volume<=0 / 错位 accepted 列的回归。"""
    # 模拟 2 股 x 5 天原始 baostock 行（600000 含 volume 缺失/0 停牌，600001 全正常）
    raw = {
        "sh.600000": [["2024-01-01", "8.79", "46604260"],
                      ["2024-01-02", "8.85", "50000000"],
                      ["2024-01-03", "8.85", ""],
                      ["2024-01-04", "8.85", "0"],
                      ["2024-01-05", "8.90", "60000000"]],
        "sh.600001": [["2024-01-01", "10.0", "1000"],
                      ["2024-01-02", "10.1", "2000"],
                      ["2024-01-03", "10.2", "3000"],
                      ["2024-01-04", "10.3", "4000"],
                      ["2024-01-05", "10.4", "5000"]],
    }
    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    ids = sorted(raw)
    T, N = 5, 2
    price = np.full((T, N), np.nan)
    halted = np.zeros((T, N), dtype=bool)
    for i, code in enumerate(ids):
        for t, row in enumerate(raw[code]):
            c, v = parse_close_volume(row)
            price[t, i] = c
            halted[t, i] = derive_halted([v])[0]
    factors = np.ones((T, N, 2), dtype=np.float32)
    d = build_corpus(price=price, factors=factors, names=["a", "b"],
                     trade_dates=dates, ids=ids, halted=halted)
    # 停牌日（volume 缺失/0）mask=False 且 price 有限（填价）；无停牌股全 True
    assert not d["mask"][2, 0] and not d["mask"][3, 0]
    assert np.isfinite(d["price"][2, 0])
    assert d["mask"][:, 1].all()
    assert d["mask"].sum() == T * N - 2


def test_rolling_ic_fwd_nan_not_counted_in_min_valid():
    """审查 F5：fwd 停牌窗口 NaN 不计入 min_valid（isfinite ∩ mask）。
    rolling_ic 契约：ok = isfinite(f) & isfinite(r) & factor_mask & fwd_mask。"""
    T, N = 10, 20
    rng = np.random.default_rng(0)
    f = rng.standard_normal((T, N)).astype(np.float32)
    r = rng.standard_normal((T, N)).astype(np.float64)
    fm = np.ones((T, N), dtype=bool)
    rm = np.ones((T, N), dtype=bool)
    # 某行 fwd 全 NaN（模拟整行停牌窗口）→ 有效 0 < min_valid → IC NaN
    r[3, :] = np.nan
    ic = fc.rolling_ic(f, r, factor_mask=fm, fwd_mask=rm, min_valid=5, device="cpu")
    assert np.isnan(ic[3]) and np.isfinite(ic[0])
    # 部分 NaN 与 mask=False 排除同一格集合 → IC 逐位一致（fwd NaN 与 mask 等价排除）
    r2 = rng.standard_normal((T, N)).astype(np.float64)
    r2[1, :4] = np.nan
    rm2 = np.ones((T, N), dtype=bool)
    rm2[1, :4] = False
    ic2 = fc.rolling_ic(f, r2, factor_mask=fm, fwd_mask=rm, min_valid=5, device="cpu")
    ic3 = fc.rolling_ic(f, r2, factor_mask=fm, fwd_mask=rm2, min_valid=5, device="cpu")
    assert np.array_equal(ic2, ic3, equal_nan=True), \
        "fwd NaN 与 fwd_mask=False 排除同一格集 → 结果逐位一致"


def test_factor_corr_mask_false_finite_excluded():
    """审查 F6：mask=False 但 price 有限（停牌填价）格不参与 corr——
    mask=False 有限 vs 手动置 NaN 应等值（mask 权威排除）。"""
    T, N, F = 40, 12, 3
    rng = np.random.default_rng(1)
    F3 = rng.standard_normal((T, N, F))
    mask = np.ones((T, N), dtype=bool)
    mask[5, 3] = False
    mask[7, 8] = False
    r_full = fc.factor_corr(F3, mask, backend="cpu")
    F3b = F3.copy()
    F3b[5, 3, :] = np.nan  # 手动置 NaN（等价排除，mask 仍 False）
    F3b[7, 8, :] = np.nan
    r_nan = fc.factor_corr(F3b, mask, backend="cpu")
    assert np.array_equal(r_full, r_nan, equal_nan=True), \
        "mask=False 但有限 vs mask=False 且 NaN 应等值（mask 权威排除）"
