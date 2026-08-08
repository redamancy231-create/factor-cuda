# -*- coding: utf-8 -*-
"""real corpus 真摄入——baostock 下载沪深 300 成分股行情 -> 标准 corpus。

链路：baostock 登录/股票池/历史日线下载 -> 价格矩阵 -> 价格派生因子 ->
build_corpus 摄入 -> write_corpus（npz+manifest）。小规模试点（20 股 x 1 年）
已验证；本脚本支持 --n-stocks/--start/--end 扩展（默认 100 股 x 2020-2024）。
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from benchmark_corpus.generate_real_corpus_v1 import (  # noqa: E402
    H, LAG, build_corpus, write_corpus)

START = "2020-01-01"
END = "2024-12-31"
N_STOCKS = 100


def parse_close_volume(r: list[str]) -> tuple[float, float]:
    """baostock 行解析（2026-08-07 审查 F4 健壮化）：strip+大小写归一；
    close 缺失/非有限 → NaN；volume 缺失/NaN/负/非有限 → 0.0（= 无成交 = 停牌候选）。
    r = [date, close, volume]。非法 token 不抛异常（返回默认），防单格中止整次抓取。"""
    def _num(s: str, default: float) -> float:
        if not isinstance(s, str):
            return default
        t = s.strip().lower()
        if t in ("", "none", "nan", "null", "na", "-"):
            return default
        try:
            return float(t)
        except ValueError:
            return default

    close = _num(r[1], np.nan) if len(r) >= 2 else np.nan
    vol = _num(r[2], 0.0) if len(r) >= 3 else 0.0
    if not np.isfinite(close):
        close = np.nan
    if not np.isfinite(vol) or vol < 0:
        vol = 0.0
    return close, vol


def derive_halted(volumes: list[float]) -> np.ndarray:
    """从 per-stock volume 序列构造 halted bool（2026-08-07 审查 F4 提取为纯函数）：
    volume<=0（含缺失→0）→ True（停牌）。"""
    return np.asarray(volumes, dtype=np.float64) <= 0.0


def derive_factors(price: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """价格派生因子（审查 R6 修正：momentum_1 正向动量命名、vol_10 用 10 个收益 as-of t）。"""
    T, N = price.shape
    logp = np.log(price)
    mom5 = np.full((T, N), np.nan)
    mom1 = np.full((T, N), np.nan)
    vol10 = np.full((T, N), np.nan)
    with np.errstate(all="ignore"):
        mom5[5:, :] = logp[5:, :] - logp[:-5, :]   # 5 日对数动量（as-of t 收盘）
        mom1[1:, :] = logp[1:, :] - logp[:-1, :]   # 1 日对数动量（正向命名，非「反转」）
    for t in range(10, T):
        # 10 个收益（price[t-10:t+1] 11 个价格），as-of t 收盘（与 mom 同时点）
        vol10[t, :] = np.nanstd(np.diff(logp[t - 10:t + 1], axis=0), axis=0)
    factors = np.stack([mom5, mom1, vol10], axis=-1).astype(np.float32)
    return factors, ["momentum_5", "momentum_1", "vol_10"]


def main() -> int:
    import baostock as bs

    query_time = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    lg = bs.login()
    if lg.error_code != "0":
        print(f"baostock login FAIL: {lg.error_code} {lg.error_msg}")
        return 1
    print(f"baostock login OK")

    rs = bs.query_hs300_stocks()
    codes = []
    while rs.error_code == "0" and rs.next():
        codes.append(rs.get_row_data()[1])  # 'sh.600000'
    codes = sorted(codes)  # 审查 R4：固定选择规则（code-asc），避免返回顺序不确定
    requested = codes[:N_STOCKS]
    print(f"HS300 components: {len(codes)}; requested first {N_STOCKS} (code-asc)")

    # 逐股下载前复权 close + volume（停牌标记用），按交易日对齐；
    # 记录接纳/拒绝清单（审查 R3/R5）；2026-08-07：volume 补停牌判定
    prices: dict[str, list[float]] = {}
    volumes: dict[str, list[float]] = {}
    dates: list[str] | None = None
    accepted: list[str] = []
    rejected: list[tuple[str, str]] = []
    for code in requested:
        rs = bs.query_history_k_data_plus(
            code, "date,close,volume", start_date=START, end_date=END,
            frequency="d", adjustflag="2")  # adjustflag=2 前复权
        rows, cols, vols = [], [], []
        while rs.error_code == "0" and rs.next():
            r = rs.get_row_data()
            rows.append(r[0])
            c, v = parse_close_volume(r)
            cols.append(c)
            vols.append(v)  # 缺失 volume → 0.0 = 无成交 = 停牌
        if not rows:
            rejected.append((code, "0 rows"))
            print(f"  {code}: 0 rows, skip")
            continue
        if dates is None:
            dates = rows
        if rows == dates:
            prices[code] = cols
            volumes[code] = vols
            accepted.append(code)
            print(f"  {code}: {len(cols)} days")
        else:
            rejected.append((code, "date axis mismatch"))
            print(f"  {code}: date axis mismatch ({len(rows)} vs {len(dates)}), skip")
    bs.logout()

    if dates is None or not prices:
        print("no data downloaded")
        return 1
    T, N = len(dates), len(prices)
    price = np.full((T, N), np.nan, dtype=np.float64)
    halted = np.zeros((T, N), dtype=bool)  # 停牌日：volume<=0（无成交）
    for i, c in enumerate(accepted):
        price[:, i] = prices[c]
        halted[:, i] = derive_halted(volumes[c])
    halted_cells = int(halted.sum())
    halted_rate = float(halted.mean())
    print(f"panel price {price.shape}, halted {halted_cells} cells "
          f"({halted_rate:.4f}), factors {derive_factors(price)[0].shape}")
    factors, names = derive_factors(price)
    print(f"  names {names}")

    d = build_corpus(price=price, factors=factors, names=names, trade_dates=dates,
                     ids=accepted, halted=halted)
    params = {"n_stocks": N, "T": T, "start": START, "end": END,
              "factors": names, "adjust": "qfq(2)",
              "selection_rule": f"HS300 current components sorted code-asc first {N_STOCKS}",
              "query_time": query_time}
    extra = {
        "selection_rule": params["selection_rule"],
        "query_time": query_time,
        "n_stocks_requested": len(requested),
        "accepted_ids": accepted,
        "rejected_ids": rejected,  # 审查 R3：接纳/拒绝清单落盘
        "mask_semantics": ("True=可交易；由 isfinite(price) & ~halted(volume==0) 派生；"
                           "停牌日 price 保留填价（mask=False 但有限，锻炼算子路径）；"
                           "volume 仅用于派生 halted 不存盘——halted 格可由 "
                           "~mask & isfinite(price) 事后识别（审计见 halted_stats）"),
        "forward_returns_rule": (f"h={H}/lag={LAG}；入场或出场停牌 → NaN（§2.5，"
                                 "防填价伪收益泄漏）；末 6 行 NaN"),
        "halted_stats": {
            "cells": halted_cells, "rate": halted_rate,
            "days_with_halt": int(halted.any(axis=1).sum()),
            "stocks_with_halt": int(halted.any(axis=0).sum()),
            "rule": "halted = (volume <= 0 or volume missing/non-finite) per (T,N)"},
        "universe_snapshot": {
            "requested_sha256": hashlib.sha256(
                "|".join(requested).encode()).hexdigest().upper(),
            "note": "query_hs300_stocks 取当前成分(code-asc first N_STOCKS)；"
                    "重新 fetch 可能因成分调整/数据修订/首股日期轴漂移而改变 "
                    "requested/accepted/T——npz 以 data_sha256 冻结为准，"
                    "重跑须对比 requested_sha256/accepted_ids（GPT-5.6-Sol 审查 F4）"},
        "generation": {"script": "benchmark_corpus/fetch_real_corpus_v1.py",
                       "script_sha256": hashlib.sha256(
                           pathlib.Path(__file__).read_bytes()).hexdigest().upper(),
                       "params": params},
        "generated_at": query_time,
    }
    manifest = write_corpus(
        d, corpus_id="corpus_real_v1", trade_dates=dates,
        source=f"baostock HS300 accepted-{N}/{len(requested)} ({START}..{END}, qfq)",
        params=params,
        data_dir=ROOT / "benchmark_corpus",
        extra_manifest=extra)
    sh = manifest["shapes"]
    print(f"corpus_real_v1 written: T={sh['T']} N={sh['N']} F={sh['F']} "
          f"sha256={manifest['hash']['data_sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
