# -*- coding: utf-8 -*-
"""rolling_ic 标签生成器 fixture v1 — 冻结时间线语义的可执行权威。

引用：reviews/_draft_contract.md §3 timeline_info_constraint / label_ownership。
契约地位：rolling_ic 的 h/入场/出场时间语义的机械验收**仅经由**本脚本 + manifest + npz
         达成（集成测试先验 hash、再运行本脚本、逐元素比较期望 npz）。
         算子黑盒只消费数组，无法（也无需）验证时间线。

冻结参数（与契约 §3 一致）：
  - h    = 5   （交易日，烘焙进 forward_returns 生成）
  - lag  = 1   （入场 = t 之后第 1 交易日收盘；出场 = t+h 之后第 1 交易日收盘）
  - 交易日历：固定连续交易日索引（此处用合成 40 个交易日；真实 corpus 由 PoC ② 提供）
  - 停牌/缺价规则：price 为 NaN 的单元在标签中置 NaN

输出：
  - 本脚本生成 tests/fixtures/rolling_ic_labels_v1.npz（期望 forward_returns + 交易日索引）
  - rolling_ic_labels_v1.json 为其 manifest（含 SHA-256）
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from typing import List

import numpy as np

VERSION = "1.0.0"
H = 5
LAG = 1
T = 40
N = 5
SEED = 20260803

HERE = pathlib.Path(__file__).resolve().parent
NPZ_PATH = HERE / "rolling_ic_labels_v1.npz"
MANIFEST_PATH = HERE / "rolling_ic_labels_v1.json"


def _trading_dates(t: int) -> List[str]:
    """合成连续交易日索引（YYYY-MM-DD 格式，跳过周末）。真实日历由 PoC ② corpus 提供。"""
    dates: List[str] = []
    day = 1
    while len(dates) < t:
        import datetime as _dt

        d = _dt.date(2024, 1, 1) + _dt.timedelta(days=day)
        day += 1
        if d.weekday() < 5:
            dates.append(d.isoformat())
    return dates


def generate() -> None:
    """生成 forward_returns 标签矩阵（h=5, lag=1）并写 npz + manifest。"""
    rng = np.random.default_rng(SEED)
    dates = _trading_dates(T)

    # 合成收盘价 [T,N]：几何随机游走，含少量停牌（NaN）单元
    log_ret = rng.normal(0.0002, 0.01, size=(T, N))
    price = np.full((T, N), np.nan, dtype=np.float64)
    price[0, :] = 100.0
    for t in range(1, T):
        price[t, :] = price[t - 1, :] * np.exp(log_ret[t, :])
    # 停牌：约 4% 单元置 NaN（mask=False 语义，标签相应置 NaN）
    halt = rng.random((T, N)) < 0.04
    price[halt] = np.nan

    # forward_returns[t] = 出场收盘/入场收盘 − 1
    #   入场 = t+LAG 收盘；出场 = t+H+LAG 收盘
    fwd = np.full((T, N), np.nan, dtype=np.float64)
    for t in range(0, T - (H + LAG)):
        entry = price[t + LAG, :]
        exit_ = price[t + H + LAG, :]
        valid = np.isfinite(entry) & np.isfinite(exit_)
        with np.errstate(all="ignore"):
            fwd[t, valid] = exit_[valid] / entry[valid] - 1.0
    # 末尾 h+lag 行无完整窗口 → 保持 NaN

    # 写 npz
    np.savez(
        NPZ_PATH,
        dates=np.array(dates, dtype=str),
        price=price,
        forward_returns=fwd,
        h=np.array([H], dtype=np.int64),
        lag=np.array([LAG], dtype=np.int64),
        seed=np.array([SEED], dtype=np.int64),
    )

    # 计算 SHA-256 并写 manifest
    digest = hashlib.sha256(NPZ_PATH.read_bytes()).hexdigest().upper()
    manifest = {
        "fixture": "rolling_ic_labels_v1",
        "version": VERSION,
        "script": str((HERE / "generate_rolling_ic_labels_v1.py").relative_to(HERE.parent)),
        "script_sha256": hashlib.sha256(
            (HERE / "generate_rolling_ic_labels_v1.py").read_bytes()
        ).hexdigest().upper(),
        "data_sha256": digest,
        "params": {"h": H, "lag": LAG, "T": T, "N": N, "seed": SEED},
        "trading_days": dates,
        "halt_rule": "price NaN -> forward_returns NaN",
        "nan_positions_rule": "末尾 h+lag 行及停牌单元为 NaN",
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"generated {NPZ_PATH} sha256={digest}")
    print(f"generated {MANIFEST_PATH}")


if __name__ == "__main__":
    generate()
