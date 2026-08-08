# -*- coding: utf-8 -*-
"""real corpus 真摄入 reader v1——A 股数据快照 -> 标准 factor-cuda corpus。

数据未到货时的框架先行（用户决策 2026-08-06）：本脚本定义 real corpus 的
**摄入管道与输入快照格式**，并可用合成 A 股风格数据做小规模试点验证链路
（load -> sha256 校验 -> 喂 fc.* 冒烟）。数据快照到货后，只需替换输入源。

输出与 corpus_synth_v1 同 schema（CORPUS_SCHEMA.md npz 数组清单 + manifest）：
  names / factors / factor_a / returns / mask / forward_returns + schema_version

## 输入快照格式（等数据到货对接；试点用合成数据）
  price        : (T, N) float64 收盘价，NaN = 缺价（mask=False 来源）
  halted       : (T, N) bool 可选，True=停牌日（无成交，volume==0）；掩蔽 price 填价
                 导致的"mask 全 True"局限（2026-08-07 修正）。停牌日 price 保留填价
                 （有限），mask=False 锻炼"mask=False 但有限"路径（schema 约定）。
  factors      : (T, N, F) float32 因子平面（每因子一张截面）
  names        : (F,) 因子名（factor_corr names 对齐 len==F）
  trade_dates  : (T,) 交易日（ISO 字符串，仅 manifest 记录，不参与计算）

## 处理管道（对齐 corpus 语义）
  mask[t]              = isfinite(price[t]) & ~halted[t]   # True=可交易（唯一权威）
  returns[t]           = price[t]/price[t-1] - 1       # 日度简单收益（t>=1）
  forward_returns[t]   = price[t+1+h]/price[t+1] - 1   # h=5/lag=1，末 h+lag 行 NaN
  factor_a             = factors[..., 0]               # cs_rank/rolling_ic 输入

用法：PYTHONIOENCODING=utf-8 python benchmark_corpus/generate_real_corpus_v1.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
H = 5      # 与 rolling_ic 契约一致（h=5）
LAG = 1
SCHEMA_VERSION = "v1"


def _sha256(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest().upper()


def _freeze(a: np.ndarray) -> np.ndarray:
    a.setflags(write=False)
    return a


def build_corpus(*, price: np.ndarray, factors: np.ndarray, names: list[str],
                 trade_dates: list[str], ids: list[str] | None = None,
                 halted: np.ndarray | None = None) -> dict:
    """核心摄入管道：输入快照 -> 标准 corpus 数组（不写盘）。

    输出对齐 CORPUS_SCHEMA.md 统一 schema（审查 R1：补齐 dates/ids/price/
    h/lag/generator_version 六数组，共 13 键，与 corpus_synth_v1 同构）。
    halted（2026-08-07）：True=停牌日（volume==0 无成交），掩蔽前复权对停牌日
    填价导致的 mask 全 True 局限；mask = isfinite(price) & ~halted。停牌日 price
    保留填价（有限），mask=False 锻炼"mask=False 但有限"路径（schema 约定）。"""
    T, N = price.shape
    F = factors.shape[2]
    if factors.shape[:2] != (T, N):
        raise ValueError(f"factors shape {factors.shape} != price (T,N)={price.shape}")
    if len(names) != F:
        raise ValueError(f"names len {len(names)} != F {F}")
    if ids is not None and len(ids) != N:
        raise ValueError(f"ids len {len(ids)} != N {N}")
    if ids is None:
        ids = [f"asset_{i}" for i in range(N)]
    if halted is not None:
        halted = np.asarray(halted, dtype=bool)
        if halted.shape != (T, N):
            raise ValueError(f"halted shape {halted.shape} != price (T,N)={price.shape}")

    mask = np.isfinite(price)  # True=可交易（唯一权威）
    if halted is not None:
        mask &= ~halted  # 停牌日掩蔽（前复权填价 → mask 全 True 局限修正）
    # returns[t] = price[t]/price[t-1] - 1（首行 NaN——无前一日）
    returns = np.full((T, N), np.nan, dtype=np.float64)
    with np.errstate(all="ignore"):
        returns[1:, :] = price[1:, :] / price[:-1, :] - 1.0
    # forward_returns[t] = price[t+1+h]/price[t+1] - 1（末 h+lag 行 NaN）
    # CORPUS_DESIGN §2.5：入场或出场缺失/停牌 → NaN（halt 日 price 保留填价但
    # fwd 必须 NaN——防填价伪收益泄漏进 rolling_ic；Workflow 审查 2026-08-07）
    fwd = np.full((T, N), np.nan, dtype=np.float64)
    for t in range(T - (H + LAG)):
        entry = price[t + LAG]
        exit_ = price[t + H + LAG]
        valid = np.isfinite(entry) & np.isfinite(exit_)
        if halted is not None:
            valid &= ~halted[t + LAG] & ~halted[t + H + LAG]
        with np.errstate(all="ignore"):
            fwd[t, valid] = exit_[valid] / entry[valid] - 1.0

    return {
        "names": np.array(names, dtype="<U20"),
        "ids": np.array(ids, dtype="<U12"),
        "dates": np.array(trade_dates, dtype="<U10"),
        "price": _freeze(np.ascontiguousarray(price, dtype=np.float64)),
        "h": np.array([H], dtype=np.int64),
        "lag": np.array([LAG], dtype=np.int64),
        "generator_version": np.array([SCHEMA_VERSION], dtype=str),
        "factors": _freeze(np.ascontiguousarray(factors, dtype=np.float32)),
        "factor_a": _freeze(np.ascontiguousarray(factors[:, :, 0], dtype=np.float32)),
        "returns": _freeze(np.ascontiguousarray(returns, dtype=np.float32)),
        "mask": _freeze(np.ascontiguousarray(mask, dtype=bool)),
        "forward_returns": _freeze(np.ascontiguousarray(fwd, dtype=np.float64)),
        "schema_version": np.array(SCHEMA_VERSION, dtype=str),
    }


def write_corpus(d: dict, *, corpus_id: str, trade_dates: list[str],
                 source: str, params: dict, data_dir: pathlib.Path,
                 extra_manifest: dict | None = None) -> dict:
    """写 npz + manifest（对齐 manifest_schema_v1.json 必需字段，审查 R2）。
    真实数据无随机种子（seeds=null）；env/stats 由调用侧补全。"""
    import platform
    import numpy as _np

    data_dir.mkdir(parents=True, exist_ok=True)
    npz_path = data_dir / f"{corpus_id}.npz"
    manifest_path = data_dir / f"{corpus_id}.manifest.json"
    np.savez(npz_path, **d)
    digest = _sha256(npz_path)
    T, N, F = d["factors"].shape
    arrays = sorted(k for k in d if k != "schema_version")
    manifest = {
        "corpus_id": corpus_id,
        "family": "real",
        "version": "1.0.0",
        "protocol_version": "1",
        "shapes": {"T": T, "N": N, "F": F},
        "arrays": arrays,
        "generation": {"script": "benchmark_corpus/fetch_real_corpus_v1.py",
                       "script_sha256": None, "params": params},
        "seeds": {"seeds_ref": None, "override": None},  # 真实数据无随机种子
        "hash": {"data_sha256": digest, "algorithm": "SHA-256"},
        "stats": {"n_dates": T, "n_stocks": N, "n_factors": F,
                  "returns_finite": int(_np.isfinite(d["returns"]).sum()),
                  "mask_true": int(d["mask"].sum())},
        "labels": {"h": H, "lag": LAG, "W": None, "benchmark_row_range": [0, T]},
        "env": {"python": platform.python_version(),
                "numpy": _np.__version__, "env_fingerprint": None, "blas": None},
        "source": source,
        "trading_days": trade_dates,
        "mask_semantics": ("True=可交易；由 isfinite(price) 派生（唯一权威）；"
                           "真实数据 baostock 前复权对停牌日可能填价 → mask 全 True "
                           "不代表无停牌，样本选择见 params（成分快照/排序规则）"),
        "forward_returns_rule": f"h={H}/lag={LAG}，末 {H+LAG} 行 NaN",
        "generated_at": None,
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    return manifest


def synthetic_trial(corpus_id: str = "corpus_real_trial_v1", *, T: int = 500,
                    N: int = 100, F: int = 3, data_dir: pathlib.Path | None = None) -> dict:
    """合成 A 股风格数据试点：跑通摄入链路（load -> sha256 -> 冒烟），
    验证 reader 框架在数据到货前的可用性。"""
    rng = np.random.default_rng(20260806)
    # 几何随机游走价格 + 少量停牌（NaN）
    log_ret = rng.normal(0.0003, 0.015, size=(T, N))
    price = np.full((T, N), np.nan, dtype=np.float64)
    price[0, :] = 100.0
    for t in range(1, T):
        price[t, :] = price[t - 1, :] * np.exp(log_ret[t, :])
    halt = rng.random((T, N)) < 0.03
    price[halt] = np.nan
    # 因子平面（随机 + 一列动量风格）
    factors = rng.normal(size=(T, N, F)).astype(np.float32)
    names = ["momentum", "reversal", "size"]
    trade_dates = [f"2024-{1 + t // 30:02d}-{1 + t % 28:02d}" for t in range(T)]

    d = build_corpus(price=price, factors=factors, names=names,
                     trade_dates=trade_dates)
    if data_dir is None:
        data_dir = HERE
    manifest = write_corpus(d, corpus_id=corpus_id, trade_dates=trade_dates,
                            source="synthetic-trial (real-corpum reader pipeline)",
                            params={"T": T, "N": N, "F": F, "h": H, "lag": LAG},
                            data_dir=data_dir)
    return {"manifest": manifest, "data": d}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", action="store_true", help="跑合成试点")
    ap.add_argument("--corpus-id", default="corpus_real_trial_v1")
    args = ap.parse_args()

    if args.trial:
        out = synthetic_trial(args.corpus_id)
        m = out["manifest"]
        d = out["data"]
        print(f"trial corpus {args.corpus_id} written:")
        print(f"  shape T={m['shapes']['T']} N={m['shapes']['N']} F={m['shapes']['F']}")
        print(f"  data_sha256={m['hash']['data_sha256'][:16]}...")
        # 链路验证：corpus_loader.load 读回 + sha256 校验 + 冒烟
        import sys
        sys.path.insert(0, str(HERE))
        from corpus_loader_v1 import load
        loaded, lm = load(args.corpus_id)
        assert lm["hash"]["data_sha256"] == m["hash"]["data_sha256"], "sha256 mismatch"
        print(f"  load back OK (corpus_loader validated sha256)")
        # 冒烟：fc.cross_sectional_rank 消费 factor_a
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
        import fc
        rk = fc.cross_sectional_rank(loaded["factor_a"], mask=loaded["mask"])
        print(f"  fc.cross_sectional_rank smoke: rank shape {rk.shape} "
              f"finite={int(np.isfinite(rk).sum())}/{rk.size}")
        return 0
    print("usage: python benchmark_corpus/generate_real_corpus_v1.py --trial")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
