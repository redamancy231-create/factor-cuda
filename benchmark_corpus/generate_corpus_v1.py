# -*- coding: utf-8 -*-
"""factor-cuda benchmark corpus 生成器 v1 — 结构保持合成 + real 摄入入口。

引用：benchmark_corpus/_corpus_design_draft.md（PoC ② manifest 设计定稿）。
契约：CLAUDE.md 操作语义契约（冻结）+ S9 可复现性 + PLAN.md §七 benchmark 协议。

设计要点（与设计定稿一致）：
  - per-role 派生流：每阶段独立 rng_k = role_rng(master, role_tag, extra)（SHA-256 无损派生），阶段内 draw 序为契约
  - 4 层因子束：f=0 momentum / f=1..F-3 特质 / f=F-2 sign / f=F-1 舍入（tie 三剖面）
  - mask 块状停牌：马尔可夫连段 + 独立缺失 + 涨跌停锁死
  - forward_returns 与 tests/fixtures/generate_rolling_ic_labels_v1.py 同规（h=5/lag=1/末 6 行 NaN）
  - np.savez ZIP_STORED 固定写序 → 同环境字节级复现；imports 白名单={numpy, hashlib, json, pathlib, argparse, datetime}

用法：
  python benchmark_corpus/generate_corpus_v1.py --mode synth --T 40 --N 100 --F 4   # smoke
  python benchmark_corpus/generate_corpus_v1.py --mode synth                        # canonical 1218x5000x12
  python benchmark_corpus/generate_corpus_v1.py --mode real --data-dir <path>       # real 摄入（需外部快照）
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

VERSION = "1.0.0"
SCHEMA_VERSION = "v1"
HERE = pathlib.Path(__file__).resolve().parent
SEEDS_PATH = HERE / "seeds.json"
FIXTURE_DIR = HERE.parent / "tests" / "fixtures"

MASTER_SEED = 20260802
ROLE_TAGS = ("mask", "price", "tie", "calibrate_probe", "subset", "parity_sample")

# 冻结的结构参数（A 股经验事实，见设计 §3.2）
HALT_P_ENTER = 0.002      # 停牌进入概率/日
HALT_P_RESUME = 1 / 6     # 恢复概率/日（均值 6 交易日连段）
GAP_P = 0.001             # 独立缺失概率
LIMIT_LOG_R = np.log1p(0.10)  # 涨跌停阈值 |logr|>=log1p(0.10)
MARKET_SIGMA_NORMAL = 0.0126
MARKET_SIGMA_HIGH = 0.025
REGIME_ENTER_P = 0.05
REGIME_EXIT_P = 0.10
BETA_LOC, BETA_SCALE, BETA_MIN, BETA_MAX = 0.9, 0.3, 0.0, 2.0
STOCK_SIGMA_MED, STOCK_SIGMA_LOGSCALE = 0.32 / np.sqrt(252), 0.35
MU_LOC, MU_SCALE = 0.0, 0.001
AR_PHI = 0.10
RET_T_DF = 4.5            # 个股收益 t 分布自由度（峰度≈10.8）
MARKET_T_DF = 12
PRICE_INIT_LOG_MED, PRICE_INIT_LOG_SCALE = 5.5, 0.6
F_CLUSTERS = 6
F_CLUSTER_RHO_LO, F_CLUSTER_RHO_HI = 0.7, 0.95
IDIO_WEIGHT = np.sqrt(1 - 0.85 ** 2)
TRAIT_G_WEIGHT = 0.25


def role_rng(master: int, role_tag: str, extra: str = "") -> np.random.Generator:
    """per-role 派生 RNG：SHA-256(master ∥ role ∥ extra) 无损派生（无低 64 位截断碰撞）。

    用完整 SHA-256 摘要作种子（numpy SeedSequence 接受任意长度 bytes），
    不同 master/role/extra 组合必然产生不同随机流（除非 SHA-256 碰撞）。
    设计 §3.1 修正：废弃低 64 位截断（int.from_bytes & 0xFF..FF 使不同 master
    在相同 role 下碰撞，实证 20260802 与 99990802 六个角色全碰撞）。
    """
    import hashlib as _hl
    raw = f"{master}:{role_tag}:{extra}".encode("utf-8")
    digest = _hl.sha256(raw).digest()
    # 将 SHA-256 摘要转为 uint32 熵数组作种子（无损；不同 master/role 必然不同流）
    entropy = np.frombuffer(digest, dtype=np.uint32).copy()
    return np.random.default_rng(entropy)


def load_seeds() -> dict:
    return json.loads(SEEDS_PATH.read_text(encoding="utf-8"))


def trading_dates_reuse_real(t: int, real_dates: Optional[List[str]] = None) -> List[str]:
    """日历：优先复用 real 实际交易日；不可得时确定性周末日历（2020-01-02 起跳过周末）。"""
    if real_dates is not None and len(real_dates) >= t:
        return list(real_dates[:t])
    dates: List[str] = []
    d = datetime.date(2020, 1, 2)
    while len(dates) < t:
        if d.weekday() < 5:
            dates.append(d.isoformat())
        d += datetime.timedelta(days=1)
    return dates


def _two_state_regime(rng: np.random.Generator, t: int) -> np.ndarray:
    """市场波动率两态马尔可夫（正常 0.0126 / 高波动 0.025）。"""
    sigma = np.full(t, MARKET_SIGMA_NORMAL, dtype=np.float64)
    state = 0
    for i in range(t):
        if state == 0:
            if rng.random() < REGIME_ENTER_P:
                state = 1
        else:
            if rng.random() < REGIME_EXIT_P:
                state = 0
        sigma[i] = MARKET_SIGMA_HIGH if state else MARKET_SIGMA_NORMAL
    return sigma


def generate_mask(rng_mask: np.random.Generator, t: int, n: int, prices_init: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """块状停牌 + 独立缺失掩码。返回 (halt, gap) 两个 bool 数组（涨跌停锁死由调用方按收益判断）。"""
    halt = np.zeros((t, n), dtype=bool)
    state = np.zeros(n, dtype=bool)
    for i in range(n):
        for day in range(t):
            if not state[i]:
                if rng_mask.random() < HALT_P_ENTER:
                    state[i] = True
            else:
                if rng_mask.random() < HALT_P_RESUME:
                    state[i] = False
            halt[day, i] = state[i]
    gap = rng_mask.random((t, n)) < GAP_P
    return halt, gap


def generate_synth(t: int, n: int, f: int, dates: List[str]) -> dict:
    """生成结构保持合成 corpus（设计 §3.2 算法）。返回 npz 数据 dict。"""
    seeds = load_seeds()
    master = seeds["master_seed"]
    rng_mask = role_rng(master, "mask")
    rng_price = role_rng(master, "price")
    rng_tie = role_rng(master, "tie")

    # ---- 收益（阶段③a：先算收益，供限停判断）----
    init_prices = np.exp(rng_price.normal(PRICE_INIT_LOG_MED, PRICE_INIT_LOG_SCALE, size=n))
    regime_sigma = _two_state_regime(rng_price, t)
    m = rng_price.standard_t(MARKET_T_DF, size=t) / np.sqrt(MARKET_T_DF / (MARKET_T_DF - 2)) * regime_sigma
    beta = np.clip(rng_price.normal(BETA_LOC, BETA_SCALE, size=n), BETA_MIN, BETA_MAX)
    stock_sigma = np.exp(rng_price.normal(np.log(STOCK_SIGMA_MED), STOCK_SIGMA_LOGSCALE, size=n))
    mu = rng_price.normal(MU_LOC, MU_SCALE, size=n)

    ret = np.empty((t, n), dtype=np.float64)
    prev = np.zeros(n, dtype=np.float64)
    for day in range(t):
        eps = rng_price.standard_t(RET_T_DF, size=n) / np.sqrt(RET_T_DF / (RET_T_DF - 2))
        r = mu + 0.6 * beta * m[day] + stock_sigma * eps + AR_PHI * prev
        r = np.clip(r, -LIMIT_LOG_R, LIMIT_LOG_R)
        ret[day] = r
        prev = r

    # ---- mask（阶段②+限停）：停牌∨缺失∨涨跌停锁死 → False ----
    halt, gap = generate_mask(rng_mask, t, n, init_prices)
    limit_lock = np.abs(ret) >= LIMIT_LOG_R  # |logr|>=log1p(0.10)，价格仍有限
    mask = ~(halt | gap | limit_lock)

    # ---- 价格（阶段③b）：停牌格 NaN，限停格价格有限（由算子排除），复牌 carry-forward×exp(r）----
    price = np.full((t, n), np.nan, dtype=np.float64)
    last_valid = np.full(n, np.nan, dtype=np.float64)
    for day in range(t):
        valid = ~halt[day]
        carry = np.where(np.isnan(last_valid), init_prices, last_valid)
        price[day, valid] = carry[valid] * np.exp(ret[day, valid])
        last_valid = np.where(valid, price[day], last_valid)

    returns = np.full((t, n), np.nan, dtype=np.float64)
    returns[~halt] = np.expm1(ret[~halt])

    # ---- forward_returns（阶段④，与 fixture 同规 h=5/lag=1）----
    h, lag = 5, 1
    fwd = np.full((t, n), np.nan, dtype=np.float64)
    for day in range(0, t - (h + lag)):
        entry = price[day + lag]
        exit_ = price[day + h + lag]
        valid = np.isfinite(entry) & np.isfinite(exit_)
        with np.errstate(all="ignore"):
            fwd[day, valid] = exit_[valid] / entry[valid] - 1.0

    # ---- 因子束（阶段⑤）----
    # f=0 momentum：trailing 5 日累计收益（NaN 按 0 求和 → 全有限）
    mom = np.zeros((t, n), dtype=np.float64)
    ret_filled = np.where(np.isnan(ret), 0.0, ret)
    for day in range(t):
        mom[day] = ret_filled[max(0, day - 4): day + 1].sum(axis=0)

    # f=1..F-3 特质因子（簇相关结构）
    rho_f = rng_price.uniform(F_CLUSTER_RHO_LO, F_CLUSTER_RHO_HI, size=f)
    U_c = rng_price.normal(0, 1, size=(n, F_CLUSTERS))
    G = rng_price.normal(0, 1, size=n)
    traits = np.zeros((t, n, f), dtype=np.float64)
    idio = rng_price.normal(0, 1, size=(t, n, f))
    # 风格簇分配：消费 rng_tie 专属流（GPT-5.6-Sol #4：tie 流此前未消费）
    cluster_assignment = rng_tie.integers(0, F_CLUSTERS, size=f)
    for k in range(f):
        c = cluster_assignment[k] % F_CLUSTERS
        # trait = ρ_f * U_c[:,c] + 0.25*sqrt(1-ρ_f²)*G   —— per-stock 静态，形状 (n,)
        trait: np.ndarray = (
            rho_f[k] * U_c[:, c] + TRAIT_G_WEIGHT * np.sqrt(1 - rho_f[k] ** 2) * G
        )
        # 因子值逐 (t,i)：静态 trait（(n,) 沿 t 广播） + i.i.d. idio（(t,n)）
        traits[:, :, k] = 0.85 * trait[None, :] + IDIO_WEIGHT * idio[:, :, k]

    # f=F-2 sign(momentum) ∈ {-1,0,+1}（稠密 tie）
    sign_factor = np.sign(mom)

    # f=F-1 特质 z：构造中等 tie（P3 修复——tie 强度须与有效 N 无关）。
    # 做法：约 tie_frac=0.5 的单元由 rng_tie 归入固定值组，其余保持连续值（tie≈0）。
    # 组数按"进组单元数"缩放：n_tie_vals ≈ (N*tie_frac)/4，使每组均值 ~4 单元，
    # 逐日"重复单元比例"稳定 ∈[0.4,0.7]，与 N 无关（N=100 与 N=5000 同强度）。
    z = idio[:, :, -1] if f > 3 else np.zeros((t, n))
    z = z.astype(np.float64)
    tie_frac = 0.5
    m_tie = max(int(n * tie_frac), 2)          # 进组单元数（按当日 N）
    n_tie_vals = max(int(m_tie / 4), 2)        # 组数：每组均值 ~4 单元 → 稳定 moderate tie
    # 固定值组：在 z 值域内确定性采样 n_tie_vals 个离散值
    z_finite = z[mask & np.isfinite(z)]
    if n_tie_vals <= z_finite.size:
        tie_vals = np.quantile(z_finite, np.linspace(0.05, 0.95, n_tie_vals))
    else:
        tie_vals = np.linspace(-2.0, 2.0, n_tie_vals)
    tie_mask_flat = rng_tie.random((t, n)) < tie_frac
    rounded = z.copy()
    rounded[tie_mask_flat] = tie_vals[rng_tie.integers(0, n_tie_vals, size=tie_mask_flat.sum())]

    # 组装因子束
    factors = np.empty((t, n, f), dtype=np.float32)
    if f >= 3:
        factors[:, :, 0] = mom
        if f >= 4:
            factors[:, :, 1:f - 2] = traits[:, :, 1:f - 2]
        factors[:, :, f - 2] = sign_factor
        factors[:, :, f - 1] = rounded
    else:
        raise ValueError(f"F 必须 ≥3，got {f}")

    # mask-aware 截面 z-score（ml-quant cs_zscore 语义）+ masked 格置 0 + cast float32
    for day in range(t):
        valid = mask[day]
        for k in range(f):
            vals = factors[day, valid, k]
            if valid.sum() >= 2:
                mean = vals.mean()
                std = vals.std()
                if std > 1e-12:
                    factors[day, :, k] = (factors[day, :, k] - mean) / std
                else:
                    factors[day, :, k] = 0.0
    factors = np.asarray(factors, dtype=np.float32)
    factors[~mask] = 0.0

    factor_a = np.ascontiguousarray(factors[:, :, 0])

    return {
        "dates": np.array(dates, dtype="<U10"),
        "ids": np.array([f"SYN{i:05d}" for i in range(n)], dtype="<U10"),
        "names": np.array(
            ["mom_20", "rev_5", "vol_20", "vol_60", "turn_20", "turn_5",
             "vr_20", "amp_20", "illiq_amihud", "dd_20", "skew_20", "kurt_20"][:f],
            dtype="<U20",
        ),
        "factors": factors,
        "factor_a": factor_a,
        "returns": np.asarray(returns, dtype=np.float32),
        "price": price,
        "mask": mask,
        "forward_returns": fwd,
        "h": np.array([h], dtype=np.int64),
        "lag": np.array([lag], dtype=np.int64),
        "schema_version": np.array(SCHEMA_VERSION),
        "generator_version": np.array(VERSION),
    }


def sha256_file(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest().upper()


def write_npz(out: pathlib.Path, data: dict) -> str:
    np.savez(out, **data)
    return sha256_file(out)


def write_manifest(manifest_path: pathlib.Path, npz_path: pathlib.Path, corpus_id: str, family: str,
                   shapes: dict, stats: dict, generation: dict, env: dict,
                   lineage: dict | None = None, calibration: dict | None = None) -> None:
    digest = sha256_file(npz_path)
    # GPT-5.6-Sol #7 修复：记录每个数组的内容级 hash（array_sha256）
    arrays = []
    with np.load(npz_path, allow_pickle=False) as z:
        for k in z.files:
            arr = z[k]
            if arr.dtype.kind in "OUS":
                # 字符串数组：用 repr 字节 hash（tobytes 对固定宽度有效）
                ah = hashlib.sha256(arr.tobytes()).hexdigest().upper()
            else:
                ah = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest().upper()
            arrays.append({"name": k, "dtype": str(arr.dtype), "shape": list(arr.shape),
                           "array_sha256": ah})
    manifest = {
        "corpus_id": corpus_id,
        "family": family,
        "version": "v1",
        "protocol_version": "1",
        "shapes": shapes,
        "arrays": arrays,
        "generation": generation,
        # P4：记录完整生成参数（variant/mode/T/N/F），使 regenerate 可恢复原命令
        "generation_params": {"mode": family,
                              "variant": (generation or {}).get("variant", "canonical"),
                              "T": shapes.get("T"), "N": shapes.get("N"), "F": shapes.get("F")},
        "seeds": {"seeds_ref": "seeds.json"},
        "hash": {"data_sha256": digest, "algorithm": "SHA-256"},
        "stats": stats,
        "labels": {"h": 5, "lag": 1, "W": 21,
                   "benchmark_row_range": {"start": 21, "stop": "T-(h+lag)", "stop_exclusive": True}},
        "env": env,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if lineage is not None:
        manifest["lineage"] = lineage
    if calibration is not None:
        manifest["calibration"] = calibration
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="factor-cuda benchmark corpus generator v1")
    ap.add_argument("--mode", choices=("synth", "real"), default="synth")
    ap.add_argument("--variant", choices=("canonical", "smoke", "sweep"), default="canonical",
                    help="GPT-5.6-Sol #6：显式身份路由——canonical 尺寸不符时拒绝")
    ap.add_argument("--T", type=int, default=1218, help="交易日数")
    ap.add_argument("--N", type=int, default=5000, help="股票数")
    ap.add_argument("--F", type=int, default=12, help="因子数（≥3）")
    ap.add_argument("--out-dir", default=str(HERE), help="输出目录")
    ap.add_argument("--data-dir", default=None, help="real 模式原始数据目录")
    ap.add_argument("--seed", type=int, default=None, help="覆写种子（须写 override）")
    args = ap.parse_args()

    if args.seed is not None:
        print("ERROR: --seed 覆写必须写入 manifest.seeds.override，否则视为生成错误")
        sys.exit(1)

    # #6：variant 自动设定尺寸（覆盖默认值），显式 --T/--N/--F 优先
    VARIANT_DEFAULTS = {
        "canonical": (1218, 5000, 12),
        "smoke": (40, 100, 4),
    }
    if args.variant in VARIANT_DEFAULTS:
        if args.T == 1218 and args.N == 5000 and args.F == 12:
            # 用户未显式指定 → 用 variant 默认
            args.T, args.N, args.F = VARIANT_DEFAULTS[args.variant]
    # canonical 尺寸不符拒绝（除非显式 sweep 覆盖）
    if args.variant == "canonical" and (args.T, args.N, args.F) != (1218, 5000, 12):
        print(f"ERROR: canonical variant 必须 (T,N,F)=(1218,5000,12)，got ({args.T},{args.N},{args.F})。"
              f"其他尺寸用 --variant sweep。")
        sys.exit(1)
    if args.variant == "smoke" and (args.T, args.N, args.F) != (40, 100, 4):
        print(f"ERROR: smoke variant 尺寸应为 (40,100,4)，got ({args.T},{args.N},{args.F})")
        sys.exit(1)

    # R1 修复（GPT-5.6-Sol 第三轮）：real 真摄入 reader 尚未实现，无条件阻塞 real 身份产物。
    # 原检查目录含扩展名文件后仍调用 generate_synth 冒充 real——内容从未被消费。
    # 真摄入需实现 dates/ids/data 映射 + raw_hash/fetch_date，属外部数据依赖，超出当前范围。
    if args.mode == "real":
        print("ERROR: real 真摄入 reader 未实现——禁止产出 family=real 身份产物。"
              "PoC ② 公平基线请使用 synthetic canonical（结构保持、确定性可复现）。"
              "real 摄入将在接入外部数据快照后实现。")
        sys.exit(1)

    out_dir = pathlib.Path(args.out_dir)
    dates = trading_dates_reuse_real(args.T)
    data = generate_synth(args.T, args.N, args.F, dates)

    # #6：sweep 变体 names 等长（factor_0..factor_{F-1}），参考 12 因子名用于 canonical
    if args.F != 12:
        data["names"] = np.array([f"factor_{i}" for i in range(args.F)], dtype="<U20")

    suffix = {"canonical": "v1", "smoke": "smoke_v1", "sweep": "sweep_v1"}[args.variant]
    corpus_id = f"corpus_{'real' if args.mode == 'real' else 'synth'}_{suffix}"
    npz_path = out_dir / f"{corpus_id}.npz"
    digest = write_npz(npz_path, data)
    print(f"generated {npz_path} sha256={digest}")

    env = {"python": "3.12.7", "numpy": "2.4.4",
           "env_fingerprint": "python-3.12.7_numpy-2.4.4",
           "blas": "OpenBLAS64"}
    lineage = None
    if args.mode == "real":
        lineage = {"source": str(pathlib.Path(args.data_dir).resolve()),
                   "fetch_date": None, "raw_hash": None, "N_select_rule": "code-asc-prefix"}
    # P5 修复：生成后计算分布统计并写入 manifest.stats（非空）
    try:
        from compute_corpus_stats_v1 import compute_stats
        data_stats = compute_stats(data)
    except Exception as e:
        print(f"WARNING: compute_stats 失败，stats 留空: {e}")
        data_stats = {}
    write_manifest(
        out_dir / f"{corpus_id}.manifest.json", npz_path, corpus_id, "real" if args.mode == "real" else "synthetic",
        {"T": args.T, "N": args.N, "F": args.F}, data_stats,
        # script recorded repo-root-relative (PII-clean; no absolute install path)
        {"script": str(pathlib.Path(__file__).resolve().relative_to(HERE.parent)),
         "variant": args.variant}, env, lineage=lineage,
    )


if __name__ == "__main__":
    main()
