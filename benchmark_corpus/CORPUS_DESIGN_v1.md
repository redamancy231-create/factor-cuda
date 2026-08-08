# PoC ② Benchmark Corpus Manifest 设计（四提案合并终稿）

> 状态：可执行设计。合并自 4 个设计 agent 提案；跨区域不一致与统一口径见 §7，开放项见 §8。
> 依据（已冻结契约）：CLAUDE.md 操作语义契约（PoC ① 冻结）+ S9 可复现性 + PLAN.md §七 benchmark 协议 + COMPETITOR_ANALYSIS.md §四。

## 1. Corpus 目录结构与文件清单

```
benchmark_corpus/
├── generate_corpus_v1.py           # 主生成器（real 快照摄入 + synth 确定性生成 + smoke/压力变体），唯一生成入口
├── compute_corpus_stats_v1.py      # 分布统计脚本（§4.1），输出写入 manifest.stats
├── verify_corpus_v1.py             # 校验脚本（§4.4 断言清单）
├── corpus_loader_v1.py             # 唯一运行时读路径（data_sha256 校验 + 子集派生），基准只经此读取
├── seeds.json                      # 种子预登记单一真源（§3.4）；manifest 只引用不重复定义
├── CORPUS_SCHEMA.md                # 人类可读 schema（与 manifest.json 双件，用户偏好）
├── manifest_schema_v1.json         # JSON Schema draft-07（stats/calibration 逐 key 枚举，verify 用 jsonschema 校验）
├── README.md                       # 协议说明：布局/复现/版本规则/许可边界
├── corpus_real_v1.npz              # 完整 real（.gitignore，本地复现，data_sha256 锚定）
├── corpus_real_v1.manifest.json    # real manifest（提交）
├── corpus_synth_v1.npz             # 完整 synth（.gitignore，确定性重生成，data_sha256 锚定）
├── corpus_synth_v1.manifest.json   # synth manifest（提交）
├── corpus_synth_smoke_v1.npz       # smoke（T=40,N=100,F=4，提交，CI/评审/loader hash 自检）
├── parity_anchors_v1.npz           # contract parity 对抗语料（小、提交，§6）
└── parity_anchors_v1.manifest.json # 锚点 manifest（提交）
```

- **版本规则**：corpus v1 为 PoC ② 唯一冻结集；任何内容/协议/脚本/种子变更 → 新版本递增（v1→v2…），旧版本保留；变更登记 CHANGELOG.md；protocol_version 随 schema/loader 变更独立递增。
- **提交范围（git）**：上述全部除 `corpus_real_v1.npz`、`corpus_synth_v1.npz` 及 `--T/--N/--F` 压力变体产物；.gitignore 新增这两类 npz；manifest JSON/脚本/smoke/锚点绝不 gitignore。完整 npz 由生成脚本本地确定性复现，由提交的 manifest data_sha256 锚定（满足 S9「corpus manifest 在 PoC ② 前提交」）。
- **目录粒度**：每 corpus 独立 manifest JSON（非单一 manifest）——real/synth/anchors 各自自包含，降低 hash 维护面；种子统一收于 seeds.json 单点。
- **smoke 与全量共用同一生成器**（仅参数不同），保证 CI 覆盖真实生成路径。

## 2. Manifest schema（字段全列出）

每个 npz 对应一个 manifest JSON（顶层字段）：

| 字段 | 类型 | 说明 |
|------|------|------|
| corpus_id | str | "corpus_real_v1" / "corpus_synth_v1" / "parity_anchors_v1" |
| family | str | "real" / "synthetic" / "parity_anchors" |
| version | str | "v1" |
| protocol_version | str | "1"（随 schema/loader 变更递增） |
| shapes | obj | {T,N,F, T_full,N_full,F_full}；real 以快照实际冻结，synth 锚定 real 参考尺寸 |
| arrays | list | 逐数组 {name, dtype, shape, array_sha256}（见 §2.1） |
| generation | obj | {script, script_sha256, stats_script_sha256, params} |
| seeds | obj | {seeds_ref:"seeds.json", override:{...}}（**唯一真源=seeds.json，manifest 只引用不内联**，防双源漂移；override 仅记录 --seed 覆写） |
| hash | obj | {data_sha256, algorithm:"SHA-256"} |
| stats | obj | §4.1 全部统计 |
| subsets | obj | {rule, seed, indices{}}（perf 前导前缀规则 + parity 固定抽样索引，见 §2.3/§6） |
| labels | obj | {h:5, lag:1, generator, generator_sha256, W:21, benchmark_row_range:{start:W, stop:"T-(h+lag)", stop_exclusive:true}}（半开区间 [W, T-6) 即 t≤T-7，不含必然 NaN 的末行） |
| lineage | obj | real 专有：{source, fetch_date, raw_hash, factor_formulas, mask_rule, adjust_rule, N_select_rule, W, mlquant_commit} |
| calibration | obj | synth 专有：{params, source:"real"\|"baseline", tolerance_table, comparison, pass} |
| env | obj | §4.3 |
| generated_at | str | ISO8601，仅信息性、不参与任何 hash |

### 2.1 npz 数组清单（real 与 synth 同一 schema）

npz 用 `np.savez`（ZIP_STORED 未压缩，allow_pickle=False，固定写序）；seeds/provenance/stats/calibration 全部在 manifest，npz 只存纯数据 + 自描述标量。

| 键 | dtype | shape | 含义 |
|----|-------|-------|------|
| dates | '<U10' | (T,) | ISO "YYYY-MM-DD" 交易日；synth 参考尺寸复用 real dates（不可得则确定性周末日历回退） |
| ids | '<U10' | (N,) | real=股票代码升序（前 5000，含 ≥1 观测行）；synth='SYN%05d' |
| names | '<U20' | (F,) | 12 因子名（mom_20/rev_5/vol_20/vol_60/turn_20/turn_5/vr_20/amp_20/illiq_amihud/dd_20/skew_20/kurt_20）；factor_corr names 参数对齐 len==F |
| factors | float32 | (T,N,F) | 因子平面全集（F 最内，C-contiguous），factor_corr 输入 |
| factor_a | float32 | (T,N) | = factors[...,0] 的连续副本（verify 断言逐位相等），cs_rank/rolling_ic/parameter_scan 零拷贝直通 |
| returns | float32 | (T,N) | 日度简单收益（停牌格 NaN），stock_corr 输入 |
| price | float64 | (T,N) | 前复权收盘价（停牌格 NaN，复牌 carry-forward），forward_returns 校验/重派生用 |
| mask | bool | (T,N) | 全操作共享，True=可交易（唯一权威） |
| forward_returns | float64 | (T,N) | h=5/lag=1 前向收益标签（与 fixture 同规），rolling_ic 输入 |
| h | int64 | (1,) | 5 |
| lag | int64 | (1,) | 1 |
| schema_version | str 标量 | () | "v1" |
| generator_version | str 标量 | () | 生成器版本 |

### 2.2 参考尺寸

**npz 体积预算**：factors 1218×5000×12×4B≈292MB 主导，全 npz ≈420MB/个（real+synth 本地约 840MB，均 .gitignore 不提交，由生成脚本复现 + data_sha256 锚定）。**sweep 变体 F'=30/100 names 规则**：`factor_0..factor_{F'-1}`（冻结命名表，factor_corr names 对齐 len==F）。

参考尺寸 (T,N,F) = **(1218, 5000, 12)**：
- T = 2020-01-01..2024-12-31 实际交易日数（目标 1218，以快照实际冻结于 manifest）；
- N = 5000（A 股全市场量级，ids 按代码升序前 5000 只、有 ≥1 观测行；不足则取全部并冻结）；
- F = 12 代表性因子（上表 names，均前复权 OHLCV+量+换手率推导，公式冻结于生成器规格并写入 lineage）。

real 与 synth 参考尺寸一致（synth 复用 real 交易日历），保证逐点对照。

### 2.3 T/N/F 参数化

性能表按 (T',N',F') 参数化，全部由 canonical 数组确定性**前导子切片**取得：dates[0:T']、ids[0:N']、factors[...,0:F']、factor_a/returns/mask/price 相应取前 N' 列；T'≤1218、N'≤5000、F'≤12。标准参数化网格：
- **N ∈ {500, 2000, 5000}、T ∈ {252, 1218}、F ∈ {5, 30, 100}**。
  - synth 提供全网格；real 因 F=12 仅 F'∈{5,12}（其余点按实际值标注，不以网格强对齐，对齐 PLAN「性能表按实际 T/N/F 参数化」）。
  - 网格上限受 RTX 4060 8GB 约束：大 F×N 组合超显存时报告分块参数（F/T 分块）。
- F'=30/100（超参考 F=12）与压力尺寸（T'=2500 复现遗留 workload、N'=10000、F'=500 供 factor_corr 压力）由 `--T/--N/--F` 确定性生成（种子=role_rng(master, "sweep", extra=f"{T}:{N}:{F}")，SHA-256 派生），benchmark 运行时物化并记录其 data_sha256 于报告（非 manifest 预登记）。
- CI smoke 用 `corpus_synth_smoke_v1.npz`（T=40,N=100,F=4，秒级）或 loader 子切片（T'=250,N'=500）。

**操作映射**：cs_rank / parameter_scan → (T,N)（主参 N、次参 T；parameter_scan 的 G=4(direction×mask_mode) 为语义轴不参与 shape 参数化）；factor_corr → (T,N,F)（主参 F/N/T，输出 (F,F)）；stock_corr → (T,N)（主参 N，输出 (N,N) 随 N² 增长）；rolling_ic → (T,N)（min_valid 固定 30，不扫描）。

### 2.4 mask 语义与生成

mask (T,N) bool，True=可交易，全操作共享，**mask 为唯一权威——严禁由数值反推 mask**；被排除格存储值可为 0.0、有限原值或 NaN（契约 §0 两种存储模式均覆盖）。
- **real**：由原始面板交易状态导出（当日有成交/未停牌 → True）；被排除格保留有限原值（前向填充至最近有效值）不置零（锻炼"mask=False 但有限"路径）；无数据/无历史格置 NaN（锻炼 isfinite 交集）。
- **synth**：per-role rng_mask 派生确定性块状停牌（马尔可夫连段 + 独立缺失 + 涨跌停锁死，公式见 §3.2），密度校准到 real 或基准（**口径统一：纯停牌率≈1.2% 容差 ±0.2pp；cell 级 mask False ≈1.8%（停牌 1.2%+限停 0.5%+缺失 0.1%），不再使用"5–15%"未定义口径**）；停牌格 price/returns=NaN、因子=0.0；涨跌停锁死格 mask=False 但价格有限（fwd 可算，由算子排除）。
- mask=True 但值 NaN 的格子自然保留（real 的因子 warm-up 行），锻炼 isfinite 交集逻辑。
- 参考尺寸每行保证有效数 ≥ min_valid=30（N=5000 下恒满足）；全 False 行不构造。

### 2.5 forward_returns 标签规则

real 与 synth 的 forward_returns 均按已提交 fixture（`tests/fixtures/generate_rolling_ic_labels_v1.py`）同一规则从价格面板推导：**h=5、lag=1、入场=t+1 收盘、出场=t+6 收盘、fwd=出场/入场−1、入场或出场缺失/停牌 → NaN、末尾 h+lag=6 行全 NaN**，float64 存储。时间线语义唯一权威仍为 fixture（契约 §3 label_ownership）；corpus 与 fixture 同规则保证 rolling_ic 基准与 oracle 测试同口径。forward_returns 由生成脚本产出并嵌入 npz，基准运行只消费不生成。

### 2.6 dtype 主线与每操作输入绑定

- factors/factor_a/returns 一律 **float32**（rank/parameter_scan 主线直通；corr/rolling_ic 接受 float32 内部提升 float64 计算）；mask 一律 bool；forward_returns **float64**（与标签 fixture 一致，锻炼 rolling_ic float64 接收路径）；price float64（内部聚合口径）。
- 每操作规范输入绑定：cs_rank=(factor_a, mask)；factor_corr=(factors, mask, names=12 因子名, len==F)；stock_corr=(returns, mask)；rolling_ic=(factor=factor_a, forward_returns, factor_mask=mask, fwd_mask=None, min_valid=30, **基准行范围半开区间 [W=21, T-(h+lag)) = [21, T-6)，即 t≤T-7**——forward_returns 有效行 t∈[0,T-7]，末 6 行 [T-6,T) 全 NaN 不入基准范围)；parameter_scan=(X=factor_a, mask=mask)。
- corpus 不提供 float64 因素副本——float64 接受路径由 oracle 小 fixture / parity 锚点覆盖。

## 3. 结构保持合成生成算法

### 3.1 确定性纪律（per-role 流）

生成器采用 **per-role 派生流**：每阶段独立 `rng_k = role_rng(master, role_tag)`，每个 `rng_k` 内的 draw 调用顺序作为契约固定。阶段：① 日历（不耗 RNG）；② mask（rng_mask）；③ 收益/价格（rng_price）；④ forward_returns（不耗 RNG，纯推导）；⑤ 因子束（rng_tie 用于 tie 注入与风格簇分配）；⑥ 自检+写盘（不耗 RNG）。**任何阶段内 RNG 调用序改动=新字节=重登记 data_sha256**；per-role 流支持「REDESIGN 合成模型（绝不改种子）」的隔离语义——改某一阶段参数只顺移该阶段下游、不污染其他阶段。numpy 版本漂移由 data_sha256 自检兜底（冻结 env=Python 3.12.7 + NumPy 2.4.4）。**imports 白名单={numpy, hashlib, json, pathlib, argparse, datetime}**；禁用 mlquant/torch/scipy。**派生编码（GPT-5.6-Sol #4 修正）**：`role_rng = default_rng(np.frombuffer(SHA-256(f"{master}:{role_tag}:{extra}"), dtype=uint32))`——用 SHA-256 无损派生，**废弃低 64 位截断**（`int.from_bytes(...) & 0xFFFFFFFFFFFFFFFF` 使不同 master 在相同 role 下碰撞，实证 20260802 与 99990802 六个角色全碰撞）；tie 流必须被消费（风格簇分配 + tie 注入均用 rng_tie）。写入 seeds.json。

### 3.2 生成公式与参数

**① 日历**：T=1218；优先复用 real 实际交易日（'<U10'）；不可得时确定性周末日历（2020-01-02 起跳过周末截取 1218 天）。日期仅作行标签，算子全部按行索引计算（契约不变式）；"结构保持 vs 可复现"边界：工作日历身份保真属 real 区域，synth 只保分布。

**② mask**：mask[t,i]=True ⟺ (非停牌) ∧ (非缺失) ∧ (非涨跌停锁死)。
- 停牌：每股双态马尔可夫，入停 p_enter=0.002/日、恢复 p_resume=1/6（均值 6 交易日，连段非 i.i.d.）；
- 独立缺失 gap~Bernoulli(0.001)；
- 涨跌停锁死：|logr| ≥ log1p(0.10) → mask=False（价格有限）。
- 结构参数硬编码为 A 股经验事实；实测参考（soft target 写 manifest）：halt≈1.29%、limit≈0.55%。

**③ 收益/价格**：
- 市场对数收益 m_t = z_t·σ_reg(t)，z~t(df=12) 方差归一；σ_reg 两态马尔可夫（正常 0.0126 / 高波动 0.025，进入 0.05 / 退出 0.10，高波动日占比≈30%）；
- 个股对数收益 r_{t,i} = μ_i + 0.6·β_i·m_t + σ_i·ε_{t,i} + φ·r_{t-1,i}；ε~t(df=4.5) 方差归一（峰度≈10.8 vs 正态 3）；β_i~N(0.9,0.3) 截断[0,2]；σ_i~Lognormal(中位 0.32/√252, σ=0.35)；μ_i~N(0,0.001)（持久漂移注入弱动量）；φ=0.10（AR(1)）。
- 钳制 r←clip(r, ±log1p(0.10))；用 numpy rng.standard_t，不依赖 scipy。
- 价格：P[0]=Init·exp(r_0)，Init~Lognormal(中位 e^5.5, σ=0.6)；P[t]=P[t-1]·exp(r_t)；停牌格 P[t]=NaN；复牌日用"上一有限价格 carry-forward×exp(r_t)"（还原复牌跳空，不传播 NaN）。returns=float32(exp(r)−1)，停牌格 NaN。

**④ forward_returns**：fwd[t]=P[t+1+5]/P[t+1]−1（与 fixture 公式逐字一致）；停牌/窗口不足 → NaN；末尾 6 行全 NaN；float64。实测 fwd 有效≈96.9%。

**⑤ 因子束 (T,N,F=12) 4 层结构**：
- f=0 momentum = trailing 5 日累计收益（ret 中 NaN 按 0 求和 → 全有限，注入弱正动量使 IC≈0.03 现实量级）；
- f=1..F-3 连续特质因子 X = 0.85·trait_f + √(1−0.85²)·idio（**idio 逐 (t,i) i.i.d.，U_c/G 为 per-stock 静态——明确时间结构，防止因子列跨日不变使 pooled 统计退化为单日截面**）；trait_f = ρ_f·U_c + 0.25·√(1−ρ_f²)·G；ρ_f~U(0.7,0.95)；K=6 风格簇；U_c 个股静态风格特质；G 全局规模特质（还原簇内相关≈0.6-0.8、簇间≈0.01-0.1）；verify 加「因子列跨日变异>0」断言；
- f=F-2 sign(momentum) ∈ {−1,0,+1}（稠密 tie ≈100%）；
- f=F-1 特质因子 z：显式分组注入（P3 修正——废弃固定 step 舍入，其在 N=5000 实测 tie≈98%）：约 50% 单元由 rng_tie 归入 `n_tie_vals ≈ N/8` 个固定值组（每组均值 ~4 单元），使逐日"重复单元比例"稳定 ∈[0.4,0.7]，与 N 无关（N=100 与 N=5000 同强度，实测 ~0.49-0.50）。
- 全部因子先 **mask-aware 截面 z-score**（mean/std 仅用有效格，公式与 ml-quant cs_zscore tensor_factors.py:83-91 一致），后 masked 格置 0.0、cast float32。F 必须 ≥3（4 层结构成立），否则 ValueError。

**tie 三剖面**（soft target，实测值写 manifest，自检容差带：dense≥0.9、moderate∈[0.2,0.8]、连续≈0）；tie 在存储的 float32 值上精确判定（与契约 tie 判定层一致）。

**⑥ 自检+写盘**：shapes/dtypes、factor_a==factors[...,0] 逐位、行有效数、tie 容差带、in-domain、fwd 末尾 6 行 NaN；统计写入 manifest；np.savez ZIP_STORED 写 npz + data_sha256 + manifest。

### 3.3 校准协议（结构保持 vs 可复现）

固定种子下迭代模型参数，生成统计对比表（real vs synth）写入 manifest.calibration。**容差表冻结**：停牌率 ±0.2pp、tie_rate 相对 ±20%、mean|offdiag| ±0.05、excess kurtosis 相对 ±30%、IC 分位点 ±0.02。未达容差 → **REDESIGN 合成模型（绝不改种子）**。校准来源（real 或文档化 A 股基准：截面 std≈1%、停牌率≈1.5%）写入 manifest。合成 corpus 保真边界=**分布统计保真（非逐样本）**；工作日历/合成 ticker 的保真损失显式文档化（对齐 methodology 实证声明范围自审）。

### 3.4 随机种子预登记

- **seeds.json** 为种子唯一真源：master=**20260802**；派生规则=role_rng(master, role_tag, extra="") = default_rng(frombuffer(SHA-256(f"{master}:{role}:{extra}"), uint32))（无碰撞，废弃低 64 位截断），roles={price, mask, tie, calibrate_probe, subset, parity_sample}；configs：canonical_v1(T=1218,N=5000,F=12)、smoke_v1(T=40,N=100,F=4)；网格/压力变体=role_rng(master, "sweep", extra=f"{T}:{N}:{F}")。
- 提交时同步登记至 CLAUDE.md S9 与 CHANGELOG；任何种子变更=新 corpus 版本；**禁止复用 fixture 种子 20260803**（rolling_ic_labels_v1 已用）；任何 --seed 覆写必须写入 manifest override 字段，否则视为生成错误（杜绝"随机数降级 corpus"，对齐 S5 反例）。

## 4. 统计协议

### 4.1 分布统计指标（manifest.stats）

**mask/停牌/缺失**：mask_coverage、halt_rate（纯停牌=马尔可夫连段，匹配 1.2% 校准目标；不含涨跌停/缺失）、limit_lock_rate（涨跌停锁死）、nan_total_rate、nan_in_tradable_rate（mask=True 且非有限，逐因子与聚合）、valid_count_by_day{min,p25,p50,p75,max}、days_below_min_valid_30（**基准行半开区间 [W,T-6) 上，t≤T-7**）。mask=False 但值有限 与 mask=True 但值 NaN 分属不同口径不混计。

**因子相关结构**：factor_corr_matrix（F×F，两两共同有效子集 pooled Pearson，float64，np.corrcoef 语义同 corr_oracle_v1）、mean_abs_offdiag、max_abs_offdiag、pct_abs_offdiag_gt_05、top_k_var_share(k=1,2,5)、effective_rank。F=1 时 offdiag 字段置空。

**IC 分布口径**：**stable ordinal 秩 + 秩的 Pearson（即 Spearman），min_valid=30，mask&finite 交集；显式不用 scipy.average**（契约 §3 显式偏离 >1e-12）。逐因子 ic_series{mean,std,icir,t_stat,skew,p05,p50,p95,pct_gt_0.01,pct_lt_neg0.01}、ic_mean_abs。compute_corpus_stats_v1.py 正确性以 rolling_ic_labels_v1 fixture 交叉校验（在 fixture 数据上 IC 逐元素等于冻结期望）——不能只信自己写的脚本。

**厚尾/tie/值域**：厚尾——逐因子 pooled {skew, excess_kurtosis, q50,q99,q999,q001, tail_asymmetry, max_abs}；tie 密集度（float32 精确）——{tie_rate_mean, tie_rate_by_day{p25,p50,p75}, max_tie_group_size_p99, days_large_ties}（**分母=参与格 mask=True∧finite，与契约 tie 判定层一致**；masked-0 格形成的存储值 tie 群属 mask 语义非缺陷，单独记录为 storage_tie_rate≈halt_rate）；值域——{max_abs_overall, min_abs_nonzero, in_corr_domain}（in_corr_domain = max|x|≤1e150 ∧ min 非零|x|≥1e-150，**仅在 float64 数组 price/forward_returns 上有信号**；float32 因子无法表示 1e150，恒在域内）。in_corr_domain=False 的因子从 corpus 剔除并在 lineage 记录原因。

### 4.2 hash 协议

统一 **SHA-256 大写十六进制**，双层：
- `hash.data_sha256` = sha256(.npz 文件字节)——运行时完整性锚；
- `arrays[].array_sha256` = sha256(arr.tobytes('C'))——内容级、跨 numpy 版本语义锚（同内容跨环境重生成验证以 array_sha256 为准）。
生成/统计/校验/加载脚本各自 script_sha256 同算法登记。npz 以 allow_pickle=False + 固定写序 + **ZIP_STORED** 生成，保证同环境字节级复现；压缩（savez_compressed）禁用——zlib 版本变化会破坏字节 hash。

### 4.3 环境指纹（manifest.env）

python="3.12.7"、numpy="2.4.4"、env_fingerprint="python-3.12.7_numpy-2.4.4"（与 corr_oracle_v1.ENV_FINGERPRINT 严格一致）、platform、numpy_config_hash=sha256(np.show_config())（大写）、blas_library、generator_commit、generated_at（ISO8601，仅信息性不参与 hash）。生成环境必须与 oracle 锁定环境一致，否则该 corpus 数据无效。real 另记录上游版本。

### 4.4 verify_corpus_v1.py 断言清单

① npz SHA-256 == manifest data_sha256；② 各脚本 SHA-256 == manifest script_sha256；③ factor_a 与 factors[...,0] 逐位相等；④ shape/dtype/C-contiguous 校验（factors float32、mask bool、forward_returns float64、dates/ids/names 字符串）；⑤ h==5、lag==1；⑥ 每行有效数 ≥ min_valid=30（**基准行半开区间 [W,T-6) 上，t≤T-7；"有效数"=操作实际交集：rolling_ic 为 isfinite(factor)∧isfinite(fwd)∧mask∧fwd_mask**，强制 NaN 末行不入断言范围）；⑦ 末尾 6 行 forward_returns 全 NaN；⑧ 全部数值 max|x|≤1e150 且 min 非零 |x|≥1e-150（correlation 数值域内；**float64 数组 price/forward_returns 上**）；⑨ --regenerate 仅对 synth 位级复现 npz（real 依赖外部快照，只做一致性+provenance 校验，不跑位级复现，对齐 S9「原始数据如因许可不入库，记录派生过程与生成脚本」）。**校验失败即 PoC ② 阻塞**。CI 显式跑 verify_corpus_v1.py --regenerate 对 synth 做字节比对（npz 字节 + array_sha256 双锚），RNG 调用序的注释约定仅作开发期文档，机器检查点在 verify --regenerate。

### 4.5 基准引用协议（corpus_loader_v1.py）

load("real"/"synth"/"parity_anchors") 返回只读数组，加载时校验 data_sha256，不匹配抛 RuntimeError；本地完整 npz 缺失 → 明确错误提示按生成协议重生成，不静默降级；基准请求超 full 尺寸 → ValueError。基准运行时不得 import 生成/统计脚本、不得调用 RNG、不得写 npz；子集参数化由 loader 按冻结规则确定性派生（perf=前导前缀；parity=seed 派生固定索引）。基准证据输出记录 corpus_id + data_sha256 前缀（PoC 决策表② 证据产物）。

## 5. 真实 vs 合成分层报告方案

### 5.1 数据层分工
- **real corpus = 端到端性能报告 + 典型路径 parity 抽样的唯一事实源**；shape 固定（1218,5000,12），性能表按实际值标注。
- **结构保持合成 corpus 仅用于 T/N/F 参数化性能分层曲线**（real shape 固定无法扫描）；统计特征保持真实（相关结构/mask 稀疏/值域/tie/停牌率），种子预登记。
- **parity 断言只发生在①语义锚点 corpus 与②真实 corpus 固定抽样两处；合成 corpus 不进入任何 parity 断言**（防止随机合成污染语义证据，对齐 S5 反例）。

### 5.2 报告结构（两层三表）
1. **契约 parity 表**：逐操作断言 vs 契约 oracle 全过 + 竞品语义对齐矩阵（一致或记录偏差数值，逐条标注口径）。QuantGplearn 无 mask → masked parity 仅 vs 契约 oracle，报告标注"覆盖 unmasked 语义"。
2. **性能表**：单算子 microbenchmark 与端到端分开成表；real 与 synth 各成表；cudaEvent+wall-clock 双口径、预热/重复/种子、冷缓存/驻留区分、异常值/置信区间、kernel/单算子/批量/端到端四层；按 op×T×N×F 逐格报告。
3. **汇总裁决表**：按操作逐项相对最佳免费替代端到端 ≥2× 判定，绝不合并为单一加速比；某操作 FAIL 单独触发 STOP，不因其他操作达标而豁免。

**contract parity 先于性能**（COMPETITOR_ANALYSIS §四）：竞品原生语义不满足契约者（QuantGplearn 无 mask、pandas average、scipy average、CuPy corrcoef 归约差异）分别报告"原生语义性能"与"契约对齐后性能"两个数字，加速比只取同语义口径。rank 整数秩 vs QuantGplearn ordinal（一致）、pandas average（tie 上记录偏差）；rolling_ic ordinal vs scipy average 分口径。某操作最佳免费替代不可对齐（如 stock_corr 的 pairwise-masked pooled 无竞品实现）→ 用最接近可对齐口径（mask 全 True 子集）并在报告显式标注。

## 6. Contract parity 语料要求

**parity_anchors_v1.npz**（小、对抗、每操作固定 shape；与 tests/fixtures 的 corr_oracle_v1.py/rolling_ic_labels_v1.* 不重复——fixtures 是 oracle 执行器+时间线 fixture，anchors 是覆盖 4 操作的对抗输入语料）。契约冻结反例逐字嵌入。

**rank 组**：tie-free / tie-dense（大并列群）/ NaN±inf 混入 / mask 与数值不一致（mask=False 但有限 / mask=True 但 NaN）；dtype float32。断言三层：① vs 契约 stable ordinal 逐位（NaN 载荷 0x7fc00000 按位断言）；② vs QuantGplearn _rank_pct_dim 序数等价（unmasked 路径）；③ vs pandas rank(method='average')（tie-free 可断言一致，tie-dense 记录最大偏差）。descending 覆盖：corpus 预存 negate 输入；**冻结反例 y=[3,3,1] descending → [1,2,3]**（契约 §1 direction 公式 desc[i]=1+#{j:y[j]>y[i] or (y[j]==y[i] and j<i)} 实测；**K−asc+1 快捷公式才是错误方向 [2,1,3]（非 [3,2,1]）**，绝不可用——嵌入前用 numpy stable descending argsort + 契约公式双重交叉验证，并加一条自动断言防再次颠倒）。

**rolling_ic 组**（八行结构）：tie-free（vs 契约 ≤1e-12 且 == scipy，无并列时 ordinal==average）；partial-tie（vs 契约 ≤1e-12，记录 vs scipy 最大|ΔIC|，文档化偏差不断言）；常量截面（双 NaN，与 scipy 一致）；min_valid 边界行（有效数 {29,30,31} 两侧）；全无效行（NaN）；**factor 散落 NaN/±inf**（vs 契约排除+输出 NaN）；**forward_returns 散落 NaN**（只被 fwd 侧剔除）；**mask=True 但 factor NaN**（无效格并入计数，不足 min_valid 行输出 NaN）——后三用例覆盖契约 nan_mask_intersection 双侧 isfinite∧双侧 mask 交集语义（rank 组的 NaN±inf 覆盖不能替代 rolling_ic 双侧交集）。tie-free 存 float64 + float32 变体（契约按接收值域判 tie）。**冻结反例 T=2,N=100, mask 全 True, min_valid=30, 因子常量 0.5、收益常量 0.1 → IC=NaN 逐字嵌入**。

**correlation 组**：n=2 不同值→|r|=1；精确零方差→NaN；无共同有效→NaN；全无效→全 NaN；域内极端（1e150/1e-150 边界）；mask=False 但存储有限值；**契约 A/B/C 反例（A=[0,0,1], B=[1,2,NaN], C=[0,1,2]：A/B 共同子集前两行 A 常量→该条目 NaN；A/C→0.866 保留）**；float32+float64 变体。断言 vs corr_oracle_v1 全条目 |Δr|≤1e-12。

**parameter_scan 组（容器锚点）**：G=4 字典序组序（ascending,masked→ascending,unmasked→descending,masked→descending,unmasked）、dict schema（spec/groups/summary 字段齐全）、result==单次 rank、failed 组 result=None、axis_values 键序——由 oracle 测试 + 本锚点最小断言覆盖（并入 rank 组旁）。

**error 组（B 档）**：域外（max|x|>1e150 / min 非零|x|<1e-150）→ 断言抛 ValueError（API 前置条件可校验；**越界值须放在 mask=True∧finite 格内才命中域校验，且仅 float64 变体——float32 无法表示 1e150 会变 ±inf 被 isfinite 排除而非触发域错误**）；shape/dtype 非法；min_valid 非法。

**三档分离**：A 档 oracle 断言（锚点组）；B 档错误路径（error 组，断言 ValueError 而非 parity）；C 档性能排除——边界输入不进 perf corpus（perf 只含典型结构），防常量/全无效截面扭曲加速比与 IC 分布。

**真实固定抽样（典型路径 parity）**：预登记 seed + 显式索引清单（前 60 个交易日 + 随机 10 组 (t,i) 切片，抽样量上限），索引派生后写入 manifest，不依赖运行时重新取样；锚点断言与抽样结果冲突时以锚点为准并记录冲突。

**corpus 数据 parity 信号**：real 因子天然 tie（数值网格）；synth 注入三档 tie；mask/NaN 非平凡结构——使"同数据同 mask 同语义"的 parity 预检（rank ordinal vs QuantGplearn ordinal、pandas average；rolling_ic ordinal vs scipy average 分口径）有真实信号。corpus 为"真实性能语料"非对抗——对抗用例归 parity_anchors + oracle 测试。

## 7. 跨区域一致性核对（发现的不一致与统一口径）

IN1 **F 参考尺寸冲突**（1:F=12 命名因子；2:canonical F=64；3:real F=8 / synth F=16；4:网格 F∈{5,30,100}）→ 统一 **参考 F_real=F_synth=12**（冻结 12 因子名，real/synth 逐点可比）；F=64/16/8 废弃；网格 F∈{5,30,100} 中 30/100 由 --F 确定性生成（超参考 F），hash 记报告。
IN2 **T/日历冲突**（2:合成周末-only 日历 1218 天 → 2024-09-02；1/3:合成复用 real 日历 → 2024-12-31）→ 统一 **synth 参考尺寸复用 real dates**（real 不可得时回退确定性周末日历并在 manifest 标注）；T 恒=1218；日期仅作行标签，不影响算子语义。
IN3 **price 是否入 npz**（1:价格不入 npz；2:入；3:arrays 无 price）→ 统一 **price float64 (T,N) 入 npz**（forward_returns 校验/重派生、halt 统计需要；约 49MB 可接受）。
IN4 **npz 压缩与 hash 稳定性**（1:savez_compressed + >300MB 回退 F=6/N=3000；2:ZIP_STORED 字节稳定）→ 统一 **np.savez ZIP_STORED**（弃 300MB 回退；array_sha256 为跨环境语义锚；体积由 .gitignore 解决）。
IN5 **dates dtype**（2:datetime64[D]；1/3/4:'<U10'）→ 统一 **'<U10' ISO 字符串**（匹配 fixture 与 loader 约定）。
IN6 **种子体系**（1:每组件字面种子，且 SEED_PRICE=20260803 与 fixture 种子冲突；2/3:master+派生）→ 统一 **MASTER_SEED=20260802 + SHA-256 角色化派生（role_rng）**；**禁止复用 fixture 种子 20260803**。
IN7 **smoke npz 是否提交**（1:仅 loader 子切片；2/3:提交 smoke npz）→ 统一 **提交 corpus_synth_smoke_v1.npz**（T=40,N=100,F=4）+ loader 子切片供 CI perf smoke（两者并存）。
IN8 **子集派生规则**（1:N'/T' 前导前缀；3:N'/F' seed 随机子集；4:parity seed 随机 (t,i)）→ 统一 **perf 网格=前导前缀**（确定性、保时间线/标签有效性）；**parity 抽样=seed 派生固定随机索引写入 manifest**；两规则按用途分离并冻结。
IN9 **manifest 粒度/命名**（1/3:每 corpus 独立 manifest；2/4:单一 manifest）→ 统一 **每 corpus 独立 manifest JSON**（corpus_real_v1.manifest.json / corpus_synth_v1.manifest.json / parity_anchors_v1.manifest.json）+ 顶层 seeds.json 种子单点。
IN10 **ids 数组**（仅 1 含）→ 统一 **ids '<U10' (N,) 入 npz**（N 子切片确定性锚、股票子宇宙标识）。
IN11 **rolling_ic 基准行 W**（1:W=21 由 mom_20 窗口+1；2:momentum=5 日）→ 统一 **W=21 冻结于 manifest**，real/synth 共用基准行范围 [W,T-6]（跨层可比基准惯例）；synth f=0 用 5 日 momentum 仅为注入 IC 信号，与 W 无耦合。
IN12 **结构保持方法**（2:显式参数化生成模型；3:对 real 校准协议；1:SEED_CALIBRATE）→ 统一 **2 的参数化模型（§3.2）作为生成结构 + 3 的校准协议（§3.3 容差表、REDESIGN 不改种子）**；2 的实测统计为 soft target 写 manifest 并由自检容差带守护。
IN13 **real F 具体值**（1:F=12；3:F=8）→ 统一 **F_real=12**（1 的 12 因子名更具体，且与 synth 参考 F 对齐）。

其余一致性已核对一致：4 操作数据需求 ↔ dtype 主线（factors/factor_a/returns float32、mask bool、forward_returns float64、price float64）；mask (T,N) bool True=可交易共享且唯一权威；T=1218（2020-2024）日期规模；forward_returns 与 fixture 同规（h=5/lag=1/末尾 6 行 NaN）；correlation 数值域（in_corr_domain）与 1e-12 oracle 容差；parity 分口径（rank ordinal vs pandas average、rolling_ic ordinal vs scipy average）。

## 8. Open items

- **real corpus 外部原始数据快照**（2020-2024 前复权日线 OHLCV+量+成交额+换手率）获取与许可——执行依赖（非设计决策）；若 PoC ② 前不可得，real npz 生成阻塞，报告按 S9 记录派生过程，synth corpus 不受影响。**real 缺失裁决分支（预注册）**：端到端≥2× 与典型路径 parity 在 synth/smoke 上以「合成数据作用域」标注运行并写入决策表（不伪装为 real 证据）；允许对确定性子切片（loader T=250,N=500）做 parity 抽查；合成 corpus 不进 parity 断言的依据是「合成是结构保持、非对抗语义语料」（非 S5 随机降级反例——synth 是 seeded+可复现，与"随机数降级"不同）。
- **real 实际 T/N 以快照冻结**（目标 1218/5000，可能微差——若 A 股上市数 <5000 取全部并冻结；实际交易日数非 1218 则取实际）。
- **synth 复用 real 日历依赖 real corpus 可用性**：不可用时回退确定性周末日历，分层报告须标注 real/synth 日期口径差异（分布可比、日历不等）。
- **性能网格 F∈{30,100} 与压力尺寸（T=2500/N=10000/F=500）运行时生成物未在 manifest 预登记**——data_sha256 在 benchmark 报告中记录；是否在 manifest 预留 grid-hash 段待落地时定。
- **12 因子公式冻结于生成器规格的具体数值实现**（real 推导公式 + synth 4 层束对应关系）——实施细节，待 generate_corpus_v1.py 落地时写入 lineage/calibration。
- **种子登记到 CLAUDE.md S9 + CHANGELOG**——由 corpus manifest 提交 agent 落地时执行（本设计只定 seeds.json）。
- **合成因子复用 real 12 names 作标注**——synth 因子为结构保持模拟、非同一经济构造；若 PoC ② 报告需更严格可比，可改 synth 独立命名（factor_0..11），factor_corr names 对齐不受影响。