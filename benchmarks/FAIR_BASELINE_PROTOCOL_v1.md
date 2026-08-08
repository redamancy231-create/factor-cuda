# PoC ② 公平基线协议 v1（预注册）

> 状态：设计稿（预注册）
> 生成模型：DeepSeek-V4-Flash (via Claude Code CLI) · 2026-08-03
> 上游：CLAUDE.md「操作语义契约」L0 Spec（冻结，HG-2 批准）+「PoC 决策表」；COMPETITOR_ANALYSIS.md §四；CORPUS_DESIGN_v1.md §6
> 判据单一真源：CLAUDE.md「PoC 决策表」PoC ② 行——相对最佳免费替代端到端 ≥2×（同数据同 mask **且同语义**），<2× → STOP

---

## 1. 目标与范围

**目标**：在 synthetic canonical 事实源上，量化 factor-cuda 的 GPU 实现相对**最佳免费替代**的可复现边际收益，作为 PoC ② 裁决与 Phase 1–4 的输入。

**范围（本协议冻结）**：
- 对比臂：① numpy/pandas（CPU 基线）② CuPy（GPU）③ QuantGplearn-Torch（GPU）④ factor-cuda GPU 实现（后续，PoC ④ 后加入）
- 对比操作：`cross_sectional_rank`（含 descending）、`factor_corr`、`stock_corr`、`rolling_ic`、`parameter_scan`（容器）
- 事实源：`benchmark_corpus/corpus_synth_v1.npz`（synthetic canonical 1218×5000×12）+ `corpus_synth_smoke_v1.npz`（40×100×4 冒烟）
- 语义判据：`benchmark_corpus/parity_anchors_v1.npz` 27 个对抗 case + `tests/fixtures/corr_oracle_v1.py`（correlation 唯一 oracle）

**不做**：real corpus 真摄入（外部快照未得，PoC ② 以 synthetic 为事实源，作用域标注「合成数据」）；factor-cuda GPU 实现的性能（未实现）；安装门槛对比（记录于报告，不进判据）。

## 2. 三臂能力映射与语义差异（contract parity 前置）

**已知语义差异（CLAUDE.md 已知坑位 + COMPETITOR_ANALYSIS §四）**：

| 操作 | numpy/pandas 原生 | CuPy 原生 | QuantGplearn-Torch 原生 | factor-cuda 契约 |
|------|------------------|-----------|------------------------|-----------------|
| rank | pandas `method="average"` | `cp.argsort`（stable，需验证） | `_rank_pct_dim` = **stable ordinal 秩 → percentile**（rank/count） | stable ordinal 整数秩 **1..K** |
| corr | `np.corrcoef`（= oracle） | `cp.corrcoef`/手写 | `batch_pearsonr`（逐截面） | oracle wrapper ≤1e-12 |
| rolling_ic | scipy `spearmanr`（average 秩） | 手写 ordinal | `batch_spearmanr`（ordinal 秩） | stable ordinal Spearman ≤1e-12 |

**关键裁决点（contract parity 的唯一难题）**：
1. **rank 输出归一化**：QuantGplearn `_rank_pct_dim` 输出 percentile `rank/count`，factor-cuda 契约输出整数秩 `1..K`。**性能对比前必须同语义**——QuantGplearn 臂输出 ×count 转回整数秩后再比较；或全部转 percentile 比较。本协议冻结：**×count 转整数秩**（保留契约语义，且整数秩无浮点误差）。
2. **pandas average vs ordinal**：numpy/pandas 原生 rank 是 average 秩（tie 时与 ordinal 不同）。**contract parity 时不能把 pandas average 当作 ordinal 断言**——须分别报告原生语义与转换后语义，不得混为同一加速比（COMPETITOR_ANALYSIS §四）。
3. **rolling_ic 秩语义**：scipy `spearmanr` 用 average 秩；契约用 ordinal 秩。无 tie 时两者一致（≤1e-12），有 tie 时偏离——parity 锚点已含 tie 用例，记录最大偏差而非断言相等。

**能力映射表（三臂对各操作的覆盖）**：

| 操作 | numpy/pandas | CuPy | QuantGplearn-Torch |
|------|--------------|------|-------------------|
| cs_rank | ✅ pandas rank / np.argsort stable | ✅ cp.argsort | ✅ `_rank_pct_dim`（ordinal，需 ×count） |
| factor_corr | ✅ np.corrcoef（= oracle） | ⚠️ cp.corrcoef（需验证 ≤1e-12） | ⚠️ batch_pearsonr（逐截面，pooled 需重聚合） |
| stock_corr | ✅ np.corrcoef 逐对 | ⚠️ 同上 | ❌ 无直接对应（面板结构 N 轴相关不原生） |
| rolling_ic | ✅ 手写 ordinal + pearson | ⚠️ 手写 | ✅ batch_spearmanr（ordinal） |
| parameter_scan | ⚠️ 循环 + 容器模拟 | ⚠️ 同上 | ⚠️ 同上 |

**结论**：三臂能力不对称。CuPy 无现成 corrcoef 的 parity 保证、QuantGplearn 无 stock_corr。**每个臂报告其原生可实现的操作集**；缺失操作标记 N/A 不测（不作假）。

## 3. Contract Parity 检查协议

**目标**：验证每个臂在 synthetic 输入上满足契约语义（或明确记录偏离），作为性能对比的前提。

**方法**：对 `parity_anchors_v1.npz` 的 27 个 case，逐 case 执行三臂对应实现，与 expected 断言：
- `tolerance="exact"`：逐位一致（rank 整数秩 / NaN 位置）
- `tolerance="1e-12"`：|Δ|≤1e-12（corr / rolling_ic）
- `tolerance="exception"`：断言抛 ValueError（error 组）
- `tolerance="schema"`：容器 schema 检查（parameter_scan）

**每臂三个断言层**（CORPUS_DESIGN §6 三档）：
- A 档：契约 oracle 断言（锚点组）
- B 档：错误路径（error 组，抛 ValueError）
- C 档：性能排除——边界输入不进 perf

**输出**：每臂 × 每 case 的 PASS/FAIL + 实测值 + 偏差。**FAIL 的处理**：若该臂语义不可对齐契约（如 pandas average 对 tie 的固有差异），标记为「已知语义偏离」，在性能表单独列「原生语义」列，不并入「同语义」加速比计算。

## 4. 性能测量协议

**corpus**：synthetic canonical（1218×5000×12），经 `corpus_loader_v1.py` 唯一读入口（校验 data_sha256）。子集经 `subset_prefix` 前导前缀派生。

**操作 × 输入**（每操作独立计时，互不混用）：
| 操作 | 输入尺寸 | 输入来源 |
|------|---------|---------|
| cs_rank | (1218, 5000) | `factor_a` |
| factor_corr | (1218, 5000, 12) | `factors` |
| stock_corr | (1218, 5000) | `returns`（⚠️ N=5000 输出 (5000,5000)，显存/内存压力大，可选子集 N'=2000 先测） |
| rolling_ic | (1218, 5000)×2 | `factor_a` + `forward_returns` |
| parameter_scan | (1218, 5000) | `factor_a` + `mask`（G=4） |

**三口径**（COMPETITOR_ANALYSIS §四）：
1. **冷调用**：每次调用新分配输入、新上传（模拟独立请求）
2. **驻留**：输入驻留 GPU/内存，重复调用只测计算（模拟批量复用）
3. **上传**：单次上传后多次计算（隔离传输与计算）

**双计时**：`cudaEvent`（GPU 端）+ wall-clock（host 端）双口径；预热 N=3，重复 M=5，取中位数 + 四分位。

**环境元数据**：CPU 型号、GPU、驱动、CUDA、Python/库版本、`np.show_config()` BLAS 指纹。

**种子纪律**：corpus 已冻结（seeds.json），基准运行不调用 RNG、不写 npz。

## 5. 判据与输出

**判据**（CLAUDE.md 决策表 PoC ②）：相对最佳免费替代**端到端** ≥2×（同数据同 mask **且同语义**）。端到端 = parameter_scan 代表流水（G 组 rank × 每组输出），或四操作加权组合（预注册加权方案，见 §6）。

**证据产物**：① 性能对比表（三臂 × 操作 × 三口径 × 双计时）② 原始计时 JSON ③ corpus hash（data_sha256）④ 环境指纹 ⑤ contract parity 报告。

**输出文件**：`benchmarks/results/parity_report_v1.{json,md}`、`benchmarks/results/perf_report_v1.{json,md}`。

## 6. 已确认决策（2026-08-03 人裁决，GATE 关闭）

1. **本轮范围 = 基线建立**：factor-cuda 正式 GPU 实现（C++/CUDA）在 PoC ④ 才做，本轮测不出真正端到端加速比。本轮交付三臂免费替代的 contract parity + 性能基线，产出对比表作为 PoC ④ factor-cuda GPU 实现的参照系。加速比判据（≥2×）留待 PoC ④ 后裁决。
2. **stock_corr 规模 = N'=500 子集（2026-08-03 两次修订）**：① 初测逐对循环不可行（N=2000 需 40–1000s）→ 缩 N'=500；② **异后端审查推翻**：pairwise-complete 相关可用 **masked-GEMM 向量化**（M^T M 求共同计数、Xm^T M 求分对和、Xm^T Xm 求分对积，ddof=1），实测 numpy N=500 36ms（vs 逐对 2689ms，**74×**）、N=2000 外推 ~0.6s，与 oracle 偏差 ≤1.1e-16。**stock_corr 免费替代高效可行，不再是 factor-cuda 价值点**。协议冻结：numpy/CuPy 臂 stock_corr 用 masked-GEMM，N'=500 实测 + N'=2000 外推。
3. **numpy/pandas rank 口径 = ordinal 同语义**：numpy/pandas 臂用 `np.argsort(kind="stable")` 实现 ordinal 秩（与契约同语义），保证同语义加速比；另附 pandas average 原生性能参考列（仅参考，不进同语义加速比）。

---

## 7. 修订记录（2026-08-04，R4/R6 实施层偏离）

**三口径语义重定义**（R4 修复——原定义标签与实际测量不符，GPT-5.6-Sol 二轮审查）：
1. **冷调用（cold）**：GPU 臂在**预上传前**测首次调用（首个 op 含 CUDA context 初始化）——原实现是预上传后测 first，非真冷。perf_bench_v1.py v2.0 的 `cold_first_ms` 即此口径。
2. **驻留（resident）**：改为**纯设备 API**（cs_rank/rolling_ic 的 `*_gpu`/`*_device` 变体，device→device，只测 device 计算）；correlation/parameter_scan 无纯设备 API → 标 `pure_device=False`（实际口径，含 CPU 回拷/重上传），不冒充纯 kernel。
3. **上传（upload）**：测该操作**全部输入**的一次性 H2D 总耗时（原只测单个 operand）。
4. **计时**：CuPy/Torch 各用 **native event**（不再跨后端混用）。

**stock_corr 免费替代实现更新**（R6 支撑）：numpy/CuPy 臂 stock_corr 在 masked-GEMM 快路径（协议 §6 决策 2）基础上加**抵消检测回退**（`_gemm_cancel_mask`：|sumx·sumy/n|、|sumx²/n|、|sumy²/n| 相对差分项超 1e3 倍 → 该 pair 回退两遍中心化 `_two_pass_corr`）——既保留 GEMM 性能（N=2000/5000 可实测），又满足大偏置锚点 |Δr|≤1e-12。§6 决策 2 的 "N'=500 实测 + N'=2000 外推"升级为 `--stock-sizes 500,2000,5000` 全实测（R6）。

**证据链**（R2）：结果写入不可变 `results/runs/<run_id>/<backend>.json`；`--render <run_id>` 从 run JSON 生成 Markdown 并校验 corpus 名/hash/shape/commit 一致性，不一致拒绝汇总；speedup 由渲染器从 JSON 复算（同语义最佳免费替代作分母）。

---

*生成模型: DeepSeek-V4-Flash (via Claude Code CLI) · 2026-08-03 · 已定稿（GATE 关闭）；2026-08-04 R4/R6 实施层修订记录（§7）*
