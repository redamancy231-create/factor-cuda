# 竞品分析 — factor-cuda

> 创建日期：2026-07-31（自 PLAN.md §三 拆出）
> 状态：活跃维护——PoC ② 公平基线结果将持续更新此文件
> 生成模型：DeepSeek-V4-Flash (via Claude Code CLI)
> 上游：PLAN.md v2 §三；方法学依据：AI 协作项目全生命周期框架 [M8][M9] 审查条目

## 一、竞品搜索（方法学修正，[M9]）

v1 的 11 组 5–6 词长短语 AND 查询（`CUDA factor computation quantitative finance` 等）全部返回 0，但**长 AND 查询在 GitHub 搜索天然假阴性**，且未记录 API 端点/认证/限流/索引范围。v2 补检（2026-07-31，`gh api search/repositories`，带认证）：

| 短关键词 | 结果数 | 相关命中 |
|---------|-------|---------|
| `gpu quant factor` | 2 | **WYFHHH/QuantGplearn**（遗传规划因子挖掘，Torch GPU 后端）、**cookfishbro/equity-factor-lab**（完整因子研究平台） |
| `cuda stock rank` | 0 | — |
| `rapids factor` | 2 | 不相关 |
| `cupy rank` | 2 | 不相关 |
| `cudf factor` | 1 | 不相关 |
| `factor zoo gpu` | 1 | 不相关 |

**结论（v2）**：v1"GitHub 上不存在以 GPU 加速量化因子截面分析为核心定位的开源项目"**不成立**。v2 结论：**在所记录检索范围内，未发现与"GPU 加速量化因子截面分析"核心定位直接重合的仓库**；因子研究相关但不直接竞争的仓库存在，须按功能而非项目名建立替代矩阵。

### v3 补检（2026-08-06，P2 待办闭合）——中文短词 + 补充英文词 + Web 全网

| 关键词（语言） | 结果数 | 相关命中 |
|---------|-------|---------|
| `因子截面` / `截面因子`（中，gh api） | 42 | 多因子选股/因子挖掘（**非 GPU 截面分析**）：gplearn_stock_dataframe、Factor-Miner、StockTrader、cross-sectional-factor-lab 等；**无 GPU 截面分析算子库** |
| `GPU 量化因子` / `GPU 因子` / `因子分析 GPU` / `GPU 量化 截面`（中，gh api） | 0 | —（**中文 GitHub 无 GPU 因子截面分析**） |
| `CUDA 量化`（中，gh api） | 20 | **全部为 LLM/CUDA 模型量化**（"量化"多义词噪音），与因子截面无关 |
| `cross-sectional factor GPU`（英） | 1 | cookfishbro/equity-factor-lab（已知） |
| `cross section factor python`（英） | 13 | 截面因子回测/研究（CPU）：cross-sectional-factor-backtest 等，非 GPU |
| `GPU alpha factor`（英） | 2 | **IIcodehub/GP-Alpha-Miner-GPU**（★7，GPU 遗传规划因子挖掘，CuPy+DEAP，非截面分析库） |
| `GPU factor research`（英） | 9 | 大多不相关（整数分解/图像）；equity-factor-lab 已知 |
| `factor IC GPU` / `factor zoo CUDA`（英） | 0 | — |
| Web 全网（中文）"GPU 量化因子 截面分析 加速" / "CUDA 因子研究 截面" | — | **Spectre（Heerozh/spectre，★818，gh api 2026-08-06 核实）**——GPU 因子计算引擎+回测器（PyTorch，含 `.rank()/.zscore()/.demean()` 截面操作，**最接近的替代**）；AurumQ-RL（A 股多因子+GPU 训练栈）；drmysore/gpu-algorithmic-trading（GPU 回测）；arXiv 2507.07107（截面排序双重 argsort GPU 化，51×）；华泰金工 CuPy/cuDF 高频因子加速（6×→100×） |

**结论（v3，2026-08-06）**：在所记录检索范围（v2 英文 + v3 中文短词 + Web 全网）内，**仍未发现与「GPU 因子截面分析算子库」（截面排序/相关/IC/参数扫描的契约级 GPU 实现）直接重合的开源项目**；但 GPU 因子计算/挖掘/回测**引擎**生态活跃——**Spectre（★818，gh api 核实）为最接近替代**（覆盖 GPU 因子计算 + rank/zscore/demean 截面标准化），须纳入替代矩阵（§三）。**P2 待办已闭合**：query manifest 存档 `COMPETITOR_SEARCH_MANIFEST_v1.json`（endpoint/query/认证/时间戳/结果数/相关命中/hash，2026-08-06）。

### v3.1 1 年活跃度核实（2026-08-06，`gh api` 实测 `pushed_at`）

| 项目 | ★ | 最后更新 | 1 年内活跃 | 定位 |
|------|---|---------|-----------|------|
| Heerozh/spectre | 818 | 2025-04-15 | **否**（>1 年停更） | GPU 因子计算+截面标准化+回测（最接近引擎级替代） |
| WYFHHH/QuantGplearn | 11 | 2026-06-13 | 是 | GPU 遗传规划因子挖掘 |
| cookfishbro/equity-factor-lab | 0 | 2026-07-05 | 是 | 因子研究平台（CPU 为主） |
| yupoet/aurumq-rl | 36 | 2026-07-24 | 是 | A 股多因子 + GPU 训练栈 |
| IIcodehub/GP-Alpha-Miner-GPU | 7 | 2026-01-15 | 是 | GPU 因子挖掘（CuPy+DEAP） |
| drmysore/gpu-algorithmic-trading | 0 | 2026-01-02 | 是 | GPU 回测 |
| rapidsai/cudf / cupy | 9723 / 12232 | 2026-08（持续） | 是 | 通用 GPU DataFrame/数组 |

补充搜索（`search/repositories sort=updated`）：GPU 因子相关最新活跃项目仅 QuantGplearn/equity-factor-lab，**无 1 年内活跃的同定位（截面分析算子库）新进入者**。

**活跃度结论**：1 年内活跃的 GPU 因子相关项目全部位于**因子挖掘/计算引擎/回测**生态位；**契约级截面分析算子库无任何 1 年内活跃项目**，最接近的 Spectre 已 >1 年停更。→ **空白生态位仅对「契约级、可复现、市场无关的截面分析算子库」定位成立**；若定位为"通用 GPU 因子引擎/工具"，则与活跃引擎（QuantGplearn 等）+ 通用库（CuPy/cuDF）重复。**差异化落点 = 冻结 oracle / 位级确定性 / 参数扫描 / 市场无关（学术复现基础设施）**。

## 二、竞品深度分析（2026-07-31，GitNexus 索引 + 源码级）

两仓库已克隆至本地 reference 目录并建立 GitNexus 索引（QuantGplearn: 580 nodes/51 flows；equity-factor-lab: 561 nodes/23 flows），供 PoC ② 公平基线直接使用。

### 2.1 QuantGplearn（WYFHHH，MIT）

遗传规划**因子挖掘**框架——进化可读因子公式，CPU(NumPy/Pandas)+GPU(**Torch**)双后端，49 算子，IC/RankIC/ICIR/Sharpe 进化目标。

源码关键点（GitNexus 核实）：
- `_rank_pct_dim`（torch_functions.py:120）：截面秩用 `stable=True` + ordinal rank，注释明示 **"Ties use stable ordinal ranks for speed on GPU"**——故意不用 pandas average 平均秩（factor-cuda tie 决策的现实参照）
- `mean_ic`（tensor_fitness.py:107）：Pearson + mask + nanmean
- 自述局限：dense `[T,N,F]` 面板内存密集、rolling `torch.unfold` 大中间张量（与 factor-cuda 显存难题同构）、GPU tie/NaN 与 Pandas 有差异
- **安装门槛低**：`pip install -e .` + PyTorch CUDA 即可，用户不需单独装 CUDA Toolkit

与 factor-cuda 关系：**功能层最接近的对照候选**——其 GPU 因子评估器覆盖 cs_rank/IC/全套截面算子；若 Torch 向量化已够用，原生 CUDA 的边际收益需被证明。

### 2.2 equity-factor-lab（cookfishbro）

**完整量化研究平台 + 诚实空结果**（S&P 500 月频 2000–2026，"Nothing survives"）。GPU 仅用于 10 万 placebo 因子模拟（PyTorch CUDA）+ XGBoost GPU；主体 pandas/numpy + scipy。

源码关键点（GitNexus 核实）：
- `rank_ic_series`（crosssection/ic.py:10）：逐月 Python 循环 + `scipy.stats.spearmanr` + `min_stocks=30` 阈值
- `run_placebo` / `placebo_pvalue`（robustness/placebo.py）+ `test_placebo_cpu_gpu_parity` 测试
- bias 控制范式：PIT universe、前瞻守卫（扰动 t 后断言信号 bit 不变，pytest guard）、DSR、Fama-MacBeth

与 factor-cuda 关系：**非直接替代**；其前瞻守卫/placebo null 范式与 factor-cuda benchmark 协议同构，可作 NRR 正确性验证参考实现。

## 三、替代方案矩阵（[M8]）

factor-cuda 的替代/对照方案——**是否值得自研，须在 PoC 阶段与这些方案公平对比**：

| 方案 | 覆盖能力 | 与 factor-cuda 的关系 |
|------|---------|----------------------|
| numpy/pandas 向量化 | rank/argsort/corr/滚动 | CPU 基线 |
| CuPy | `cp.argsort/cp.cov` 等一行 GPU 化 | 强替代——需证明 factor-cuda 有 CuPy 做不到的边际收益 |
| **QuantGplearn（Torch）** | cs_rank/IC/全套截面算子 GPU 评估 | **功能层最接近的对照候选**——PoC ② 必须对比 |
| cuDF/RAPIDS | pandas 兼容 rank/rolling/corr | 替代方案，非"并行不竞争" |
| Numba | 定制内核 | 部分替代 |
| **Spectre（Heerozh）** ★818 | GPU 因子计算引擎+回测器（PyTorch 张量化，`.rank()/.zscore()/.demean()` 截面标准化；GPL-3.0；last push 2025-04） | **最接近的引擎级替代（v3 发现）**——覆盖 GPU 因子计算与截面标准化；但非契约级截面分析算子库（无冻结 oracle/位级确定性/参数扫描），PoC ④ 需对照 |
| **GP-Alpha-Miner-GPU（IIcodehub）** ★7 | GPU 遗传规划因子挖掘（CuPy+DEAP，RankIC 适应度，~100×） | 与 QuantGplearn 同类（因子挖掘）——覆盖因子生成，非截面分析（v3 发现） |
| AurumQ-RL（yupoet） | A 股多因子选股 + GPU 强化学习训练栈（polars 因子引擎，截面 IC 实证） | 间接（因子引擎+训练）；截面 rank-z/IC 过拟合实证可参考（v3 发现） |
| drmysore/gpu-algorithmic-trading | GPU 回测/组合优化/VaR（H100，100-500×） | 间接（回测侧），非截面算子（v3 发现） |
| equity-factor-lab | 完整研究流水线（bias 控制/验证） | 非直接替代；验证范式参考 |

## 四、PoC ② 对比要求

- **协议**：同数据同 mask、同操作集；冷调用/驻留显存/每次上传三种口径；`cudaEvent`+wall-clock 双计时
- **contract parity（关键）**：各后端先做语义契约对齐（如 QuantGplearn Torch=stable ordinal vs NumPy=average tie；CuPy 的 tie/NaN/百分位定义）——**不能原生满足契约者，分别报告原生语义性能与转换后性能，不得混为同一加速比**
- **对照方**：numpy/pandas（CPU 基线）、CuPy、QuantGplearn-Torch、factor-cuda GPU 实现
- **判定**：相对最佳免费替代的端到端加速比 ≥ 预注册门槛（≥2×，目标 ≥5×），否则 STOP/REDESIGN
- **安装门槛对比**：记录各方案用户安装成本（PyTorch 免装 CUDA Toolkit vs factor-cuda 需装）

## 五、PoC ② 公平基线实测（2026-08-03 审查修复版，synthetic canonical 1218×5000×12）

> 完整报告：`benchmarks/results/perf_report_v1.md`；parity 三臂（numpy 27/27、cupy 27/27、qgplearn 15 PASS+12 N/A，共享算子层）。环境：RTX 4060 / torch 2.13.0+cu132 / cupy 14.1.1。
> **GPT-5.6-Sol 审查修复**（6 高/6 中/1 低）：共享算子层统一 parity/perf（H4）、stock_corr 用 returns（H1）、rolling_ic 带 mask（H2）、QG rolling_ic 标 known-deviation 排除（H3）、稳健中心化+退化对角+三角镜像（H5/M1）、**程序化比值修正 13.8×→1.99×**（H6）。

| 操作 | numpy/pandas | CuPy | QuantGplearn-Torch | 同语义最佳 vs numpy |
|------|:---:|:---:|:---:|:---:|
| cs_rank | 288.9 ms | 52.5 ms | **27.6 ms** | **10.5×** |
| cs_rank_desc | 409.3 ms | 51.3 ms | **33.3 ms** | **12.3×** |
| factor_corr (F=12) | 4831.8 ms | **4273.6 ms** | N/A | 1.1× |
| stock_corr (N'=500) | 106.4 ms | **53.6 ms** | N/A | **2.0×** |
| rolling_ic | 953.2 ms | **151.7 ms** | 124.0* | 6.3× |
| parameter_scan (G=4) | 1688.3 ms | 219.9 ms | **144.3 ms** | **11.7×** |

\* QG rolling_ic 为 **known-deviation**（原生 float32，非 ≤1e-12 同语义），**不纳入同语义最佳替代**（H3）。同语义 rolling_ic 最佳 = CuPy 151.7ms。

**要点**：
- QuantGplearn-Torch 为排序/扫描类最佳免费替代（cs_rank 10.5×、parameter_scan 11.7×）
- **stock_corr masked-GEMM 向量化后 GPU 加速 2.0×**（修正初版 13.8× 算术错误）——GPU 加速真实但幅度远小于初版声称
- factor_corr 无显著 GPU 优势（1.1×，FP64 吞吐瓶颈）
- rolling_ic 同语义最佳 = CuPy 151.7ms（6.3×）
- **状态 = 候选/待复核**（非"已锁定"）；P1 计时协议（cudaEvent/三口径/更多样本）未完成
- PoC ④ factor-cuda 门槛 = 相对**同语义**最佳免费替代 ≥2×（cs_rank ≤13.8ms；stock_corr ≤26.8ms；parameter_scan ≤72.2ms）

---

*生成模型: DeepSeek-V4-Flash (via Claude Code CLI) · 2026-08-03 · 上游: PLAN.md v2 §三 + PoC ② 实测*
