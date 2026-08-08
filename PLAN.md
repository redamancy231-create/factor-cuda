# factor-cuda 项目方案（v2 修订版）

> ⚠️ **历史方案文档（superseded）**：本项目已发布 **v1.1.0**（PoC ①-④ 全闭合 + Phase 1-4 完成 + P3 适配层自动缓存）。本文档为 PoC 前的方案设计稿，仅供历史/方法参考，**不作为当前状态**。当前状态见 `README.md`；操作语义契约见 `CLAUDE.md`（L0 Spec 冻结）。
>
> 创建日期：2026-07-30（v1）；修订：2026-07-31（v2，PoC 先行主线）
> 状态：设计阶段（v2 整改草案，待复核）——v1 已通过异后端独立审查（21 条）；v2 为整改产物（落实状态：解决4 / 部分11 / 未解决3 / 新增3，见复核报告）；PoC 前 HOLD Phase 1–4
> 中文名（定稿）：**CUDA 因子截面分析加速**（GPU 加速的量化因子截面分析工具）
> 生成模型：DeepSeek-V4-Flash (via Claude Code CLI)

> **文档分工**：本文档 = 方案设计。操作语义契约 / 停止条件 / 成功标准 / 评估计划 / 可复现性 → `CLAUDE.md`（L0 Spec）；变更记录 → `CHANGELOG.md`；风险 → `RISK.md`；独立审查 → `reviews/`。

## 审查基线（摘要）

- GPT-5.6-Sol 独立审查（2026-07-31，21 条：🔴6 / 🟠11 / 🟡4），完整报告：`reviews/plan_review_gpt56sol_2026-07-31.md`
- 裁决：不直接进入完整 Phase 1–4；先预注册 PoC（语义/基线/显存/端到端），再按 GO / REDESIGN / STOP 三态裁决
- 本 v2 按"PoC 先行"主线组织；每节标注对应审查条目（C#/M#/m#）
- **v2 复核**（2026-07-31，17 条）：`reviews/plan_claude_md_review_gpt56sol_2026-07-31.md`——21 条落实状态 **解决4 / 部分11 / 未解决3 / 新增3**；CLAUDE.md 据此降为 DRAFT

## 一、项目定位

用 CUDA C++ 实现量化因子**截面分析**常用操作的 GPU 加速，与 ashare-mcp（数据获取）和 ml-quant-trading（因子生产）形成完整链路。

**中文名（定稿）**：CUDA 因子截面分析加速。不采用"CUDA 加速量化因子计算"——本项目**不做因子计算**。

**是**：GPU 加速的因子截面分析工具——截面排序、相关性、IC、参数扫描
**不是**：因子计算引擎（ml-quant-trading 已做）、回测框架、数据获取层

**边界能力矩阵**（[C1][C2]）：因子重算、收益生成、回测的责任归属必须写清，防止"不是回测"被参数扫描内部逻辑突破：

| 操作 | 责任方 | 说明 |
|------|--------|------|
| 因子值计算 | ml-quant-trading（用户侧） | factor-cuda 输入已是因子值矩阵 |
| 收益/forward_returns 生成 | 用户侧传入 | factor-cuda 只消费，不生成 |
| 截面排序/相关性/IC/参数扫描 | factor-cuda | 本项目范围 |
| 分层统计（top_n 分组收益等） | **待 PoC 定义** | 若纳入则更新本表，否则归用户侧 |

## 二、三项目关系

保留"接口耦合、代码独立"原则，但修正数据流示例（[C6]）。

**真实接口（2026-07-31 已核实 ml-quant-trading 源码）**：
- 模块路径：`from mlquant.features import compute_legacy_set`（**不是** `mlquant.factors`）
- 返回：`(factors, mask, names)`，factors 为 **PyTorch Tensor** `(T, N, F)`，**不是 numpy**
- → 需要**输入适配层**（Tensor→numpy 或 DLPack，PoC 验证）；**不再宣称"零拷贝"**；适配契约见 CLAUDE.md「操作语义契约」

## 三、市场验证

**市场定位结论**：在所记录检索范围内，未发现与"GPU 加速量化因子截面分析"核心定位直接重合的开源仓库（v1"零竞品"结论已修正）。因子研究相关仓库存在——其中 **QuantGplearn**（Torch GPU 因子评估器）在功能层最接近，是 PoC ② 必须对比的候选。

- 竞品搜索记录、GitNexus 源码级深度分析（QuantGplearn / equity-factor-lab）、替代方案矩阵、PoC ② 对比协议 → **`COMPETITOR_ANALYSIS.md`**
- **PoC 必须回答**：给定 A 股日频 2500×5000 参数扫描 workload，factor-cuda 相比最佳免费替代（CuPy / QuantGplearn-Torch）的边际收益（性能/精度/mask/内存/安装门槛）是否值得维护 C++/CUDA 代码库？

## 四、使用场景与目标人群

### 核心场景：因子参数扫描（待验证）

v1 的 **"8s/组 → 100 组 800s → GPU 45s → 18×"** 叙事**撤回**（[C2]）：CPU 数字无测量协议且高估（2026-07-31 **探索性测量** pandas rank 1.18s / argsort 0.18s / 截面IC 0.58s / 分层回测 0.68s——脚本/corpus/日志未归档，PoC ② 前补可复现证据，**当前不作为已核实基线**）；分层回测计入加速账但无实现；GPU 45s 无原型。**性能卖点改为待验证假设，不预支 18×**。

### 目标人群（需求假设降级，[m2]）

| 人群 | 需求 | 证据等级 |
|------|------|---------|
| A 股个人量化交易者 | 待验证假设 | 无调研/社区证据；多数个人研究者用 300–2000 只子集，瓶颈常在数据下载与因子计算 |
| 量化金融学生 | 待验证假设 | 毕设/课程价值成立，需可负担安装门槛 |
| 小私募研究员 / 深度学习者 | 待验证 / 未验证 | 无访谈/issue/下载证据 |

验证指标（PoC 收集）：愿意安装 CUDA wheel 的用户数、可接受等待时间、实际扫描规模。数据收集前不写死"需求强"。

## 五、操作语义契约

**契约全文已提升至 `CLAUDE.md`「操作语义契约」**。本节保留索引：

| 契约 | 关键点 | 位置 |
|------|--------|------|
| 输入适配 | 两级 dtype（rank/scan float32 主线；correlation/rolling_ic 内部 float64）；ml-quant Tensor 需适配层；mask 契约 | CLAUDE.md |
| cross_sectional_rank | ties/NaN/升降序/稳定性 | CLAUDE.md |
| correlation | `factor_corr` (T,N,F)→(F,F) / `stock_corr` (T,N)→(N,N)；FP64 CPU fallback | CLAUDE.md |
| rolling_ic | 每日截面 Spearman；防未来函数 | CLAUDE.md |
| parameter_scan | 扫描自身操作参数；组合展开待 PoC ① | CLAUDE.md |

## 六、技术架构

### 关键技术决策

| 决策 | 选择 | 修正 |
|------|------|------|
| 绑定 | pybind11 + CUDA C++ | 候选（非已验证） |
| GPU 后端 | CUDA Toolkit + thrust + cuBLAS | FP64 项允许 CPU fallback |
| 输入/输出 | rank/scan float32 主线；(T,N)；correlation/rolling_ic 内部 float64、输出 float64 | 撤"与 ml-quant 零拷贝一致"；oracle=仓库内 wrapper（tests/fixtures/corr_oracle_v1.py） |
| 编译 | CMake（**Ninja** 推荐）+ nvcc | VS generator 亦可用（2026-07-31 全新目录验证）；历史 CUDA 探测失败为限定事件（根因未确认，坑位见 CLAUDE.md）；CMP194 不存在 |
| Python 包 | `pip install factor-cuda` | 待定——分发策略取决于架构兼容方案 |

### 模块拆分

```
factor-cuda/
├── src/
│   ├── cross_sectional_rank.cu    # 截面排序（分段排序方案 PoC 定）
│   ├── correlation_matrix.cu      # factor_corr / stock_corr（SYRK 候选）
│   ├── rolling_ic.cu              # 每日截面秩 IC
│   ├── parameter_scan.cu          # 参数扫描（语义见 CLAUDE.md）
│   └── bindings.cpp               # pybind11 模块注册
├── python/factor_cuda/
│   ├── __init__.py
│   ├── adapters.py                # 输入适配层（Tensor→numpy/DLPack）
│   └── benchmark.py               # CPU vs GPU 对照
├── tests/
│   ├── test_correctness.py        # GPU vs CPU oracle
│   └── test_contracts.py          # 语义契约测试
├── benchmark_corpus/              # 版本锁定 corpus（.npz）
├── CMakeLists.txt
├── pyproject.toml
└── README.md
```

### 与 etf-pattern-match-pybind11 的复用（修正，[M7]）

2026-07-31 已核实其 CMake 为 **纯 C++（`LANGUAGES CXX`）、无 CUDA kernel**——"直接复用 CUDA CMake 模板"**不成立**。但核查其 `src/cpp/etf_core.cpp` 存在 8 处 `py::gil_scoped_release`，**GIL 释放经验确实存在**（v2 此前误否）。三层结论：[M7] ① 纯 C++ CMake 不能证明 CUDA 构建可复用；② pybind11 绑定与同步 CPU 计算的 GIL release 模式可参考；③ CUDA stream/异步完成/Python 对象生命周期/延迟错误传播须独立验证。

## 七、实现阶段：PoC 先行 + 三态裁决

### Phase 0：工具链与环境（[M6]）

**本机现状（2026-07-31 精核实）**：

| 工具 | 状态 |
|------|------|
| CUDA Toolkit | ✅ **v13.3 已装**（2026-07-31 验证，nvcc V13.3.73；RISK.md R001 已缓解） |
| MSVC 19.51 + 19.44 | ✅ 已装（VS2026 Community） |
| CMake（VS 内置） | ✅ 已装 |
| GPU | ✅ RTX 4060 Laptop（sm_89），驱动 610.88 / CUDA UMD 13.3 |

版本：**本机锁定** CUDA 13.3 / MSVC 19.51（Phase 0 smoke 已实测 PASS，2026-07-31）；**产品支持范围**（13.2+、其他 MSVC/驱动/GPU 架构）为待定义矩阵（P1 待办），未验证前不宣称兼容。`CUDA_ARCHITECTURES` 见 CMakeLists（`89`，CMake 对无后缀架构自动含 virtual；历史 `89;89-virtual` 曾致重复 codegen 已修）。安装步骤与验证命令见 CHANGELOG 对应条目及本文件历史版本。

### PoC 阶段（预注册）

| 验证项 | 内容 | 三态判据（详见 CLAUDE.md「停止条件」） |
|-------|------|------|
| ① 语义 | 冻结操作语义契约 + oracle 测试 | 契约全部可运行 |
| ② 公平基线 | 同数据同 mask：numpy/pandas、CuPy、QuantGplearn-Torch、GPU 实现 | 相对最佳免费替代有可复现边际收益（≥2×） |
| ③ 显存 | 峰值公式 + cudaMemGetInfo 校准 | 峰值 ≤ 可用显存−安全余量（非固定 8GB）且可分块 |
| ④ 端到端 | 最小参数扫描流水（F/T 分块、H2D、归并） | 端到端收益成立或明确 STOP |

> **Amend（2026-08-06，pybind11 绑定 + PoC ④ 会话）**：PoC ④ 行的「F/T 分块」要素已由**最小证明①/②**（2026-08-05）设备级位级断言闭合——rank 类（rolling_ic/cs_rank/parameter_scan，含 CUB 段重切）与 factor_corr continuation 分块均与不分块**位级一致**（证据：`reviews/poc3_ft_minproof1_review_gpt56sol_2026-08-05.md` / `poc3_ft_minproof2_review_response_gpt56sol_2026-08-05.md`）。端到端流水（`benchmarks/poc4_e2e_v1.py`）因此采用**非分块路径**——当前显存余量（校准 max 2381 / 可用 7676 MiB）足够，分块非必要；显存受限时可切分块（正确性已证，逐位等价）。**端到端收益判据（含传输/归并，相对最佳免费替代 ≥2×，目标 5×）不变**。H2D 单次上传 + parameter_scan 4 组合并 + IC 堆叠即「H2D、归并」要素。

三态：**GO**（①∧②∧③∧④）→ Phase 1–4；**REDESIGN**（①/③ 不成立）；**STOP**（②/④ 不成立，记负结果 NRR）。

### Phase 1–4（PoC GO 后，各带验收门槛）

| Phase | 内容 | 验收门槛 |
|-------|------|---------|
| 1 | 截面排序 | oracle 精确一致（ties/NaN/mask/边界）；排序+scatter 端到端 ≤ 预注册上限 |
| 2 | 相关性矩阵与 IC | numpy.corrcoef 对照；FP64 项 CPU fallback 就绪；无未来函数测试 |
| 3 | 参数扫描 | 组合展开与输出 schema 符合契约；端到端可复现 |
| 4 | benchmark + NRR | 固定 corpus 可复现；统计协议完备；NRR 按预注册判据 |

> **Amend（2026-08-06，Phase 2-3 验收闭合）**：Phase 1-3 验收门槛已全部闭合——`benchmarks/acceptance_v1.py` **五门全 PASS**（`gate_2_semantics/memory/no_lookahead/schema/e2e`，证据 `benchmarks/results/acceptance_v1.{json,md}`，commit `1a814f2`），**解锁 Phase 4**。Phase 2（factor_corr/stock_corr/rolling_ic：正确性+性能≤gate+显存+无未来函数 timeline 集成测试 `tests/test_timeline_no_lookahead_v1.py`）与 Phase 3（parameter_scan：G=4 字典序+schema+group_status+可复现）验收通过。**stock_corr 双口径裁决**：fast（全有效面板，v2 同面板 gate，N=500/2000/5000 5.3/37.3/199.4ms BEATS）+ general（corpus 同面板 gate，N=500/2000 **2.41×/2.32× PASS**——纠正早前 `poc3_stock_corr_perf.cu` 合成面板跨数据比较造成的 0.42× FAIL 假象，同面板重基线 `runs/stock_corr_general_gate_20260806/`）。已知缺口诚实记录：below_5x（cs_rank/parameter_scan/rolling_ic/general 2.3-3.3×）入 Phase 4/NRR below_5x_note。

### benchmark 协议（[M10][m1][m4]）

- **corpus**：`benchmark_corpus/` 版本锁定 `.npz` + hash + 生成脚本（**manifest 在 PoC ② 前提交**；不依赖运行时 ml-quant）+ 结构保持合成 corpus
- **日期规模**：2020–2024 约 **1218 个交易日**（非 2500）；性能表按实际 T/N/F 参数化
- **统计**：环境元数据、预热/重复/种子、`cudaEvent`+wall-clock 双口径、冷缓存/驻留区分、异常值/置信区间、kernel/单算子/批量/端到端四层
- 单算子 microbenchmark 与端到端分开报告

## 八、风险评估

完整风险登记见 **`RISK.md`**（附录F 8 列，HG 触发规则）。当前触发 Human Gate 的项：

- **R001** CUDA Toolkit 未装（H×H）——**已缓解**（v13.3 已装，2026-07-31）
- **R003** FP64 吞吐硬伤（H×H）——CPU fallback 为正式后端
- **R004** 免费替代足够（H×M）——PoC ② <2× 则 STOP
- **R005** dense 面板显存溢出（H×M）——F/T 分块
- **R007** parameter_scan 语义未冻结（H×H）——PoC ① 冻结

其余（R002 工具链兼容 / R006 分发门槛 / R008 基线公平 / R009 GPU 侧阻塞）见 RISK.md。

## 九、验收与三态裁决

**判据单一真源为 `CLAUDE.md`「PoC 决策表」**（PoC ①–④ PASS/FAIL/最多重试/证据产物）。PoC 预注册阈值：正确性（契约 oracle 全过）、边际收益（≥2×，目标 ≥5×）、显存（≤可用显存−安全余量，非固定 8GB）、安装成功率 ≥80%、NRR 负结果预登记。

## 十、构建参考（压缩，[m3]）

mfaktc（GPL-3.0，21 stars，2026-07-31 核实）：借鉴"cl + nvcc 可共存"通用构建模式、CUDA streams 重叠调度、GPU 自检思路。**(MSVC 19.51 × CUDA 13.3) 组合已于 2026-07-31 Phase 0 自检实测通过**（nvcc 编译 + GPU vec_add PASS）；不复制源码，仅借鉴公开模式。

---

*生成模型: DeepSeek-V4-Flash (via Claude Code CLI) · 2026-07-31 · 修订来源: GPT-5.6-Sol 独立审查报告 (reviews/plan_review_gpt56sol_2026-07-31.md)*
