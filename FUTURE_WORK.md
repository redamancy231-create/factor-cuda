# FUTURE_WORK — factor-cuda 未来修改方向

> **状态**：v1.1.0 已发布（2026-08-08），**本轮已批准待办闭合**（多架构/multi-GPU 等列为 future candidate/backlog，见 §五）。本文档记录**可能的修改方向**，按「依据强度 × 价值/成本」分层，供后续决策参考。
> **不承诺实现**——列为方向 ≠ 批准做。每个方向标注：依据（来自何处）、价值、成本、风险。
> 当前权威状态见 `README.md` / `CLAUDE.md`（L0 Spec 冻结契约）；历史方案见 `PLAN.md`（superseded）。

---

## 一、已有项目证据支撑的优化（低成本，建议优先评估）

这些方向直接从已记录的项目观察推导，改动小、验证路径清晰。

### 1.1 适配层加性开销定位（cs_rank 缓存收益被稀释到 1.20x）
- **依据**：`benchmarks/results/ws_py_cache_v1` —— Python 端无缓存 cs_rank **37.35ms**，有缓存 31.06ms（消除设备分配 ~6.3ms）。加性开销 = Python 端 vs C++ driver 的**跨口径估算 ~21ms**（C++ 基线 16.4ms 来自独立 driver `poc3_cs_rank_perf`：workspace-cached、不同输入/同步——**不可直接相减得精确固定成本，需同一进程/输入/mask/口径下分层验证**）。自动缓存把收益从理论 ~1.8x 稀释到实测 1.20x。
- **价值**：cs_rank 是高频算子（截面扫描最常用），每省 1ms 端到端都直接放大所有下游。
- **成本**：需三口径 profiling（高层 API / raw pybind / C++ driver，同一输入与同步）分别计时转换、同步、分配，确认 ~21ms 估算的实际构成。中。
- **风险**：低。纯测量先行，不改契约。

### 1.2 parameter_scan 缓存收益（1.18-1.23x，口径跨门槛）
- **依据**：C++ workspace 先例 parameter_scan 35.90→29.13ms（=**1.23x**，满足 ≥1.2x gate）；但 P3 证据 disclosure 记录「C++ workspace 收益 1.18x 未达 1.2x」——**来源口径冲突**（CHANGELOG 记 1.18-1.23× 均 BEATS，disclosure 记 1.18x）。**两口径的 commit/原始时序/估计量未并列，性能归类待统一重算**；缓存至少是正确性收益（消除 per-call 分配）。
- **价值**：同 1.1——若适配层开销定位后，parameter_scan 可能随 cs_rank 一起改善。
- **成本**：低（复用 1.1 的结果）。
- **风险**：低。性能门槛非硬目标。

### 1.3 fcad-3：f64 路径 isfinite 冗余
- **依据**：2026-08-07 f32 domain 审查记录「fcad-3：f64 路径 isfinite 冗余 ~1 次 bool 全面板」——冗余在 **Python 适配层**（`fc/correlation.py` `_gpu_domain_check` 先 `np.isfinite` 又调 `in_corr_domain`），非 CUDA kernel 内；记录不处置。
- **价值**：极小（单次 bool 扫描 vs 全计算）。低优先。
- **成本**：需确认适配层冗余 isfinite 与绑定层 upcast 的职责边界（避免误删 domain 检查）。低。
- **风险**：低。不建议为极小收益动 review-closed kernel；若随 1.1 的适配层 profiling 一起评估则几乎零成本。

### 1.4 F128 streaming 物理余量薄（GPU 并发占用时 vram-exhausted）
- **依据**：`benchmarks/results/factor_stream_hwm_v1` —— F128 streaming 物理余量当前 JSON 实测 **26.25 MiB**；166 MiB 仅见于历史 CHANGELOG（早期 artifact git_dirty）——**跨 artifact 的"26-166 MiB 波动"未统一口径验证，暂仅保留当前实测 26.25 MiB**；发布文档已披露。
- **价值**：仅当用户需要 GPU 并发跑多个大面板时才有意义（余量工程：更紧的分块/预释放）。
- **成本**：中（改 streaming 分块策略 + 重测）。
- **风险**：中（余量随驱动/WDDM 波动，机制未隔离验证——见 §五）。

---

## 二、性能方向（需先做基准，勿直接投入）

### 2.1 fp64 吞吐瓶颈（混合精度 / 单精度参考路径）
- **依据**：RTX 4060 手写 fp64 实测 ~124 GFLOPS（`CHANGELOG.md` 2026-08-04 PoC③；spec 上限 ~184 GFLOPS 记录于 review prompt）；stock_corr general N=5000 gate 需 ~143 GFLOPS——**相对当前手写实现缺口 1.15×（143/124），但仅达设备规格 78%（143/184），存在实现优化空间，非「物理不可达」**。（ic_stack 0.72× 是端到端组件负结果，非 fp64 吞吐证据；FP64 ceiling 为未确认候选机制。）
- **价值**：corr 类算子（stock_corr/factor_corr）是当前最大计算量；且 143<184 提示**先挖手写实现优化**（统一 FLOP 计数与频率/boost 口径后重测），fp32 混合精度是另一条路。
- **成本**：高。**先做 scratch parity（fp32 计算能否在既有 |Δr|≤1e-12 容差内匹配冻结 oracle）——若满足既有容差则无需新契约**；失败后才需降精度参考路径设计。
- **风险**：中。契约是逐元素 |Δr|≤1e-12 容差（HG-2，非跨后端位级匹配）——fp32 可行性是**可预研验证的实证问题**，非必然推翻契约。**建议先做精度影响预研**（scratch 实验）再决定。

### 2.2 factor_corr 适配层开销（已基本闭合，低价值）
- **依据**：Phase 4 fresh 实测（2026-08-07，P1 f32 upcast 优化之后）适配层 factor_corr 开销 **+1.37%**（adapter 332.3ms vs binding 327.9ms）；历史上曾达 +238%（2026-08-06 修复前值），但已在 P1 f32 upcast 优化（净省 79%）中基本消除。
- **价值**：低——主要开销已闭合；残留 +1.37% 不值得单独追。若做 1.1 的适配层 profiling，可作为顺带确认项。
- **成本**：低。
- **风险**：低。

### 2.3 多架构支持（compute capability 其他 SM）
- **依据**：`docs/support_matrix.json` 明确「compute capability 8.9 单架构实测」；CMakeLists 当前 `CMAKE_CUDA_ARCHITECTURES="89"` **单架构**，多架构扩展是记录在案的 P1 待办。
- **价值**：扩大硬件覆盖（数据中心 A100/H100 等 fp64 强卡反而可能消除 2.1 的 fp64 瓶颈——值得注意的互补关系）。
- **成本**：中高。需配置多架构构建 + **各 SM 实测验证**（不只是重编译）+ support_matrix 更新。
- **风险**：低。不改代码，只扩验证范围。

### 2.4 设备驻留零拷贝（torch-CUDA 输入避免 CPU 中转）
- **依据**：`CLAUDE.md` 记录 torch-CUDA 输入当前经历 **D2H→H2D→D2H→H2D 四次拷贝**（fc/_util.py 将 CUDA 输入转 CPU NumPy、结果再传回 CUDA）。这是**独立的结构性开销**，不能被 §1.1 的一般 profiling 替代。
- **价值**：CUDA-resident 输入（torch tensor 已在显存）是常见用法，消除中转直接省 4 次设备↔主机传输。
- **成本**：高。需绑定层 device-pointer 路径（DLPack device / cuPointerGetAttribute）+ 冻结 device/stream/生命周期/输出契约。
- **风险**：中（device 指针生命周期 / stream 同步，契约级风险）。与 §1.1 profiling 互补而非替代。

---

## 三、功能扩展方向

### 3.1 把已验证的「内存三件套」提升为生产接口
- **依据**：`factor_corr_gpu_stream`（输入流式化）、`factor_corr_gpu_fblock`（F 分块）、`stock_corr_gpu_nblock`（N 分块）目前都是 **selfcheck PoC driver**，未暴露为 `fc.*` 生产 API；生产路径仍是非分块版（Phase 4 用）。
- **价值**：**结构性机会（限定大 F 场景）**——factor_corr F=128 production **模型峰值 12.6 GiB（未物化模型）超预算**；实测口径：production **7106.0 MiB（显存耗尽）** vs streaming **7079.75 MiB fits**——**measured-vs-measured 才是可比口径**，跨口径「12.6→6.9」不做节省比例断言。**注意**：N=22600 的 stock nblock **从未实际运行**——PoC harness 同时持有多个 N² host buffer（~8.4 GB）是**当前 harness 实现成本，非生产下界**；生产化需 caller-owned 输出 / 逐 tile 落盘（mmap）后另行判定可行性。
- **成本**：高。涉及绑定层暴露 + fc 适配层接入 + 契约（分块版与生产版的位级一致性已各自独立验证，但接口语义/错误码需设计）+ 双审查。
- **风险**：中。**位级正确性由各分块路径独立 selfcheck 闭合**（streaming/fblock/nblock，非「最小证明①②」——那是 T 延续证明，判语范围明确排除三件套）；**streaming 路径物理余量薄（实测 ~26 MiB）**，生产化前需余量工程。

### 3.2 截面分析算子扩展
- **依据**：项目定位「不做因子计算/回测/数据获取」（ml-quant-trading、ashare-mcp 分工）；现有 5 算子（rank/parameter_scan/corr×2/rolling_ic）。
- **候选**（与定位一致）：IC 面板聚合（rolling_ic 的跨行聚合）、分位数/分层（复用 rank）、信息比率面板。**注意**：因子正交化/行业中性化需**截面回归残差化（新 kernel）**——现有 kernel 集只有排序/pearson/去均值，**不能「复用 corr 管线」**，需独立评估。
- **价值**：扩大「截面分析工具」覆盖，与分工边界一致。
- **成本**：每个算子 = kernel + 绑定 + 适配层 + 契约 + 审查，约一个 P 级迭代。
- **风险**：中（新契约需冻结流程）。**建议先明确需求优先级**（用户/研究场景驱动），勿一次性铺开。

### 3.3 real corpus 扩充
- **依据**：`corpus_real_v1` = 93 只沪深 300 × 1212 交易日（2020-2024），`data_sha256=41BB9EF4`；`benchmark_corpus/fetch_real_corpus_v1.py` 已建 baostock 管道。
- **价值**：更大 universe（全沪深 300 / 中证 800）+ 更长历史 → benchmark 更可信。
- **成本**：低——管道已建，主要成本是数据下载与重跑时间（均低，拆分后无中量级项）。
- **风险**：低。注意成分股/停牌/复权口径漂移（已披露的 corpus 变更风险）。

---

## 四、工程方向

### 4.1 CI GPU 臂
- **依据**：`.github/workflows/ci.yml` 无 GPU（标准 runner），当前 CPU/契约臂 85 passed / 68 skipped（NEED_CUDA 跳过）。
- **价值**：GPU 相关测试（大量位级断言）在 CI 里被跳过——合并 GPU 臂能防 GPU 路径回归（本会话多次靠本地 GPU 测试发现竞态/泄漏）。
- **成本**：需要自托管 runner 或云 GPU（GitHub 无免费 GPU）。中。
- **风险**：低（不改代码）。是质量基建。

### 4.2 证据/benchmark 统一编排
- **依据**：多个 fail-closed 证据生产者（phase4/acceptance/factor_stream/stock_nblock/factor_fblock/ws_py_cache）各自独立；provenance/closure 模式已统一但无统一编排入口。
- **价值**：一键重跑全部证据（如 corpus 更新后），减少"哪个证据该重跑"的决策成本。
- **成本**：中（编排器 + 依赖图）。
- **风险**：低。

### 4.3 fc 包分发（wheel/PyPI）
- **依据**：当前源码构建（CMake + pybind11）；README 提供构建步骤。
- **价值**：免构建使用；但 CUDA 绑定跨环境难（需要 per-CUDA-version wheel）。
- **成本**：高（CUDA 版本矩阵、打包、签名）。
- **风险**：中（多 CUDA 版本兼容）。**建议暂缓**——单架构 + 单 CUDA 版本下分发价值有限。**注意 CUDA 口径三态**（support_matrix 单一真源）：build-tested=13.3 / declared support=13.2+ / benchmark runtime=cu132——表述兼容范围时须区分，勿混淆。

### 4.4 文档自动化（数字漂移根治）
- **依据**：本会话发现 README CI 数字（75）随测试增长过时；三语 README 手工同步易漂移（`methodology-doc-sync-drift`）。
- **价值**：防「文档声称的数字与实测不符」。
- **成本**：中（manifest 单一事实源 + 三语模板生成，框架仓库已有 `methodology-manifest-single-truth` 先例）。
- **风险**：低。

---

## 五、已知限制（不一定是方向，但决策时需知情）

| 限制 | 状态 | 影响 |
|------|------|------|
| WDDM 低余量 delta 波动 | 机制**未隔离验证**（`reviews/b32_delta_iso_verification_2026-08-07.md` 降级记录） | 大面板显存预算的精确值不可信，fits 判定留安全余量 |
| F128 streaming 物理余量薄 | 已披露（实测 ~26 MiB） | GPU 并发占用大面板时可能 vram-exhausted |
| Python 适配层加性开销 | **估算**（cs_rank ~21ms，跨口径，待同口径验证） | 稀释高频算子收益（估算非实测） |
| 单算子跨会话测量方差 | 历史端点 rolling_ic 1.99→6.94×（CHANGELOG）；当前 phase4 记录「最多约 2×」——端点/估计量未并列 | 单点收益数字不可当作稳定证据；跨会话比较须同估计量 |
| N=22600 stock nblock host 成本 | 从未实测（~8.4 GB 是**当前 PoC harness 实现成本**，非生产下界） | 生产化需 caller-owned 输出 / 逐 tile 落盘后另行判定 |
| real corpus 规模 | 93 股 × 5 年 | benchmark 外部效度有限 |
| 单架构支持 | sm_89 实测，其他未验证（多架构为 P1 待办） | 数据中心 GPU 需自行重编译 |
| multi-GPU / 双卡 | 记录在案待办（「单卡多 GPU 待双卡」），多 GPU 测试跳过 | device 路由/跨卡场景未验证 |

---

## 决策建议（右尺寸检查）

- **短期（低成本、高确定性）**：§1.1 适配层开销定位（三口径 profiling，确认跨口径估算 ~21ms 的实际构成）→ 可能连带 §1.2；§4.4 文档自动化防漂移。
- **中期（结构性价值，需设计）**：§3.1 内存三件套提升为生产接口（**限定大 F 场景**，N=22600 不可行需先澄清 host 侧）；§4.1 CI GPU 臂；**multi-GPU/双卡验证**（记录在案待办，需双卡硬件）。
- **长期/需预研**：§2.1 fp64（先 scratch parity——若满足 |Δr|≤1e-12 则无需新契约；**且 143<184 提示先挖手写实现**）；§2.4 设备驻留零拷贝（CUDA 输入避免 CPU 中转，契约级）；§4.3 wheel 分发（CUDA 矩阵成本高，暂缓）。
- **不建议独立做**：§1.3 fcad-3（收益极小，除非随 1.1 一起）；§1.4 余量工程（除非有真实 GPU 并发场景）；§2.2 factor_corr 适配层（主要开销已闭合，残留 +1.37% 不值得单独追）。

> 任何方向进入实施前：按项目流程走 CLAUDE.md（L0 Spec 冻结，变更走 HG-2）+ 双审查 + fail-closed 证据。
