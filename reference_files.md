# 关键文件索引

> 最后更新: 2026-08-08

## 代码
- [phase0_selfcheck.cu](poc/phase0_selfcheck.cu) — Phase 0 GPU 自检
- [CMakeLists.txt](CMakeLists.txt) — CMake 构建骨架
- [dev-build.bat](dev-build.bat) — 构建脚本（Ninja，6 targets）
- [cross_sectional_rank.cuh/.cu](src/cross_sectional_rank.cuh) — cs_rank v0 kernel（审查闭合；**2026-08-05 workspace 分配缓存**：`cs_rank_workspace` 显式缓存 API 消除每次调用 malloc/free 开销，steady-state 9.17ms BEATS gate 13.926ms；**审查响应闭合**：owner_tracker 绑定 + device 入键 + d_mask 懒分配 + 7 参 overload 保旧 ABI，perf 9.25ms 无回归）
- [mem_tracker.h/.cu](src/mem_tracker.h) — HWM 分配器（strict 所有权，审查闭合）
- [rolling_ic.cuh/.cu](src/rolling_ic.cuh) — rolling_ic v0 kernel（审查闭合；**2026-08-05 F/T 最小证明①**：3 kernel 移至文件作用域体零改动 + `rolling_ic_gpu` 加可选 rank 输出 `h_rank_f_out/h_rank_r_out` 供 chunked-vs-non-chunked rank 位级断言）
- [rolling_ic_impl.cuh](src/rolling_ic_impl.cuh) — rolling_ic 3 生产 kernel 外部声明（F/T 最小证明①共享头，chunked driver 经此调用生产 kernel）
- [factor_corr.cuh/.cu](src/factor_corr.cuh) — factor_corr v0 kernel（两遍中心化 + Kahan 触发；**2026-08-05 F1 对角 var 下溢修复**：writeback 对角改读计算路径值 `isfinite→1.0/NaN`；**F/T 最小证明②重构**：kernel 移文件作用域 + reduce_p1/p2 循环体/树归约改调共享内联函数（factor_corr_impl.cuh，`__forceinline__` 机械等价，3 形状 fc/b 实测位级一致）+ `factor_corr_gpu` 加可选第 8 参 `h_trigger_out`）
- [factor_corr_impl.cuh](src/factor_corr_impl.cuh) — **F/T 最小证明②共享头**（ABI Partial1/2/K1/K2 static_assert + KahanAcc/safe_pearson + 内联累加算子 accum_p1/2_cell + 固定树归约 tree_reduce_p1/2_store + 生产/延续 kernel 声明；生产与延续编译期绑定同一逻辑防 F4；头部标注重构等价三证据源）
- [factor_corr_pybind.cpp](src/factor_corr_pybind.cpp) — **factor_corr pybind11 绑定层（2026-08-05）**（`factor_corr_f64(F3, mask=None)` GIL release + NumPy c_style|forcecast upcast；CMake `BUILD_PYBIND11` option 默认 OFF；smoke 测试 ALL PASS）
- [stock_corr.cuh/.cu](src/stock_corr.cuh) — stock_corr v2 kernel（**双路径 dispatch**：全列 count==T → **fast path**（demean_kernel 串行 Kahan 均值+S2 + fast_gemm 1 累加 Sxy 分层归约，de-mean Gram ≡ two-pass 参考位级一致，无抵消检测）；部分有效 → **general path**（6 累加 + 抵消检测 ratio3 + 分离 fallback_kernel）+ 数值域 -4；fast path **N=500 ~6.1ms / N=2000 ~37ms / N=5000 ~200ms，BEATS 同面板 v2 gate 26.35/359.35/2382.37ms（4.3×/9.6×/11.9×）**；**2026-08-05 可选 `StockCorrRunStats*` out 参**；**2026-08-08 N-blocking 重构**：7 kernel 移入 `namespace stock_corr_impl`（共享头 stock_corr_impl.cuh，防 F4 自参照）——重构等价验证（selfcheck/parity/calibration 不变））
- [parameter_scan.cuh](src/parameter_scan.cuh) — parameter_scan v0（G=4 字典序参数扫描，实现于 cross_sectional_rank.cu；单次 H2D + 逐组 D2H + **`group_status[4]` 逐组部分成功输出**（契约两级失败语义）；**GPT-5.6-Sol 审查闭合** F8/F9/F10；perf 35.9ms BEATS 55.34ms gate）
- [poc3_cs_rank_selfcheck.cu](poc/poc3_cs_rank_selfcheck.cu) — cs_rank 位级自检（150 case + **2026-08-05 workspace 路径 13 例**：ws vs 非 ws 逐位一致/形状切换/mask 开关/clear 复用/错误后可用）
- [poc3_cs_rank_perf.cu](poc/poc3_cs_rank_perf.cu) — cs_rank 性能/显存探针（2026-08-05 改测 **workspace 稳态路径**：9.17ms BEATS gate 13.926ms，诚实报 cold 首调 18.1ms；leak 检查前 clear）
- [poc3_mem_tracker_selfcheck.cu](poc/poc3_mem_tracker_selfcheck.cu) — 三口径显存校准
- [poc3_rolling_ic_selfcheck.cu](poc/poc3_rolling_ic_selfcheck.cu) — rolling_ic 自检（85 case + **F/T 最小证明①**：`rolling_ic_gpu_chunked`（chunk-local 缓冲 + CUB 段重切 + rank dump）+ 3 正用例「IC+ranks」位级 + null-mask + 负控（break_offsets_side）+ min-valid 边界 + **2026-08-10 常量行 GPU→NaN 用例**）
- [poc3_rolling_ic_perf.cu](poc/poc3_rolling_ic_perf.cu) — rolling_ic 性能/显存探针
- [poc3_factor_corr_selfcheck.cu](poc/poc3_factor_corr_selfcheck.cu) + [corr_anchors.h](poc/corr_anchors.h) — factor_corr 自检（16 corpus anchors + 随机 + 确定性 + **2026-08-05 F1 对角回归 5 例** + **F/T 最小证明②**：延续 kernel（reduce_p1/2_cont + finalize_pX_from_pp）+ `factor_corr_gpu_chunked` driver + 8 用例「corr+trigger」位级 + expected_K 硬断言 + 负控（fresh-start/bad-chunk））
- [poc3_factor_corr_perf.cu](poc/poc3_factor_corr_perf.cu) — factor_corr 性能/显存探针（215ms BEATS 1543.52ms）
- [poc3_stock_corr_selfcheck.cu](poc/poc3_stock_corr_selfcheck.cu) — stock_corr 自检（锚点 + 随机含 N>256 + **长 T 对抗至 T=262144 + tiny-scale + 域 -4** + 确定性 + 错误 smoke + **v2 dispatch 8 例**：全有效/full-ones mask/单格 masked 边界/bias 偏移/近常量列/masked bias/N257；CPU 参考两遍 Kahan + **2026-08-10 mask-shape 缺口 3 例**：棋盘交错遮挡/整列全 False mask/masked-out 极端值域校验）
- [poc3_stock_corr_perf.cu](poc/poc3_stock_corr_perf.cu) — stock_corr 性能/显存探针（v2：fast path 全有效面板 N=500 ~6.1ms / N=2000 ~37ms / N=5000 ~200ms，BEATS 同面板 v2 gate 26.35/359.35/2382.37ms；general 路径走 make_panel 合成面板 ~52ms 为**非代表测量**——general 同面板裁决见 2026-08-06 重基线 `runs/stock_corr_general_gate_20260806`）
- [generate_stock_corr_panel_v1.py](benchmark_corpus/generate_stock_corr_panel_v1.py) — v2 全有效 returns 面板生成器（numpy 确定性，连续正态 std~0.02 无精确零，5000 列切片 N=500/2000/5000）
- [rebaseline_stock_corr_gate_v1.py](benchmarks/rebaseline_stock_corr_gate_v1.py) — stock_corr v2 gate 同面板重新基线（CuPy 全有效面板 → exact_half，证据 `benchmarks/results/runs/stock_corr_v2_rebaseline_20260805/gate.json`；**不改 gate_config_v1.json**——v2 gate 独立文档化，corpus gate 保持）
- [poc3_parameter_scan_selfcheck.cu](poc/poc3_parameter_scan_selfcheck.cu) — parameter_scan 自检（error smoke + **group_status 断言** + 字典序锚点 + vs CPU + vs 单次 cs_rank + 确定性）
- [poc3_parameter_scan_perf.cu](poc/poc3_parameter_scan_perf.cu) — parameter_scan 性能/显存探针（corpus 1218×5000 35.9ms BEATS 55.34ms）
- [poc3_calibration.cu](poc/poc3_calibration.cu) — **五操作三口径显存校准（2026-08-05）**：cs_rank/parameter_scan/rolling_ic/factor_corr/stock_corr × canonical 1218×5000 + N=10000（stock N=2000/5000/10000），理论公式(align256 全 buffer) vs tracker HWM vs 驱动采样三口径，CUB temp 取 tracker 单次 max（跨 rep 求和曾致假 FAIL）
- [poc3_corpus_parity.cu](poc/poc3_corpus_parity.cu) — **冻结 corpus 跨端 parity GPU 端（2026-08-05）**：加载冻结 .bin 面板跑 factor_corr/stock_corr，dump GPU 输出供 Python 对冻结 wrapper 对比（关闭 stock_corr v2 审查 F4 放行门槛）

## stock N-blocking + HWM 实测 + factor fblock 移植（2026-08-08 新增）
- [stock_corr_impl.cuh](src/stock_corr_impl.cuh) — stock_corr 7 生产 kernel 共享声明/常量/类型（`namespace stock_corr_impl`，防 F4；N-blocking driver 经此调用生产 kernel）
- [poc3_stock_corr_selfcheck.cu](poc/poc3_stock_corr_selfcheck.cu) — 更新：`stock_corr_gpu_nblock`（pair 轴 N-blocking driver，[B|A] 布局保 safe_pearson 参数序）+ `run_stock_nblock_case` 位级断言 + **`--hwm` 模式**（设备 HWM 实测：sampler + GUARD fast+masked）
- [stock_nblock_hwm_v1.py](benchmarks/stock_nblock_hwm_v1.py) + [stock_nblock_hwm_v1.json](benchmarks/results/stock_nblock_hwm_v1.json) — **stock N-blocking HWM 证据生产者 + 证据**（fail-closed 深度：完整字段/case 集/派生重算/guard r1r2/删旧/provenance git+exe；N22600 device 峰值 22.0 MiB fits，nblock 列闭合为实测）
- [memory_budget_v1.py](benchmarks/memory_budget_v1.py) + [memory_budget_v1.json](docs/memory_budget_v1.json) — 更新：nblock/fblock 列均闭合为实测（N22600 nblock 22.0 MiB、F128 fblock 6588.11 MiB）+ load_fblock/load_stock_nblock 严格 fail-closed（重算派生/整数校验/异常干净拒绝）
- [factor_fblock_hwm_v1.py](benchmarks/factor_fblock_hwm_v1.py) — 更新：fail-closed 深度同步（kind 校验/白名单/guard 唯一/free_before 缺失拒/unexpected+异常 gate/provenance git_dirty/disclosure 全动态/时区）
- [stock_nblock_hwm_v1.py](benchmarks/stock_nblock_hwm_v1.py) — 更新：parse_fblc kind+白名单校验、guard 唯一、driver_peak≤fb、git_dirty/时区

## 数据
- **benchmark corpus 已提交**（2026-08-03）：`benchmark_corpus/`——生成器 `generate_corpus_v1.py`、校验器 `verify_corpus_v1.py`、加载器 `corpus_loader_v1.py`、统计 `compute_corpus_stats_v1.py`、parity 锚点 `generate_parity_anchors_v1.py`、seeds.json（MASTER_SEED=20260802）、manifest_schema_v1.json、CORPUS_DESIGN_v1.md、smoke npz + parity anchors npz（提交）；完整 npz real/synth 不提交（.gitignore）
- 测试 fixture：
  - `tests/fixtures/corr_oracle_v1.py` — correlation 唯一 oracle（锁定 numpy corrcoef，2026-08-03）
  - `tests/fixtures/generate_rolling_ic_labels_v1.py` — rolling_ic 标签生成器（h=5/lag=1）
  - `tests/fixtures/rolling_ic_labels_v1.json` / `.npz` — 标签 manifest + 期望数据（SHA-256 记录于 manifest）
  - `tests/fixtures/corr_corpus_v1.*` — 对抗数值 corpus（v1.1，12 case：偏置/近±1/近零方差/稀疏 mask/ULP 扰动/退化；branch 由 trigger 状态机算、真 dtype、严格 manifest，R3-02）
  - `tests/fixtures/test_cases_v1.json` — 测试实体矩阵（v1.1，50 cases/7 targets，recoverable 白名单 + fatal 10 stages，R3-08/F5-09）
  - `tests/fixtures/corr_math_v1.py` — correlation 数学单一真源（CompensatedSum/BinaryFrontier/safe_pearson/is_aliasable/checked，F5-01/02/04/08/11）+ `corr_math_trace_v1.json`（机械 trace）
  - `tests/fixtures/calibration_v1.py` + schema + trace — reserve 校准（p99=650M、fail/fail/pass、原子写，F5-12）
  - `tests/fixtures/validate_self_fix_v1.py` — 自修复验证单入口（validate_implementation/corr_corpus/test_cases/workspace/calibration，15/15 PASS）

## 文档
- [PLAN.md](PLAN.md) — 方案设计（v2 整改草案）
- [CLAUDE.md](CLAUDE.md) — L0 Spec（已冻结，PoC ① 2026-08-03 HG-2 批准；2026-08-05 HG-2 二次修订：correlation 归约顺序敏感输入豁免条款）
- [IMPLEMENTATION.md](docs/IMPLEMENTATION.md) — 实现设计（**v0.7 自包含**，GPT-5.6-Sol 自修复闭合六审 15 项，`validate_self_fix_v1.py` 15/15；Phase 1-4 占位）
- [workspace_v1.json](docs/workspace_v1.json) — workspace/solver 自动计算（`benchmarks/compute_workspace_v1.py` v2：逐分配 Timeline + live-byte HWM + solver 12/36/8/16 scenarios）
- [gate_config_v1.json](docs/gate_config_v1.json) — PoC ④ Gate 机器配置（`benchmarks/generate_gate_config_v1.py` 从 canonical run JSON 自动生成，raw/2 全精度 + SHA-256）
- [COMPETITOR_ANALYSIS.md](COMPETITOR_ANALYSIS.md) — 竞品分析（§五 PoC ② 实测）
- [CHANGELOG.md](CHANGELOG.md) — 变更记录
- [RISK.md](RISK.md) — 风险登记
- [README.md](README.md) — 项目入口
- [FUTURE_WORK.md](FUTURE_WORK.md) — 未来修改方向（v1.1.0 后，双审查闭环）
- [PROJECT_ASSESSMENT.md](PROJECT_ASSESSMENT.md) — 项目价值评估（内部，已 gitignore）
- [project_status.md](project_status.md) — 内部状态（已 gitignore）
- [reviews/](reviews/) — 独立审查报告
- [prompts/](prompts/) — 审查提示词

## PoC ③ 校准 + 冻结 corpus parity（2026-08-05 新增）
- [poc3_calibration_v1.py](benchmarks/poc3_calibration_v1.py) — 校准汇总报告生成器（跑 exe / `--from` 读捕获输出 → `results/poc3_calibration_v1.{json,md}`；预算交叉核对 + 模型设计口径说明）
- [results/poc3_calibration_v1.json](benchmarks/results/poc3_calibration_v1.json) / [.md](benchmarks/results/poc3_calibration_v1.md) — **五操作三口径校准证据（11 例全 PASS）**：理论公式==HWM 精确（delta 0）、driver overhead ∈ [0,64MiB]、无泄漏；最大 HWM 2381 MiB（factor_corr N=10000）、stock_corr N=10000 输出 O(N²)=800MiB 实测确认
- [corpus_parity_v1.py](benchmarks/corpus_parity_v1.py) — 冻结 corpus 跨端 parity 编排（corpus_loader 校验 → 导出 f64/u8 冻结面板 → 跑 GPU → 冻结 wrapper corr_oracle_v1.py 逐 pair 对比 |Δr|≤1e-12/NaN + 逐 pair joint-mask bias/尺度守卫；**v1.1 响应 GPT-5.6-Sol 审查**：selected_path/fallback_count 执行证据断言 + 冻结 degenerate 对角/低偏置 fallback 用例 + 全链路 hash provenance + 复合 `gate_closed=comparisons∧coverage∧provenance`，无 --skip-run；**2026-08-10 corpus_parity 检查工具 v1.2 代**：stock_corr 默认**全列 5000**、流式逐 pair 比较防 O(N²) 物化、`--n-sub` 仅资源受限降级且 gate 保持 false）
- [results/corpus_parity_v1.json](benchmarks/results/corpus_parity_v1.json) / [.md](benchmarks/results/corpus_parity_v1.md) — **F4 放行门槛关闭证据（v1.2 全列）**：5 用例全 PASS——factor_corr 2.65e-14 / stock general 6.66e-16 / fast 2.22e-16 / degenerate 对角 1.11e-16 / 低偏置 fallback 1.39e-17；**general/fast 各 12,502,500 对（全列 N=5000）全 PASS**；dispatch 执行证据（general/fast 全匹配、fallback_count 10318/10）；bias 有限 pair 全 <1e3
- 临时面板/输出：`scratch/corpus_parity/`（gitignored）

## Phase 1/2 fc.* 契约适配层 + 验收测试套件（2026-08-06 新增）
- [fc/__init__.py](fc/__init__.py) — fc 包入口（6 公共操作懒加载绑定，CPU-only 可 import）
- [fc/_util.py](fc/_util.py) — 适配共享助手（容器检测/DLPack/严格 bool mask/device 路由/错误映射/`_dtype_ok` torch-aware + `mask_must_match_device` 消费 capsule 解析 device，2026-08-06 双修复）
- [fc/_cpu_core.py](fc/_cpu_core.py) — **CPU oracle 单一真源**（np_cs_rank/np_factor_corr/np_stock_corr/np_rolling_ic/np_parameter_scan + `_two_pass_corr` Kahan 守卫，2026-08-06 HG-2 修复）
- [fc/cross_sectional_rank.py](fc/cross_sectional_rank.py) — cross_sectional_rank + factor_plane
- [fc/correlation.py](fc/correlation.py) — factor_corr + stock_corr（cpu/cuda backend + 数值域前置）
- [fc/rolling_ic.py](fc/rolling_ic.py) — rolling_ic（min_valid 严格 int + device 路由）
- [fc/parameter_scan.py](fc/parameter_scan.py) — parameter_scan（G=4 字典序 + active-group 子集 + **非白名单 status→扫描级 RuntimeError 硬化** 2026-08-06）
- [tests/test_adapter_v1.py](tests/test_adapter_v1.py) — **Phase 2 验收测试套件（7 类 105 测试，2026-08-06）**：F01-F18 七项处置 + Workflow/GPT-5.6-Sol 双审查闭合；`pytest tests/ -q` 105 passed / 3 skipped
- [tests/conftest.py](tests/conftest.py) + [pytest.ini](pytest.ini) — pytest 路径配置（pythonpath=root）
- [prompts/gpt56sol_test_adapter_review_prompt.md](prompts/gpt56sol_test_adapter_review_prompt.md) — 测试套件外部审查提示词（gitignored）

## PoC ② 公平基线（2026-08-03 新增）
- [FAIR_BASELINE_PROTOCOL_v1.md](benchmarks/FAIR_BASELINE_PROTOCOL_v1.md) — 公平基线协议（GATE 关闭，定稿）
- [parity_check_v1.py](benchmarks/parity_check_v1.py) — contract parity 三臂 × 27 case + ext 边界检查 8 例（含 **`ext_corr_diag_boundary`**：5 类对角边界列 × stock/factor × np/cp 对冻结 oracle 断言，2026-08-05 回归 _two_pass_corr 乘积型分母；2026-08-06 加 gpu 臂 = 四臂）
- [perf_bench_v1.py](benchmarks/perf_bench_v1.py) — 性能基准三臂 × 6 操作
- [results/](benchmarks/results/) — 证据产物（`runs/<run_id>/` 不可变 run 记录：每 backend 独立 JSON 含 CI/cold/upload/resident/显存/命令；`--render` 从 run JSON 生成 perf_report_v1.md 并做一致性校验）

## Phase 2-3 验收门槛 + 三项 P1（2026-08-06 新增）
- [acceptance_v1.py](benchmarks/acceptance_v1.py) — 统一验收编排器 v2（六门全 PASS + unlock_phase4）
- [phase23_acceptance_spec_v1.md](docs/phase23_acceptance_spec_v1.md) — 验收门槛设计 spec（Workflow 7 agent）
- [test_timeline_no_lookahead_v1.py](tests/test_timeline_no_lookahead_v1.py) — 无未来函数集成测试（6 例）
- [test_corpus_halt_v1.py](tests/test_corpus_halt_v1.py) — **real corpus 停牌标记测试（12 例，2026-08-07）**：mask 不 all-True 守卫 / fwd 停牌窗口 NaN / fetch 接线端到端 / rolling_ic mask∩isfinite / factor_corr mask=False 但有限 / parse_close_volume 边界
- [rebaseline_stock_corr_general_gate_v1.py](benchmarks/rebaseline_stock_corr_general_gate_v1.py) — stock general 同面板 gate（2.4×）
- [memory_budget_v1.py](benchmarks/memory_budget_v1.py) + [memory_budget_v1.json](docs/memory_budget_v1.json) — 内存静态预算模型（校准 delta≈0；F=128 需分块；**fblock(项③) 列已实测闭合**：block=32 driver 6529.09 MiB fits，分配链 probe delta 3 MiB）
- [poc3_factor_corr_selfcheck.cu](poc/poc3_factor_corr_selfcheck.cu) — fblock driver + 边界 cases（pair 轴分块位级证明）+ **`--hwm-f128` 模式**（F=128 设备 HWM 实测：逐 case sampler + 健康握手 + GUARD 位级守卫 load-bearing）
- [factor_fblock_hwm_v1.py](benchmarks/factor_fblock_hwm_v1.py) + [factor_fblock_hwm_v1.json](benchmarks/results/factor_fblock_hwm_v1.json) — fblock HWM 报告生成器 + 证据（FBLC\|/GUARD\| 解析 + **fail-closed 两层判据**：fit=driver_peak≤预算 / 显存耗尽=margin≈0 / OOM=rc≠0；`closure_status` + 下游复验 + provenance）
- [fetch_real_corpus_v1.py](benchmark_corpus/fetch_real_corpus_v1.py) + [generate_real_corpus_v1.py](benchmark_corpus/generate_real_corpus_v1.py) — real corpus 摄入管道（baostock）
- [corpus_real_v1.manifest.json](benchmark_corpus/corpus_real_v1.manifest.json) — 真实 A 股 corpus provenance（93 股×1212 交易日）

## Phase 4 benchmark + NRR（2026-08-06 新增）
- [phase4_bench_v1.py](benchmarks/phase4_bench_v1.py) — Phase 4 基准编排器（G1 corpus 校验 / G2 git 钉扎 / G3 双 corpus 流水位级确定性 / G4 real parity / G5a binding 级单算子统计 + 热采样 / G5b e2e fresh+跨 run 门 / G6 证据 self-hash；gate 持久化 + `--fresh` 复用；**CUDA 上下文 segfault 修复**：torch/cupy 先于 pybind kernel 初始化，fc/poc4 惰性导入）
- [results/phase4_bench_v1.json](benchmarks/results/phase4_bench_v1.json) / [.md](benchmarks/results/phase4_bench_v1.md) — Phase 4 基准证据（**E2E F=12 2.940–3.035× PASS-partial，未达 ≥5× 目标 → NRR-2026-024**；统一估计量 ratio-of-medians；单算子 binding 级 cs_rank 1.59×/parameter_scan 2.28×/rolling_ic 1.99×/factor_corr 13.15×/stock general 2.31–2.65×（跨会话方差异常波动，边界断言仅指示性）；**2026-08-07 基于停牌 corpus `41BB9EF4` 重跑 fresh**：parity gate=True、cross-run delta 3.13% 稳定、**stale 标注解除**；**经 GPT-5.6-Sol 审查 13 发现修复**：fail-closed 门 + 统一估计量 + CUDA 子命令覆盖 + gate provenance envelope）

## 发布流程（2026-08-08 新增）
- [README.md](README.md) / [README.en.md](README.en.md) / [README.zh-Hant.md](README.zh-Hant.md) — 三语 README（交叉链接/badge/Quick Start/性能表/示例图/API 矩阵/文档索引/Mermaid 架构）
- [support_matrix.json](docs/support_matrix.json) — 构建/运行环境支持矩阵（单一真源；CUDA 13.3/MSVC 19.51/Python 3.12.7/CC 8.9；build_tested/declared_support/benchmark_runtime 三口径）
- [LICENSE](LICENSE) / [CITATION.cff](CITATION.cff) — MIT 许可 + 学术引用
- [CONTRIBUTING.md](CONTRIBUTING.md) / [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) / [SECURITY.md](SECURITY.md) / [SUPPORT.md](SUPPORT.md) — 社区健康文件（贡献/行为准则/安全/支持）
- [ci.yml](.github/workflows/ci.yml) + [requirements-ci.txt](requirements-ci.txt) — CI（windows-latest + Python 3.12，CPU/契约臂 75 passed；numpy/pytest 锁版本）
- [.github/ISSUE_TEMPLATE/](.github/ISSUE_TEMPLATE/) + [PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md) — Issue/PR 模板
- [factor_stream_hwm_v1.py](benchmarks/factor_stream_hwm_v1.py) + [factor_stream_hwm_v1.json](benchmarks/results/factor_stream_hwm_v1.json) — streaming(项②) HWM 证据（fail-closed 深度：分配下界/provenance source=live+git_head 校验/原子写/GUARD K>0；F128 实测 7,079.75 MiB fits，物理余量薄 ~26-166 MiB）
- [docs/img/](docs/img/) — 示例图（rolling_ic/factor_corr/perf_speedup，确定性合成面板真实输出）
