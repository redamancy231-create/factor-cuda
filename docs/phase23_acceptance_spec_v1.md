# Phase 2-3 验收门槛方案 spec（综合定稿，2026-08-06）

> 生成：2026-08-06 · Workflow 7 agent 多视角设计（acceptance-scope / stock-verdict / script-design / evidence-provenance / independent-adversary + challenge 魔鬼代言人 + synthesize）
> 生成模型：DeepSeek-V4-Flash (via Claude Code CLI) · 973K tokens · 159 tool calls
> 本 spec 为内部追踪 + 验收实现依据，不公开推送。

## 0. 定稿依据与裁决要点

本 spec 综合 5 设计输出 + 魔鬼代言人挑战，并经文件实证核实。三处决定性纠正：

- **C1/C2 成立（最关键）**：`poc/poc3_stock_corr_perf.cu:159-166` general 路径用 `make_panel("returns")` 合成面板（~6% NaN、mask=nullptr），对 corpus 面板 CuPy gate（`docs/gate_config_v1.json` stock_corr(N=500) exact_half 21.806ms）报 52.19ms NOT beat —— 这是**跨数据比较**，违反 FAIR_BASELINE 同数据同 mask 纪律，0.42× 判 FAIL 无效。同项目自证 `benchmarks/results/poc4_e2e_v1.json:437-484`：**同一 general kernel 在 corpus returns+mask 面板上 N=500 gpu=27.05ms vs cupy=53.09ms = 1.96×、N=2000 = 1.99×**（快于 CuPy raw，非"比免费替代还慢"）。52.19 vs 27.05 的 2× 差异（同 kernel 两面板）**无人解释**，须同面板重基线后裁决，此前任何 general 判定均为 over-claim。
- **C3 成立（BLOCKER）**：Phase 2「无未来函数测试」无执行链。`tests/fixtures/rolling_ic_labels_v1.json`（h=5/lag=1/T=40/N=5/seed=20260803，data_sha256=660F07…、script_sha256=FC34…）已冻结但 **grep 全仓库 tests/ 零测试引用**；test_adapter_v1.py F07=CUDA 同步边界、F17=parity_anchors_v1 manifest（`:705`），均不涉时间线 fixture。设计输出声称 F07/F17 覆盖为误判。
- **C4 成立**：`docs/gate_config_v1.json` 是冻结 formal gate（run_id poc2_baseline_20260804c，schema 1.1.0，generator_sha256）。**禁止**验收日 `--generate` 重生成（会移动靶、与 Phase 1 不可比）；只调 `generate_gate_config_v1.py --check`（fail-closed）。新增 gate 走 `benchmarks/rebaseline_stock_corr_gate_v1.py` 独立同面板模式（不改 gate_config_v1.json）。

仍成立的设计结论（纳入正文）：perf exe 退出码不反映 BEATS（`poc3_cs_rank_perf.cu:177` return 0 恒定），验收须解析 stdout/gate JSON 复算；gate 硬编码已末位漂移（.cu:157 26.348400 vs gate.json 26.34840000064287）；general 负结果入 NRR 单列、不自动 REDESIGN；reference_files.md 陈旧数字（:26 记 27.27ms vs gate.json 26.3484）随验收一次性订正；F18 用 monkeypatch fake 绑定（test_adapter_v1.py:920-1009）须在报告标注 adapter 逻辑层 vs 绑定行为层分层。

---

## 1. 验收范围（逐操作五维聚合）

判据单一真源：CLAUDE.md PoC 决策表 §179-190（①语义②公平基线≥2×③显存④端到端）+ PLAN.md:148-153 Phase 1-4 门槛 + CLAUDE.md:56-57 确定性/oracle 一致性条款。

每操作五维判定规则（固化进验收脚本）：

1. **正确性**：C++ selfcheck ALL PASS（rc==0）∧（corr 类：corpus_parity gate_closed=true 且 |Δr|≤1e-12/NaN parity + dispatch selected_path 断言；rank 类：独立 oracle 位级一致 + NaN 载荷）+ pytest 105 passed / 0 failed。
2. **性能**：median < gate exact_half → BEATS；gate 来源按 op 锚定（下表）。
3. **显存**：max HWM ≤ available − 安全余量（aggregation 层显式减 512 MiB；校准脚本不改）。
4. **Phase 专属**：Phase 2 = timeline 集成测试（缺口 P0 补后）；Phase 3 = G=4 字典序 + 输出 schema + group_status 逐组部分成功 + 端到端可复现。
5. **可复现**：git HEAD + corpus data_sha256 + gate source+sha256 + env 指纹。

| 操作 | gate 锚定（单一真源） | 正确性证据锚 | 性能现状（2026-08-06 fresh） |
|---|---|---|---|
| cs_rank（复核，Phase 1） | corpus `gate_config_v1.json` cs_rank exact_half 13.926 | selfcheck 163 case（poc/poc3_cs_rank_selfcheck.cu）+ F01 系列 | 9.34ms BEATS（vs raw 27.85 = 2.98×） |
| factor_corr | corpus `gate_config_v1.json` 1543.521 | selfcheck anchors+最小证明② + corpus_parity factor 2.65e-14 + F10/HG-2 | 217.69ms BEATS（14.2× vs raw 3087.04） |
| stock_corr fast（全有效面板） | **v2 同面板 gate** `benchmarks/results/runs/stock_corr_v2_rebaseline_20260805/gate.json` | selfcheck v2 dispatch 8 例 + corpus_parity fast 2.22e-16 + F10/HG-2 | 6.09/37.34/200.30ms BEATS 26.35/359.35/2382.37（4.3×/9.6×/11.9×） |
| stock_corr general（NaN/mask 面板） | **待建同面板 gate**（corpus returns[:, :N]+mask，CuPy masked-GEMM exact_half，对称 reps） | corpus_parity general 4.44e-16 + F10/HG-2 | **未裁决**（见 §2；初步同面板 1.96×/1.99×） |
| rolling_ic | corpus `gate_config_v1.json` 77.183 | selfcheck 85+最小证明① + F01 系列 | 48.43ms BEATS（3.19× vs raw 154.37） |
| parameter_scan | corpus `gate_config_v1.json` parameter_scan(G=4) 55.343 | selfcheck（G=4 字典序+group_status）+ F18（adapter 逻辑层） | 35.82ms BEATS（3.09× vs raw 110.69） |

PASS/FAIL 规则：五维全 PASS 记 PASS；性能 <2×（相对公平基线 raw）记负结果入 NRR 不自动 REDESIGN；正确性/显存任一 FAIL 记 FAIL 阻塞 Phase 4。

---

## 2. stock_corr 双口径裁决（含 0.42× 纠正）

**纠正既有口径**：设计输出中「general 0.42× FAIL」不成立——`poc/poc3_stock_corr_perf.cu:159-166` general 用合成面板，对 corpus gate 属跨数据比较；「general 比 CuPy raw 还慢 ~1.2×」同样被 `poc4_e2e_v1.json:437-484` 直接证伪（同面板 1.96×/1.99× 快于 CuPy）。

**双路径并列判定**（禁平均、禁混用 gate、禁以 fast 覆盖 general）：

- **stock_corr fast**：性能主判据 PASS（全有效面板，v2 同面板 gate，N≥2000 超 5× 目标）。声明限定「全有效面板」。
- **stock_corr general**：**状态 = UNADJUDICATED（待同面板重基线）**，不是 PASS 也不是 FAIL。真实 A 股 returns 几乎必走 general（corpus 前 500 列仅 2/500 全有效），**general 才是主路径**，fast 是特化回归。

**待裁决动作（P0）**：仿 `benchmarks/rebaseline_stock_corr_gate_v1.py` 协议，导出 corpus returns[:, :500]+mask 为冻结 .bin（消除 `poc3_stock_corr_perf.cu:159-166` 合成面板与 `poc4_e2e_v1.py` corpus 面板之间 52.19 vs 27.05 的 2× 矛盾），CuPy masked-GEMM exact_half 同面板对称 reps 重测 GPU general。裁决规则：≥2× 记 PASS（附边界注释）；<2× 记 NRR 负结果（硬件 FP64 天花板 1/64 结构性归因 + kernel 效率次因，禁止写「可实现 2×」），不阻塞 Phase 2/3 语义门。

**正确性**：PLAN Phase 2 门槛（numpy.corrcoef 对照/FP64 CPU fallback/无未来函数）双路径 PASS——corpus_parity general 4.44e-16/fast 2.22e-16 + F10 oracle 直连 + HG-2 bias ≤1e-12 + fc/_cpu_core.py `_two_pass_corr` Kahan 守卫。general <2× 是性能负结果，非正确性失败，**不构成 Phase 2 阻塞**。

---

## 3. 统一验收机制（右尺寸结论）

**形态**：写 thin orchestrator `benchmarks/acceptance_v1.py`（~300-400 行），**子进程复用全部现有证据生产者，不重写任何验证逻辑**。这是最小充分形态——现有 5 个 perf exe + 4 个 Python 生产者 + pytest + 10 个 selfcheck exe 无「一跑全绿」入口，纯手写报告会重蹈 README 数字漂移（methodology manifest_single_truth）。

**职责边界**：只拥有 ①编排（subprocess 按序跑生产者、rc==0 信号）②gate 重推（perf 判定从 `gate_config_v1.json` + v2 `gate.json` 读 exact_half 复算，**不信任 exe 硬编码 BEATS**——退出码恒 0 且硬编码已漂移）③聚合（逐 op 五维判定 + 双层/三层报告）。

**阶段流水线**：①env/provenance → ②correctness_cpp（6 selfcheck exe rc==0）→ ③pytest tests/（rc==0、0 failed、3 skipped 列因白名单 {infeasible caps, multi-GPU}）→ ④parity_corpus（corpus_parity_v1.py gate_closed）→ ⑤parity_arms（parity_check_v1.py 四臂 27/27+ext 8/8）→ ⑥memory（poc3_calibration_v1.py all_pass + 显式减 512 MiB 余量）→ ⑦perf（5 exe stdout 正则 median + JSON 重推）→ ⑧e2e（poc4_e2e_v1.py verdict_main_F12）→ ⑨聚合发双件 + 退出码。

**右尺寸收敛（防过度建造）**：不建缓存/增量层（Gate 语义要求每次 fresh 证据）、不建 --fast/--strict 迭代模式（Gate 无迭代语义，fresh 全量即默认）、不对每个证据文件建 sha256 引用矩阵（用既有 run JSON env + corpus_parity 全链路 hash + 单个 np.show_config() 摘要 hash 即可）。`--no-run` 仅供审阅既有产物且 provenance 标 'reused'，不得冒充 fresh。timeline 集成测试是唯一必须新建的验证逻辑。

---

## 4. 报告结构（JSON + MD 双件）

`benchmarks/results/acceptance_v1.{json,md}`，**JSON 为单一真源、MD 由渲染器从 JSON 现算生成，零手工同步**（迁移 perf_bench_v1.py `--render` 模式 + FAIR_BASELINE §5）。

JSON schema：
- `meta`: schema_version、generated_at、git HEAD+dirty、python/numpy/torch/cupy 版本、GPU 名/CC/总显存、nvcc/driver、np.show_config() 摘要 hash、corpus data_sha256（经 corpus_loader_v1.load 校验）
- `evidence`: 既有证据引用矩阵 `{key: {path, sha256, role}}`（引用不重算）
- `gates`: 五门 verdict，每门逐条 evidence key 引用：
  - `gate_2_semantics`（oracle 对照 + FP64 CPU fallback）
  - `gate_2_memory`（max HWM ≤ available−512 MiB）
  - `gate_2_no_lookahead`（timeline 集成测试，缺口补后）
  - `gate_3_schema`（G=4 字典序 + 输出 schema + group_status 部分成功）
  - `gate_3_e2e`（corpus data_sha256 + seeds.json 可复现）
- `perf_status`: 每 op 一行 `{op, gate_source, gate_exact_half, median_ms, verdict, speedup_vs_raw}`；stock_corr 两行并列（fast/v2 gate、general/待裁决）
- `known_gaps[]`: 每项 `{id, title, evidence, decision}`（stock_corr general = "carried to Phase 4/NRR"）
- `overall_acceptance`: exit 0/1 反映五门；gap 不吞没于绿

**provenance 纪律**：perf 判定由编排器从 (exe stdout median, gate JSON) 复算，被评对象自报的 BEATS 仅终端展示；数字去化（case 数/pytest 数不硬编码、引用证据文件）；pytest 3 skipped 显式列因（T*N>INT32_MAX、parameter_scan 4GiB host 预算、NEED_MULTI_GPU），不得写成 108 passed。报告覆盖矩阵标注 F18 为 adapter 逻辑层（monkeypatch fake）vs C++ selfcheck 为绑定行为层。

---

## 5. Phase 4 解锁条件

**定义**：Phase 2-3 验收通过 = 五门全 PASS（`gate_2_semantics` ∧ `gate_2_memory` ∧ `gate_2_no_lookahead` ∧ `gate_3_schema` ∧ `gate_3_e2e`），其中：
- `gate_2_no_lookahead` 以新 timeline 集成测试为准（§6 P0-1）
- stock_corr general 性能缺口以同面板重基线裁决为准；≥2× 记 PASS、<2× 记 NRR 负结果，**两者均不阻塞 Phase 4 解锁**（性能负结果非语义门失败，且 fast 路径 5-12× 大胜 + 正确性全 PASS）

**下一步（Phase 4）**：固定 corpus 可复现 + 统计协议完备 + NRR 按预注册判据（≥5×）。NRR 须单列：stock_corr general 负结果条目（硬件 FP64 归因）+ fast 正结果条目；cs_rank 2.98×/parameter_scan 3.09×/rolling_ic 3.19×/factor_corr 14.2× 中 ≥2× 但 <5× 者统一标注 below_5x_note，禁以 fast 的高加速掩盖 general 缺口。E2E 判据范围排除 stock_corr（poc4_e2e_v1.py verdict_scope 明示）。

---

## 6. 缺口清单（按优先级）

**P0（阻塞验收）**
1. **timeline 集成测试（BLOCKER）**：新建 pytest，读 `tests/fixtures/rolling_ic_labels_v1.json` → 校验 data_sha256 与 .npz 匹配 → 重跑 `generate_rolling_ic_labels_v1.py` 逐元素断言等于冻结 npz → 断言 forward_returns[t]=price[t+1+h]/price[t+1]−1（h=5/lag=1）→ 断言末尾 h+lag 行及停牌单元 NaN → fc.rolling_ic 行对齐冒烟。这是 PLAN.md:151 门槛的必要执行证据。
2. **stock_corr general 同面板重基线 + 裁决**：corpus returns[:, :500]+mask 冻结 .bin，CuPy masked-GEMM exact_half 对称 reps，重测 GPU general，解释 52.19 vs 27.05 矛盾；产出独立同面板 gate（仿 v2 模式，不改 gate_config_v1.json）。

**P1**
3. `benchmarks/acceptance_v1.py` + 双件报告 + 五门退出码（§3/§4）。
4. `reference_files.md:26` 陈旧数字订正（27.27→26.3484 等，随验收落地一次性完成）。

**P2**
5. 后续 perf .cu 改从 gate JSON 读 gate（消除硬编码漂移源）；现阶段仅交叉核对记录偏差。
6. 显存安全余量显式化：聚合层断言 max HWM(2381 MiB) ≤ 可用(7676 MiB)−512 MiB，不改校准脚本逻辑。

---

**诚实声明**：本 spec 中所有 PASS 均有文件证据锚定（§1 表）；stock_corr general 性能未裁决、52.19 vs 27.05 的 2× 矛盾未解释，均如实呈现；无任何以 fast 成绩掩盖 general 缺口之处。
