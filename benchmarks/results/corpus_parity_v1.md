# 冻结 corpus 跨端 parity 重测（v1.1）—— stock_corr v2 审查 F4 放行门槛

> 生成：2026-08-08 · benchmarks/corpus_parity_v1.py v1.1 (DeepSeek-V4-Flash via Claude Code CLI, 2026-08-05)
> 语料：corpus_synth_v1（T=1218 N=5000 F=12，data_sha256 已校验）+ all-valid 冻结面板前缀 N=200
> v1.1 响应 GPT-5.6-Sol 审查：执行证据（selected_path/fallback_count）断言 + NaN/退化对角 + 全链路 hash + 逐 pair bias/尺度 + 复合 gate_closed

## 结论

**`gate_closed = comparisons_ok AND coverage_ok AND provenance_ok` = ✅ 关闭**
- comparisons_ok = **✅** 实现（GPU kernel）对冻结 wrapper corr_oracle_v1.py 逐元素满足 |Δr| ≤ 1e-12 / NaN parity
- coverage_ok = **✅** 全部 4 个 stock 用例的**实际 dispatch 路径**与预期一致 + fallback_count>0 + ✅ degenerate 用例同时覆盖有限对角（正常列 1.0）与 NaN/退化对角（常量列）
- provenance_ok = **✅** 全链路 SHA-256（corpus/all-valid 面板/导出输入/GPU 输出/exe）已记录且本轮 fresh 运行

| 用例 | pair 数 | 有限 ok/总 | NaN 匹配 | max|Δr| | max pair bias | fallback | 判定 |
|---|---|---|---|---|---|---|---|
| factor_corr | 78 | 78/78 | 0/0 | 2.653e-14 | 0.0000 |  | ✅ |
| stock_corr_general | 20100 | 20100/20100 | 0/0 | 4.441e-16 | 0.2130 | fb=55 | ✅ |
| stock_corr_fast | 20100 | 20100/20100 | 0/0 | 2.220e-16 | 0.0938 | fb=0 | ✅ |
| stock_corr_degenerate_diag | 10 | 6/6 | 4/4 | 1.110e-16 | 0.1297 | fb=0 | ✅ |
| stock_corr_fallback | 10 | 6/6 | 4/4 | 1.388e-17 | 2.1302 | fb=10 | ✅ |

## Dispatch 执行证据（GPU 端回传，非命名推断）

| 用例 | 预期路径 | 实际 selected_path | fallback_count |
|---|---|---|---|
| stock_corpus | general | general | 55 |
| stock_fast | fast | fast | 0 |
| stock_degen | fast | fast | 0 |
| stock_fallback | general | general | 10 |

## Bias / 尺度证据（逐 pair joint mask，HG-2 strict parity 适用性）

| 用例 | max_abs(mean)/sigma | 退化 pair 数 | max_abs | min_nonzero_abs | 下溢 pair | HG-2 阈值 |
|---|---|---|---|---|---|---|---|
| factor_corr | 0.0000 | 0 | 8.376e+00 | 6.426e-09 | 0 | 1000.0 |
| stock_corr_general | 0.2130 | 0 | 9.998e-02 | 3.256e-08 | 0 | 1000.0 |
| stock_corr_fast | 0.0938 | 0 | 1.018e-01 | 7.274e-08 | 0 | 1000.0 |
| stock_corr_degenerate_diag | 0.1297 | 1 | 2.808e+00 | 1.534e-02 | 0 | 1000.0 |
| stock_corr_fallback | 2.1302 | 1 | 4.976e+00 | 1.618e-02 | 0 | 1000.0 |

- **全部有限比较 pair 的 max_abs(mean)/sigma < 1000.0 且无下溢尺度** → 归约顺序敏感豁免
  不适用于任一有限比较 → strict wrapper parity（|Δr|≤1e-12）判据成立（`strict_parity_applies` 全 ✅）。
- 退化 pair（常量列，σ==0 → bias 未定义）以 **NaN parity** 判定（GPU/oracle 同判 NaN），
  不进入 strict parity 适用域，其 NaN 匹配数见上表 `NaN 匹配` 列。

## 用例说明

- **factor_corr**：全 corpus (T,N,F) masked pooled 相关 → (F,F)，含对角 1.0/NaN。
- **stock_corr_general**：corpus returns 前缀，~199/200 列部分有效（含 NaN/mask False）→ general path。
- **stock_corr_fast**：all-valid 冻结面板前缀（全列 count==T）→ fast path（de-mean Gram）。
- **stock_corr_degenerate_diag**：冻结面板（常量列 + 正常列，全有效）→ fast path；覆盖**有限对角与 NaN/退化对角**两类。
- **stock_corr_fallback**：冻结低偏置面板（独立 N(2,1) 列触发抵消检测 + 常量列 + mask 强制 general）→ general path 且 **fallback 实际命中**（fallback_count>0），结果对冻结 wrapper 有限/NaN 双判据通过。

## Provenance（全链路 SHA-256）

- corpus `corpus_synth_v1` data_sha256：`0A7D7AB5A6AD1F7E…`
- all-valid 面板源 `benchmark_corpus\stock_corr_panel_v1_5000.bin`：`934D6F927093A4E6…`
- exe `build\poc3_corpus_parity.exe`：`A054097F462029D7…`
- 导出输入/GPU 输出 hash 详见 `corpus_parity_v1.json` `provenance` 节（fresh 运行，无 --skip-run 复用）。

## 复现

    PYTHONIOENCODING=utf-8 python benchmarks/corpus_parity_v1.py [--n-sub 200]

*生成模型: benchmarks/corpus_parity_v1.py v1.1 (DeepSeek-V4-Flash via Claude Code CLI, 2026-08-05)*
