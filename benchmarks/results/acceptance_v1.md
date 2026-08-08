# factor-cuda Phase 2-3 验收报告（acceptance_v1 v2）

> 生成：2026-08-06T16:57:29+08:00 · git 7f0ee95626be (clean)
> 环境：python 3.12.7 · numpy 2.4.4 · cupy 14.1.1
> GPU：NVIDIA GeForce RTX 4060 Laptop GPU (cc 8.9) · corpus corpus_synth_v1 data_sha256 0A7D7AB5A6AD1F7E...

## 总体裁决

**✅ PASS** —— 六门全 PASS；**unlock_phase4 = True**（git worktree clean）

| 门 | verdict | 证据 |
|---|---|---|
| gate_2_semantics | ✅ | selfcheck_cpp, pytest, parity_corpus, parity_arms |
| gate_2_memory | ✅ | calibration, nvidia-smi |
| gate_2_perf | ✅ | perf_exes, gate_config_v1.json, v2_gate.json, general_gate.json |
| gate_2_no_lookahead | ✅ | pytest_timeline, rolling_ic_labels_v1.json |
| gate_3_schema | ✅ | selfcheck_parameter_scan, pytest_F18 |
| gate_3_e2e | ✅ | poc4_e2e_v1.json, corpus_seeds |

## 性能判定（gate 由 JSON 复算 + 身份校验，不信任 exe 硬编码 BEATS）

| 操作 | 路径 | gate 源 | gate exact_half (ms) | median (ms) | rc | verdict |
|---|---|---|---|---|---|---|
| cs_rank | workspace | corpus | 13.9265 | 9.2527 | 0 | BEATS |
| parameter_scan | canonical | corpus | 55.3428 | 36.6345 | 0 | BEATS |
| rolling_ic | canonical | corpus | 77.1825 | 49.1063 | 0 | BEATS |
| factor_corr | canonical | corpus | 1543.5209 | 219.6155 | 0 | BEATS |
| stock_corr | fast(N=500) | v2-same-panel | 26.3484 | 5.9540 | 0 | BEATS |
| stock_corr | fast(N=2000) | v2-same-panel | 359.3518 | 38.7625 | 0 | BEATS |
| stock_corr | fast(N=5000) | v2-same-panel | 2382.3669 | 203.7185 | 0 | BEATS |
| stock_corr | general(N=500) | general-same-panel-20260806 | 31.0500 | 24.5875 | 0 | BEATS |
| stock_corr | general(N=2000) | general-same-panel-20260806 | 288.9532 | 255.1424 | 0 | BEATS |

> synthetic make_panel('returns') general N=500 median=50.986ms is a non-representative panel (~6% non-finite + integer/zero-point distribution); general verdict uses fresh corpus same-panel measure vs frozen gate (20260806).

## 已知缺口（不吞没于绿）

- **below_5x** [cs_rank workspace 3.01]：>=2x but <5x speedup target line → carried to Phase 4/NRR below_5x_note
- **below_5x** [parameter_scan canonical 3.021]：>=2x but <5x speedup target line → carried to Phase 4/NRR below_5x_note
- **below_5x** [rolling_ic canonical 3.143]：>=2x but <5x speedup target line → carried to Phase 4/NRR below_5x_note
- **below_5x** [stock_corr general(N=500) 2.526]：>=2x but <5x speedup target line → carried to Phase 4/NRR below_5x_note
- **below_5x** [stock_corr general(N=2000) 2.265]：>=2x but <5x speedup target line → carried to Phase 4/NRR below_5x_note
- **stock_corr_general_synthetic_panel**：stock_corr general perf exe uses synthetic make_panel (non-representative) → general verdict uses fresh corpus same-panel measure vs frozen gate; synthetic-panel line kept as informative note only

## 复现

    PYTHONIOENCODING=utf-8 python benchmarks/acceptance_v1.py

*生成模型: DeepSeek-V4-Flash (via Claude Code CLI) · acceptance_v1.py v2 (GPT-5.6-Sol 审查 15 条全处置)*
