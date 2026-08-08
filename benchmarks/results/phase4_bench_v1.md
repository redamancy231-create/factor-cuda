# factor-cuda Phase 4 benchmark (phase4_bench_v1)

- git HEAD: `a4064c60f15a23bac5d851d8be7e05d9a1fd261d`
- date: 2026-08-07T19:15:23+08:00
- env: NVIDIA GeForce RTX 4060 Laptop GPU | driver 610.88 | python 3.12.7 numpy 2.4.4 cupy 14.1.1
- corpora: synth_v1 sha256=0A7D7AB5A6AD1F7E... | real_v1 sha256=41BB9EF49DD4ADF3...

> **Why corpus_real_v1 is not a perf corpus**: N=93 is far below the scale at which GPU launch/transfer overhead amortizes; its timing (cold ~0.23s / warm ~0.04s) is dominated by fixed costs. It serves ONLY as a correctness + bitwise-determinism anchor on real A-share data. Real-corpus mask (since 2026-08-07): mask = isfinite(price) & ~halted(volume==0); halt days keep fill prices but mask=False, and forward_returns are NaN on entry/exit-halt windows (§2.5). **Phase 4 evidence was re-run FRESH on 2026-08-07 against this halting corpus (real_v1 sha256 41BB9EF4); the G4/parity leg is therefore CURRENT (the earlier 2026-08-06 evidence, recorded hash CF7497 mask all-True, was stale and is superseded).**

## G1-G3 reproducibility gates

- `corpus_synth_v1`: verified=True T x N x F = 1218x5000x12 | determinism bitwise_equal=True (run1 BE1A44943F2F... / run2 BE1A44943F2F...)
- `corpus_real_v1`: verified=True T x N x F = 1212x93x3 | determinism bitwise_equal=True (run1 A7F0C4FF0C38... / run2 A7F0C4FF0C38...)
- determinism gate: PASS

## G4 parity on real corpus (fc vs numpy oracle)

| check | result |
|---|---|
| rank_bitwise_equal | True |
| rank_max_abs_dr | 0.0 |
| rolling_ic_max_abs_dr | 0.0 |
| factor_corr_max_abs_dr | 1.1102230246251565e-16 |
| stock_corr_max_abs_dr | 4.440892098500626e-16 |
| gate | True |

## G5a single-op perf on corpus_synth_v1 (median + bootstrap 95% CI)

| op | panel | gpu ms | best-free ms | speedup | grade | CI width |
|---|---|---:|---:|---:|---|---:|
| cs_rank | synth-full | 34.7 | 51.4 | 1.48x | gpu-timing-wide | 7.6% |
| parameter_scan(G=4) | synth-full | 87.2 | 204.2 | 2.34x | gpu-timing-precise | 2.3% |
| rolling_ic | synth-full | 61.3 | 124.1 | 2.02x | gpu-timing-precise | 3.3% |
| factor_corr | synth-full | 325.2 | 4256.1 | 13.09x | gpu-timing-precise | 0.9% |
| stock_corr general(N=500) | synth-prefix-500 | 24.9 | 84.2 | 3.38x | gpu-timing-precise | 4.2% |
| stock_corr general(N=2000) | synth-prefix-2000 | 256.1 | 590.8 | 2.31x | gpu-timing-precise | 0.6% |

- best-free = min(numpy 2.4.4, cupy 14.1.1); qgplearn NOT installed (missing baseline arm -> reproducibility level 'partially-reproducible', schema-valid).

> **CROSS-SESSION VARIANCE (review fix, mandatory read)**: single-op medians on this RTX 4060 Laptop GPU vary substantially across sessions (cross-session variance of the speedup ratio up to ~2x observed across this project's runs; the CI widths above are WITHIN-BLOCK precision only). Per-op boundary claims (e.g. 'cs_rank <2x') are NOT cross-session stable and must not be read as decision-grade. The PREREGISTERED DECISION rests on the E2E F=12 criterion (below), which is stable across rounds (delta 3.1%).

### Layer disclosure (kernel-resident vs binding incl. transfer)

The single-op table above is the **binding level (includes H2D transfer + D2H)**, consistent with the e2e layer and what a Python user actually pays. The acceptance perf numbers were **kernel-resident** (GPU compute only, transfers excluded). The two columns come from DIFFERENT sessions with different baselines, so they are a disclosure of layer/session difference, NOT a causal claim that kernel-resident overstates user-facing speedup (review F12: this session's binding numbers are not consistently lower, e.g. cs_rank 3.11x binding vs 3.01x kernel; a causal transfer-cost claim would require a same-session, same-input, layer-only paired experiment).

| op | kernel-resident (acceptance) | binding incl. transfer (this run) |
|---|---|---|
| cs_rank | 3.01x (9.3ms) | 1.48x (34.7ms) **<2x minimum at product level** |
| parameter_scan(G=4) | 3.02x (36.6ms) | 2.34x (87.2ms) |
| rolling_ic | 3.14x (49.1ms) | 2.02x (61.3ms) |
| factor_corr | (219.6ms kernel) | 13.09x (325.2ms) |
| stock_corr general(N=500) | 2.53x | 3.38x (24.9ms) |
| stock_corr general(N=2000) | 2.27x | 2.31x (256.1ms) |

**Adapter overhead (product Python layer vs raw binding):**
- cs_rank: adapter 32.9ms vs binding 31.3ms -> +5%
- factor_corr: adapter 332.3ms vs binding 327.9ms -> +1%
- note: factor_corr adapter overhead is dominated by the contract f32->f64 Python upcast (adapter converts in numpy before the binding's faster internal forcecast upcast)


## G5b e2e F=12 (pipeline only; stock_corr EXCLUDED from verdict)

- committed (poc4_e2e_v1.json @ 7f0ee956): speedup 3.035x | per-op {"parameter_scan": 2.66, "rolling_ic": 2.03, "factor_corr": 13.14, "ic_stack": 0.72}
- fresh round (6 samples/arm, arm order rotated per sample, thermal recorded): gpu 4.57s (CI 4.54-4.61) / numpy 51.69s / cupy 13.43s -> speedup 2.940x (CI 2.911-2.964; best-free = cupy)
- thermal: 55.0C -> 54.0C (clocks 2250.0MHz)
- absolute drift (committed @ 7f0ee956 -> fresh): gpu 4.51s -> 4.57s (+1%) -- ratio is stable via common-mode cancellation; absolute timings do NOT overlap sessions
- absolute drift (committed @ 7f0ee956 -> fresh): numpy 50.68s -> 51.69s (+2%) -- ratio is stable via common-mode cancellation; absolute timings do NOT overlap sessions
- absolute drift (committed @ 7f0ee956 -> fresh): cupy 13.68s -> 13.43s (-2%) -- ratio is stable via common-mode cancellation; absolute timings do NOT overlap sessions
- cross-run ratio stability: delta 3.1% (<= 10%, ratio-only, weak with n=2) -> STABLE; speedup range 2.940-3.035x
- producer-commit consistency (review F04): DIFFERENT commits
  - note: committed reference produced at git 7f0ee956; fresh round at df547f28 -- the cross-run delta spans producer versions and is reported as directional, not a same-code stability proof (review F04).

**Component-level negatives (surfaced, not buried):**
- ic_stack speedup 0.719x is worse than baseline

## Verdict (preregistered criterion)

- E2E F=12 speedup range 2.940-3.035x vs target 5.0x / min-acceptable 2.0x
- min_met=True, target_met=False -> verdict `PASS-partial`
- **below_5x target line not met -> NRR-2026-024**
- verdict_scope: pipeline only (parameter_scan -> rolling_ic -> factor_corr + IC merge); stock_corr EXCLUDED

## Provisos (honesty)

- E2E verdict scope EXCLUDES stock_corr (poc4_e2e_v1.py verdict_scope); the exclusion is repeated in every verdict/NRR context and fast-path positives never enter the main judgement.
- All performance numbers come ONLY from corpus_synth_v1; corpus_real_v1 (N=93) is a correctness/determinism anchor (speedup_computed=false; launch/transfer dominated).
- qgplearn is NOT installed -> best_free = min(numpy, cupy); one planned baseline arm missing; reproducibility level = 'partially-reproducible' (schema-valid; not 'fully-reproducible').
- The factor_corr >=5x PASS (12-15x) is BINDING-level (pybind, incl. transfer). At the PRODUCT adapter level the contract-mandated f32->f64 Python upcast adds measurable overhead (adapter block records the adapter-vs-raw-binding delta, +178%~+266% across sessions). No product-level factor_corr speedup is claimed as a precise number: the adapter overhead and the best-free baseline are measured in different blocks, so combining them into a 'product-layer 3.98x' would be a mechanical combination without a clean structured basis (review F10). The 'kernels reach 5x' positive control holds at binding level only.
- stock_corr fast path is an all-valid degenerate-mask synthetic-panel special case; it is never the positive control for 'kernels reach 5x'.
- The RTX 4060 FP64 ceiling (FP64 ~= FP32/64) is a CANDIDATE mechanism (untested as a confirmatory experiment) for the float64 aggregation ops (rolling_ic, factor_corr reductions, stock_corr general); the float32 ops below 5x (cs_rank, parameter_scan) are attributed to launch/transfer/occupancy overhead -- 'FP64 ceiling' must not sweep float32 gaps.
- DECISION STATISTIC (review fix): both min_met and target_met use the WORST-CASE min-of-run-medians (conservative bound), symmetric across gates. Single-op = median of 20 samples in one run + bootstrap CI (within-block precision only). E2E = two run medians; the range and the min (worst-case) are the decision inputs. min-of-medians is used for decisions only as this worst-case bound, never to inflate a result.
- CROSS-SESSION DRIFT (review fix): decision CIs are WITHIN-BLOCK precision only. Same-machine absolute timings drift across sessions (the exact committed-vs-fresh per-arm drift % is rendered in the e2e section from the structured fields -- not hardcoded here): the speedup RATIO is stable via common-mode cancellation (cross-run delta is small, rendered in the e2e section) but absolute timings do NOT overlap across sessions. No 'timing CI-overlap' claim is made.
- E2E fresh round records per-block nvidia-smi thermal and ROTATES the arm order per sample (no arm always runs warmest); a GPU outlier (5.98s vs 7.84s median) is reported in raw samples.
- Alternative explanations (higher-FP64 GPU, FP32 aggregation path, kernel tuning) are labeled UNTESTED HYPOTHESES, not feasibility claims.
- Component-level negatives (any e2e sub-op < 1x, e.g. ic_stack) are surfaced explicitly, never buried inside the aggregate >=2x PASS.
- Evidence self-hash (G6) is a REAL gate: it fails if any hashed source file is missing or dirty in git; raw single-op samples are persisted in the gate JSON for auditability.
- Cross-machine reproducibility scope: same machine + same commit + same corpus -> bitwise output determinism; TIMING is NOT transferable across machines or sessions (perf artifact carries GPU identity + thermal state).

## Reproduction

```bash
git rev-parse HEAD && git status --porcelain  # must be clean
PYTHONIOENCODING=utf-8 python benchmarks/phase4_bench_v1.py --fresh  # --fresh REQUIRED for a publishable reproduction (review F02)
```

- A plain `python benchmarks/phase4_bench_v1.py` (no --fresh) reuses persisted gates from `runs/phase4_gates/` for fast iteration; it is NOT a fresh measurement and must not be used as publication evidence. Gate artifacts are bound to the producer commit + corpus hash at save time.
