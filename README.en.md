# factor-cuda — CUDA-Accelerated Cross-Sectional Factor Analysis

> [简体中文](README.md) · [English](README.en.md) · [繁體中文](README.zh-Hant.md)

> GPU-accelerated cross-sectional analysis for quantitative factors — cross-sectional ranking, correlation, IC, and parameter scans.
> Together with ashare-mcp (data acquisition) and ml-quant-trading (factor production), it forms a free A-share quantitative-research pipeline.
> **This project does NOT compute factors, backtest, or fetch data.**

## Status

**Released v1.0.1 (2026-08-08)** — PoC items ①–④ closed out; Phases 1–4 complete.
- The operation-semantics contract is frozen (L0 Spec, CLAUDE.md); all 126 Phase 2 acceptance tests passed locally on GPU / 3 skipped (GitHub Actions runs the CPU/contract arm without GPU: 75 passed); all six Phase 3 acceptance gates are marked PASS.
- Phase 4 benchmark end-to-end **~3.0×** (vs the best same-semantics free alternative; below the pre-registered 5× superiority threshold → **registered as negative result NRR-2026-024**).
- The memory-model "divisible" trilogy (F-blocking / streaming / N-blocking) is validated as a PoC / memory-feasibility path: factor F=128 model peak 12,645.61 MiB (over budget) → streaming measured 7,079.75 MiB (fits).

## Features

GPU-accelerated cross-sectional operators (pybind11 bindings + pure-Python backends):

| Operator | Description |
|----------|-------------|
| `cross_sectional_rank` | stable ordinal cross-sectional ranking (float32, CUB radix sort) |
| `parameter_scan` | parameter scan over four direction × mask groups (single H2D transfer) |
| `factor_corr` | factor correlation matrix (F×F; a trigger for Kahan recomputation + F-blocking/streaming) |
| `stock_corr` | stock correlation matrix (N×N; separate fast and general paths + N-blocking) |
| `rolling_ic` | rolling Spearman IC (a single unified float64 path) |

Also: a static GPU-memory budget model (`docs/memory_budget_v1.json`), workspace allocation caching, corpus parity verification.

## Architecture

```mermaid
graph LR
    A["Factor panel (T,N,F)"] --> B["Adapter fc.*<br/>dtype / mask / device normalization"]
    B --> C["pybind11 bindings"]
    C --> D["CUDA kernel pipeline"]
    D --> E["cross_sectional_rank<br/>· parameter_scan"]
    D --> F["factor_corr<br/>· stock_corr"]
    D --> G["rolling_ic"]
    E --> H["Cross-sectional rank"]
    F --> I["Correlation matrix"]
    G --> J["Spearman IC"]
    H & I & J --> K["Cross-sectional results"]
```

## Build

- Environment support matrix in `docs/support_matrix.json` (single source of truth; tested CUDA Toolkit 13.3 / VS2026 MSVC 19.51 / Python 3.12.7 / compute capability 8.9, single-arch declaration).
- CMake + Ninja (C++20); PoC tools built via `pwsh -NoProfile -File _build_poc3.ps1 <target>`.

## Documentation Index

| File | Content |
|------|---------|
| `PLAN.md` | design (positioning / market validation / technical architecture / implementation phases / risks) |
| `CLAUDE.md` | project spec L0 (operation-semantics contract / stop conditions / success criteria / evaluation / reproducibility / pitfalls) |
| `CHANGELOG.md` | change log |
| `docs/support_matrix.json` | build/runtime environment support matrix |
| `docs/memory_budget_v1.json` | static GPU-memory budget model (validated empirically) |
| `reviews/` | independent review reports (gitignored, not published) |

## Competitors & Baselines

- **QuantGplearn**: genetic-programming factor mining, Torch GPU backend — the strongest candidate for the fair baseline in PoC ② (optional local baseline, not bundled)
- **equity-factor-lab**: a full factor research platform, a reference for bias-control practices

## Related Projects

- [ashare-mcp](https://github.com/redamancy231-create/ashare-mcp) — A-share data acquisition
- [ml-quant-trading](https://github.com/redamancy231-create/ml-quant-trading) — factor production
- [etf-pattern-match-pybind11](https://github.com/redamancy231-create/etf-pattern-match-pybind11) — a reference project using the same stack (pybind11 + CMake)
- [negative-results-registry](https://github.com/redamancy231-create/negative-results-registry) — negative-result registry (NRR-2026-024)
- [redamancy231-create](https://github.com/redamancy231-create/redamancy231-create) — personal profile / full project index

*Not investment advice.*
