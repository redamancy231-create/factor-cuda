# factor-cuda — CUDA 因子橫斷面分析加速

> [简体中文](README.md) · [English](README.en.md) · [繁體中文](README.zh-Hant.md)

> GPU 加速的量化因子**橫斷面分析**工具——橫斷面排序、相關性、IC、參數掃描。
> 與 ashare-mcp（資料取得）、ml-quant-trading（因子生產）組成 A 股量化研究免費管線。
> **本專案不做因子計算、不做回測、不做資料取得。**

## 狀態

**PoC ①–④ 均已完成驗證 + Phase 1-4 完成（2026-08）**。
- 操作語義契約已凍結（L0 Spec，CLAUDE.md）；Phase 2 驗收測試套件 126 項全數通過；Phase 3 六項驗收關卡全數 PASS。
- Phase 4 benchmark 端到端 **~3.0×**（vs 同語義最佳免費替代；未達 5× 優勢門檻 → **負結果已登記 NRR-2026-024**）。
- 記憶體模型的三項分塊／串流機制均已通過實測驗證：F-blocking / streaming（輸入串流化）/ N-blocking——F=128 情境由超出預算的 12.6 GiB 降至約 6.9 GiB，符合預算。
- **發布準備中**。

## 功能

GPU 加速橫斷面分析運算子（pybind11 綁定 + 純 Python 後端）：

| 運算子 | 說明 |
|------|------|
| `cross_sectional_rank` | 穩定序數橫斷面排序（float32，CUB radix sort） |
| `parameter_scan` | 四組方向×mask 參數掃描（單次 H2D） |
| `factor_corr` | 因子相關矩陣（F×F，含 Kahan 重算觸發 + F-blocking/streaming） |
| `stock_corr` | 股票相關矩陣（N×N，fast/general 雙路徑 + N-blocking） |
| `rolling_ic` | 滾動 Spearman IC（float64 統一單一路徑） |

配套：GPU 記憶體靜態預算模型（`docs/memory_budget_v1.json`）、workspace 分配快取、corpus parity 驗證。

## 建置

- 環境支援矩陣見 `docs/support_matrix.json`（單一真實來源；實測 CUDA Toolkit 13.3 / VS2026 MSVC 19.51 / Python 3.12.7 / compute capability 8.9，僅宣告單一架構）。
- CMake + Ninja（C++20）；PoC 工具經 `pwsh -NoProfile -File _build_poc3.ps1 <target>` 建置。

## 文件索引

| 文件 | 內容 |
|------|------|
| `PLAN.md` | 方案設計（定位/市場驗證/技術架構/實作階段/風險） |
| `CLAUDE.md` | 專案規範 L0 Spec（操作語義契約/停止條件/成功標準/評估/可重現性/常見陷阱） |
| `CHANGELOG.md` | 變更記錄 |
| `docs/support_matrix.json` | 建置/執行環境支援矩陣 |
| `docs/memory_budget_v1.json` | GPU 記憶體靜態預算模型（含實測驗證） |
| `reviews/` | 獨立審查報告（gitignored，不發布） |

## 競品與對照

- **QuantGplearn**：遺傳規劃因子探勘，Torch GPU 後端——PoC ② 公平基線最強對照候選（可選本機基線，不隨套件提供）
- **equity-factor-lab**：完整因子研究平台，偏誤控制方法的參考

## 相關專案

- [ashare-mcp](https://github.com/redamancy231-create/ashare-mcp) — A 股資料取得
- [ml-quant-trading](https://github.com/redamancy231-create/ml-quant-trading) — 因子生產
- [etf-pattern-match-pybind11](https://github.com/redamancy231-create/etf-pattern-match-pybind11) — 同技術堆疊（pybind11+CMake）的參考專案
- [negative-results-registry](https://github.com/redamancy231-create/negative-results-registry) — 負結果登記（NRR-2026-024）

*非投資建議。*
