# Corpus Schema v1 — factor-cuda benchmark corpus

> 人类可读 schema（与 `manifest_schema_v1.json` JSON Schema 双件）。依据：`_corpus_design_draft.md`（PoC ② manifest 设计定稿）。

## npz 数组清单（real 与 synth 同一 schema）

| 键 | dtype | shape | 含义 |
|----|-------|-------|------|
| `dates` | `<U10` | (T,) | ISO "YYYY-MM-DD" 交易日 |
| `ids` | `<U10` | (N,) | real=股票代码升序；synth=`SYN%05d` |
| `names` | `<U20` | (F,) | 因子名（factor_corr names 对齐 len==F） |
| `factors` | float32 | (T,N,F) | 因子平面全集（F 最内，C-contiguous） |
| `factor_a` | float32 | (T,N) | = `factors[...,0]` 连续副本（cs_rank/rolling_ic/parameter_scan 输入） |
| `returns` | float32 | (T,N) | 日度简单收益（stock_corr 输入） |
| `price` | float64 | (T,N) | 前复权收盘价（fwd 校验/重派生用） |
| `mask` | bool | (T,N) | 全操作共享，True=可交易（唯一权威） |
| `forward_returns` | float64 | (T,N) | h=5/lag=1 前向收益标签（rolling_ic 输入） |
| `h` | int64 | (1,) | 5 |
| `lag` | int64 | (1,) | 1 |
| `schema_version` | str | () | "v1" |
| `generator_version` | str | () | 生成器版本 |

## Manifest JSON 字段

见 `manifest_schema_v1.json`（JSON Schema draft-07，机器可解析）。

## 关键约定

- **dtype 主线**：factors/factor_a/returns float32；mask bool；forward_returns/price float64
- **mask 语义**：True=可交易；被排除格存储值可为 0.0/有限原值/NaN，严禁由数值反推 mask
- **forward_returns 规则**：h=5/lag=1，入场=t+1 收盘、出场=t+6 收盘，末 6 行全 NaN
- **基准行范围**：半开区间 `[W, T-(h+lag)) = [21, T-6)`（t≤T-7，不含必然 NaN 末行）
- **hash**：`data_sha256` = npz 文件字节 SHA-256；`array_sha256` = 数组内容级锚
- **环境指纹**：`python-3.12.7_numpy-2.4.4`
- **种子**：唯一真源 `seeds.json`，manifest 只引用
