# benchmark_corpus — factor-cuda 基准语料

> PoC ② 公平基线 / 性能基准的版本锁定数据源。依据：CLAUDE.md S9 可复现性 + `_corpus_design_draft.md`（manifest 设计定稿）。

## 目录结构与提交范围

```
benchmark_corpus/
├── generate_corpus_v1.py           # 主生成器（real 摄入 + synth 确定性生成 + smoke/压力变体）
├── compute_corpus_stats_v1.py      # 分布统计（manifest.stats）
├── verify_corpus_v1.py             # 校验脚本（设计 §4.4 断言清单）
├── corpus_loader_v1.py             # 唯一运行时读路径（data_sha256 校验 + 子集派生）
├── seeds.json                      # 种子预登记单一真源
├── manifest_schema_v1.json         # JSON Schema draft-07
├── CORPUS_SCHEMA.md                # 人类可读 schema（双件）
├── README.md                       # 本文件
├── corpus_real_v1.npz              # 完整 real（.gitignore，本地复现，data_sha256 锚定）
├── corpus_real_v1.manifest.json    # real manifest（提交）
├── corpus_synth_v1.npz             # 完整 synth（.gitignore，确定性重生成）
├── corpus_synth_v1.manifest.json   # synth manifest（提交）
├── corpus_synth_smoke_v1.npz       # smoke（T=40,N=100,F=4，提交）
├── parity_anchors_v1.npz           # contract parity 对抗语料（小，提交）
└── parity_anchors_v1.manifest.json # 锚点 manifest（提交）
```

**提交范围**：脚本/manifest/schema/smoke/anchors 提交；`corpus_real_v1.npz`、`corpus_synth_v1.npz` 及 `--T/--N/--F` 压力变体 `.gitignore`（由生成脚本本地复现 + data_sha256 锚定，满足 S9「corpus manifest 在 PoC ② 前提交」）。

## 快速开始

```bash
# 生成 smoke（秒级，CI/验证用；--variant 显式身份路由）
python benchmark_corpus/generate_corpus_v1.py --mode synth --variant smoke --out-dir benchmark_corpus

# 生成 canonical（1218×5000×12，合成，确定性；尺寸不符拒绝）
python benchmark_corpus/generate_corpus_v1.py --mode synth --variant canonical

# 生成 sweep 变体（F'=30 等，names=factor_0..）
python benchmark_corpus/generate_corpus_v1.py --mode synth --variant sweep --T 100 --N 500 --F 30 --out-dir benchmark_corpus

# 校验（含 schema + --regenerate 位级复现）
python benchmark_corpus/verify_corpus_v1.py --npz benchmark_corpus/corpus_synth_smoke_v1.npz \
    --manifest benchmark_corpus/corpus_synth_smoke_v1.manifest.json --regenerate

# 加载（基准唯一读入口）
python -c "import sys; sys.path.insert(0,'benchmark_corpus'); from corpus_loader_v1 import load; d,m=load('corpus_synth_smoke_v1')"
```

## 版本与变更规则

- corpus v1 为 PoC ② 唯一冻结集；任何内容/协议/脚本/种子变更 → 新版本递增（v1→v2…），旧版本保留
- 变更登记 CHANGELOG.md；`protocol_version` 随 schema/loader 变更独立递增
- 种子变更=新 corpus 版本；`--seed` 覆写必须写入 manifest.seeds.override，否则视为生成错误
- 生成环境必须与 oracle 锁定环境一致（Python 3.12.7 + NumPy 2.4.4），否则数据无效

## 真实 vs 合成

- **real（已实现，2026-08-06 摄入；2026-08-07 停牌标记）**：`corpus_real_v1` = 93 只沪深 300 × 1212 交易日（2020-2024）baostock 前复权日线（`--mode real` / `fetch_real_corpus_v1.py`）。**mask = isfinite(price) & ~halted(volume==0)**（停牌日 price 保留填价但 mask=False，锻炼"mask=False 但有限"路径；实测停牌率 0.15%，167 cells/25 股）；**forward_returns 对入场/出场停牌窗口置 NaN**（§2.5，防填价伪收益泄漏）。npz 本地（data_sha256 锚定），manifest 提交。**停牌率口径**：0.15% 为 real 样本经验值（沪深 300 大盘股）；synth 校准目标 1.2%±0.2pp 是**压力基准**（小盘/极端停牌），非 real 经验校准（审查 F9）。**可复现性**：npz 以 data_sha256 冻结；重新 fetch 可能因成分调整/数据修订漂移（`query_hs300_stocks` 取当前成分），accepted_ids 落盘审计（审查 F4/F7）。
- **合成（结构保持）**：T/N/F 参数化性能分层曲线；复现 A 股截面分布/因子相关/mask 稀疏/tie/停牌率；种子预登记（SHA-256 派生），确定性重生成。**PoC ② 公平基线当前以 synthetic canonical 为事实源**。
- **parity**：只发生在语义锚点 corpus 与合成固定抽样；边界输入归 parity_anchors。

## 许可边界

原始数据若因许可不入库，按 S9 记录来源/获取日期/hash 与派生过程；合成 corpus 完全自主可复现。

*非投资建议。*
