# PoC ④ 端到端最小参数扫描流水测量结果（poc4_e2e_v1）

> 日期：2026-08-06（DeepSeek-V4-Flash via Claude Code CLI）；证据封装：`poc4_e2e_v1.json` `evidence`（ISO 时间戳 / git HEAD `49a97f3` / 命令行 / corpus data_sha256）
> 判据（CLAUDE.md 停止条件 ④ / PLAN.md 预注册）：**端到端加速比（含传输/归并，含 (4F,T) IC 堆叠，非单算子）相对最佳免费替代 ≥2× PASS、≥5× 优；<2× STOP**。
> 基线：`best_free_total = min` over 严格白名单 `{numpy, cupy, qg}`（**不含 gpu**；qgplearn 本机未装 → best = min(numpy, cupy)）。同数据同 mask 同语义。
> **GO 范围（审查 2026-08-06 裁定）**：主判据 = **PoC ④ 最小参数扫描流水**（parameter_scan→rolling_ic→factor_corr + IC 归并）；stock_corr 为**独立支路**，不进主判据，其 <2× 值单独诚实报告。
> 测量协议（审查 2026-08-06 闭合）：**三臂对称 reps=3**（中位数 + min/max + 逐样本）；IC 堆叠在**每臂计时区内实际构造**；fresh run。

## 结果摘要（三臂端到端，含 IC 堆叠归并）

| 规模 | GPU 端到端（med） | [min,max] | numpy | cupy | best_free | 加速比 |
|------|-----------|-----------|-------|------|-----------|--------|
| F=4 | 1.510 s | [1.482, 1.596] | 16.20 s | 4.258 s | 4.258 s (cupy) | **2.82×** |
| F=8 | 2.938 s | [2.911, 3.004] | 32.99 s | 8.715 s | 8.715 s (cupy) | **2.97×** |
| **F=12（主口径）** | **4.432 s** | [4.431, 4.463] | 48.89 s | 13.17 s | **13.17 s (cupy)** | **2.97×** |

**VERDICT：PASS（F=12 主口径 2.97× ≥ 2×）**——含 IC 堆叠归并计时后仍 ≥2×，端到端收益成立；但**低于 5× 优线**（recorded，未虚报）。

## 逐算子分解（F=12，GPU vs best-free，含各算子传输）

| 算子 | GPU | best_free (cupy) | 加速比 |
|------|-----|------------------|--------|
| parameter_scan | 1.198 s | 3.220 s | **2.69×** |
| rolling_ic | 2.912 s | 6.023 s | **2.07×** |
| factor_corr | 0.323 s | 3.968 s | **12.29×** |
| ic_stack（归并） | 0.00015 s | 0.00016 s | 1.09× |

> **ic_stack 注**：`np.stack((4F,T))` 是三臂共用的 NumPy 操作（GPU 臂无优势，1.09×≈1），归并成本微乎其微但已实际进入端到端计时（审查②修复项）。
> **factor_corr 表述精确化（审查②）**：12.29× 是**完整实现臂（CUDA kernel + 绑定）对 CuPy 参考实现臂**的差异，不能表述为「CUDA kernel 本身快 12.29×」——CuPy 的 factor_corr 参考实现慢（3.97 s），且 factor_corr 的 masked-GEMM 与两遍中心化属不同算法实现，非语义不公平。

## stock_corr 独立支路（全 T，3 reps 中位数；**不进主判据**）

| N | GPU | best_free (cupy) | 加速比 |
|---|-----|------------------|--------|
| 500 | 27.1 ms | 53.1 ms | **1.96×** |
| 2000 | 260.6 ms | 518.6 ms | **1.99×** |

> stock_corr 支路 <2×（边界弱项，RTX4060 FP64 吞吐瓶颈），按审查②如实呈现：**不据此声称「五 kernel 全部端到端 ≥2×」**；GO 仅覆盖最小参数扫描流水。N=5000 全量仅 GPU-only 可运行性演示（CPU 基线不可行，不作为比较证据）。

## 诚实解读

- **端到端收益成立（F=12 主口径 2.97× ≥ 2×，PASS）**，但**未达 5× 优线**。对称 reps=3 后离散度极小（GPU [4.431, 4.463]s）。
- 加速比主要由 **factor_corr（12.29×，完整实现臂 vs CuPy 参考臂）** 贡献；**parameter_scan 2.69× / rolling_ic 2.07×** 更接近真实边际（排序/IC 类相对 CuPy 向量化）。
- **趋势随 F 扩展持平/近稳**（2.82 / 2.97 / 2.97×，尤其 F=8→12 几乎不变）——**不表述为随因子数显著增长**（审查②修正）。
- best_free 恒为 CuPy（非 numpy）→ **无「numpy 巨大差距垫高」的 trivially-padded PASS**；qgplearn 未装（若纳入 QG 排序/IC 更快，主加速比会略降——诚实记录）。
- 端到端口径含传输/归并：Python wall-clock 覆盖全部 H2D/D2H（GPU 臂每算子经绑定）+ parameter_scan 4 组合并 + **(4F,T) IC 堆叠**；cudaEvent 级 kernel 计时在 PoC ③ perf 程序（C++ 侧）单独报告。
- 与 PoC ② 基线对照：单算子 GPU 均 BEATS QG 门槛（cs_rank ~9ms vs ≤13.8ms、parameter_scan ~37ms vs ≤72.2ms、stock_corr N500 ~5.5ms vs ≤26.8ms）。
- **数据规模事前性**：F={4,8,12} 与全量 1218×5000×12 在脚本（commit `03dbdd2` 起）中固定，可追溯；精确的「早于测量确定」时间线无法从本证据包单独证明（审查②标注）。

## 三态（④ 端到端）

**GO**（收益成立 2.97×，≥2× 判据 PASS；未达 5× 优线）——**范围限定为最小参数扫描流水**；stock_corr 支路 <2× 为独立边界弱项（诚实记录，不影响主判据）。由用户结合 ①语义 / ②公平基线 / ③显存 综合裁决是否进入 Phase 1–4。
