# PoC ③ 五操作三口径显存校准（v1）

> 生成：2026-08-08 · benchmarks/poc3_calibration_v1.py (DeepSeek-V4-Flash via Claude Code CLI, 2026-08-05)
> 协议：poc34_workload_estimate.md §3.3 校准纪律

## 结论

**11 例 × 三口径（理论公式 / tracker HWM / 驱动采样）全部 PASS。**
- 偏差口径1 公式 vs HWM：全部 `delta_formula == 0`（理论 + CUB temp == HWM 精确，分配确定性验证）
- 偏差口径2 HWM vs 驱动：全部 `overhead ∈ [0, 64 MiB]`（驱动/分配器开销）
- 无泄漏（final_live == 0）、strict tracker 无 unknown free

- **预算交叉核对**：全部 11 例 HWM ≤ 可用预算 7676 MiB：✅ 全部满足，最大 HWM 2381 MiB、最大 driver 开销 20.99 MiB。

## 校准明细（MiB；delta_formula=HWM−(理论+temp)，overhead=driver−HWM）

| op | T×N×F | 理论(无temp) | CUB temp | HWM | 驱动峰值 | Δ公式 | overhead | 判定 |
|---|---|---|---|---|---|---|---|---|
| cs_rank | 1218×5000×0 | 151.0 | 0.00 | 151.0 | 172.0 | +0 B | +20.99 | ✅ |
| cs_rank | 1218×10000×0 | 302.0 | 0.00 | 302.0 | 314.0 | +0 B | +11.98 | ✅ |
| parameter_scan | 1218×5000×0 | 151.0 | 0.00 | 151.0 | 158.0 | +0 B | +6.99 | ✅ |
| parameter_scan | 1218×10000×0 | 302.0 | 0.00 | 302.0 | 314.0 | +0 B | +11.98 | ✅ |
| rolling_ic | 1218×5000×0 | 482.1 | 0.00 | 482.1 | 500.0 | +0 B | +17.89 | ✅ |
| rolling_ic | 1218×10000×0 | 964.2 | 0.00 | 964.2 | 982.0 | +0 B | +17.83 | ✅ |
| factor_corr | 1218×5000×12 | 1190.6 | 0.00 | 1190.6 | 1194.0 | +0 B | +3.37 | ✅ |
| factor_corr | 1218×10000×12 | 2381.2 | 0.00 | 2381.2 | 2386.0 | +0 B | +4.75 | ✅ |
| stock_corr | 1218×2000×0 | 119.2 | 0.00 | 119.2 | 130.0 | +0 B | +10.78 | ✅ |
| stock_corr | 1218×5000×0 | 526.9 | 0.00 | 526.9 | 536.0 | +0 B | +9.07 | ✅ |
| stock_corr | 1218×10000×0 | 1816.8 | 0.00 | 1816.8 | 1824.0 | +0 B | +7.19 | ✅ |

## 说明

- 理论公式 = 各 kernel 分配布局的 align256 逐 buffer 和（源码 AllocOrTrack 链推导）。
- CUB temp（cs_rank/parameter_scan/rolling_ic）为唯一理论盲点，取 tracker 单次调用最大值
  （校准脚本初版曾跨 rep 求和 temp 导致 -512/-1024 假 FAIL，已修复为 max 单次）。
- stock_corr N=10000 输出矩阵 O(N²)=800 MiB 不可约，实测 HWM 1816.8 MiB 仍远低于预算
  ——O(N²) 输出不可约的下界主张得到实测确认。
- workspace_v1.json 的 `theoretical_workspace` 锚点是计划中 **per-pair 设计**的全 pair 常驻
  归约 workspace 上界；实现 kernel 为 tile 常驻，两者是不同设计口径，不做逐 buffer 直接比较。
  本校准的预算交叉核对（全部 HWM ≤ 7676 MiB 可用预算）即为模型预算主张的实测验证。

## 复现

    cd factor-cuda
    "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build --target poc3_calibration
    build\poc3_calibration.exe             # 直接运行；或
    PYTHONIOENCODING=utf-8 python benchmarks/poc3_calibration_v1.py

*生成模型: benchmarks/poc3_calibration_v1.py (DeepSeek-V4-Flash via Claude Code CLI, 2026-08-05)*
