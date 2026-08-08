# factor-cuda factor_corr streaming(项②) 设备 HWM 实测(v1)

> 生成:2026-08-08 · benchmarks/factor_stream_hwm_v1.py (DeepSeek-V4-Flash via Claude Code CLI, 2026-08-08)
> 判据(两层,替代 5-op 校准的 delta_formula==0——streaming 无 MemTracker):
> **运行期 fits**:driver_peak ≤ 7676 MiB 可用预算 且 margin>8 MiB(WDDM margin≈0 非 fit);driver_peak−模型 delta 未独立归因(cudaMemGetInfo MiB 量化 ±1 MiB)
> **显存耗尽**:margin≈0 且 rc=0(与 WDDM 超分配/共享内存回退一致,机制未隔离验证;非可用 fit——仅 production 对照合法)

## 结论

**4 例 FBLC 记录,3 例判定为可用 fit。**
- **streaming F=128 (N=5000, tt=4096) 实测 driver_peak 7079.75 MiB**(模型 6863.6 MiB,delta +216.15 MiB)→ **fits**。**余量双口径**:预算余量 596.25 MiB(相对 7676 名义预算)/ **物理余量 26.25 MiB**(相对运行期 free_before 7106.0 MiB)——free_before 7106 vs 预算 7676 存在 570 MiB 缺口,物理余量才是本机真实空间 → **streaming(项②) 列闭合为实测(本机 RTX 4060, fb=7106 MiB)**
- **对比口径(诚实标注)**:模型 current 12.6 GiB(未实测,calibration-校) vs 实测 streaming 7079.75 MiB——5.7 GiB 节省为**模型-实测**比较(本机未物化 12.6 GiB);**measured-vs-measured**:production 对照 7106.0 MiB(vram-exhausted)→ streaming 7079.75 MiB,物理差 26.25 MiB
- **F=12 锚点**:N=5000 636.0 MiB(模型 634.98 MiB,delta 1.02 MiB)、N=10000 1270.0 MiB(模型 1268.03 MiB,delta 1.97 MiB)——模型-实现一致性校准
- **production F=128 对照** 7106.0 MiB(margin≈0,**vram-exhausted** 非 fit)——确认模型 current 峰值 12.6 GiB 超预算声明真实
- **测量披露**:本 run 存在非 MiB 整数倍 driver_peak(如 F128 7079.75 MiB)——free 内存量化粒度随驱动/时刻变化,非恒定 1 MiB;F=128 delta +216.15 MiB 远大于 F=12 锚点(delta ≤2 MiB),与 fblock B32 低余量 WDDM 运行期波动同量级,未独立归因;本机 F=128 物理余量 26.3 MiB(相对 free_before 7106.0 MiB,接近耗尽档,delta 含分配粒度/驱动开销)

## 环境

- GPU:NVIDIA GeForce RTX 4060 Laptop GPU total 8187.5 MiB
- 运行期 free_before ≈ 7106.0 MiB;模型预算口径 7676 MiB(8188−512)

## 明细

| kind | T×N×F | 规模 | driver(MiB) | 模型(MiB) | Δ(MiB) | margin(MiB) | rc | 判定 | 注 |
|---|---|---|---|---|---|---|---|---|---|
| stream | 1218×5000×12 | tt=4096 (max_transpose_rows) | 636.00 | 634.98 | 1.02 | 6470.0 | 0 | ✅ | fit: 预算余量 7040.0 MiB, 物理余量 6470.0 MiB |
| stream | 1218×10000×12 | tt=4096 (max_transpose_rows) | 1270.00 | 1268.03 | 1.97 | 5836.0 | 0 | ✅ | fit: 预算余量 6406.0 MiB, 物理余量 5836.0 MiB |
| stream | 1218×5000×128 | tt=4096 (max_transpose_rows) | 7079.75 | 6863.6 | 216.15 | 26.25 | 0 | ✅ | fit: 预算余量 596.3 MiB, 物理余量 26.3 MiB |
| production | 1218×5000×128 | production (non-streamed) | 7106.00 | 12645.61 | — | 0.0 | 0 | ❌ | 物理显存耗尽(margin≈0, rc==0);与 WDDM 超分配/共享内存回退一致,机制未隔离验证;模型峰值超设备物理显存 |

## 闭合范围(诚实边界)

- 实测闭合**仅 streaming(项②) 列**(F=128 N=5000 锚定);`current_peak`(production)仍为模型预测(calibration 已校)
- **streaming 路径 device 峰值随 F 线性**(d_Xt+d_valid+d_pp 全驻留),无"峰值 N 无关"简化——N=10000 F=12 锚点实测校准,N=10000 F=128 未实测(host 面板 11.9 GiB 不可行)
- streaming 无 MemTracker → 无 delta_formula==0;driver_peak−模型 delta 未独立归因(cudaMemGetInfo MiB 量化 ±1 MiB;F=128 +123 MiB 与 fblock B32 低余量 WDDM 波动同量级)
- 逐 sub-chunk host 上传/range 转置为 host 成本(HWM 只测 device VRAM)
- d_pp 延续状态(F=128 ~161 MiB)是 streaming/chunked 路径固有成本,计入模型

## 复现

    cmake --build build --target poc3_factor_corr_selfcheck
    build\poc3_factor_corr_selfcheck.exe --hwm-stream   # 或
    PYTHONIOENCODING=utf-8 python benchmarks/factor_stream_hwm_v1.py

*生成模型:benchmarks/factor_stream_hwm_v1.py (DeepSeek-V4-Flash via Claude Code CLI, 2026-08-08)*
