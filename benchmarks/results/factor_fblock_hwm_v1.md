# factor-cuda F=128 fblock 设备 HWM 实测(v1)

> 生成:2026-08-08 · benchmarks/factor_fblock_hwm_v1.py (DeepSeek-V4-Flash via Claude Code CLI, 2026-08-07)
> 判据(两层,替代 5-op 校准的 delta_formula==0——fblock 无 MemTracker):
> **分配链验证**(probe `scratch/alloc_probe.cu`):实测 drop 6328 MiB vs 模型 6325.11 MiB,delta +3 MiB → 模型分配链精确
> **运行期 fits**:driver_peak ≤ 7676 MiB 可用预算;driver_peak−分配链为运行期 delta(低 free 余量时现象已验证增大,WDDM 机制未隔离验证)
> **显存耗尽**:margin≈0 且 rc=0(与 WDDM 超分配/共享内存回退一致,机制未隔离验证;非可用 fit)
> **测量披露**(synced to stock external F1/contract, F2-hardened):free-memory sampling granularity is driver/time-dependent on this WDDM: 7/7 fb snapshots and 6/7 driver_peak values are exact MiB multiples, block=32 (6588.105 MiB) is NOT a MiB multiple -- so small deltas carry sampling-granularity error and are not precisely attributable to driver overhead

## 结论

**7 例 FBLC 记录,5 例判定为可用 fit。**
- F=128 (N=5000) block=32 实测 driver_peak 6588.11 MiB(**fits**:预算口径余量 1087.89 MiB,实际运行余量 517.89 MiB)→ **fblock(项③) 峰值闭合为实测**;分配链模型 6325.11 MiB(probe 6328,delta 3 MiB);总 delta +263.0 MiB(driver−model)。**运行期 delta 现象已验证**(2026-08-07 隔离验证:同路径低余量对照 block=8 +pad 压余量 delta 从 +4.4 跳 +223;纯分配恒 ~+3-5),**WDDM 具体机制未隔离验证**(reviews/b32_delta_iso_verification_2026-08-07.md)
- block=64 / production F=128 实测**显存耗尽(margin≈0),非 OOM**——与 WDDM 超分配/共享内存回退一致(具体机制未隔离验证,review F4);模型 12645 MiB 超物理显存被实证;两者**不可用(非 fit)**
- F=12 block=6 交叉验证锚点:driver_peak ≈ 模型 ≈ 校准 current 1191/2381(fblock 模型在 block≤F/2 区间钉死)

## 环境

- GPU:NVIDIA GeForce RTX 4060 Laptop GPU total 8187.5 MiB
- 运行期 free_before ≈ 7106.0 MiB(total 8188,实际占用 ~1082 MiB);模型预算口径 7676 MiB(8188−512)→ 实际可用比口径少 570 MiB,block=32 实际余量 517.89 MiB
- host staging(**估算,非实测**——仅 device VRAM 被 cudaMemGetInfo 采样,review h4):面板 F3 全量 5947.3 MiB + 逐 tile 临时最大 2973.6 MiB(M6 注记)

## 明细

| kind | T×N | 规模 | driver(MiB) | 模型(MiB) | Δ(MiB) | margin(MiB) | rc | 判定 | 注 |
|---|---|---|---|---|---|---|---|---|---|
| fblock | 1218×5000 | 12 block=6 | 1194.0 | 1190.63 | 3.37 | 5912.0 | 0 | ✅ | fit 余量 6482.0 MiB |
| fblock | 1218×10000 | 12 block=6 | 2386.0 | 2381.24 | 4.76 | 4720.0 | 0 | ✅ | fit 余量 5290.0 MiB |
| fblock | 1218×5000 | 128 block=8 | 1590.0 | 1585.57 | 4.43 | 5516.0 | 0 | ✅ | fit 余量 6086.0 MiB |
| fblock | 1218×5000 | 128 block=16 | 3170.0 | 3165.38 | 4.62 | 3936.0 | 0 | ✅ | fit 余量 4506.0 MiB |
| fblock | 1218×5000 | 128 block=32 | 6588.1 | 6325.11 | 263.0 | 517.89 | 0 | ✅ | fit 余量 1087.9 MiB |
| fblock | 1218×5000 | 128 block=64 | 7106.0 | 12645.05 | — | 0.0 | 0 | ❌ | 物理显存耗尽(margin≈0, rc==0);与 WDDM 超分配/共享内存回退一致,具体机制未隔离验证(review F4);模型峰值超设备物理显存 |
| production | 1218×5000 | 128 (production) | 7106.0 | 12645.61 | — | 0.0 | 0 | ❌ | 物理显存耗尽(margin≈0, rc==0);与 WDDM 超分配/共享内存回退一致,具体机制未隔离验证(review F4);模型峰值超设备物理显存 |

## 闭合范围(诚实边界)

- 实测闭合**仅 fblock(项③)** 路径;`current_peak`(production)仍为模型预测(12645 MiB,已实测显存耗尽确认超预算)、`streaming_peak`(项②)无实现仍为预测
- fblock 无 MemTracker → 无 delta_formula==0;分配链用 probe 单独验证(delta 3 MiB),运行期 driver_peak 用预算判 fit
- block=64/production 报"显存耗尽 margin≈0(与 WDDM 超分配/共享内存回退一致,机制未隔离验证)"而非"12.6 GiB 实测";模型峰值 12645 MiB 为设备不可达的预算断言
- block=1 场景省略(8256-tile host 循环 ~30min,无决策价值;block=8/16/32 递减曲线已覆盖)

## 复现

    cmake --build build --target poc3_factor_corr_selfcheck
    build\poc3_factor_corr_selfcheck.exe --hwm-f128   # 或
    PYTHONIOENCODING=utf-8 python benchmarks/factor_fblock_hwm_v1.py

*生成模型:benchmarks/factor_fblock_hwm_v1.py (DeepSeek-V4-Flash via Claude Code CLI, 2026-08-07)*
