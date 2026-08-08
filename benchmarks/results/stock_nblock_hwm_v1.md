# factor-cuda stock_corr N-blocking 设备 HWM 实测(v1)

> 生成:2026-08-08 · benchmarks/stock_nblock_hwm_v1.py (DeepSeek-V4-Flash via Claude Code CLI, 2026-08-08)
> 判据(两层,替代 5-op 校准的 delta_formula==0——nblock 无 MemTracker):
> **运行期 fits**:driver_peak ≤ 7676 MiB 可用预算;driver_peak−模型 delta 未独立归因(cudaMemGetInfo MiB 量化 ±1 MiB,无 stock allocation-chain probe)
> **显存耗尽**:margin≈0 且 rc=0(与 WDDM 超分配/共享内存回退一致,机制未隔离验证;非可用 fit)

## 结论

**5 例 FBLC 记录,5 例判定为可用 fit。**
- **nblock block=256 实测 driver_peak 22.0 MiB**(模型 16.9 MiB,delta +5.1 MiB)→ **fits(预算余量 7654.0 MiB)** → **N-blocking 峰值闭合为实测**
- **峰值与 N 无关**(N≥2*block 时 max_cols=2*block,tile 缓冲按 max_cols 非 N)→ **N=22600 的 device 峰值 = 22.0 MiB**(同 block=256),M4 闭合成立;**N=22600 实际运行未实测**(host 峰值 ~8.4 GB——两个 N*N 输出 buffer(每 ~3.9 GiB)并存 + 面板,非单个 ~4 GiB,review F7;+ O(N²·T) 计算,诚实披露)
- block 阶梯递减:256→14.0 (128)→6.0 (64)→2.0 MiB (32);production N=5000 530.0 MiB(no-mask 模型 521.13 MiB,delta +8.87 MiB——校准 stock_corr driver overhead 实测 7-11 MiB 范围,一致)
- **测量披露**:cudaMemGetInfo free 内存 ~1 MiB 量化(driver_peak 均为 MiB 整数倍),小 delta 归因含 ±1 MiB 误差(review F1/contract)

## 环境

- GPU:NVIDIA GeForce RTX 4060 Laptop GPU total 8187.5 MiB
- 运行期 free_before ≈ 7106.0 MiB;模型预算口径 7676 MiB(8188−512)

## 明细

| kind | T×N | 规模 | driver(MiB) | 模型(MiB) | Δ(MiB) | margin(MiB) | rc | 判定 | 注 |
|---|---|---|---|---|---|---|---|---|---|
| nblock | 1218×5000 | block=256 | 22.0 | 16.9 | 5.1 | 7084.0 | 0 | ✅ | fit 余量 7654.0 MiB |
| nblock | 1218×5000 | block=128 | 14.0 | 7.95 | 6.05 | 7092.0 | 0 | ✅ | fit 余量 7662.0 MiB |
| nblock | 1218×5000 | block=64 | 6.0 | 3.85 | 2.15 | 7100.0 | 0 | ✅ | fit 余量 7670.0 MiB |
| nblock | 1218×5000 | block=32 | 2.0 | 1.89 | 0.11 | 7104.0 | 0 | ✅ | fit 余量 7674.0 MiB |
| production | 1218×5000 | production (non-blocked) | 530.0 | 521.13 | 8.87 | 6576.0 | 0 | ✅ | fit 余量 7146.0 MiB |

## 闭合范围(诚实边界)

- 实测闭合**仅 N-blocking 路径**(N=5000 锚定,峰值 N 无关);`current_peak`(production)仍为模型预测(calibration 已校)
- **N=22600 实际运行未实测**:host 峰值 ~8.4 GB(两个 N*N 输出 buffer 并存 + 面板,非单个 ~4 GiB)+ O(N²·T) 计算不可行;device 峰值 N 无关故闭合,N=22600 host 侧成本单独披露
- nblock 无 MemTracker → 无 delta_formula==0;driver_peak−模型 delta 未独立归因(cudaMemGetInfo MiB 量化 ±1 MiB + 无 stock allocation-chain probe;block=256 +5.1 MiB 含量化误差)
- 逐 tile host 抽列/写回为 host 成本(HWM 只测 device VRAM)

## 复现

    cmake --build build --target poc3_stock_corr_selfcheck
    build\poc3_stock_corr_selfcheck.exe --hwm   # 或
    PYTHONIOENCODING=utf-8 python benchmarks/stock_nblock_hwm_v1.py

*生成模型:benchmarks/stock_nblock_hwm_v1.py (DeepSeek-V4-Flash via Claude Code CLI, 2026-08-08)*
