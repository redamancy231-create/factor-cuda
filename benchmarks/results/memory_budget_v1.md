# factor-cuda 内存静态预算 v1（当前实现精确模型 + 项②流式化 + 项③F-blocking）

> 预算：可用 7676 MiB (total 8188 − reserve 512)

## 校准验证（模型 vs 实测 HWM，delta≈0 通过）

| 操作 | 规模 | 校准 HWM (MiB) | 模型 (MiB) | delta |
|---|---|---|---|---|
| factor_corr | (5000, 12) | 1191 | 1190.63 | -0.37 ✅ |
| factor_corr | (10000, 12) | 2381 | 2381.25 | 0.25 ✅ |
| stock_corr | (2000,) | 119 | 119.22 | 0.22 ✅ |
| stock_corr | (5000,) | 527 | 526.93 | -0.07 ✅ |
| stock_corr | (10000,) | 1817 | 1816.81 | -0.19 ✅ |

**all_match = True**

## 场景对比（当前 / 项②流式化 / 项③F-blocking）

### factor_corr canonical (N=5000, F=12)

- 当前实现峰值：**1190.63 MiB** ✅
- 项②流式化峰值：**634.98 MiB** ✅
- **实测(项② streaming tt=4096)**：driver_peak **636.0 MiB** ✅ fits（模型 634.98 MiB,delta 1.02 MiB,margin 6470.0 MiB,rc=0）
- 项③F-blocking（block=6, 3 tiles）：**1190.63 MiB** ✅
- **实测(项③ fblock block=6)**：driver_peak **1194.0 MiB** ✅ fits（模型 1190.63 MiB,delta 3.37 MiB,margin 5912.0 MiB）
- d_pp continuation：78 pairs × 256 lanes × 80B = **1.5 MiB**
- 注：审查 M1 修正：fblock 含 d_F_tile 输入缓冲（与 d_Xt_tile 同驻留）；F128/B64 超预算（~12.6 GiB），B32 可 fit；生产 F-blocking 若逐 tile 流式上传+转置+释放 d_F_tile（项②机制）才命中更小峰值。 fblock(项③) 已实测闭合（block=6, 1194.0 MiB vs 模型 1190.63 MiB）。 streaming(项②) 已实测闭合（tt=4096, 636.0 MiB vs 模型 634.98 MiB）。

### factor_corr F128 (N=5000, F=128)

- 当前实现峰值：**12645.61 MiB** ❌ 超预算
- 项②流式化峰值：**6863.6 MiB** ✅
- **实测(项② streaming tt=4096)**：driver_peak **7079.75 MiB** ✅ fits（模型 6863.6 MiB,delta 216.15 MiB,margin 26.25 MiB,rc=0）
- 项③F-blocking（block=32, 10 tiles）：**6325.11 MiB** ✅
- **实测(项③ fblock block=32)**：driver_peak **6588.11 MiB** ✅ fits（模型 6325.11 MiB,delta 263.0 MiB,margin 517.89 MiB）
- d_pp continuation：8256 pairs × 256 lanes × 80B = **161.2 MiB**
- 注：审查 M1 修正：fblock 含 d_F_tile 输入缓冲（与 d_Xt_tile 同驻留）；F128/B64 超预算（~12.6 GiB），B32 可 fit；生产 F-blocking 若逐 tile 流式上传+转置+释放 d_F_tile（项②机制）才命中更小峰值。 fblock(项③) 已实测闭合（block=32, 6588.1 MiB vs 模型 6325.11 MiB）。 streaming(项②) 已实测闭合（tt=4096, 7079.8 MiB vs 模型 6863.60 MiB）。

### factor_corr N10000 (N=10000, F=12)

- 当前实现峰值：**2381.25 MiB** ✅
- 项②流式化峰值：**1268.03 MiB** ✅
- **实测(项② streaming tt=4096)**：driver_peak **1270.0 MiB** ✅ fits（模型 1268.03 MiB,delta 1.97 MiB,margin 5836.0 MiB,rc=0）
- 项③F-blocking（block=6, 3 tiles）：**2381.24 MiB** ✅
- **实测(项③ fblock block=6)**：driver_peak **2386.0 MiB** ✅ fits（模型 2381.24 MiB,delta 4.76 MiB,margin 4720.0 MiB）
- d_pp continuation：78 pairs × 256 lanes × 80B = **1.5 MiB**
- 注：审查 M1 修正：fblock 含 d_F_tile 输入缓冲（与 d_Xt_tile 同驻留）；F128/B64 超预算（~12.6 GiB），B32 可 fit；生产 F-blocking 若逐 tile 流式上传+转置+释放 d_F_tile（项②机制）才命中更小峰值。 fblock(项③) 已实测闭合（block=6, 2386.0 MiB vs 模型 2381.24 MiB）。 streaming(项②) 已实测闭合（tt=4096, 1270.0 MiB vs 模型 1268.03 MiB）。

### stock_corr N5000 (N=5000)

- 当前实现峰值：**526.93 MiB** ✅
- 项③N-blocking（block=256, 210 tiles）：**16.9 MiB** ✅
- **实测(N-blocking block=256, N=5000)**：driver_peak **22.0 MiB** ✅ fits（模型 16.9 MiB,delta 5.1 MiB,margin 7084.0 MiB,rc=0）
- 注：审查 M4 修正：output d_corr+d_out 各分配 N*N*8（完整矩阵，非上三角），N=22600 两输出合计 ~7793 MiB 已超预算 → N-blocking 须同时分块/stream 输出（改变输出驻留契约），不能只分输入/工作区；N-blocking 模型基于 poc3_stock_corr_selfcheck.cu 的 stock_corr_gpu_nblock 分配链（tile 级驻留 + 逐 tile 输出流式化，无 N*N device 输出缓冲）；**nblock 峰值已实测闭合**（N=5000 block=256 driver 22.0 MiB vs 模型 16.90 MiB，device 峰值 N 无关（N≥2*block 限定）→ N=22600 同值；N=22600 实际运行未实测——host 峰值 ~8.4 GB（两个 N*N 输出 buffer 并存）+ O(N²·T) 计算）；审查 M4-2 披露：本预算只量化 device 驻留，nblock 额外付出 host pass-1 O(T*N) 预扫描 + 逐 tile host 抽列/写回（生产路径无此 host 成本）——device 内存换 host 预扫描的权衡未计入。

### stock_corr N10000 (N=10000)

- 当前实现峰值：**1816.81 MiB** ✅
- 项③N-blocking（block=256, 820 tiles）：**16.9 MiB** ✅
- **实测(N-blocking block=256, N=5000)**：driver_peak **22.0 MiB** ✅ fits（模型 16.9 MiB,delta 5.1 MiB,margin 7084.0 MiB,rc=0）
- 注：审查 M4 修正：output d_corr+d_out 各分配 N*N*8（完整矩阵，非上三角），N=22600 两输出合计 ~7793 MiB 已超预算 → N-blocking 须同时分块/stream 输出（改变输出驻留契约），不能只分输入/工作区；N-blocking 模型基于 poc3_stock_corr_selfcheck.cu 的 stock_corr_gpu_nblock 分配链（tile 级驻留 + 逐 tile 输出流式化，无 N*N device 输出缓冲）；**nblock 峰值已实测闭合**（N=5000 block=256 driver 22.0 MiB vs 模型 16.90 MiB，device 峰值 N 无关（N≥2*block 限定）→ N=22600 同值；N=22600 实际运行未实测——host 峰值 ~8.4 GB（两个 N*N 输出 buffer 并存）+ O(N²·T) 计算）；审查 M4-2 披露：本预算只量化 device 驻留，nblock 额外付出 host pass-1 O(T*N) 预扫描 + 逐 tile host 抽列/写回（生产路径无此 host 成本）——device 内存换 host 预扫描的权衡未计入。

### stock_corr N22600 (N=22600)

- 当前实现峰值：**8451.08 MiB** ❌ 超预算
- 项③N-blocking（block=256, 4005 tiles）：**16.9 MiB** ✅
- **实测(N-blocking block=256, N=5000)**：driver_peak **22.0 MiB** ✅ fits（模型 16.9 MiB,delta 5.1 MiB,margin 7084.0 MiB,rc=0）
- 注：审查 M4 修正：output d_corr+d_out 各分配 N*N*8（完整矩阵，非上三角），N=22600 两输出合计 ~7793 MiB 已超预算 → N-blocking 须同时分块/stream 输出（改变输出驻留契约），不能只分输入/工作区；N-blocking 模型基于 poc3_stock_corr_selfcheck.cu 的 stock_corr_gpu_nblock 分配链（tile 级驻留 + 逐 tile 输出流式化，无 N*N device 输出缓冲）；**nblock 峰值已实测闭合**（N=5000 block=256 driver 22.0 MiB vs 模型 16.90 MiB，device 峰值 N 无关（N≥2*block 限定）→ N=22600 同值；N=22600 实际运行未实测——host 峰值 ~8.4 GB（两个 N*N 输出 buffer 并存）+ O(N²·T) 计算）；审查 M4-2 披露：本预算只量化 device 驻留，nblock 额外付出 host pass-1 O(T*N) 预扫描 + 逐 tile host 抽列/写回（生产路径无此 host 成本）——device 内存换 host 预扫描的权衡未计入。

## Kahan 决策 v2 四项闭合

- **d_valid_lifetime**：d_valid 与 d_Xt 同驻留（Kahan 重跑读两者）；流式化/F-blocking 均须含 d_valid
- **transpose_overlap**：峰值 = 转置前 d_F/d_X + d_Xt + d_valid + d_mask 重叠（校准 2381 MiB 实证）
- **dpp_scale**：d_pp = P×blockDim×struct_bytes_lane(Partial1+Partial2=80B/lane)；F=128 → 161.25 MiB（见 scenarios dpp）
- **cross_block_pair**：F-blocking 2D tile 驻留 (B_a+B_b)×R×(8+1) + tile pair 工作区（见 scenarios fblock）

## 审查口径（GPT-5.6-Sol 2026-08-06，15 项处置后）

- 场景峰值：**fblock(项③) 已于 2026-08-07 实测闭合**（factor_corr F=12/F=128 见各场景 `measured_fblock`，证据 `benchmarks/results/factor_fblock_hwm_v1.json`；分配链 probe delta 3 MiB、B32 本次总 delta 见 `measured_fblock.delta_MiB`（另一次 245.09 MiB 单样本未纳入证据，review F7））；**N-blocking 已于 2026-08-08 实测闭合**（stock N5000/N10000/N22600 见各场景 `measured_nblock`，证据 `benchmarks/results/stock_nblock_hwm_v1.json`；N=5000 block=256 锚定，device 峰值 N 无关（N≥2*block）→ N=22600 同值 22.0 MiB；N=22600 实际运行未实测——host 峰值 ~8.4 GB（两个 N*N 输出 buffer）+ 计算，诚实披露）；**streaming(项②) 已于 2026-08-08 实测闭合**（F=128 见各场景 `measured_stream`，证据 `benchmarks/results/factor_stream_hwm_v1.json`；F128 driver 6987 MiB fits——当前 12.6 GiB 超预算→streaming 6.9 GiB fits；F=12 锚点 636/1270 MiB 与模型 delta ≤2 MiB；production F=128 对照 vram-exhausted margin≈0）
- fblock 已含 d_F_tile 输入缓冲（M1）：F128/B64 超预算、B32 可 fit；生产 F-blocking 若逐 tile 流式上传+释放 d_F_tile 命中更小峰值
- d_pp 按实际 ABI struct（Partial1+Partial2=80B/lane）计（M2），F128=161.25 MiB
- host staging 未计入 GPU 预算（M6）：fblock driver 逐 tile F_tile host 内存 F128/B64 ~5.9 GiB，生产须有界/pinned staging
- stock N-blocking 须同时分块/stream 输出（M4）：d_corr+d_out 各 N*N*8，N=22600 已超预算

## 复现

    PYTHONIOENCODING=utf-8 python benchmarks/memory_budget_v1.py

*生成模型: DeepSeek-V4-Flash (via Claude Code CLI) · memory_budget_v1.py*
