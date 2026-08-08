# 风险登记: factor-cuda

> 模板来源：AI 协作项目全生命周期框架 附录F。**HG触发规则**：影响=H 且 概率≥M → 必须触发 Human Gate。更新频率：每次 Retrospect 时顺带检查。

| ID | 风险描述 | 影响 | 概率 | 缓解措施 | 触发信号 | Plan B | 状态 |
|----|---------|------|------|---------|---------|--------|------|
| R001 | CUDA Toolkit 未安装（Phase 0 阻塞） | H | H | ✅ 已装 **v13.3**（2026-07-31，nvcc V13.3.73） | 触发已过 | — | 已缓解 |
| R002 | nvcc × MSVC 19.51 host 编译器兼容性 | H | M | ✅ 本机 smoke 通过（2026-07-31，当前命令/配置：nvcc 13.3×MSVC 19.51 编译+GPU PASS）；**完整工具链矩阵（其他 CUDA/MSVC/驱动/架构）未验证**，为开放风险 | 本机触发已过 | 19.44 兜底 | 部分缓解 |
| R003 | RTX 4060 FP64 吞吐硬伤（DGEMM 慢于 CPU numpy/OpenBLAS） | H | H | CPU fallback 设为正式后端；精度档位管理 | correlation 基准慢于 CPU | FP32/TF32 + 误差预算（与 FP64 oracle 对照） | 监控中 |
| R004 | 免费替代足够（CuPy / QuantGplearn-Torch）→ 自研无边际收益 | H | M | PoC ② 预注册门槛：端到端 ≥2× | PoC ② 加速比 < 2× | STOP，记负结果 NRR | 监控中 |
| R005 | dense 面板显存溢出（全量 (T,N,F) 超可用显存） | H | M | F/T 分块 + 字节级峰值模型 + 可用显存−安全余量 + cudaMemGetInfo 校准（PoC ③ 交付） | 峰值 > 可用显存−安全余量 | REDESIGN 分块 | 监控中 |
| R006 | 分发门槛（用户需自行装 CUDA Toolkit + 编译） | M | M | wheel/conda 策略评估；目标硬件范围声明 | PoC 安装测试 | 源码编译或限定硬件范围 | 监控中 |
| R007 | parameter_scan 扫描语义未冻结（lookback 属因子职责冲突） | H | H | PoC ① 冻结契约（扫描自身操作参数） | 契约不可运行 | REDESIGN 接口 | 监控中 |
| R008 | 竞品对照的 tie/mask 语义差异导致公平基线失真 | M | M | 同数据同 mask 协议；CPU/GPU parity 测试 | 基线与 oracle 不一致 | 记录差异并分口径报告 | 监控中 |
| R009 | 本机无 GPU 前无法做 GPU 侧 PoC 验证 | M | M | ✅ 已缓解（RTX 4060 可用，sm_89，24 SM，vec_add PASS） | 触发已过 | — | 已缓解 |

> 触发 Human Gate 的当前项（影响=H 且 概率≥M）：**R003、R004、R005、R007**。（R001/R002/R009 已于 2026-07-31 缓解）
