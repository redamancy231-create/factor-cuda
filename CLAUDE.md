# CLAUDE.md — factor-cuda（CUDA 因子截面分析加速）

> 项目类型：软件工程（C++/CUDA + Python，量化因子**截面分析**加速工具）
> **状态：L0 Spec（已冻结，当前维护规则）**——PoC ① 契约冻结经三轮异后端复核（GPT-5.6-Sol，34 条发现）闭合，2026-08-03 HG-2 人工批准恢复 Spec 状态；项目已发布 v1.0.x。任何变更走 HG-2。
> 本文档是 L0 Spec 入口文件（冻结契约）。方案细节 → `PLAN.md`（历史方案文档，superseded）；变更记录 → `CHANGELOG.md`；风险 → `RISK.md`；审查 → `reviews/`。
> 中文名定稿：「CUDA 因子截面分析加速」。**不是"因子计算"**——本项目不做因子计算。

## Agent 边界

- **可派**：`PLAN.md` 范围内的截面分析加速设计/实现/审查/基准，**PoC 优先**（PoC ①–④ 通过并经人裁决前不进入 Phase 1–4）。
- **禁止**：
  - 因子计算（ml-quant-trading 职责）、数据获取（ashare-mcp 职责）、回测框架（分层统计是否纳入待 PoC 定义）
  - 承诺未实测的性能数字（18×/45s 类叙事已撤回，一切性能以 PoC ② 实测为准）
  - 修改本文件（Spec/DRAFT）——变更走 HG-2（见「更新协议」）

## 环境与命令

- 运行环境：Windows 11 / Git Bash（非 darwin）
- 工具链（2026-07-31 本机实测，版本限定）：VS2026 Community（MSVC **19.51.36248** 优先 / 19.44 兜底）、VS 内置 CMake 4.3.1、**CUDA Toolkit v13.3**（nvcc V13.3.73）、RTX 4060 Laptop（sm_89，总显存 8188 MiB）。Phase 0 基本 smoke PASS（**本机当前配置**，非全矩阵验证）。
- 关键命令（不可从代码推导）：
  - `nvcc --version`
  - CMake：VS2026 自带 CMake（`cmake.exe`，随 VS 安装；Ninja generator）
  - MSVC：`vcvars64.bat` 设置 cl 环境（VS2026 Community：`VC\Auxiliary\Build\vcvars64.bat`，相对 VS 安装根）
  - **构建**：`cmd /c dev-build.bat`（CMake **Ninja** 配置+构建+运行自检；Ninja 为推荐路径，非唯一——VS generator 在 2026-07-31 全新目录亦验证可用）
  - 竞品对照（PoC ② 基线）：QuantGplearn、equity-factor-lab（本地开发索引，发布后不随包分发）

## 操作语义契约（冻结，PoC ①）

> 本段替换原「操作语义契约（DRAFT）」草案。冻结要求已满足：每操作定死轴/排序方向/tie/NaN/mask/dtype/输出 shape/空集与常量输入/错误类型/确定性，无"二选一/须定义/待冻结"措辞。本段经 HG-2 批准后生效；生效后任何变更走 HG-2。
>
> 证据锚点（供 oracle/parity 测试对照；⚠️ **本地开发环境锚点**——引用本地克隆的 ml-quant-trading/QuantGplearn/equity-factor-lab 仓库路径，仅作契约设计依据，非发布契约、不随包分发）：`compute_legacy_set` 返回 `(T,N,F)` float32 Tensor + joint bool mask + list(names)——`ml-quant-trading/src/mlquant/features/legacy_factors.py:102-119`；数值字段 `[T,N]` float32、mask True=可交易——`ml-quant-trading/src/mlquant/data/panel.py:5-12,51`；cs_rank average/percentile（`tensor_factors.py:46-91`）、**cs_zscore masked 格置 0 仅当 neutralize=True 路径**——`ml-quant-trading/src/mlquant/features/tensor_factors.py:83-91`；`legacy_factors.py:113` 仅将非有限值替换为 0（**不保证 mask=False 且有限格为 0**，masked 置 0 只发生在 cs_zscore 归一化路径）——故"mask 才是权威"主规则不受影响；stable ordinal 升序秩、isfinite mask、非有限填 inf、输出 NaN——`reference/QuantGplearn/QuantGplearn/torch_functions.py:121-129`；pandas average pct、空掩码全 NaN——`reference/QuantGplearn/QuantGplearn/functions.py:142-149`；min_stocks=30、valid=notna 交集、不足置 NaN——`reference/equity-factor-lab/factorlab/crosssection/ic.py:11-24`；**支持 skip-day 与 point-in-time 标签方向**（参数为 horizon_months、端点围绕月末，**非 rolling_ic 日频 h 公式的直接实现证据**；日频 h 标签的权威来源为本契约 §3 标签生成器 fixture）——`reference/equity-factor-lab/factorlab/bias/asof.py:3-11,27-56`；numpy.corrcoef 两遍中心化 ddof=1 为 CPU oracle（numpy 版本须与实现锁定，见 §2 oracle 归约顺序冻结）。

### 0. 输入适配与全局约定

- **dtype 主线（两级口径，全局唯一）**：规范 dtype 分两级。① 因子值张量与截面排序类操作（`cross_sectional_rank` 及其参数扫描）——float32 主线：ml-quant 输出 float32 `(T,N,F)`（legacy_factors.py:118）为零转换直通主路径；float64 输入（numpy float64 或 torch float64）由适配层按 IEEE 754 舍入到最近偶数下转 float32；其他数值 dtype（int/uint/bool/complex/float16）一律抛 ValueError。② 聚合统计类操作（`factor_corr`/`stock_corr`/`rolling_ic`）——输入接受 float32/float64（见各操作 dtype 项），内部一律提升 float64 计算，输出 float64（与 numpy oracle 精度对齐）。各操作内部允许将累加器提升 float64（仅内部，不改变契约规定的输入/输出 dtype）。RTX 4060 FP64 吞吐≈FP32 的 1/64（CLAUDE.md:70 坑位）：任何热路径 kernel 不得以 float64 为规范因子 dtype。
  - edge cases：float64→float32 下转产生新 tie（两不同 float64 值在 float32 下相等）——tie 一律在 float32 规范张量上判定，CPU oracle 与 GPU 后端消费同一 float32 张量，故两后端逐位一致；ml-quant masked 格填充的 0.0 保持 float32，不因 dtype 转换改变；累加器 float64 提升仅内部，输出仍按契约 dtype。
- **因子束规范输入**：统一"因子束" = (values, mask, names)：单因子截面操作（`cross_sectional_rank`、`stock_corr`、`rolling_ic` 的因子侧）规范 values 为 `(T,N)`；批量操作（`factor_corr`）规范 values 为 `(T,N,F)`；mask 一律 `(T,N)` bool；names 为长度 F 的序列（仅批量形态需要）。T=日期数、N=股票数、F=因子数，允许为 0 表示空集（空集语义由各操作契约定义）。ml-quant 适配入口 `compute_legacy_set` 直接产出此束（legacy_factors.py:102-119）。单因子操作收到 ndim=3 输入时抛 ValueError，并提示用适配层 `factor_plane` 取单面，不做隐式取面。
  - edge cases：T=0 或 N=0 或 F=0：适配层透传不拦截，空集处置由各操作契约决定（四个操作均对 T/N/F<1 抛 ValueError）；批量形态 F>1：names 可选，提供但 len≠F 抛 ValueError，不提供则输出按 F 索引；`(T,N,F)` 中 F=1：仍按批量形态处理；`stock_corr` 的 `(T,N)` 输入：N 维为股票，输出 `(N,N)`。
- **mask 语义（全局唯一）**：True=可交易/有效格，False=排除格（不参与排序、统计、相关、IC 的任何计算）。mask 为唯一权威——**被排除格的存储值可能为 0.0，也可能保留有限原值**（`legacy_factors.py:113` 仅将非有限值替换为 0，不保证 mask=False 且有限的格为 0；masked 置 0 仅发生在 cs_zscore/neutralize 路径，见证据锚点限定），**任何路径均严禁由数值反推 mask**。有效格 = mask==True 且 isfinite(values)；二者不一致时（mask=True 但值 NaN/±inf）该格按无效计，输出格按各操作的空值规则（如 rank 输出该格为 NaN）。mask 允许为 None（视为全 True，仅受 isfinite 约束；rolling_ic 的因子/收益两侧 mask 缺省均视为全 True）。
  - edge cases：mask 全 False（该截面无有效股）→ 该截面输出按各操作空集/常量截面规则（如 rank 全 NaN）；mask 与 values 前两维不一致 → ValueError；mask dtype 非 bool → ValueError，不自动 0/1 转换；mask=True 但值含 NaN：按无效计，输出为 NaN/空值；ml-quant joint mask 已排除不可交易格，适配层不改写 mask。
- **dtype 接受与转换（适配层单一白名单）**：适配层接受 numpy 数组或 torch.Tensor，dtype ∈ {float32, float64}；以及 DLPack 胶囊。float64→float32 下转（IEEE 754 舍入到最近偶数）；float32 原样零复制。mask 只接受 bool dtype（numpy bool_ 或 torch.bool），非 bool 抛 ValueError，不自动 0/1 转换。int/uint/bool/complex/float16 数值 dtype 一律抛 ValueError（对全部公共操作生效，无 per-op 例外）。注：float64→float32 下转仅适用于 float32 主线操作（rank/parameter_scan）；聚合类操作（correlation/rolling_ic）接受 float32/float64 后内部提升 float64 计算、输出 float64，不执行下转（见各操作 dtype 项）。
  - edge cases：float64 numpy 输入：torch.from_numpy 后 .float()（复制）；非连续 float32 输入：.contiguous()（复制）；float16 输入：不支持，抛 ValueError；bool 因子值（非 mask）：不支持，抛 ValueError；DLPack 张量 dtype 非 float32/float64：抛 ValueError（消费式转换已发生）。
- **内存布局**：规范内存布局 = C-contiguous（行主序）：`(T,N)` 中 N 为最内维；`(T,N,F)` 中 F 为最内维。所有进入 kernel 的输入与所有输出均为 C-contiguous。非 C-contiguous 输入（视图/转置/切片/F-contiguous）由适配层复制一次转为 C-contiguous（.contiguous() / np.ascontiguousarray）。ml-quant 的 torch.stack(cols, dim=-1) 输出即 C-contiguous（F 最内），零复制直通。适配层提供 `factor_plane(factors, f)`：校验 0≤f<F，返回因子 f 的 `(T,N)` C-contiguous 副本（`(T,N,F)` 中 N 面 stride=F 不连续，故必为副本；ndim=2 输入要求 f==0 并返回其连续副本）。
  - edge cases：numpy 转置视图（如 factors.T）→ 复制为 C-contiguous；torch 切片（factors[:, :5, :]）→ 复制；F-contiguous `(T,N,F)`（N 最内）→ 复制为 C-contiguous；输入已满足规范（float32+C-contiguous+目标 device）→ 零拷贝直通（不额外分配）；`factor_plane(factors, f)` 对 f 越界（f<0 或 f≥F）→ ValueError。
- **device 策略**：执行 device 由调用方显式 device 参数指定，缺省 = torch.cuda.current_device()（CUDA 不可用且未显式指定时抛 RuntimeError）。结果张量 mirror 输入因子张量的 device：输入在 CPU → 结果在 CPU（适配层负责 CPU→GPU 上传与结果回拷，传输成本计入端到端口径）；输入已在 CUDA → 结果留在该 CUDA device。mask 必须与 values 同 device，不同则抛 ValueError（不自动迁移）。**device 优先级表（对全部操作，按序裁决）**：①操作公开签名若有显式 device/backend 参数，其值优先于全局缺省；②无显式参数或参数为 None 时，按全局缺省（current_device）；③**操作级例外**（rolling_ic 的 device=None 在 CUDA 不可用时自动 CPU、correlation backend='cpu' 恒 CPU、parameter_scan 恒 GPU）覆盖全局缺省；④结果 device 一律 mirror 输入因子张量的 device（见例外①），"device 指定端"不得理解为输出落到执行端（除非输入本就在该端）。**correlation 的 cpu backend 例外**：cpu 后端（含默认）下 mask/values 异 device 合法——适配层在拷贝前不做同 device 校验，统一拷贝到 CPU 后再算（见 §2 adapter_alignment）；仅 backend='cuda' 时强制 values 与 mask 同 device，否则 ValueError。例外① correlation：backend 参数默认 cpu（CPU fallback 为正式 oracle 后端），backend='cuda' 时输出仍 mirror 输入因子张量的 device（仅计算在 GPU，输出落到输入所在端），见 §2；例外② parameter_scan：为 GPU 计时语义要求 GPU（无 CPU 回退），其各组 result 恒为 CPU（D2H 物化，见 §4 output）。
  - edge cases：CPU 输入 + device 缺省（CUDA 可用）→ 上传执行 + 结果回拷到 CPU；CUDA 不可用 + device 缺省 → RuntimeError（提示显式指定 device='cpu'）；mask 在 CPU 而 values 在 CUDA → ValueError（correlation cpu backend 除外，见例外①）；输入为 cuda:0 而 device 参数为 cuda:1 → 迁至 cuda:1 执行，结果回 cuda:0。
- **复制策略**：只读契约：适配层与 kernel 均不修改输入张量/数组；任何需要的表示变换（dtype/device/layout）都产生新副本，绝不在原地改写。返回张量均为新分配，与输入内存无共享。输入已满足规范（float32 + C-contiguous + 目标 device）时走零拷贝直通路径（kernel 直接读输入，不额外复制），但 kernel 输出始终是新张量。numpy 输入经 torch.from_numpy 进入（必要时复制转换），返回张量与 numpy 数组无内存共享。requires_grad 输入在适配层 .detach() 处理（本项目输出无梯度，不参与 autograd 图）。**零拷贝适用范围（HG-2 澄清 2026-08-06）**：零拷贝直通仅适用于 host-resident 输入（numpy 数组 / torch-CPU / DLPack-host）；torch-CUDA 输入经 D2H→H2D→D2H→H2D 传输，语义正确但非零拷贝（显式承诺，非隐藏偏离）；设备驻留零拷贝为 Phase 4 性能项（绑定 device-pointer API 扩展 + 审查）。
- **实现维度上限（HG-2 冻结 2026-08-06，超出抛 ValueError）**：截面排序类（`cross_sectional_rank`/`parameter_scan`）`T*N ≤ INT32_MAX` 且 `N ≤ 2^24`（float32 秩精确表示上限）；`stock_corr` `N*N ≤ INT32_MAX`（N ≤ 46340）；`factor_corr` `T*N ≤ INT32_MAX` 且 `F ≤ 128`（pair 网格）；`parameter_scan` 宿主输出预算 `4·T·N·4 B ≤ 4 GiB`。上限源于 32 位索引/网格与宿主输出约束，超出按契约抛 ValueError（消息指明超限维度）。
  - edge cases：同 buffer 的 mask 与 values 为各自独立张量，互不共享；多次调用同一输入：每次返回独立新张量（不缓存/复用输出内存）；输出为新分配 → 调用方任意顺序释放均安全；requires_grad 输入 → .detach() 后进入计算，输出 requires_grad=False。
- **DLPack 所有权与同步**：DLPack 胶囊经 torch.utils.dlpack.from_dlpack 消费式转换。**统一所有权措辞**：适配层在**调用期间**消费该 capsule（from_dlpack 后底层 storage 由新 tensor 持有），并持有**必要引用**至同步返回完成；**同步返回后不跨调用保留引用**——下次调用需重新提供 capsule。capsule 为单次消费：同一 capsule 传入两次，第二次**抛 ValueError**（消息指明 capsule 已消费；所有权已于首次转移，producer tensor 不得在消费前释放）；适配层在清理时不得二次释放（deleter 由 torch 管理）。同步契约：所有公共操作对调用方是同步的——操作入口对输入所在 stream 执行同步（确保输入就绪），操作返回前完成同步（结果已物化、可在调用方当前 stream 上读取）；内部多 stream/异步分块（PoC ③/④ 优化）不改变对外同步契约。
  - edge cases：DLPack 张量在 CUDA 异步 stream：入口同步保证就绪；同一 DLPack 胶囊传入两次 → 第二次**抛 ValueError**（非未定义行为，错误类型冻结）；from_dlpack 后底层内存由新 tensor 持有，deleter 由 torch 管理，适配层不二次释放；输出在返回前已同步 → 调用方无需额外 synchronize。
- **names 与 F 对齐**：names 为可选标注，仅在批量操作（factor_corr）中生效——提供时须 len(names) == F 否则抛 ValueError，不提供时输出行列按 F 索引（0..F-1）；names[i] 标签 factors[...,i]。names 仅作标注不参与计算；适配层按引用透传 ml-quant 返回的 list（不排序、不去重、不改写）。factor_corr 输出 `(F,F)` 的行/列顺序与 names 顺序一致（out[i,j] = names[i] 与 names[j] 两因子的相关）。names 允许重复。
  - edge cases：names 提供但长度 != F → ValueError；names 为 None → 输出按 F 索引，无标注；names 含 None 或空串：允许（仅标注）；names 重复：允许，输出行列标签可能相同但数值有定义。
- **适配错误行为（全局统一）**：适配/校验失败统一抛标准 Python 异常，类型映射唯一，覆盖全部公共操作（含 rolling_ic/correlation 的 dtype 错误，一律 ValueError，不归 TypeError）。① 形状/维度错误（ndim∉{2,3}、mask 前两维≠values、names 长度≠F、单因子操作收到 ndim=3、factor_plane 的 f 越界、mask 与 values 不同 device）→ ValueError；② dtype 不支持（因子非 float32/float64、mask 非 bool、int/uint/bool/complex/float16/object）→ ValueError；③ 传入非 tensor/非 numpy 数组/非 DLPack 胶囊对象 → TypeError；④ CUDA 不可用、cudaMalloc/内核启动失败、显存不足 → RuntimeError（全操作统一，含 correlation backend='cuda' 与 parameter_scan，禁止改归 ValueError）。禁止抛 AssertionError 或裸 Exception；跨语言边界（C++/CUDA）错误经 pybind11 转为上述 Python 异常，绝不静默吞错也不让段错误越界。空 shape（T=0/N=0/F=0）适配层透传不拦截，空集处置由各操作契约决定（四个操作均对 T/N/F<1 抛 ValueError）。
  - edge cases：ndim=1 或 4 → ValueError；mask 为 float 数组 → ValueError；传入 list/str 等非张量对象 → TypeError；CUDA 驱动失败 → RuntimeError（含可读上下文）；批量形态 F>1 提供 names 但长度≠F → ValueError，不提供 names → 按 F 索引（非错误）。
- **确定性与 oracle 一致性**：确定性：同一输入（字节级一致）+ 同一参数 + 同一 device，重复调用输出逐位一致；实现必须固定归约顺序与分块/合并顺序，禁止依赖 stream 完成时序的不确定归约。oracle 一致性（对全部后端统一，含 GPU kernel、CPU fallback、numpy 参考实现）：比较型操作（rank 的秩/argsort/tie 判定）在相同 float32 输入上与 CPU oracle 位一致；累加型操作（相关、IC、方差）相对误差 ≤1e-5（以 oracle 为基准的通用下限），操作级更严容差覆盖之——correlation |Δr|≤1e-12、rolling_ic IC ≤1e-12（均以 float64 内部累加达成）；超判据视为契约违背。
  - edge cases：常数截面/全无效截面：输出确定（各 op 定义，如 rank 全 NaN），仍逐位可复现；tie 密集输入：stable ordinal 输出确定且与 oracle 位一致；跨运行 GPU 温度/时钟不影响结果（只影响耗时）；分块（PoC ③）后归并顺序固定 → 结果与不分块位级一致。

### 1. cross_sectional_rank

- **axis**：秩沿 instruments 轴计算，在规范布局 `(T,N)` 中即 axis=1；每个时间点 t 的截面独立排名，排名不使用跨时间的任何信息；操作不接受 axis 参数，也不接受 3D `(T,N,F)` 输入——批量多因子请经适配层 `factor_plane` 逐因子调用。
  - edge cases：`(T,N)` 单布局；N=1 单股票截面（每截面 K≤1）；T=1 单日面板；ndim=3 输入 → ValueError。
- **direction**：提供 descending 布尔参数，默认 False（升序）。升序秩 asc[i] = 1 + #{j : y[j] < y[i] 或 (y[j] == y[i] 且 j < i)}；降序秩定义为对元素逐个取负后的升序秩，即 desc[i] = 1 + #{j : y[j] > y[i] 或 (y[j] == y[i] 且 j < i)}（并列内按截面内列索引递增打破、先出现的列获得更优秩）。禁止用 desc = K − asc + 1 快捷公式（有并列时方向翻转，如 y=[3,3,1] 得 [2,1,3] 而非 [1,2,3]）。
  - edge cases：升/降序下并列打破方向（先出现列更优）；降序下常量截面（与升序同为 1..K 按索引，非 K..1 反转）；K=1 时升降序秩均为 1。
- **ties**：tie 处理定死为 stable ordinal：参与单元按值精确比较排序，相等值按截面内列索引递增打破并列，每个单元获得互不相同的 1..K 序位（公式见 direction）；CPU oracle 与 GPU 内核必须都实现该公式，禁止任一实现悄悄退化为 average。
  - edge cases：全相等截面；二元/三元并列；+0.0 与 -0.0（IEEE 相等→并列按索引）；整数型输入值的精确相等。
- **exact_comparison**：排名与 tie 判定一律使用 IEEE-754 对 float32 值的精确相等与精确严格比较，禁用任何 epsilon/容差/近似比较；即使输入值表现为整数（如已是整数秩），也按精确值比较，不因浮点接近而并入并列。
  - edge cases：相邻 float32 可表示值的差异不得误判并列；-0.0/+0.0；整型值的浮点表示；接近但不等的值。
- **base**：输出秩为 1-based：截面内最小（升序）/最大（降序）参与单元的秩为 1，最大序位为 K；禁止输出 0-based 秩。
  - edge cases：K=1 → 秩 1；K=0 → 全 NaN（见 empty_section）；最大序位恰为 K。
- **nan**：非有限值（NaN、+inf、-inf）一律不参与排名、不参与计数 K；参与条件 = mask 参与 且 isfinite(x)；非参与单元输出 NaN。
  - edge cases：NaN/+inf/-inf 混入；全 NaN 行；mask=True 但值 NaN（不参与，输出 NaN）；mask=False 但值有限（不参与，输出 NaN）。
- **mask**：mask 参数允许为 None，提供时必须是 bool 张量且 shape 与输入完全一致 `(T,N)`，True=参与该单元；mask 为 None 时表示全部单元参与（仅受 isfinite 约束）；参与条件 = (mask 为 None 或 mask[i]==True) 且 isfinite(x[i])。mask 校验基于与 values 前两维完全匹配；不匹配 → ValueError。
  - edge cases：mask 全 False 行；mask=False 但值有限；mask=True 但值 NaN；mask shape 与输入不一致（抛 ValueError）；mask 非 bool dtype（抛 ValueError）。
- **dtype**：规范输入 dtype = float32；接受 float32/float64（float64 经适配层按 IEEE 754 舍入到最近偶数下转为 float32 规范张量后计算，tie 在该 float32 张量上判定，CPU oracle 与 GPU 后端消费同一张量）；输出 dtype = float32，秩为 1..K 的精确整数值（float32 可精确表示整数至 2^24），非参与单元输出 NaN；N>2^24 → ValueError（秩精确表示保证不适用于超界）。
  - edge cases：float64 输入 → 适配层下转后计算（非错误）；float16/int/complex 输入 → ValueError；输出 NaN 采用 IEEE 754 默认 quiet NaN 载荷（float32 0x7fc00000）——oracle 按位断言对 NaN 单元按载荷比较（reinterpret 为 uint32 后比较，不启用 equal_nan 归并）；N>2^24 → ValueError。
- **output_shape**：输出 shape 与输入完全一致：输入 `(T,N)` → 输出 `(T,N)`；无降维、无 keepdim 参数；不接受 `(T,N,F)` 输入。
  - edge cases：N=1；与 mask 同 shape 对齐；ndim=3 输入 → ValueError。
- **output_normalization**：操作输出原始整数秩（1..K），不做百分位归一化（除以 K）；百分位秩属下游操作（如 rolling_ic 内 Spearman 或分位变换）职责，不在本操作内实现。注：这是对 ml-quant cs_rank 的 percentile (0,1] 输出（tensor_factors.py:46-80）与 QuantGplearn `_rank_pct_dim` 的 percent 输出的显式偏离；因子分析如需百分位，可 rank/K 一键互转。
  - edge cases：K 变化时秩上界；下游对整数秩的精确处理；rank/K 可由下游一键互转。
- **empty_section**：单个截面 K=0（无任何参与单元）时，该截面输出全 NaN，不抛错、不影响其他截面；结构性空输入（N==0 或 T==0）抛 ValueError。
  - edge cases：单行 mask 全 False；单行全 NaN；N==0（抛 ValueError）；T==0（抛 ValueError）；混合行（部分截面正常、部分全无效）。
- **constant_section**：参与值全部相等（常量截面）不特殊处理：stable ordinal 下各单元按截面内列索引递增获得互异秩 1..K（升序与降序输出相同），不产生 NaN、不报错；实现者不得"修正"为平均秩或全 NaN。
  - edge cases：全相等升序/降序（均 1..K 按索引）；两相等+一不同；K=1 常量（秩 1）。
- **determinism**：操作必须逐位确定：相同输入在任何运行、任何设备、任何线程调度下产生位级相同输出；GPU 实现必须用稳定排序（如 thrust::stable_sort / cub 稳定分段排序）保证并列按输入列索引顺序打破；oracle 测试按位断言 GPU 输出 == CPU oracle 输出。
  - edge cases：大 tie 群；同一输入多次运行；CPU 与 GPU 交叉运行；多个 (t,·) 截面并行的确定（每截面独立定秩，跨截面无依赖）。
- **errors**：输入错误统一抛 ValueError（带指明问题域的 message）：输入 ndim≠2、N==0、T==0、N>2^24、mask 提供但 shape 与输入不一致、mask 非 bool、输入 dtype 非 float32/float64（float16/int/complex 等）。数据级无效（全无效截面、常量截面、含 NaN）不抛错。
  - edge cases：1D/4D 输入；N==0/T==0；N>2^24；mask shape 错误；mask 为 int 型；float64 输入（合法，适配层下转）。

### 2. correlation（factor_corr / stock_corr）

- **api_split**：`factor_corr` 与 `stock_corr` 是两个独立操作，签名分别为 `factor_corr(数据 (T,N,F), 可选 mask (T,N) bool, 可选 names, backend='cpu') → 形状 (F,F) 的 float64 相关矩阵`，与 `stock_corr(数据 (T,N), 可选 mask (T,N) bool, backend='cpu') → 形状 (N,N) 的 float64 相关矩阵`；二者各自独立实现、各自独立 oracle 测试，互不调用。factor_corr 的 names 可选——提供时须 len(names)==F 否则 ValueError，不提供则行列按 F 索引。**backend 参数即执行端选择（无独立 device 参数）**：backend='cpu' 恒 CPU；backend='cuda' 时执行端由输入 device 或 torch.cuda.current_device() 唯一选定——输入在 CUDA 则用之，输入在 CPU 则用 current_device，**不接受额外 device 参数**（这是对全局"显式 device 参数"的显式例外）。**输出容器**：backend='cpu' 时输出 numpy float64 数组；backend='cuda' 时输出 torch float64 张量且 mirror 输入因子张量的 device（见 output_shape_dtype）。backend 参数与全局 device 策略缺省正交（见 §0 device 策略例外①）。
  - edge cases：factor_corr 输入必须是 3-D，stock_corr 必须是 2-D，维度错误抛 ValueError；F=1 或 N=1 时输出 (1,1) 矩阵，不视为错误。
- **aggregation**：两个操作的每个输出条目都是两条输入序列在各自有效观测集上的 Pearson 相关系数，聚合口径统一为 pooled 且不加权：factor_corr 条目 (f,g) 的有效观测集 = { (t,i) : mask[t,i]=True 且 factors[t,i,f] 与 factors[t,i,g] 均有限 }，在该 pooled 样本上算相关；stock_corr 条目 (i,j) 的有效观测集 = { t : mask[t,i]=True 且 mask[t,j]=True 且两序列在 t 均有限 }，在有效 t 序列上算相关。
  - edge cases：T=1 时 pooled 退化为该日截面相关；mask 全 False 或全非有限 → 全体 NaN；单对序列无共同有效观测 → 该条目 NaN。
- **uncentered_gram**：所有后端都必须计算中心化 Pearson 相关，输出必须与对**同一原始有效子集**调用**仓库内冻结 oracle wrapper**（`tests/fixtures/corr_oracle_v1.py`，锁定 Python 3.12.7 + NumPy 2.4.4 + 构建指纹）的结果满足逐元素 |Δr|≤1e-12——**该 wrapper 是唯一 oracle**（直接执行 `np.corrcoef(np.stack([xv,yv]))[0,1]`，含其 NaN 行为）；禁止把未中心化的 XᵀX 或其任何归一化当作相关输出。**禁止强制 max-abs 缩放**：缩放虽不改变 Pearson 实数值，但在浮点上与 numpy 直接路径不等价（实测域内可差 ~3e-2），故实现**不得**以缩放改变计算路径；若实现内部使用缩放仅作防溢出的数值手段，须保证其结果与 wrapper 仍 ≤1e-12（见 oracle wrapper 条款）。
  - **数值域 = API 前置条件（对全部后端统一）**：有效观测的绝对量级须满足 max|x| ≤ 1e150 且 min 非零 |x| ≥ 1e-150。**域内输入**承诺与 oracle wrapper ≤1e-12。**域外输入（任一量级越界）统一抛 ValueError**（支持域是可校验的 API 前置条件；不要求 GPU/CPU 复制 numpy 的偶发 overflow/underflow 行为，也不承诺域外的任何数学或 NaN 结果）。典型量化因子 |x|∈[1e-3,1e3] 完全落在域内（实测 corr 有限正常），本前置条件不影响实际使用。
  - **快捷规则优先级（数学规则让位于 wrapper）**：`n=2 且两值不同 → |r|=1`、`对角线 → 1.0`、`精确零方差 → NaN` 等数学快捷规则**仅当 oracle wrapper 对同一子集返回有限值时才适用**；若 wrapper 返回 NaN（如平方溢出），则所有快捷规则一律让位，输出保持 NaN。wrapper 是分支优先级的唯一裁决者，实现与 oracle 测试均以 wrapper 输出为准。
  - edge cases：均值非零的常量平移输入；接近 ±1 的高相关输入；大数×小数尺度差异的输入；[1e200,-1e200]（域外→**抛 ValueError**，不承诺任何相关值）；[0,5e-324]（域外→**抛 ValueError**）；1e15 偏置输入（域内但归约顺序敏感，见归约顺序敏感输入条款）。
  - **归约顺序敏感输入（HG-2 变更 2026-08-05）**：域内存在浮点归约顺序敏感输入，其 corr 值无法由非 numpy 后端位级复现 wrapper——因为 wrapper 的值本身携带 numpy 私有归约（np.mean/np.dot，含 SIMD/BLAS，跨构建可变）的特定舍入误差。实证（GPT-5.6-Sol 独立审查 2026-08-05 + 本机复现）：**大偏置**输入（某参与列 |mean| > 1e3×σ，如 1e12 偏置）下 mean 舍入误差（np.mean pairwise ~1e-4 绝对）被放大到 corr（~1e-7 级，>1e-12 契约）；**var 下溢风险**输入（中心化后平方和接近 float64 下溢，如 max|有效值| ≲ 1e-140 的相邻值）下 var 是否下溢取决于 dot 归约顺序。**处置**：归约顺序敏感输入（上述两类）**不承诺与 wrapper ≤1e-12**；所有后端（含 CUDA）须与**高精度参考**（串行 Kahan 或更高精度，相对数学真值 ≤1e-12）逐元素一致；wrapper 在此类输入上仅作诊断参考，不作 parity 硬判据。**低偏置且无下溢风险的典型输入保持严格 ≤1e-12 parity**（实测 corpus 因子 |mean|/σ ≤ ~0.3、量级 1e-3..1e3，GPU 与 wrapper 差 ~1e-17）。edge cases：判定边界 |mean|≈1e3×σ 附近按敏感处理（保守）；selfcheck 对敏感用例以高精度参考为硬判据并标注（见 `poc3_stock_corr_selfcheck.cu` 注释）。
- **ddof**：采用样本协方差约定 ddof=1（与 numpy.cov/numpy.corrcoef、pandas 一致）；有效样本数 n<2 时该条目输出 NaN；相关系数本身对 ddof 不敏感，ddof 冻结只锁定内部协方差约定。
  - edge cases：n=2 且两值不同 → |r|=1；n=1 → NaN；n=0 → NaN。
- **mask**：mask 是形状 `(T,N)` 的布尔张量，True=该单元可交易（含入观测），False=无论存储值如何一律排除；未提供 mask 时有效判定仅依赖数值有限性；一个单元计入有效观测当且仅当 mask[t,i]=True 且数值有限（非 NaN/inf），二者取交集；每个输出条目按 pairwise-complete（各自共同有效观测）计算，不强制所有 F 或所有 N 共享同一有效集。
  - edge cases：mask False 处存非零有限值（如填 0）→ 被 mask 排除；未 mask 处存 NaN → 被有限性排除；mask=None → 仅按有限性判定。
- **constant_column（条目级退化）**：退化判断冻结到**每个输出条目的共同有效子集**，不跨条目传播。某条目 (i,j) 在**该 pair 的共同有效观测**上，若任一操作数方差精确为零（所有共同有效值相同）→ **仅该条目及其镜像 (j,i) 输出 NaN**；其他 pair 不受影响（pairwise-complete 下同一列对不同 pair 拥有不同有效子集，可能在某 pair 常量、在另一 pair 非常量）。对角线 (i,i) 另按该序列自身有效集判断（非退化→1.0，退化→NaN）。不做任何 epsilon 截断或伪值钳制，仅在精确零方差时输出 NaN，方差非零但很小仍按正常公式计算。
  - edge cases：A=[0,0,1],B=[1,2,NaN],C=[0,1,2]——A/B 共同子集为前两行、A 常量→仅 A/B 条目 NaN，A/C 共同子集三行、A 非常量→0.866 保留；仅两值不同 → 正常 |r|=1；整列全 masked → 涉及该列的条目按各自共同子集判退化。
- **empty_or_degenerate**：输入为空集或全无效（mask 全 False、全部非有限、或所有条目有效样本 n<2）时不抛异常，返回全部条目为 NaN 的、形状正确的 float64 矩阵；只有形状、dtype、mask 形状、backend 取值等参数错误才抛异常。
  - edge cases：T·N≥1 但有效观测为 0；F=1 且全无效 → (1,1) [[NaN]]；T<1 或 N<1 或 F<1 属参数错误 → ValueError（非数据退化）。
- **output_shape_dtype**：factor_corr 输出形状 `(F,F)`、stock_corr 输出形状 `(N,N)`，dtype 恒为 float64；CPU 后端输出 numpy float64 数组，CUDA 后端输出 torch float64 张量且 mirror 输入因子张量的 device（输入 CPU→结果 CPU，输入 CUDA→结果 CUDA，仅计算在 GPU）；对角线对非退化序列精确置 1.0（显式赋值而非计算产生）；对称性通过只计算严格下三角再镜像复制到上三角实现，保证 r[i,j]==r[j,i] 逐位相等。
  - edge cases：F=1 → (1,1) [[1.0]] 或 [[NaN]]；N=1 → (1,1)；退化序列对角线 → NaN。
- **backend_dispatch**：factor_corr 与 stock_corr 各有且仅有两个后端取值：cpu（默认）与 cuda；cpu 是正式 oracle 后端（numpy float64 两遍中心化实现，始终可用、始终作为 parity 参照）；cuda 为显式 opt-in，必须在冻结验证语料上对 cpu oracle 逐元素 |Δr|≤1e-12 且重复调用逐位相同，否则不得作为该操作的后端；契约不设自动分派阈值，GPU 是否/何时更快属 PoC ② 需实测的事实，任何自动分派不得改变语义且引入前须重测 parity。注：correlation 的 backend 默认 cpu 独立于全局 device 策略缺省——CPU fallback 为正式 oracle 后端；backend='cuda' 时按全局 device 策略在目标 device 执行。
  - edge cases：小输入走 cpu 更快（契约不承诺速度）；无 cuda 设备时 backend="cuda" 抛 RuntimeError（CUDA 不可用属环境失败，全局映射统一）；backend 非法值抛 ValueError。
- **determinism**：同一后端、同一输入的任何两次调用返回逐位相同的结果；归约求和采用固定顺序（禁止 atomicAdd 或任何顺序不确定的归约，必须用固定序的分块部分和 + 固定序树归约）；对称性由三角形镜像保证；GPU 与 CPU 跨后端仅要求 ≤1e-12 容差，不要求逐位一致。
  - edge cases：并行块数随输入规模变化时归约顺序保持固定。**不承诺 permutation invariance**：共同观测的置换在实数 Pearson 上不变，但会改变浮点 mean/dot 的归约顺序，从而改变与 wrapper 的末位差——故**删除"mask 内单元重排不影响结果"承诺**；有效集成员由 mask/finite 交集唯一决定，规范观测顺序即输入的 `(t,i)` 行主序，仅字节级相同的输入承诺逐位重复（与 wrapper 的 |Δr|≤1e-12 不因置换而放松或收紧）。
- **errors**：参数错误抛确定性异常：形状不匹配（factor_corr 输入非 3-D、stock_corr 输入非 2-D、mask 形状≠(T,N)、T/N/F<1）抛 ValueError；mask 非布尔 dtype 抛 ValueError；非实数 dtype（complex、object）抛 ValueError（全局白名单统一，不归 TypeError）；backend 取值不在 {cpu, cuda} 抛 ValueError；输入数值 dtype 遵循全局单一白名单（float32/float64 接受，int/uint/bool/complex/float16 抛 ValueError）；数据退化（空/常量/全无效）不抛异常、输出 NaN。
  - edge cases：提供了 mask 但形状错 → ValueError；F=0 → ValueError；int 因子输入 → ValueError（全局白名单，不做隐式提升）。
- **adapter_alignment**：操作接收 numpy 或 torch 输入，内部一律提升为 float64 计算；输入对象只读、绝不原地修改，非 C-contiguous 输入复制为 C-contiguous 内部缓冲；mask 与数据按 `(T,N)` 轴对齐，来自 ml-quant 的 joint mask（legacy_factors.py:117 的 joint 且运算结果）原样传入；**DLPack 措辞（与 §0 统一）**：DLPack capsule 消费后由内部新 tensor 持有底层 storage 至同步返回完成，**不跨调用保留引用**（与 §0「DLPack 所有权与同步」一致，此处不再表述为"不持有所有权"）；已消费 capsule 的**二次传入抛 ValueError**（消息指明 capsule 已消费），返回后不保留任何输入引用，输出为全新对象。
  - **oracle wrapper（唯一可执行 oracle，取代一切归约算法描述）**：本操作不以"pairwise/固定树"等描述性算法作为契约——那些允许多种实现且不能保证与 numpy 一致。**唯一 oracle = 仓库内冻结 wrapper** `tests/fixtures/corr_oracle_v1.py`：对每个输出条目，按该 pair 的共同有效子集、以规范 `(t,i)` 行主序切片后，直接执行 `np.corrcoef(np.stack([xv, yv]))[0, 1]`。**具体冻结**：①wrapper 锁定 Python 3.12.7、NumPy 2.4.4 与 `np.show_config()` 记录的平台/BLAS 构建指纹；②测试只约束 wrapper 输出的最终值/NaN 与绝对误差 |Δr|≤1e-12，**不声称 GPU 内部复现 numpy 的私有归约树**（mean/dot 的具体实现属 numpy 内部，跨 BLAS 构建可能变化，oracle 测试锁定同一环境即可）；③实现的分块/归约策略是内部自由，只要最终输出满足 wrapper parity 与确定性命约（固定归约顺序，见 determinism 条款）。
  - edge cases：torch 输入在 CUDA 上但请求 cpu 后端 → values 与 mask 一并拷贝到 CPU 再算（无 device 冲突）；backend='cuda' 时 values 与 mask 须同 device，不同 → ValueError（全局规则，不自动迁移）；非连续输入（转置视图）→ 复制；names 提供但长度≠F → ValueError，不提供则按 F 索引。

### 3. rolling_ic

- **signature**：`fc.rolling_ic(factor, forward_returns, factor_mask=None, fwd_mask=None, min_valid=30, device=None) → (T,) float64`——factor 与 forward_returns 为必填 `(T,N)` 数组（numpy 或 torch），factor_mask/fwd_mask 可选 `(T,N)` bool，min_valid 可选整数（缺省 30），device 可选（**缺省 None 的语义：CUDA 可用→current_device，CUDA 不可用→自动 CPU——这是对全局 device 策略"缺省不可用即 RuntimeError"的显式例外**）；参数一律可按关键字传。**输出容器与 device（优先级表 §0 统一裁决）**：输出 device 恒 mirror factor 输入张量的 device（factor 为 CPU→输出 CPU；factor 为 CUDA→输出 CUDA）；device 参数只决定执行端，不决定输出端。容器：若输入均为 numpy 且执行端为 CPU → 返回 numpy float64 数组；若任一输入为 torch 或执行端为 CUDA → 返回 torch float64 张量。**跨 device 输入**：factor 与 forward_returns 异 device → 适配层统一拷贝到执行 device（不抛错），输出仍 mirror factor 端。收益侧参数名全局统一为 `forward_returns`（不使用 `fwd` 缩写）。
  - edge cases：factor 与 forward_returns shape 不一致 → ValueError；factor_mask/fwd_mask 提供但 shape ≠ `(T,N)` → ValueError；min_valid 缺省 30；CPU-only 环境调用且未传 device → 返回 numpy（非 RuntimeError）；显式传 device='cuda' 但 CUDA 不可用 → RuntimeError；factor CPU + returns CUDA + device 缺省 → 拷贝到执行端计算，输出 mirror factor 端（CPU）。
- **timeline_info_constraint**：信息约束只作用于因子构造：factor[t,:] 仅可使用截至 t 日收盘的可用信息构造（t 为 panel 行索引）；forward_returns[t,:] 是调用方预先生成的离线评估标签，天然使用 t 日之后价格，这是标签的固有属性而非模型输入，不构成未来函数，也不受信息约束。**契约地位（非黑盒可验证）**：时间线条款（h、lag=1、入场/出场时点、防未来函数）是**调用方语义前置条件 + 标签生成器 fixture 契约**，**不计入 rolling_ic 黑盒 oracle PASS**——算子只能消费传入的数组，无法从 factor/forward_returns 数值判别其是否由合规时间线生成（两个数值完全相同、但一个由正确 lag 流程生成的标签数组与一个错移一日的标签数组，对算子不可区分）。验收方式：标签生成器单独冻结（见 label_ownership 的 fixture 条款），在集成测试层用固定日期索引 + 固定生成脚本验证时间线；算子 oracle 仅验证数组黑盒语义。
  - edge cases：t=0 首行因子仅能用第0行及以前信息；末尾 h+lag 行 forward_returns 无完整窗口 → 无标签 → 该行 IC=NaN。
  - **标签生成器 fixture（已提交实体，本条款引用）**：本契约**不内联**标签生成实现；其唯一权威来源为已提交的 fixture 文件（仓库相对路径）：`tests/fixtures/generate_rolling_ic_labels_v1.py`（生成脚本）、`tests/fixtures/rolling_ic_labels_v1.json`（manifest：固定交易日索引、停牌/缺价规则、价格输入、h=5、lag=1、期望 forward_returns 数组、NaN 位置、脚本版本与 SHA-256）、`tests/fixtures/rolling_ic_labels_v1.npz`（期望标签数据）。rolling_ic 的 h/入场/出场时间语义的机械验收**仅经由**该 fixture 的集成测试达成：CI 先验 manifest hash，再运行生成脚本并逐元素比较（见 label_ownership）。
- **horizon_h**：h 为调用方在标签生成期设定的正整数（单位=交易日，与 panel 时间轴一致），取值范围 {1,2,...}；rolling_ic 的 API 不接收 h 参数，h 的数值被烘焙进 forward_returns[t] 的生成过程，算子对 h 取值不敏感；契约按此定义 forward_returns 的语义，oracle 与 parity 测试按同一 (h, lag=1) 约定生成标签。
  - edge cases：h=1（次日收益，最小情形）；h 大于剩余窗口导致末尾若干行无标签；h 极大使序列几乎全 NaN——仍合法，非错误。
- **execution_lag**：成交时点固定：入场= t 之后第 1 个交易日（lag=1，等价于 equity-factor-lab 的 skip_days=1）的收盘价；lag 冻结为常量 1 交易日，不开放为参数。
  - edge cases：t 之后无交易日（panel 末行）→ 无入场 → 无标签 → IC=NaN；停牌/缺失价导致入场价缺失（调用方在标签中置 NaN）→ 该单元无效。
- **return_interval**：收益区间固定：出场= (t+h) 之后第 1 个交易日（即 t+h+lag = t+h+1，lag=1）收盘；forward_returns[t] = 出场收盘/入场收盘 − 1（累计简单收益）；若出场日索引超出 panel 末行，该行无标签，视为无效。
  - edge cases：末尾 h+lag 行窗口不完整 → 无标签 → 该行 IC=NaN；入场或出场价格为 0/NaN → 标签 NaN → 该单元无效。
- **axis**：计算轴固定为股票维：对每个时点 t，取 factor[t,:] 与 forward_returns[t,:]（跨全部 N 列）在有效子集上计算截面 Spearman；输出行 t 与输入行 t 严格对齐；秩升序，最小值为秩 1，无升降序选项。
  - edge cases：N=1（单只股票）→ 每行有效数 ≤1 < min_valid(≥2) → 全 NaN；不同 t 行有效股票数不同——逐行独立判定，互不影响。
- **spearman_rank_ties**：秩采用 stable ordinal，与 cross_sectional_rank 冻结的语义完全一致（常量截面处理为例外：本操作经显式前置分支输出 NaN，见 constant_all_invalid，与 cross_sectional_rank 的 constant_section 互异秩 1..K 语义不同）：升序，最小值为秩 1；并列值按原始列索引升序（stable）分配连续整数秩 1..m；秩为精确整数，只用精确比较、不用浮点容差；仅在每行有效子集上定秩。tie 在**接收值域**上判定：rolling_ic 不执行 cs_rank 的 float32 下转规范化，float64 输入按 float64 原值精确判 tie（同一 float64 输入经两操作可能得到不同 tie 模式，属契约语义而非缺陷）。IC[t] = 有效子集上两组 ordinal 秩的 Pearson 相关（即 Spearman）。
  - edge cases：常量截面（有效因子或有效收益全等值）→ 显式前置分支输出 NaN（见 constant_all_invalid，不走到秩计算）；部分并列 → 并列组内按原始列索引序分配连续秩；单调递减关联 → IC 为负值（[-1,0)）。
  - oracle 口径：本操作 oracle = 同一 stable ordinal 秩对在 float64 下做 Pearson（与契约同实现）；本地 equity-factor-lab 仓库的 `factorlab/crosssection/ic.py` 仅作 min_stocks=30、valid=notna 交集、不足置 NaN 的**设计依据锚点**（本地开发引用，非发布契约）；tie 语义不取自 scipy.spearmanr（其 average 秩在并列/常量输入上与 ordinal 偏离 >1e-12，故与 scipy 不具 parity，属显式偏离）。
- **nan_mask_intersection**：每单元有效的充分必要条件：isfinite(factor[t,i]) 且 isfinite(forward_returns[t,i]) 且 (factor_mask[t,i]，缺省视为 True) 且 (fwd_mask[t,i]，缺省视为 True)；±inf 视为非有限即无效；mask 为 `[T,N]` bool，True=有效。
  - edge cases：mask 全 False 的行 → 有效数 0 → IC=NaN；仅 factor 或仅收益有 NaN → 交集剔除；±inf 单元 → 无效；mask 缺省（未传）→ 视为全 True。
- **min_valid_stocks**：min_valid 为命名参数，缺省 30（参照 equity-factor-lab 的 min_stocks=30，ic.py:12,20-22）；取值须为整数且 ≥2，否则 ValueError；某行有效单元数 < min_valid 时 IC[t]=NaN（非错误）；min_valid 大于 N 时输出全 NaN（非错误）。
  - edge cases：min_valid=2（最小合法值）；有效数恰等于 min_valid → 正常计算；min_valid > N → 全 NaN；min_valid=1 或非整数 → ValueError。
- **constant_all_invalid**：常量与全无效截面一律输出 IC[t]=NaN，不抛错误。全无效=有效数 < min_valid；常量=有效子集上因子或收益全部等值——以显式前置分支在 ordinal 定秩前判定并输出 NaN（这是对纯 ordinal 的显式偏离：ordinal 下常量截面本可得 1..m 互异秩，但无信息截面的 Spearman 语义上未定义）。固定反例（写入 oracle 测试）：T=2、N=100、mask 全 True、min_valid=30，行 t 因子=0.5 常量、收益=0.1 常量 → IC=NaN。
  - edge cases：因子整行同一值；收益整行同一值（如全 0）；有效数=1（min_valid≥2 强制下必被 NaN 分支捕获）；双常量截面 → NaN（非 +1.0）。
- **output_shape**：输出为 `(T,)` float64 一维序列，行 t = 时点 t 的截面 IC；因不做滚动窗口聚合，输出长度固定为 T（不是 T-W+1），与输入行严格对齐，末尾无标签行以 NaN 占位。
  - edge cases：T=1（单日）→ 输出长度 1；末尾无标签行 → NaN 占位保持长度 T。
- **rolling_aggregate**：不做任何滚动/窗口聚合：输出就是每日截面 Spearman 原始序列；对 IC 序列再做滚动均值/方差/ICIR 等属于调用方后处理，超出本操作范围。
  - edge cases：无——本操作无窗口概念。
- **dtype**：输入 factor 与 forward_returns 接受 float32 或 float64 的 `(T,N)` 二维数组，其余 dtype → ValueError（遵循全局单一白名单）；mask 为 bool `(T,N)`，其余 → ValueError；输出恒为 float64 `(T,)`；秩内部用精确整数，Pearson 相关在 float64 下计算以保证 oracle parity。
  - edge cases：int 数组输入 → ValueError（不隐式转换）；float16/bfloat16 → ValueError；整数型输入无 NaN（整数无 NaN 概念）。
- **determinism**：算子逐位确定：相同输入（含 mask 与参数）与相同设备下，多次调用输出逐位一致，且与 block/grid 配置、stream 数、调度顺序无关；秩用 stable ordinal（索引序 tie-break）保证，Pearson 归约使用固定次序（固定归约树），禁止以依赖调度顺序的非确定性原子操作累积结果。
  - edge cases：同设备不同 launch 配置 → 逐位一致；跨设备不做 bitwise 承诺（仅同设备内承诺）。
  - **验收三层分离（全契约适用）**：①黑盒语义测试（输出断言——可观察行为）；②静态/IR 检查（源码级约束——ddof=1 的协方差约定、两遍中心化、只算下三角镜像、禁 atomicAdd、固定归约树、factor_corr/stock_corr 互不调用）；③运行插桩（allocator/stream/cudaEvent/NVTX 剖面——H2D 恰一次、D2H 顺序、同步边界）。ddof=1 因 Pearson 中 n 与 n−1 消去而不可从输出观察，故其验收归静态检查②；仅依赖黑盒测试会漏检"实现不合约但数值全绿"的情况。
- **errors**：错误类型明确：factor/forward_returns 非二维或 shape 非 `(T,N)` → ValueError；factor 与 forward_returns shape 不一致 → ValueError；mask shape ≠ `(T,N)` → ValueError；mask 非 bool → ValueError；factor/forward_returns 非 float32/float64 → ValueError；min_valid 非整数或 <2 → ValueError；T=0 或 N=0 → ValueError。数值层面（NaN/常量/全无效/全 NaN 输入）不抛错，按规则输出 NaN。
  - edge cases：T=0/N=0 空面板；factor 一维 vs forward_returns 二维；factor_mask/fwd_mask 与输入 shape 错位。
- **label_ownership**：forward_returns 由调用方生成并经输入适配层传入；factor-cuda 只消费、不生成、不校验其数值是否与 (h, lag=1) 约定一致（无法从标签逆推 h）。**fixture 契约（实体已提交）**：时间线语义的机械验收不依赖算子，而依赖已提交的**标签生成器 fixture**（`tests/fixtures/generate_rolling_ic_labels_v1.py` + `rolling_ic_labels_v1.json` manifest + `rolling_ic_labels_v1.npz`，见 timeline_info_constraint 的 fixture 条款）——manifest 冻结交易日索引、停牌/缺价规则、价格输入、h=5、lag=1、期望 forward_returns 数组、NaN 位置、脚本版本与 SHA-256；集成测试先验 hash、再跑生成脚本、逐元素断言输出等于冻结 npz。契约对标签语义的约定是文档化约定，供调用方与 oracle/parity 测试遵循。
  - edge cases：调用方传错 h 的标签 → 数值错误但算子不报错（契约外误用）；标签与 factor 行未对齐 → shape 相同但语义错位（调用方责任）。

### 4. parameter_scan

- **domain**：parameter_scan 的 PoC ① 扫描对象集合冻结为单一操作 cross_sectional_rank，可扫描轴仅两项：direction ∈ {ascending, descending}（ascending=秩1给最小有效值；descending=对输入取负后走 ascending，秩1给最大有效值）、mask_mode ∈ {masked, unmasked}（masked=应用 rank 契约冻结的 mask 参与规则，仅 mask==True 单元参与且其余输出 NaN；unmasked=不读取不校验 mask，全部有限单元参与，非有限值排除与输出 NaN 仍按 rank 契约 NaN 规则）；轴规约可省略任一轴，省略轴取契约默认值（direction 默认 ascending，mask_mode 默认 masked）；correlation（factor_corr/stock_corr）与 rolling_ic 的参数在其各自契约中已冻结为单值，不在本扫描域内；域（新增操作或新增轴）的任何扩展须走 HG-2 变更。**spec 语义（user_spec 规范化→effective spec）**：**用户输入（user_spec）在校验后先规范化到契约定义的冻结轴序（direction 在前、mask_mode 在后）**，再用规范化后的 effective spec 统一做组合展开、计时与输出。`spec` 输出恒为 effective spec（含完整两轴绑定，轴序固定 direction→mask_mode）；各组 `axis_values` 恒含完整两轴绑定且键序与 spec 一致。user_spec 不单独作为输出字段；group_index 与组序一律按 effective spec 的 canonical 序。
  - edge cases：轴名不在冻结域内（如 lookbacks、top_ns 等因子/选股参数）→ ValueError 并列出允许轴名；轴值不在该轴允许枚举内 → ValueError 并列出允许值；轴规约中出现重复轴名 → ValueError；任一轴的值列表为空 → ValueError；省略全部轴 → 等价于单组、全默认（ascending,masked）。
- **expansion**：组合展开固定为各轴值列表的笛卡尔积：每组由各轴各取一个值唯一确定，组数 G = 各轴长度之积；**组序按 effective spec（规范化后的冻结轴序）的字典序展开**（首个轴 direction 变化最慢，mask_mode 次之）；禁止自定义组合器或用户指定子集；任一轴值列表为空或 G==0 → ValueError。
  - edge cases：单轴扫描 → G=len(values)；多轴扫描 → G=各轴长度之积；轴值列表含重复值 → ValueError；**用户颠倒轴给定顺序（如 mask_mode 在前）→ 先规范化再展开，组序与 group_index 不受用户轴序影响（与未颠倒完全一致）**。
- **input**：单次扫描接受且只接受一个输入张量 X（`(T,N)`）与一个可选 mask（`(T,N)` bool，True=可交易，语义取自 ml-quant Panel mask）；**不接收 device 参数**——执行端恒为 torch.cuda.current_device()（GPU 计时语义，见 device 策略例外②）；X 经适配层规范为 float32（接受 float32/float64，float64 下转；非 C-contiguous 由适配层自动连续化复制，见全局 dtype/内存布局）；X.shape[0]>=1 且 X.shape[1]>=1，T<1、N<1、shape/dtype 不符或 mask 与 X 的 shape 不一致 → ValueError；mask_mode=masked 时 mask 必须提供否则 ValueError；mask_mode=unmasked 时不读取也不校验 mask；GPU 侧只上传一次 X 与 mask（不逐组上传）；`(T,N,F)` 多因子张量不被 PoC ① 接受，调用方按 F 在 Python 层逐因子循环调用。**mask 校验 override（对全局适配层校验的显式例外）**：仅当**有效组合中含 masked 组**（扫描的任一组合使用 mask）时，mask 才被完整校验（dtype/shape/device）且须提供；**全部组合均为 unmasked** 时，mask 参数被完全忽略——不校验 dtype/shape/device，即使传非 bool/错误 shape/非 tensor 也不抛错，等价于未传。混合 masked/unmasked 扫描：mask 在任何组执行前按 masked 组的需要完成校验（扫描级，见 failure）。
  - edge cases：T=0 或 N=0 → ValueError；mask shape 与 X 不一致 → ValueError（仅 masked 模式；全 unmasked 时被忽略）；X 含 NaN/Inf：非错误，由 rank 契约 NaN 规则处理（排除+输出 NaN）；mask_mode=masked 但 mask 缺失 → ValueError；X 非 C-contiguous → 适配层自动连续化（非错误）。
- **steps**：每组 g（字典序索引 0..G-1）执行固定步骤：①从轴规约物化该组参数绑定；②在该绑定下调用 cross_sectional_rank 内核（descending=对输入取负后执行 ascending 冻结语义；unmasked=内部以全有效掩码跳过 mask 应用）；③用 cudaEvent 记录设备端起止并同时记录 wall-clock 组耗时；④D2H 复制该组结果并按字典序追加到输出；所有组共享同一次 H2D 上传（上传耗时计入 elapsed_ms 但不计入任何单组 time_ms）；组间语义上严格串行且记录顺序固定，任何多流并行（PoC ② 性能手段）不得改变任一组输出或组记录顺序。
  - **计时定义（时间线）**：`time_ms`（wall-clock）= 本组步骤②（kernel launch）开始到步骤④（D2H 完成 + cudaDeviceSynchronize）之间的 host 墙钟毫秒，含本组 kernel 与 D2H、不含共享 H2D；`time_gpu_ms`（cudaEvent）= 步骤②内核 launch 起点 event 到内核完成 event 的设备端毫秒，不含 D2H。`total_time_ms` = 各 ok 组 time_ms 之和（不含 failed 组）；`total_time_gpu_ms` = 各 ok 组 time_gpu_ms 之和；`elapsed_ms` = 整次扫描 wall-clock（从入口到返回，含 H2D、全部组、结果聚合）。**failed 组计时**：恒为 `0.0`（time_ms 与 time_gpu_ms 均为 0，不取失败前部分值）。**error_stage（固定字段，非可选）**：`error_stage` 是每个组记录的固定键——`status="ok" → error_stage=None`；`status="failed" → error_stage="launch"`（唯一合法值，因组级降级白名单仅含内核 launch 检查点错误，见 failure 条款）；event/d2h/alloc/unknown 阶段失败均属扫描级 RuntimeError（不返回 groups），故不得作为组级 error_stage 合法值。
  - edge cases：G 很大时分批执行：为 PoC ③ 显存模型内的内部行为，语义透明、不改变输出 schema 与组序；单组内核失败：该组置 failed，后续组照常执行；计时数值属测量，允许运行间波动；结果与组序不因测量而异。
- **output**：返回单个 dict：{"spec": effective spec（轴名+值列表，**轴序 = direction 在前、mask_mode 在后**，包含全部已冻结轴的完整有效绑定；不暗示任何容错去重——重复轴名/重复轴值已在 failure 条款抛 ValueError），"groups": 长度为 G 的组记录列表，"summary": 汇总 dict}；**组序/group_index 唯一依据 = effective spec 的 canonical 序（用户轴序在规范化后不再影响任何输出字段）**。每个组记录固定字段：group_index（0-based 字典序 int）、axis_values（该组 {轴名:值} 绑定）、result（该组操作输出；PoC ① 下为 `(T,N)` float32，shape/dtype 与单次 cross_sectional_rank 冻结输出一致，parameter_scan 不重定义）、status（"ok"|"failed"）、error（None 或错误消息 str）、time_ms（wall-clock float 毫秒）、time_gpu_ms（cudaEvent 设备耗时 float 毫秒）；summary 固定字段：total_groups（int）、n_failed（int）、total_time_ms（float）、total_time_gpu_ms（float）、elapsed_ms（整体 wall-clock float）；组记录严格按字典序排列，禁止乱序；result 恒为 `(T,N)`，direction 与 mask_mode 不改变输出 shape。**各组 result 恒为 CPU**（`(T,N)` float32，D2H 物化）——parameter_scan 对全局 device 策略的显式例外（GPU 计时语义 + host 端结果聚合；result 不 mirror 输入 device）。
  - edge cases：n_failed>0 时其余 ok 组结果仍有效返回；某组失败时该组 result 为 None；全部组失败 → 返回 G 个 failed 组且 summary.n_failed==G；result 的 dtype/NaN 语义沿用 cross_sectional_rank 契约，parameter_scan 不改变。
- **failure**：失败分两级。**扫描级错误**（在任一组执行前抛异常且不返回任何部分结果）：①结构错误，异常类型逐项映射——未知轴名、轴值不在允许枚举、轴值重复、空轴 → ValueError；X/mask 的 shape/dtype 不符 → ValueError；masked 模式缺 mask → ValueError；T<1 或 N<1 → ValueError；N>2^24 → ValueError（与 rank 契约一致）；②环境错误——GPU 不可用、cudaMalloc/显存不足、context loss、illegal address、device assert、launch failure → RuntimeError（致命，终止整批，不返回部分结果）。**组级运行错误（白名单，穷尽枚举）**：仅当单个**内核 launch** 在**明确的 launch 检查点同步捕获**、且满足以下任一 error code 时，才降级为该组 status="failed"：`cudaErrorInvalidConfiguration`（设备永远无法满足的 launch 配置）；`cudaErrorLaunchOutOfResources`（launch 资源/配置错误）。**禁止**将以下归为组级：`cudaErrorInvalidValue`（通用 API 参数错误——须按发生 API 分类，发生在 setup/事件/D2H/allocator 一律扫描级 RuntimeError）；异步错误（illegal address、device assert、launch failure）须**扫描级终止**（context 已不安全，继续启动后续组在运行时层面不可行）；未列入白名单的任何 CUDA 错误一律扫描级 RuntimeError。**同步检查点**：每组在 launch 后、进入下一组前执行 cudaGetLastError + 同步，确保错误归属到正确的组；仅凭 D2H 或下一组上报的错误不得降级继续。**override 声明**：本契约对全局错误映射（§0 ④ CUDA 失败→RuntimeError）的覆盖范围=仅限上述白名单（InvalidConfiguration/LaunchOutOfResources 于内核 launch 检查点）；其余一律扫描级 RuntimeError。setup/H2D/事件创建/D2H/结果分配等**非内核**步骤失败均按致命错误处理（RuntimeError，非组级）。扫描从不静默丢弃任何组。
  - edge cases：全部组失败（G 个 failed，n_failed==G，仅当全部为可恢复错误）；扫描级错误时 summary/groups 均不返回（调用方收到异常）；组级错误不影响其他组结果与记录顺序；context loss 后继续启动后续组在运行时层面不可行 → 必须扫描级终止。
- **determinism**：在相同输入、相同轴规约、相同硬件下，parameter_scan 每次调用的组顺序、每组 axis_values 与每组 result 逐位一致（操作内核确定、无随机数、归约顺序固定；稳定序秩 argsort 本身确定）；计时数值属测量允许运行间波动，但组记录顺序与结果不变。**跨 device bitwise 承诺：继承 cross_sectional_rank**——因 rank 输出为 1..K 整数秩（float32 可精确表示）与固定 NaN 载荷 0x7fc00000，与 rank 契约的跨 device 位级一致承诺一致，本操作**不降级**：相同输入+相同轴规约在任意 GPU 上逐位一致（结果无浮点归约，故无跨硬件位差来源）。
  - edge cases：重复调用同输入同硬件 → 结果逐位一致；同输入不同 GPU → 逐位一致（继承 rank，非仅语义等价）；计时量不参与确定性承诺。
- **example（最小可执行示例，冻结）**：fc.parameter_scan(axes=[("direction",["ascending","descending"]),("mask_mode",["masked","unmasked"])], X=X, mask=mask)，其中 X 为 `(T,N)` float32（或 float64，适配层下转）、mask 为 `(T,N)` bool；G=4，组序固定为 (ascending,masked)、(ascending,unmasked)、(descending,masked)、(descending,unmasked)；每组 result 为 `(T,N)` float32 秩矩阵：ascending 秩1给最小有效值、descending（取负后）秩1给最大有效值，masked 仅 mask==True 单元参与且其余输出 NaN，unmasked 忽略 mask 由全部有限单元参与；该语义与 QuantGplearn `_rank_pct_dim`（torch_functions.py:121-129）的**秩排序语义一致（stable ordinal + NaN 排除），非其 percentile 数值输出**（本契约 result 为整数秩 1..K，属 output_normalization 的显式归一化偏离）；与 `_rank_pct`（functions.py:142-149）的一致仅限"空掩码返回全 NaN 而非报错"路径。
  - edge cases：masked 模式下某截面 mask 全 False：该截面结果全 NaN（非错误，与 functions.py:146-147 空掩码返回全 NaN 一致）；unmasked 模式下 X 含 NaN 的单元：仍按 rank 契约 NaN 规则排除并输出 NaN；axis_values 顺序与组记录一一对应，调用方可直接按键取值。

## PoC 决策表（唯一真源）

> 跨文件（PLAN/RISK/竞品分析）对 PoC 判据的表述以本表为准；其他文档只引用、不重复定义。

| 验证项 | PASS | FAIL → 动作 | 最多重试 | 证据产物 |
|-------|------|------------|---------|---------|
| ① 语义 | 契约全部可执行，oracle 测试过 | 契约不可运行 → REDESIGN | 2 | 冻结契约 + oracle 测试 |
| ② 公平基线 | 相对最佳免费替代端到端 ≥2×（同数据同 mask **且同语义**） | <2× → STOP | 1 | 性能对比表 + 原始计时 + corpus hash |
| ③ 显存 | 峰值 ≤ 可用显存 − 安全余量，且可分块 | 超限不可分块 → REDESIGN | 2 | 字节级峰值模型 + cudaMemGetInfo 实测（预测 vs 实测偏差） |
| ④ 端到端 | 端到端收益成立（含传输/归并，非单算子） | 不成立 → STOP | 1 | 端到端耗时分解 |

GO = ①∧②∧③∧④ 全 PASS（人裁决）。NRR 阈值 ≥5× 预注册；未达记负结果 NRR。安装成功率：干净环境 ≥80%（Windows/Python 3.12 矩阵，PoC 实测）。

## 成功标准（S5）

- **主指标**：端到端加速比（同数据同 mask **同语义** vs 最佳免费替代）——目标 ≥5×，最低可接受 ≥2×
- **辅指标**：契约 oracle 测试全过；显存峰值 ≤ 可用显存−安全余量；干净环境安装成功率 ≥80%
- **反例**（不算成功）：只加速单次小截面、无参数扫描收益；用不可复现 corpus 得出的性能；随机数降级 corpus 的基准；混入语义差异（ordinal vs average）的加速比

## 评估计划（S6）

- 三层分工：**AI 自评每轮 + 独立模型里程碑审查 + 人类关键决策**（PoC 三态裁决由人拍板）
- 已执行：① GPT-5.6-Sol 首轮审查 v1（21 条）→ PLAN v2；② 复核 v2+CLAUDE.md（17 条，`reviews/plan_claude_md_review_gpt56sol_2026-07-31.md`）→ 本文件降 DRAFT；③ PoC ① 契约冻结：内部 2 轮（30 条）+ GPT-5.6-Sol 3 轮独立复核（34 条）闭合 → **2026-08-03 HG-2 批准恢复 L0 Spec**
- 里程碑：PoC ①语义 / ②公平基线 / ③显存 / ④端到端

## 可复现性（S9）

- Python 3.12.7；**corpus manifest 已提交**（2026-08-03）：`benchmark_corpus/`——生成器/校验器/加载器/统计脚本 + manifest JSON（含 data_sha256 + array_sha256）+ seeds.json（种子单一真源：MASTER_SEED=20260802 + SHA-256 无损派生 `role_rng(master,role,extra)=default_rng(frombuffer(SHA-256(...),uint32))`，禁复用 fixture 种子 20260803）+ smoke/parity 锚点。完整 npz（real/synth 1218×5000×12）不提交（.gitignore），由生成脚本确定性复现 + data_sha256 锚定
- 性能基准：环境元数据（CPU/GPU/驱动/时钟）+ `cudaEvent`+wall-clock 双口径 + warmup/重复/种子 + 冷缓存/驻留区分 + 异常值/置信区间；corpus 经 `corpus_loader_v1.py` 唯一读入口（校验 data_sha256）
- 发布前 pip freeze + 数据快照（记录来源/获取日期/hash；原始数据如因许可不入库，记录派生过程与生成脚本）

## 已知坑位

- **RTX 4060 FP64 吞吐 ≈ FP32 的 1/64**——DGEMM 相关性可能慢于 CPU numpy/OpenBLAS；`correlation_matrix` 的 CPU fallback 是正式后端之一（端到端影响待 PoC ② 实测）
- `mlquant.factors` 不存在（真实 `mlquant.features`，返回 Tensor 三值）
- `CUDA_ARCHITECTURES "89"` 只覆盖 Ada——**非多架构方案**；**支持矩阵已定义（单一真源）**：`docs/support_matrix.json`（CUDA 13.3 实测/MSVC 19.51/Python 3.12.7/CC 8.9 实测，声明范围见矩阵）；版本号从矩阵导出，不在 README 等重复硬编码（2026-08-08 闭合 P1）
- GPU ordinal rank ≠ pandas average rank——**公平基线须先 contract parity 再比性能**（QuantGplearn：Torch=ordinal / NumPy=average）
- **CMake policy 注意**：CMP194 **不存在**；CMP0194 是 Windows ASM policy（非 CUDA 探测）。历史 VS generator CUDA 探测失败为**限定事件**（根因未确认；2026-07-31 全新目录 + VS generator 已复现成功）；Ninja 为推荐路径
- **nvcc + UTF-8 中文注释**：特定字符（如 U+4E57，GBK 尾字节 0x5C）在 CP936 解析下吞换行致 LNK1561；ASCII-only 为保守策略，或统一源/执行字符集 UTF-8（`/utf-8`）已验证可行——**非所有中文必然失败**

## 更新协议

- 本文件是 **Spec（已冻结，L0）**：任何变更走 HG-2（AI 建议 → 人批准），不区分大小改
- **PoC ① 契约冻结已完成**（2026-08-03 HG-2 批准）；操作语义契约为冻结版本，变更须走 HG-2
- 更新后同步 `CHANGELOG.md`；只在操作指令变更时修改（版本历史由 git log / CHANGELOG 追踪）
