# factor-cuda 实现设计（Implementation Design）v0.7

> 状态：**v0.7 自包含实现规格**——GPT-5.6-Sol 六审 15 项 finding 自修复 + 异后端核对闭合
> 生成模型：DeepSeek-V4-Flash (via Claude Code CLI) · 2026-08-04，星期二，Asia/Hong_Kong
> 审查依据：`reviews/implementation_review_gpt56sol_review6_2026-08-04.md`（裁决需修复，非 REDESIGN）· 自修复由 GPT-5.6-Sol 执行 + Claude 异后端核对
> 语义判据单一真源：CLAUDE.md L0 Spec + `tests/fixtures/corr_oracle_v1.py` + `tests/fixtures/corr_math_v1.py`（correlation 数学单一真源）
> 本文档为**自包含**规范：所有 normative 语义在本版本内可解析，不引用旧版本正文

---

## 0. 状态与闭合声明（implementation-status-v1）

<!-- implementation-status-v1:begin -->
```json
{
  "schema_version": "1.0.0",
  "document_version": "0.7",
  "execution_date": "2026-08-04",
  "timezone": "Asia/Hong_Kong",
  "findings": [
    {
      "id": "F5-01",
      "status": "closed",
      "evidence": [
        {
          "path": "tests/fixtures/validate_self_fix_v1.py",
          "sha256": "33c99b28def1f035c03f36f1f1c42ff733071b79653aa54d506e9e483b2c3d57",
          "command": "python tests/fixtures/validate_self_fix_v1.py"
        },
        {
          "path": "tests/fixtures/corr_math_v1.py",
          "sha256": "70cb26d773765887c46bf82a5b6283a1da889e8b542ea78ef74968c04095e2a4",
          "command": "python tests/fixtures/validate_self_fix_v1.py"
        }
      ]
    },
    {
      "id": "F5-02",
      "status": "closed",
      "evidence": [
        {
          "path": "tests/fixtures/validate_self_fix_v1.py",
          "sha256": "33c99b28def1f035c03f36f1f1c42ff733071b79653aa54d506e9e483b2c3d57",
          "command": "python tests/fixtures/validate_self_fix_v1.py"
        },
        {
          "path": "tests/fixtures/corr_math_trace_v1.json",
          "sha256": "1c8f910457ae53a35487b39ef14772971fe1ae8eb2bab3774206e0a82c027ea2",
          "command": "python tests/fixtures/validate_self_fix_v1.py"
        }
      ]
    },
    {
      "id": "F5-03",
      "status": "closed",
      "evidence": [
        {
          "path": "tests/fixtures/validate_self_fix_v1.py",
          "sha256": "33c99b28def1f035c03f36f1f1c42ff733071b79653aa54d506e9e483b2c3d57",
          "command": "python tests/fixtures/validate_self_fix_v1.py"
        },
        {
          "path": "benchmarks/compute_workspace_v1.py",
          "sha256": "a47bd35afd20a53b8bc5ce989481681c6cc2fcaaa0e56f109ed362c2104d9ebf",
          "command": "python tests/fixtures/validate_self_fix_v1.py"
        }
      ]
    },
    {
      "id": "F5-04",
      "status": "closed",
      "evidence": [
        {
          "path": "tests/fixtures/validate_self_fix_v1.py",
          "sha256": "33c99b28def1f035c03f36f1f1c42ff733071b79653aa54d506e9e483b2c3d57",
          "command": "python tests/fixtures/validate_self_fix_v1.py"
        },
        {
          "path": "tests/fixtures/corr_corpus_v1.manifest.json",
          "sha256": "5190747494f7052494ca93b3c3a68e365848d5127cd06de4dcc636b7044ea0f5",
          "command": "python tests/fixtures/validate_self_fix_v1.py"
        }
      ]
    },
    {
      "id": "F5-05",
      "status": "closed",
      "evidence": [
        {
          "path": "tests/fixtures/validate_self_fix_v1.py",
          "sha256": "33c99b28def1f035c03f36f1f1c42ff733071b79653aa54d506e9e483b2c3d57",
          "command": "python tests/fixtures/validate_self_fix_v1.py"
        },
        {
          "path": "docs/workspace_v1.json",
          "sha256": "a68781bac584ec58c15377d163803fd42704854a39867a132ef1b588b0b00d4e",
          "command": "python tests/fixtures/validate_self_fix_v1.py"
        }
      ]
    },
    {
      "id": "F5-06",
      "status": "closed",
      "evidence": [
        {
          "path": "tests/fixtures/validate_self_fix_v1.py",
          "sha256": "33c99b28def1f035c03f36f1f1c42ff733071b79653aa54d506e9e483b2c3d57",
          "command": "python tests/fixtures/validate_self_fix_v1.py"
        },
        {
          "path": "docs/workspace_v1.json",
          "sha256": "a68781bac584ec58c15377d163803fd42704854a39867a132ef1b588b0b00d4e",
          "command": "python tests/fixtures/validate_self_fix_v1.py"
        }
      ]
    },
    {
      "id": "F5-07",
      "status": "closed",
      "evidence": [
        {
          "path": "tests/fixtures/validate_self_fix_v1.py",
          "sha256": "33c99b28def1f035c03f36f1f1c42ff733071b79653aa54d506e9e483b2c3d57",
          "command": "python tests/fixtures/validate_self_fix_v1.py"
        },
        {
          "path": "benchmarks/compute_workspace_v1.py",
          "sha256": "a47bd35afd20a53b8bc5ce989481681c6cc2fcaaa0e56f109ed362c2104d9ebf",
          "command": "python tests/fixtures/validate_self_fix_v1.py"
        }
      ]
    },
    {
      "id": "F5-08",
      "status": "closed",
      "evidence": [
        {
          "path": "tests/fixtures/validate_self_fix_v1.py",
          "sha256": "33c99b28def1f035c03f36f1f1c42ff733071b79653aa54d506e9e483b2c3d57",
          "command": "python tests/fixtures/validate_self_fix_v1.py"
        },
        {
          "path": "tests/fixtures/corr_math_v1.py",
          "sha256": "70cb26d773765887c46bf82a5b6283a1da889e8b542ea78ef74968c04095e2a4",
          "command": "python tests/fixtures/validate_self_fix_v1.py"
        }
      ]
    },
    {
      "id": "F5-09",
      "status": "closed",
      "evidence": [
        {
          "path": "tests/fixtures/validate_self_fix_v1.py",
          "sha256": "33c99b28def1f035c03f36f1f1c42ff733071b79653aa54d506e9e483b2c3d57",
          "command": "python tests/fixtures/validate_self_fix_v1.py"
        },
        {
          "path": "tests/fixtures/test_cases_v1.json",
          "sha256": "255243001a8df5f9c176baa968f88b0b0fd594622a4da32ceda71158de94f22e",
          "command": "python tests/fixtures/validate_self_fix_v1.py"
        }
      ]
    },
    {
      "id": "F5-10",
      "status": "closed",
      "evidence": [
        {
          "path": "tests/fixtures/validate_self_fix_v1.py",
          "sha256": "33c99b28def1f035c03f36f1f1c42ff733071b79653aa54d506e9e483b2c3d57",
          "command": "python tests/fixtures/validate_self_fix_v1.py"
        },
        {
          "path": "docs/gate_config_v1.json",
          "sha256": "5f2b43079d58d2d11241280853eac331583cafa29d955ba4c5a1bb7ab8f3fe31",
          "command": "python tests/fixtures/validate_self_fix_v1.py"
        }
      ]
    },
    {
      "id": "F5-11",
      "status": "closed",
      "evidence": [
        {
          "path": "tests/fixtures/validate_self_fix_v1.py",
          "sha256": "33c99b28def1f035c03f36f1f1c42ff733071b79653aa54d506e9e483b2c3d57",
          "command": "python tests/fixtures/validate_self_fix_v1.py"
        },
        {
          "path": "tests/fixtures/corr_math_v1.py",
          "sha256": "70cb26d773765887c46bf82a5b6283a1da889e8b542ea78ef74968c04095e2a4",
          "command": "python tests/fixtures/validate_self_fix_v1.py"
        }
      ]
    },
    {
      "id": "F5-12",
      "status": "closed",
      "evidence": [
        {
          "path": "tests/fixtures/validate_self_fix_v1.py",
          "sha256": "33c99b28def1f035c03f36f1f1c42ff733071b79653aa54d506e9e483b2c3d57",
          "command": "python tests/fixtures/validate_self_fix_v1.py"
        },
        {
          "path": "tests/fixtures/calibration_trace_v1.json",
          "sha256": "fe27ae06fff0e82763443dde4102692739c477b6b20f1fefb554ea1950c98ccd",
          "command": "python tests/fixtures/validate_self_fix_v1.py"
        }
      ]
    },
    {
      "id": "F5-13",
      "status": "closed",
      "evidence": [
        {
          "path": "tests/fixtures/validate_self_fix_v1.py",
          "sha256": "33c99b28def1f035c03f36f1f1c42ff733071b79653aa54d506e9e483b2c3d57",
          "command": "python tests/fixtures/validate_self_fix_v1.py"
        },
        {
          "path": "tests/fixtures/validate_implementation_v1.py",
          "sha256": "b586ce06819916e22b1ee7562aaf12d524c56078118484bc87c32fb1d0a934af",
          "command": "python tests/fixtures/validate_self_fix_v1.py"
        }
      ]
    },
    {
      "id": "N6-01",
      "status": "closed",
      "evidence": [
        {
          "path": "tests/fixtures/validate_self_fix_v1.py",
          "sha256": "33c99b28def1f035c03f36f1f1c42ff733071b79653aa54d506e9e483b2c3d57",
          "command": "python tests/fixtures/validate_self_fix_v1.py"
        },
        {
          "path": "docs/workspace_v1.json",
          "sha256": "a68781bac584ec58c15377d163803fd42704854a39867a132ef1b588b0b00d4e",
          "command": "python tests/fixtures/validate_self_fix_v1.py"
        }
      ]
    },
    {
      "id": "N6-02",
      "status": "closed",
      "evidence": [
        {
          "path": "tests/fixtures/validate_self_fix_v1.py",
          "sha256": "33c99b28def1f035c03f36f1f1c42ff733071b79653aa54d506e9e483b2c3d57",
          "command": "python tests/fixtures/validate_self_fix_v1.py"
        },
        {
          "path": "tests/fixtures/validate_implementation_v1.py",
          "sha256": "b586ce06819916e22b1ee7562aaf12d524c56078118484bc87c32fb1d0a934af",
          "command": "python tests/fixtures/validate_self_fix_v1.py"
        }
      ]
    }
  ]
}
```
<!-- implementation-status-v1:end -->

---

## 1. 范围与输出契约（N6-01 修正）

本设计覆盖 PoC ③ 五操作的 CUDA 内核 + 显存模型。**输入/输出契约按 L0 冻结，不修改 CLAUDE.md**：

- **backend='cpu'**：NumPy 输入，结果 `NumPy float64` 容器，输出镜像 CPU host
- **backend='cuda'**：输入可来自 CPU host 或 CUDA device
  - `CPU 输入走 CUDA 后结果回 CPU`
  - `CUDA 输入结果留在原 CUDA device`，输出 `torch float64`，**`mirror 输入 device`**
- **PoC ③ 输入范围**：`NumPy-only API 子集`——只实现 NumPy host 输入路径（f32/f64 白名单），CUDA tensor 直通属 Phase 1
- **诚实边界（N6-01）**：NumPy-only 单一路径**`不能证明完整 solver`**；factor/stock 的 capacity 结论必须分别对 CPU 输出与 CUDA 同设备输出两条路径建模，不得用单一路径代替最终 solver

---

## 2. cs_rank（F-06/F-13 已闭合）

- CUB `SortPairs` DoubleBuffer：`d_keys` 与 `d_values` 独立，排序后分别 `d_keys.Current()`（key）与 `d_values.Current()`（payload）
- host checked multiply：`T*N ≤ INT32_MAX`（`checked_mul`）
- 门槛 `≤ 13.926450`（gate_config_v1.json `exact_half`）

---

## 3. correlation 内核

### 3.1 共享数学单一真源（F5-02/04/08/11 修正）

**所有数值核心实现收敛到 `tests/fixtures/corr_math_v1.py`**（本文档与 generator/validator 共用）：

- **CompensatedSum（F5-02 符号修正）**：Kahan 更新 `y=x−c; t=sum+y; c=(t−sum)−y; sum=t` 的 state 代数表示恒为 **`sum - c`**；`merge_state(left, right)` 依次 `add(right.sum)` 与 **`-right.c`**（非 `right.c`）。机械反例：`[-1e16] + [1e16, 1]` → `+1`；错误符号重放 → `-1`（corr_math_trace_v1.json 可区分）
- **safe_pearson（F5-04 精确表达式）**：`corr = (Sxy/sqrt(Sxx)) / sqrt(Syy)`——逐次除法，避免 `Sxx*Syy` 中间乘积溢出。normal 与 Kahan 两遍**共用同一 finalize**。corpus `pearson_overflow`（x=y=[-1e150,1e150]）必须返回 1.0
- **ordinal key（F5-11 ±0）**：先 **`canonicalize ±0`**（`if x==0: bits = positive_zero_bits`），再做单调 bit transform；`stable ordinal` tie 按原列索引。f32/f64 分别 `canonical_ordinal_key_f32/f64`，invalid 直写 sentinel
- **checked arithmetic（F5-11）**：`checked_add`/`checked_mul`/`checked_scatter_out_base`/`checked_global_element_offset`/`checked_byte_offset`——`scatter_out_base = checked_mul(row_base, N)`，global element/byte offset 全部 checked，禁止先乘后查
- **is_aliasable（F5-08）**：`factor_is_aliasable(view, kernel_layout)` 与 `stock_is_aliasable(view, kernel_layout)` 独立判定（dtype/strides/alignment/offset/device/ownership/lifetime）；不满足 → gather 为 packed。路径枚举：`f32_conversion` / `f64_alias` / `f64_gather`
- **BinaryFrontier（F5-01）**：标准连续全局叶序 binary carry，`slot = (leaf_index >> level) & 1` 在对应 level 单 occupancy 槽位落/合并，含 leaf capacity guard 与跨 chunk carry；`corr_math_trace_v1.json` 覆盖 0..17 叶 + 跨 chunk 机械 trace

### 3.2 参考函数（完整独立，自包含）

#### factor_corr_reference
```
def factor_corr_reference(F3, mask, *, kernel_layout):
    validate_input(F3, mask)                                   # ndim=(T,N,F)、dtype f32/f64、contiguity
    path = select_factor_input_path(F3, kernel_layout)         # f32_conversion / f64_alias / f64_gather
    first_pass_state = {}                                       # Partial1: count/sum_x/sum_y/min/max
    second_pass_state = {}                                      # Partial2: sxx/syy/sxy
    for chunk in deterministic_chunks(E, T, N, T_chunk):        # 跨 chunk frontier carry
        frontier = BinaryFrontier(Partial1)                     # 规范归约树
        for leaf in chunk:
            first_pass_state = merge1(first_pass_state, leaf1(F3, mask))
            if bias_metric > 1e8 or abs(r) > 1 or not finite(r):
                fallback(CompensatedSum)                        # Kahan 两遍（同一固定树括号）
        μ_x, μ_y = first_pass_finalize(first_pass_state)
        for leaf in chunk:
            second_pass_state = merge2(second_pass_state, leaf2(F3, μ_x, μ_y))
        corr = (Sxy/sqrt(Sxx)) / sqrt(Syy)                      # safe Pearson（单一真源）
    if corr 非有限 or 零方差 or count<2:
        corr = NaN（退化分支）
    writeback(corr, (i, j)); mirror(corr, (j, i))
    if frontier capacity exceeded or CUDA fatal error:
        raise RuntimeError("扫描级终止，不返回部分 groups")
```

#### stock_corr_reference
```
def stock_corr_reference(X, mask, *, kernel_layout):
    validate_input(X, mask)                                     # ndim=(T,N)、dtype f32/f64
    path = select_stock_input_path(X, kernel_layout)            # f32_conversion / f64_alias / f64_gather
    for chunk in deterministic_chunks(T, N, T_chunk):
        frontier = BinaryFrontier(Partial1)                     # leaf = t，T=1218，current=5/next=3
        for (bi, bj) in lower_triangle_block_pairs(widths):     # off-diag pairs=N_ci*N_cj；diag=N_ci(N_ci+1)/2
            first_pass_state = {}                                # count/sum_left/sum_right（pair-specific）
            second_pass_state = {}
            for leaf in chunk:
                if bias_metric > 1e8:
                    fallback(CompensatedSum)
            corr = (Sxy/sqrt(Sxx)) / sqrt(Syy)
            if corr 非有限 or 退化:
                corr = NaN
            writeback(corr, tile); mirror(corr)
        if frontier capacity exceeded:
            raise RuntimeError("扫描级终止")
```

#### rolling_ic_reference
```
def rolling_ic_reference(f, r, factor_mask, fwd_mask, min_valid, *, kernel_layout):
    validate_input(f, r, factor_mask, fwd_mask)                 # 4 种 mixed dtype：(f32,f32)/(f32,f64)/(f64,f32)/(f64,f64)
    for chunk in deterministic_chunks(T, N, T_chunk):
        first_pass_state = {}                                    # 前置：valid_count<min_valid → NaN；常量 → NaN
        for row in chunk:
            if valid_count(row) < min_valid or constant(row):    # Partial1 min==max
                writeback(NaN, row); continue
            key = canonical_ordinal_key_f64 if dtype_f64 else canonical_ordinal_key_f32
            # full-N sentinel：invalid → sentinel，CUB SortPairs 串行复用 DoubleBuffer
            rank_f, rank_r = stable_ordinal_rank(f, r)          # 双侧 ordinal 秩 1..K
            second_pass_state = pearson_state(rank_f, rank_r)
            corr = (Sxy/sqrt(Sxx)) / sqrt(Syy)
            if bias_metric > 1e8 or abs(corr) > 1 or not finite(corr):
                fallback(compensated_pearson)                     # Kahan/安全路径（同一固定树括号）
            if corr 非有限:
                corr = NaN
            writeback(corr, row)
        if frontier capacity exceeded:
            raise RuntimeError("扫描级终止")
```

### 3.3 struct ABI（F5-03 修正：extrema 全 double）

| struct | sizeof(B) | alignof(B) | offset 布局 |
|---|---|---|---|
| Partial1 | 56 | 8 | count=0, sum_x=8, sum_y=16, min_x=24, max_x=32, min_y=40, max_y=48 |
| Partial2 | 24 | 8 | sxx=0, syy=8, sxy=16 |
| PartialK1 | 40 | 8 | count=0, sum_x=8, c_x=16, sum_y=24, c_y=32 |
| PartialK2 | 48 | 8 | sxx=0, c_xx=8, syy=16, c_yy=24, sxy=32, c_xy=40 |

`Partial1` 的 extrema 用 **double**（F5-03）——`f64_adjacent_ulp`（[1.0, nextafter(1.0,2.0)]）不得窄化 f32 误判常量。`corr_math_trace_v1.json::struct_abi` 机械断言 56/24/40/48。

### 3.4 主对角退化决策树（R3-03/F5-03）

```
diagonal(i)：序列 i 自身有效集（isfinite_i ∧ mask_i）
  count < 2                  → NaN
  zero_variance（double min==max）→ NaN
  !isfinite(corr)            → NaN
  否则                        → 1.0
```

---

## 4. 显存模型与 solver（F5-03/05/06/07 闭式）

**`benchmarks/compute_workspace_v1.py` 是可执行单一真源**，生成 `docs/workspace_v1.json`（schema 2.0.0）：

- **逐分配 Timeline + live-byte HWM**：每 buffer 记录 alloc/free 事件，`hwm = max(各 pass/path 同时存活)`——peak 取**四遍最大值**（normal/Kahan × 第一/第二遍），非单 pass 指定
- **theoretical 锚点**（available = `8188 MiB − 512 MiB = 7676 MiB`）：factor 155,872,080 / stock_500 56,112,000 / stock_2000 896,448,000 / stock_5000 5,601,120,000 / rolling 2,046,240
- **reserve 唯一消费点**：`available = free − reserve` 仅此处消费一次；candidate `required_bytes` **不再包含 reserve**
- **solver 覆盖**：factor 12 scenarios、stock 36（N=500/2000/5000 × 输出模式 × dtype path）、rolling 8 scenarios × normal/Kahan 2 variants = 16——每个候选输出 chosen / runner-up / first_infeasible_candidate；CPU 与 CUDA device 两种 output_mode 分别建模
- **rolling 4 mixed dtype**：每种 `(prediction_dtype, label_dtype)` 组合单独 alloc/free 时间线；CUB `query_api`/`temp_bytes` 为具名 solver 输入（不允许"≤256MB"估计）

---

## 5. parameter_scan（F-09/H6 已闭合）

- needs_mask 规则；白名单 `cudaErrorInvalidConfiguration` / `cudaErrorLaunchOutOfResources` 仅两个错误码可降级组失败（`error_stage="launch"`，time 全 0）；其余（allocator/event/h2d/d2h/sync/async/context/illegal address/device assert/launch failure/result allocation）→ 扫描级 `RuntimeError`，不返回部分 groups
- `elapsed_ms` 对齐 CLAUDE.md：入口第一条可执行语句 → groups/summary 完成即将返回前，调用内 H2D/分配/event/同步/D2H/聚合全计入
- 门槛 `≤ 55.342775`（gate_config exact_half）

---

## 6. Gate 与资产（gate-config-v1 自动块）

<!-- gate-config-v1:begin -->
本块由 `python benchmarks/generate_gate_config_v1.py <run-id>` 自动生成；禁止手改。
canonical run：`poc2_baseline_20260804c`。run-id 中的日期片段是冻结 provenance 标签，不是本文执行日期。
配置 payload SHA-256：`daa2556a5221174a6130259b52ec4b1715d23aaf34bac396c762cba80e2d5a99`；generator SHA-256：`297ff625776f6bad134fc57a735a2d84b98aaf7e6f54911cb2db71477b456293`。
机器 Gate 只比较全精度 `exact_half = raw_wall_ms / 2`；`display` 是向负无穷取整两位的小数，且必须满足 `display <= exact_half`。

| scope | operation | backend | raw_wall_ms | exact_half | display | canonical source |
|---|---|---:|---:|---:|---:|---|
| formal | `cs_rank` | `qgplearn` | 27.85290000247187 | 13.926450001235935 | 13.92 | `benchmarks/results/runs/poc2_baseline_20260804c/qgplearn.json` |
| formal | `cs_rank_desc` | `qgplearn` | 27.52580000378657 | 13.762900001893286 | 13.76 | `benchmarks/results/runs/poc2_baseline_20260804c/qgplearn.json` |
| formal | `factor_corr` | `cupy` | 3087.0417500009353 | 1543.5208750004676 | 1543.52 | `benchmarks/results/runs/poc2_baseline_20260804c/cupy.json` |
| formal | `rolling_ic` | `cupy` | 154.3650500025251 | 77.18252500126255 | 77.18 | `benchmarks/results/runs/poc2_baseline_20260804c/cupy.json` |
| formal | `parameter_scan(G=4)` | `qgplearn` | 110.68555000019842 | 55.34277500009921 | 55.34 | `benchmarks/results/runs/poc2_baseline_20260804c/qgplearn.json` |
| formal | `stock_corr(N=500)` | `cupy` | 43.611249995592516 | 21.805624997796258 | 21.80 | `benchmarks/results/runs/poc2_baseline_20260804c/cupy.json` |
| extension | `stock_corr(N=2000)` | `cupy` | 537.4797500007844 | 268.7398750003922 | 268.73 | `benchmarks/results/runs/poc2_baseline_20260804c/cupy.json` |
| extension | `stock_corr(N=5000)` | `cupy` | 2558.881699998892 | 1279.440849999446 | 1279.44 | `benchmarks/results/runs/poc2_baseline_20260804c/cupy.json` |

canonical source SHA-256：
- `numpy`：`benchmarks/results/runs/poc2_baseline_20260804c/numpy.json` → `8be7a4cd0760a6750c967c05c8989a056a297e7b5242fa78129e6db774329d53`
- `cupy`：`benchmarks/results/runs/poc2_baseline_20260804c/cupy.json` → `5e8e3a2d47269394bccddf443d8c379ef475a9682d432782756daf3de3c6c427`
- `qgplearn`：`benchmarks/results/runs/poc2_baseline_20260804c/qgplearn.json` → `0a1de3f983072b77a44d2a1153e82062bfd457f2307249606a82886166a04047`
<!-- gate-config-v1:end -->

`benchmarks/generate_gate_config_v1.py` 生成 `docs/gate_config_v1.json`（schema 1.1.0），并支持 `--check`（fail-closed 校验 JSON/source SHA/payload SHA/本文档 gate 块一致）。机器 Gate 只比较全精度 `exact_half = raw_wall_ms / 2`。

---

## 7. 校准（F5-12 闭式）

- `reserve = max(536,870,912, calibrated_p99)`；`nearest-rank p99`（8 有序样本取最大值）
- 采样公式 `max(0, free_before - min_free_during)`；7 个样本锚点 `calibrated_p99_bytes = 650,000,000`
- cache key 六字段（GPU UUID/驱动/CUDA runtime/总显存/allocator 版本/生成时间）
- 原子写：**`tmp + flush + fsync + close + replace`**（UTF-8 无 BOM、LF only、last_complete_rename_wins、temp leftovers 清理）
- budget 边界：`free≤reserve → fail`；`budget≤base → fail`；否则 pass

---

## 8. 测试与验证映射

- **corr corpus**：`corr_corpus_v1.*`（v1.2.0，16 cases，含 `pearson_overflow`/`f64_adjacent_ulp`/`stable_zero_f32`/`stable_zero_f64`），branch 由状态机计算，NPZ/manifest/generator/math SHA 绑定
- **corr math trace**：`corr_math_trace_v1.json`（BinaryFrontier 0..17 叶 + 跨 chunk、CompensatedSum +1/-1、safe Pearson 可区分、struct ABI）
- **test manifest**：`test_cases_v1.json`（manifest 1.1.0，50 cases / 7 targets，recoverable 白名单精确两错误码，fatal 10 stages 穷尽）
- **validator**：`validate_self_fix_v1.py` 单入口（调 validate_implementation/validate_corr_corpus/validate_test_cases/validate_workspace/validate_calibration），N6-01/N6-02 分别验证契约与自包含

---

## 9. 开工前置（Gate）

1. `validate_self_fix_v1.py` 全绿（本文档 status 块绑定其 SHA）
2. PoC ② 基线定稿（canonical run `poc2_baseline_20260804c`）
3. 工具链确认

---

*生成模型: DeepSeek-V4-Flash (via Claude Code CLI) · 2026-08-04，星期二，Asia/Hong_Kong · 状态 v0.7 自包含 · 范围 PoC ③④，Phase 1-4 占位*
