# -*- coding: utf-8 -*-
"""PoC ② 公平基线——contract parity 检查 v1（三臂 × 27 parity anchors case）。

设计依据：
- parity_anchors_v1.manifest.json 的 case_schema（27 case，operation/inputs/masks/params/expected/tolerance）
- CLAUDE.md「操作语义契约」L0 Spec（stable ordinal 秩 1..K / corr oracle ≤1e-12 / rolling_ic ordinal Spearman ≤1e-12）
- FAIR_BASELINE_PROTOCOL_v1.md §3

三臂（BACKENDS）：numpy/pandas（CPU）、CuPy（GPU）、QuantGplearn-Torch（GPU）。
关键语义裁决（协议 §2）：
- QuantGplearn `_rank_pct_dim` 输出 percentile（rank/count）→ ×count 转整数秩 1..K 后比较
- numpy/pandas 臂用 np.argsort(kind="stable") 实现 ordinal 秩（同语义口径），不直接用 pandas average
- rolling_ic：scipy average 秩仅记录偏差，不作断言；契约 ordinal 秩断言 ≤1e-12
- stock_corr：parity case 集含 factor_corr 无 stock_corr，本轮 parity 覆盖 factor_corr

用法：
    PYTHONIOENCODING=utf-8 python parity_check_v1.py [--backend numpy|cupy|qgplearn|all]
输出：results/parity_report_v1.json + 终端摘要
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
CORPUS_DIR = HERE.parent / "benchmark_corpus"
FIXTURES_DIR = HERE.parent / "tests" / "fixtures"
RESULTS_DIR = HERE / "results"

VERSION = "1.0.0"
BACKENDS = ["numpy", "cupy", "qgplearn"]
# Phase 1 (F16): fc.* adapter GPU arm, registered only when CUDA is available
# (preserves the L-01 no-unconditional-CUDA-dependency rule).
try:
    import torch  # noqa: F401
    if torch.cuda.is_available():
        sys.path.insert(0, str(HERE.parent))  # repo root -> fc package
        BACKENDS.append("gpu")
except Exception:
    pass

# 共享算子层（H4：parity 与 perf 必须同一实现入口）
from backends import (  # noqa: E402
    np_cs_rank, np_factor_corr, np_rolling_ic, np_parameter_scan,
    cp_cs_rank, cp_factor_corr, cp_rolling_ic, cp_parameter_scan,
    qg_cs_rank, qg_rolling_ic, qg_parameter_scan,
)


def _load_manifest() -> tuple[list, dict]:
    """加载 parity anchors case_schema + npz 数据。"""
    man = json.loads((CORPUS_DIR / "parity_anchors_v1.manifest.json").read_text(encoding="utf-8"))
    npz = np.load(CORPUS_DIR / "parity_anchors_v1.npz", allow_pickle=False)
    data = {k: npz[k] for k in npz.files}
    return man["case_schema"], data


def _json_safe(obj):
    """递归规范化 NumPy 标量/数组为 JSON 原生类型（S1：不以 default=str 静默变字符串）。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _collect_env() -> dict:
    """环境指纹（脚本生成，可复现——审查修复：原手工附加不可复现）。"""
    import platform
    env: dict[str, object] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    for mod, attr in (("pandas", "__version__"), ("scipy", "__version__"),
                      ("torch", "__version__"), ("cupy", "__version__")):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, attr)
        except Exception:
            env[mod] = "n/a"
    try:
        import torch
        env["torch_cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        env["torch_cuda"] = False
    return env


# ---------------------------------------------------------------------------
# 三臂实现
# ---------------------------------------------------------------------------

def _ordinal_rank_1d(x: np.ndarray) -> np.ndarray:
    """契约 cs_rank 单截面 stable ordinal 整数秩 1..K（非有限值 → 不参与 → 输出 NaN）。

    x: 1D float32 数组。返回 float32 秩（1..K）或 NaN。
    语义对齐 CLAUDE.md §1：asc[i] = 1 + #{j: y[j] < y[i] or (y[j]==y[i] and j<i)}。
    """
    x = np.asarray(x, dtype=np.float32)
    out = np.full(x.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return out
    xs = x[finite]
    # stable argsort（保留并列内索引序）→ ordinal 秩
    order = np.argsort(xs, kind="stable")
    r = np.empty(len(xs), dtype=np.float32)
    r[order] = np.arange(1, len(xs) + 1, dtype=np.float32)
    out[finite] = r
    return out


def _apply_mask_rank_1d(x: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """mask 参与规则：参与 = (mask None 或 mask True) 且 isfinite；其余输出 NaN。"""
    x = np.asarray(x, dtype=np.float32)
    out = np.full(x.shape, np.nan, dtype=np.float32)
    if mask is None:
        part = np.isfinite(x)
    else:
        part = np.isfinite(x) & np.asarray(mask, dtype=bool)
    if not part.any():
        return out
    xs = x[part]
    order = np.argsort(xs, kind="stable")
    r = np.empty(len(xs), dtype=np.float32)
    r[order] = np.arange(1, len(xs) + 1, dtype=np.float32)
    out[part] = r
    return out


def _oracle_rank_panel(x: np.ndarray, mask: np.ndarray | None, descending: bool) -> np.ndarray:
    """独立 ordinal 秩 oracle（S2，parameter_scan 内容比对）：panel (T,N) 或 1D，逐行 stable argsort。

    与 backends.py 共享层**实现无关**（只用本文件的 _apply_mask_rank_1d），
    descending = 取负后 ascending（契约语义）。
    """
    x = np.asarray(x)
    if x.ndim == 1:
        return _apply_mask_rank_1d(-x if descending else x, mask)
    T, N = x.shape
    out = np.full(x.shape, np.nan, dtype=np.float32)
    for t in range(T):
        mrow = mask[t] if mask is not None else None
        out[t] = _apply_mask_rank_1d(-x[t] if descending else x[t], mrow)
    return out


def _in_corr_domain(x: np.ndarray, mask: np.ndarray | None = None) -> bool:
    """契约 §2 数值域前置校验（**仅有效子集**）：max|x| ≤ 1e150 且 min 非零 |x| ≥ 1e-150。

    域外（任一量级越界）→ 契约要求抛 ValueError（corr API 前置条件，oracle 不校验）。
    审查修正（2026-08-03）：契约措辞为"有效观测的绝对量级须满足"——域校验只针对
    mask 交集 + isfinite 的有效子集，mask=False 格的极端值不参与域判定（不误拒）。
    注意：min 非零 |x| 判断须排除精确 0；corr_underflow=[0, 5e-324] 的 5e-324 < 1e-150 → 域外。
    """
    x = np.asarray(x, dtype=np.float64)
    valid = np.isfinite(x)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    xv = x[valid]
    if xv.size == 0:
        return True  # 空有效子集域内（后续 n<2 → NaN，非错误）
    amax = np.max(np.abs(xv))
    nonzero = xv[xv != 0.0]
    amin = np.min(np.abs(nonzero)) if nonzero.size else None
    if amax > 1e150:
        return False
    if amin is not None and amin < 1e-150:
        return False
    return True


def _numpy_param_scan(x: np.ndarray, mask: np.ndarray | None) -> list:
    """parameter_scan G=4 字典序：(ascending,masked)(ascending,unmasked)(descending,masked)(descending,unmasked)。"""
    nb = NumpyBackend()
    return nb.parameter_scan(x, mask)


def _spearman_ordinal(a: np.ndarray, b: np.ndarray) -> float:
    """契约 rolling_ic：stable ordinal 秩对在 float64 下做 Pearson（CLAUDE.md §3）。"""
    ok = np.isfinite(a) & np.isfinite(b)
    av, bv = a[ok], b[ok]
    if av.size < 2:
        return float("nan")
    ra = _ordinal_rank_1d(av.astype(np.float64))
    rb = _ordinal_rank_1d(bv.astype(np.float64))
    with np.errstate(all="ignore"):
        return float(np.corrcoef(np.stack([ra, rb]))[0, 1])


class NumpyBackend:
    """numpy/pandas 臂（CPU）——调用共享算子层 backends（H4：parity/perf 同一实现）。"""

    name = "numpy"

    def cs_rank(self, x: np.ndarray, mask: np.ndarray | None, descending: bool) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        # 1D → (1,N) panel 适配到共享层
        x2 = x.reshape(1, -1) if x.ndim == 1 else x
        m2 = mask.reshape(1, -1) if (mask is not None and np.asarray(mask).ndim == 1) else mask
        out = np_cs_rank(x2, m2, descending)
        return out.reshape(-1) if x.ndim == 1 else out

    def factor_corr(self, xa: np.ndarray, xb: np.ndarray | None = None, mask: np.ndarray | None = None) -> float:
        """契约 factor_corr：pairwise pooled 有效子集 → 共享层 np_factor_corr。

        xb=None（单输入）→ 形状错误 ValueError（契约 §2 errors：factor_corr 输入须 3-D）。
        parity 1D case 用单行 panel（T=1）适配到共享层。
        """
        if xb is None:
            raise ValueError("factor_corr 输入形状错误：需要 (T,N,F) 3-D 面板或两条等长序列")
        # 1D 对 → (1,N,2) panel（两序列作为两个 factor）
        a = np.asarray(xa); b = np.asarray(xb)
        if a.ndim == 1:
            F3 = np.stack([a, b], axis=1).reshape(1, -1, 2)  # (1,N,2)
            m2 = mask.reshape(1, -1) if mask is not None else None
            mat = np_factor_corr(F3, m2)  # (2,2)
            return float(mat[0, 1])
        # panel 输入
        return float(np_factor_corr(np.asarray(xa), mask)[0, 1])

    def rolling_ic(self, f: np.ndarray, r: np.ndarray, min_valid: int,
                   factor_mask: np.ndarray | None = None, fwd_mask: np.ndarray | None = None) -> float:
        f = np.asarray(f); r = np.asarray(r)
        f2 = f.reshape(1, -1) if f.ndim == 1 else f
        r2 = r.reshape(1, -1) if r.ndim == 1 else r
        fm = factor_mask.reshape(1, -1) if (factor_mask is not None and np.asarray(factor_mask).ndim == 1) else factor_mask
        rm = fwd_mask.reshape(1, -1) if (fwd_mask is not None and np.asarray(fwd_mask).ndim == 1) else fwd_mask
        out = np_rolling_ic(f2, r2, fm, rm, min_valid)
        return float(out[0])

    def parameter_scan(self, x: np.ndarray, mask: np.ndarray | None) -> list:
        x = np.asarray(x, dtype=np.float32)
        x2 = x.reshape(1, -1) if x.ndim == 1 else x
        m2 = mask.reshape(1, -1) if (mask is not None and np.asarray(mask).ndim == 1) else mask
        res = np_parameter_scan(x2, m2)
        if x.ndim == 1:
            return [r.reshape(-1) for r in res]
        return res


class CupyBackend:
    """CuPy 臂（GPU）——调用共享算子层 backends（H4：parity/perf 同一实现）。"""

    name = "cupy"

    def __init__(self):
        import cupy as cp
        self.cp = cp

    def cs_rank(self, x: np.ndarray, mask: np.ndarray | None, descending: bool) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        x2 = x.reshape(1, -1) if x.ndim == 1 else x
        m2 = mask.reshape(1, -1) if (mask is not None and np.asarray(mask).ndim == 1) else mask
        out = cp_cs_rank(x2, m2, descending)
        return out.reshape(-1) if x.ndim == 1 else out

    def factor_corr(self, xa: np.ndarray, xb: np.ndarray | None = None, mask: np.ndarray | None = None) -> float:
        if xb is None:
            raise ValueError("factor_corr 输入形状错误：需要 (T,N,F) 3-D 面板或两条等长序列")
        a = np.asarray(xa); b = np.asarray(xb)
        if a.ndim == 1:
            F3 = np.stack([a, b], axis=1).reshape(1, -1, 2)
            m2 = mask.reshape(1, -1) if mask is not None else None
            mat = cp_factor_corr(F3, m2)
            return float(mat[0, 1])
        return float(cp_factor_corr(np.asarray(xa), mask)[0, 1])

    def rolling_ic(self, f: np.ndarray, r: np.ndarray, min_valid: int,
                   factor_mask: np.ndarray | None = None, fwd_mask: np.ndarray | None = None) -> float:
        f = np.asarray(f); r = np.asarray(r)
        f2 = f.reshape(1, -1) if f.ndim == 1 else f
        r2 = r.reshape(1, -1) if r.ndim == 1 else r
        fm = factor_mask.reshape(1, -1) if (factor_mask is not None and np.asarray(factor_mask).ndim == 1) else factor_mask
        rm = fwd_mask.reshape(1, -1) if (fwd_mask is not None and np.asarray(fwd_mask).ndim == 1) else fwd_mask
        out = cp_rolling_ic(f2, r2, fm, rm, min_valid)
        return float(out[0])


class QGplearnBackend:
    """QuantGplearn-Torch 臂（GPU）——调用共享算子层 backends。

    rolling_ic 用原生 float32 batch_spearmanr（known-deviation，不具同语义资格）。
    factor_corr/stock_corr 无原生算子 → N/A（能力映射）。
    """

    name = "qgplearn"

    def __init__(self):
        import torch
        self._torch = torch

    def cs_rank(self, x: np.ndarray, mask: np.ndarray | None, descending: bool) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        x2 = x.reshape(1, -1) if x.ndim == 1 else x
        m2 = mask.reshape(1, -1) if (mask is not None and np.asarray(mask).ndim == 1) else mask
        out = qg_cs_rank(x2, m2, descending)
        return out.reshape(-1) if x.ndim == 1 else out

    def factor_corr(self, xa: np.ndarray, xb: np.ndarray | None = None, mask: np.ndarray | None = None) -> float:
        """QuantGplearn 无 pooled factor_corr 原生算子 → N/A（能力映射，不调 NumPy oracle 充数）。"""
        raise NotImplementedError("QuantGplearn 无 factor_corr 原生算子（能力映射 N/A）")

    def rolling_ic(self, f: np.ndarray, r: np.ndarray, min_valid: int,
                   factor_mask: np.ndarray | None = None, fwd_mask: np.ndarray | None = None) -> float:
        """QG 原生 float32 batch_spearmanr——known-deviation（披露），非 ≤1e-12 同语义。"""
        f = np.asarray(f); r = np.asarray(r)
        f2 = f.reshape(1, -1) if f.ndim == 1 else f
        r2 = r.reshape(1, -1) if r.ndim == 1 else r
        fm = factor_mask.reshape(1, -1) if (factor_mask is not None and np.asarray(factor_mask).ndim == 1) else factor_mask
        rm = fwd_mask.reshape(1, -1) if (fwd_mask is not None and np.asarray(fwd_mask).ndim == 1) else fwd_mask
        out = qg_rolling_ic(f2, r2, fm, rm, min_valid)
        return float(out[0])


def _run_case(backend, case: dict, data: dict) -> dict:
    """执行单个 case。返回 {id, backend, pass, actual, expected, err, detail}。"""
    op = case["operation"]
    tol = case.get("tolerance", "exact")
    expected = case["expected"]
    inputs = [data[k] for k in case["inputs"]]
    mask = data[case["masks"]["mask"]] if case.get("masks") else None
    params = case.get("params", {})

    result = {"id": case["id"], "operation": op, "backend": backend.name,
              "tolerance": tol, "expected": expected}

    try:
        if op == "cs_rank":
            actual = backend.cs_rank(inputs[0], mask, params.get("descending", False)).tolist()
        elif op == "factor_corr":
            if len(inputs) == 1 and (isinstance(expected, str) and expected == "ValueError"):
                # error 组：单输入 + expected ValueError → 形状错误（xb=None）
                actual = backend.factor_corr(inputs[0], None, mask)
            elif len(inputs) == 1:
                # self-corr：inputs 仅 1 个、expected 非 ValueError（如 corr_mask_false_finite）
                actual = backend.factor_corr(inputs[0], inputs[0], mask)
            else:
                actual = backend.factor_corr(inputs[0], inputs[1], mask)
        elif op == "rolling_ic":
            # 支持 factor_mask / fwd_mask 独立传入（GPT R3 修复）
            f_m = None
            r_m = None
            if case.get("masks"):
                f_m = data[case["masks"].get("factor_mask")] if case["masks"].get("factor_mask") else None
                r_m = data[case["masks"].get("fwd_mask")] if case["masks"].get("fwd_mask") else None
            actual = backend.rolling_ic(inputs[0], inputs[1], params.get("min_valid", 30), f_m, r_m)
        elif op == "parameter_scan":
            # 真实 schema 校验（协议 §3 schema 档）：G=4 字典序 + result shape/dtype。
            # S2 增强：逐元素比对独立 oracle（_oracle_rank_panel），防"harness 自证、任何同形数组即过"。
            x = inputs[0]
            mask_arg = data[case["masks"]["mask"]] if case.get("masks") else None
            res = backend.parameter_scan(x, mask_arg) if hasattr(backend, "parameter_scan") else _numpy_param_scan(x, mask_arg)
            exp_groups = ["(ascending,masked)", "(ascending,unmasked)", "(descending,masked)", "(descending,unmasked)"]
            got = []
            ok = len(res) == 4
            content_err = None
            for g in range(4):
                desc = g // 2 == 1
                use_mask = g % 2 == 0
                got.append(f"({'descending' if desc else 'ascending'},{'masked' if use_mask else 'unmasked'})")
                r = res[g]
                if not isinstance(r, np.ndarray) or r.shape != x.shape or r.dtype != np.float32:
                    ok = False
                    content_err = f"组 {g} shape/dtype 不符"
                    break
                oracle = _oracle_rank_panel(x, mask_arg if use_mask else None, desc)
                o_nan = np.isnan(oracle)
                r_nan = np.isnan(r)
                if not np.array_equal(o_nan, r_nan):
                    ok = False
                    content_err = f"组 {g}({got[-1]}) NaN 位置不符 oracle"
                    break
                if not o_nan.all() and not np.array_equal(
                        r[~o_nan].astype(np.float32), oracle[~o_nan].astype(np.float32)):
                    ok = False
                    content_err = f"组 {g}({got[-1]}) 秩值不符 oracle"
                    break
            if got != exp_groups:
                ok = False
                if content_err is None:
                    content_err = f"期望组序 {exp_groups} 实测 {got}"
            result["actual"] = {"groups": got, "schema_ok": ok, "content_checked": True}
            result["pass"] = ok
            if not ok:
                result["err"] = content_err or f"期望组序 {exp_groups} 实测 {got}"
            return result
        else:
            result["pass"] = False
            result["err"] = f"未知操作 {op}"
            return result
    except ValueError as e:
        result["actual"] = "ValueError"
        result["err"] = str(e)[:200]
        result["pass"] = (tol == "exception" and expected == "ValueError")
        # S2：断言异常来源符合 case 语义——防"任意 ValueError 即过"（只对冻结 error 组检查）
        if result["pass"]:
            _exc_kw = {
                "corr_shape_error": ["3-D", "形状", "factor_corr 输入须"],
                "corr_underflow_domain": ["数值域外", "1e-150"],
                "err_out_of_domain": ["数值域外", "1e150"],
            }.get(case.get("id"), [])
            if _exc_kw and not any(k in str(e) for k in _exc_kw):
                result["pass"] = False
                result["err"] = f"异常消息与 case 语义不符: {str(e)[:120]}"
        return result
    except NotImplementedError as e:
        # 能力映射 N/A（如 QG 无 factor_corr/stock_corr）——不计 PASS 也不计 FAIL（GPT H4）
        result["actual"] = "N/A"
        result["err"] = str(e)[:200]
        result["pass"] = False
        result["na"] = True
        return result
    except Exception as e:  # noqa: BLE001
        result["actual"] = f"EXC:{type(e).__name__}"
        result["err"] = str(e)[:200]
        result["pass"] = False
        return result

    # 断言
    if tol == "exception":
        result["pass"] = (actual == "ValueError")  # noqa: SIM108 实际由 except 捕获
        result["actual"] = str(actual)
        return result
    if isinstance(expected, list):
        exp = [float("nan") if isinstance(v, str) and v == "nan" else float(v) for v in expected]
        act = [float(v) for v in actual]
        # M2 修复：先断言长度一致（zip 会静默截断）
        if len(act) != len(exp):
            result["actual"] = actual
            result["pass"] = False
            result["err"] = f"长度不符：期望 {len(exp)} 实测 {len(act)}"
            return result
        if tol == "exact":
            # 逐位：NaN 位置 + 数值 exact。若 case 声明 nan_payload（契约 NaN 载荷冻结），
            # 对 NaN 单元按位校验载荷（reinterpret uint32）——审查补强（2026-08-03）。
            payload = case.get("nan_payload")
            exp_arr = np.asarray(exp, dtype=np.float32)
            act_arr = np.asarray(act, dtype=np.float32)
            ok = True
            for e, a in zip(exp_arr, act_arr):
                if np.isnan(e) and np.isnan(a):
                    if payload is not None:
                        # 按位比较 NaN 载荷
                        e_bits = np.asarray(e, dtype=np.float32).view(np.uint32).item()
                        a_bits = np.asarray(a, dtype=np.float32).view(np.uint32).item()
                        if e_bits != a_bits:
                            ok = False
                            break
                elif np.isnan(e) != np.isnan(a):
                    ok = False
                    break
                elif e != a:
                    ok = False
                    break
        else:
            atol = float(tol)
            ok = all(
                (np.isnan(e) and np.isnan(a)) or (not np.isnan(e) and not np.isnan(a) and abs(e - a) <= atol)
                for e, a in zip(exp, act)
            )
        result["actual"] = actual
        result["pass"] = ok
        if not ok:
            result["err"] = f"期望 {exp} 实测 {act}"
        return result

    # 标量 expected
    exp_scalar = float("nan") if isinstance(expected, str) and expected == "nan" else float(expected)
    act_scalar = float("nan") if isinstance(actual, str) and actual == "nan" else float(actual)
    if tol == "exact":
        if np.isnan(exp_scalar) and np.isnan(act_scalar):
            result["pass"] = True
        else:
            result["pass"] = (abs(exp_scalar - act_scalar) == 0.0)
    else:
        atol = float(tol)
        if np.isnan(exp_scalar) and np.isnan(act_scalar):
            result["pass"] = True
        else:
            result["pass"] = (abs(exp_scalar - act_scalar) <= atol)
    result["actual"] = act_scalar
    if not result["pass"]:
        result["err"] = f"期望 {exp_scalar} 实测 {act_scalar}"
    return result


class GpuBackend:
    """fc.* 适配层 GPU 臂（Phase 1，F16）——路由到 fc 的 GPU/backend='cuda' 路径。

    factor_corr 由两条 1D 序列构造 F=2 面板 (1,N,2) → fc.factor_corr(backend='cuda');
    rolling_ic 走 device=None（CUDA 可用 → GPU 执行）；parameter_scan 返回 4 组
    (T,N) float32（契约 dict 序）。
    """

    name = "gpu"

    def __init__(self):
        import fc
        self.fc = fc

    def cs_rank(self, x: np.ndarray, mask: np.ndarray | None, descending: bool) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        x2 = x.reshape(1, -1) if x.ndim == 1 else x
        m2 = mask.reshape(1, -1) if (mask is not None and np.asarray(mask).ndim == 1) else mask
        out = self.fc.cross_sectional_rank(x2, m2, descending)
        out = np.asarray(out, dtype=np.float32)
        return out.reshape(-1) if x.ndim == 1 else out

    def factor_corr(self, xa: np.ndarray, xb: np.ndarray | None = None,
                    mask: np.ndarray | None = None) -> float:
        if xb is None:
            raise ValueError("factor_corr 输入形状错误：需要 (T,N,F) 3-D 面板或两条等长序列")
        a = np.asarray(xa); b = np.asarray(xb)
        if a.ndim == 1:
            F3 = np.stack([a, b], axis=1).reshape(1, -1, 2)
            m2 = mask.reshape(1, -1) if mask is not None else None
            mat = self.fc.factor_corr(F3, m2, backend="cuda")
            return float(np.asarray(mat)[0, 1])
        return float(np.asarray(self.fc.factor_corr(np.asarray(xa), mask,
                                                    backend="cuda"))[0, 1])

    def rolling_ic(self, f: np.ndarray, r: np.ndarray, min_valid: int,
                   factor_mask: np.ndarray | None = None,
                   fwd_mask: np.ndarray | None = None) -> float:
        f = np.asarray(f); r = np.asarray(r)
        f2 = f.reshape(1, -1) if f.ndim == 1 else f
        r2 = r.reshape(1, -1) if r.ndim == 1 else r
        fm = (factor_mask.reshape(1, -1) if (factor_mask is not None
                                             and np.asarray(factor_mask).ndim == 1)
              else factor_mask)
        rm = (fwd_mask.reshape(1, -1) if (fwd_mask is not None
                                          and np.asarray(fwd_mask).ndim == 1)
              else fwd_mask)
        out = self.fc.rolling_ic(f2, r2, fm, rm, min_valid, device=None)
        return float(np.asarray(out)[0])

    def parameter_scan(self, x: np.ndarray, mask: np.ndarray | None) -> list:
        x = np.asarray(x, dtype=np.float32)
        x2 = x.reshape(1, -1) if x.ndim == 1 else x
        m2 = mask.reshape(1, -1) if (mask is not None and np.asarray(mask).ndim == 1) else mask
        res = self.fc.parameter_scan(
            axes=[("direction", ["ascending", "descending"]),
                  ("mask_mode", ["masked", "unmasked"])], X=x2, mask=m2)
        groups = [np.asarray(g["result"], dtype=np.float32) for g in res["groups"]]
        if x.ndim == 1:
            return [r.reshape(-1) for r in groups]
        return groups


def run_backend(name: str) -> dict:
    if name == "numpy":
        backend = NumpyBackend()
    elif name == "cupy":
        backend = CupyBackend()
    elif name == "qgplearn":
        backend = QGplearnBackend()
    elif name == "gpu":
        backend = GpuBackend()
    else:
        raise ValueError(f"未知 backend {name}")

    cases, data = _load_manifest()
    results = [_run_case(backend, c, data) for c in cases]
    passed = sum(1 for r in results if r["pass"])
    na = sum(1 for r in results if r.get("na"))
    failed = len(results) - passed - na
    # M4 扩展锚点（边界健壮性，不进冻结 manifest）
    ext = run_extended_checks(backend)
    ext_passed = sum(1 for c in ext if c["pass"])
    return {
        "backend": name,
        "version": VERSION,
        "total": len(results),
        "passed": passed,
        "na": na,
        "failed": failed,
        "cases": results,
        "extended": ext,
        "extended_passed": ext_passed,
        "extended_total": len(ext),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def run_extended_checks(backend) -> list:
    """M4 扩展锚点：关键契约边界（descending+mask、F=1、单侧 mask、min_valid>N、stock_corr 矩阵、GEMM 对抗）。

    这些 case 不进冻结 manifest（避免改 corpus），作为脚本内置的边界健壮性检查。
    返回 [{id, backend, pass, actual, expected, err}]。
    L-01：不无条件依赖 CuPy（NumPy baseline 无 CUDA 环境也可独立运行）。
    """
    import backends as _B
    checks = []

    def _rec(id_, pass_, actual, expected, err=None, na=False):
        # S1：统一 bool(pass_)——np.bool_ 经 json.dumps(default=str) 会变字符串 "True"，破坏 schema
        checks.append({"id": id_, "backend": backend.name, "pass": bool(pass_),
                       "actual": actual, "expected": expected, "err": err, "na": na})

    try:
        # 1. cs_rank descending + mask（契约 §1 direction×mask 组合）
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        m = np.array([True, False, True, True])
        r = backend.cs_rank(x, m, True)
        exp = [3.0, np.nan, 2.0, 1.0]
        ok = (len(r) == 4 and np.isnan(r[1]) and abs(r[0]-3) < 1e-6
              and abs(r[2]-2) < 1e-6 and abs(r[3]-1) < 1e-6)
        _rec("ext_cs_rank_desc_mask", ok, r, exp, None if ok else f"期望 {exp} 实测 {r}")
    except Exception as e:
        _rec("ext_cs_rank_desc_mask", False, f"EXC:{type(e).__name__}", "正常", str(e)[:80])

    try:
        # 2. factor_corr F=1 → (1,1)
        F3 = np.random.default_rng(0).normal(size=(5, 10, 1)).astype(np.float32)
        mat = _factor_corr_panel(backend, F3, None)
        ok = (mat.shape == (1, 1) and (np.isnan(mat[0,0]) or abs(mat[0,0]-1) < 1e-12))
        _rec("ext_factor_corr_F1", ok, mat.shape, "(1,1)", None if ok else f"shape={mat.shape} val={mat}")
    except NotImplementedError:
        _rec("ext_factor_corr_F1", False, "N/A", "(1,1)", "QG 无 factor_corr", na=True)
    except Exception as e:
        _rec("ext_factor_corr_F1", False, f"EXC:{type(e).__name__}", "(1,1)", str(e)[:80])

    try:
        # 3. factor_corr 全 mask False → 全 NaN
        F3b = np.random.default_rng(1).normal(size=(5, 10, 2)).astype(np.float32)
        mb = np.zeros((5, 10), dtype=bool)
        mat_b = _factor_corr_panel(backend, F3b, mb)
        ok = bool(np.isnan(mat_b).all())
        _rec("ext_factor_corr_allmask", ok, mat_b, "全 NaN", None if ok else f"{mat_b}")
    except NotImplementedError:
        _rec("ext_factor_corr_allmask", False, "N/A", "全 NaN", "QG 无 factor_corr", na=True)
    except Exception as e:
        _rec("ext_factor_corr_allmask", False, f"EXC:{type(e).__name__}", "全 NaN", str(e)[:80])

    try:
        # 4. rolling_ic 单侧 mask（factor_mask 提供，fwd_mask None）——GPT R3 修复：真传 mask + 比较 oracle
        f = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float64)
        r = np.array([[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]], dtype=np.float64)
        fm = np.array([[True, True, True, False, True, True]])
        ic_masked = backend.rolling_ic(f, r, 2, fm, None)
        ic_none = backend.rolling_ic(f, r, 2, None, None)
        # 独立 ordinal-Spearman oracle（factor-only mask）
        fv = f.ravel()[fm.ravel()]; rv = r.ravel()[fm.ravel()]
        ra = _ord_float64(fv); rb = _ord_float64(rv)
        oracle_val = np.corrcoef(np.stack([ra, rb]))[0, 1]
        ok = (abs(ic_masked - oracle_val) < 1e-12) and (abs(ic_masked - ic_none) > 1e-12)
        _rec("ext_rolling_ic_single_mask", bool(ok), float(ic_masked),
             float(oracle_val), None if ok else f"masked={ic_masked} oracle={oracle_val} none={ic_none}")
    except Exception as e:
        _rec("ext_rolling_ic_single_mask", False, f"EXC:{type(e).__name__}", "oracle", str(e)[:80])

    try:
        # 5. rolling_ic min_valid > N → NaN
        f5 = np.ones((1, 3)); r5 = np.ones((1, 3))
        ic5 = backend.rolling_ic(f5, r5, 30)
        ok = bool(np.isnan(ic5))
        _rec("ext_rolling_ic_minvalid_gt_N", ok, ic5, "NaN", None if ok else f"{ic5}")
    except Exception as e:
        _rec("ext_rolling_ic_minvalid_gt_N", False, f"EXC:{type(e).__name__}", "NaN", str(e)[:80])

    try:
        # 6. stock_corr correctness（GPT R3：manifest 0 case，补扩展回归）——returns + NaN/inf + mask + 大偏置
        import backends as _B
        if backend.name == "numpy":
            _sc = _B.np_stock_corr
        elif backend.name == "cupy":
            _sc = _B.cp_stock_corr
        elif backend.name == "gpu":
            _sc = lambda X, M: np.asarray(backend.fc.stock_corr(X, M, backend="cuda"))
        else:
            _sc = None
        if _sc is not None:
            # returns 输入（协议）+ mask 排除 NaN 格
            rng = np.random.default_rng(7)
            X = rng.normal(size=(20, 8)).astype(np.float64)
            X[3, 2] = np.nan; X[10, 5] = np.inf  # 非有限格
            M = np.ones((20, 8), dtype=bool); M[3, 2] = False; M[10, 5] = False
            corr = _sc(X, M)
            # 与两遍 oracle 抽样对比
            ok = True; maxdev = 0.0
            for i in range(8):
                for j in range(i + 1, 8):
                    o = np.isfinite(X[:, i]) & np.isfinite(X[:, j]) & M[:, i] & M[:, j]
                    if o.sum() < 2:
                        if not np.isnan(corr[i, j]):
                            ok = False
                        continue
                    a = X[o, i]; b = X[o, j]
                    am = a.mean(); bm = b.mean()
                    ac = a - am; bc = b - bm
                    den = np.sqrt((ac * ac).sum() * (bc * bc).sum())
                    ref = float((ac * bc).sum() / den) if den != 0 else np.nan
                    maxdev = max(maxdev, abs(ref - corr[i, j]))
            ok = ok and maxdev < 1e-12 and np.array_equal(corr, corr.T)
            _rec("ext_stock_corr_correctness", bool(ok), maxdev, "≤1e-12 + 对称",
                 None if ok else f"maxdev={maxdev}")
        else:
            _rec("ext_stock_corr_correctness", False, "N/A", "≤1e-12", "QG 无 stock_corr", na=True)
    except NotImplementedError:
        _rec("ext_stock_corr_correctness", False, "N/A", "≤1e-12", "QG 无 stock_corr", na=True)
    except Exception as e:
        _rec("ext_stock_corr_correctness", False, f"EXC:{type(e).__name__}", "≤1e-12", str(e)[:80])

    try:
        # H-01 回归（三轮审查反例）：GEMM 抵消检测对抗输入——ratio=28, rho=-0.99。
        # 阈值 100 必须触发回退使 |Δr|≤1e-12；此 case 在阈值 1e3 时会超差（1.14e-12）。
        T_adv = 100; ratio_adv = 28.0; rho_adv = -0.99
        rng = np.random.default_rng(0)
        z = rng.normal(size=(T_adv, 2))
        a = z[:, 0]; b0 = rho_adv * a + np.sqrt(1 - rho_adv * rho_adv) * z[:, 1]
        a = (a - a.mean()) / a.std(); b0 = (b0 - b0.mean()) / b0.std()
        Xadv = np.column_stack([a + ratio_adv, b0 + ratio_adv * (1 + 1e-6)])
        if backend.name == "numpy":
            got_adv = _B.np_stock_corr(Xadv, None)[0, 1]
        elif backend.name == "cupy":
            got_adv = _B.cp_stock_corr(Xadv, None)[0, 1]
        elif backend.name == "gpu":
            got_adv = float(np.asarray(
                backend.fc.stock_corr(Xadv, None, backend="cuda"))[0, 1])
        else:
            got_adv = None
        if got_adv is not None:
            a2, c2 = Xadv[:, 0], Xadv[:, 1]
            a2 = a2 - a2.mean(); c2 = c2 - c2.mean()
            den = np.sqrt((a2 * a2).sum() * (c2 * c2).sum())
            ref_adv = float((a2 * c2).sum() / den)
            ok_adv = abs(ref_adv - got_adv) <= 1e-12
            _rec("ext_gemm_cancel_adversarial", bool(ok_adv), float(got_adv), float(ref_adv),
                 None if ok_adv else f"err={abs(ref_adv - got_adv):.3e} > 1e-12")
        else:
            _rec("ext_gemm_cancel_adversarial", False, "N/A", "≤1e-12", "QG 无 stock_corr", na=True)
    except NotImplementedError:
        _rec("ext_gemm_cancel_adversarial", False, "N/A", "≤1e-12", "QG 无 stock_corr", na=True)
    except Exception as e:
        _rec("ext_gemm_cancel_adversarial", False, f"EXC:{type(e).__name__}", "≤1e-12", str(e)[:80])

    try:
        # 7. corr 对角边界（异后端审查 2026-08-05 发现）：正次正规方差（S>0 且 S*S 下溢为 0）、
        #    zero-variance（dx^2 下溢）、常量、count<2、正常 —— np/cp × stock/factor 四后端
        #    对角必须与冻结 oracle（np.corrcoef）同判（oracle 有限 → 1.0；oracle NaN → NaN）。
        #    回归 _two_pass_corr 原乘积型分母 sqrt(sxx*syy) 在 S*S 下溢时把有限判成 NaN 的 bug。
        if backend.name == "numpy":
            _sc, _fc = _B.np_stock_corr, _B.np_factor_corr
        elif backend.name == "cupy":
            _sc, _fc = _B.cp_stock_corr, _B.cp_factor_corr
        elif backend.name == "gpu":
            _sc = lambda X, M: np.asarray(
                backend.fc.stock_corr(X, M, backend="cuda"))
            _fc = lambda F3, M: np.asarray(
                backend.fc.factor_corr(F3, M, backend="cuda"))
        else:
            _sc = _fc = None
        xt0 = np.float64(1e-150)
        boundary = {
            "subnormal_var": np.array([1e-150, 1e-150 + 1e-161], dtype=np.float64),
            "zero_var": np.array([xt0, np.nextafter(xt0, np.float64(np.inf))], dtype=np.float64),
            "constant": np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float64),
            "count1": np.array([1.0], dtype=np.float64),
            "normal": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64),
        }
        ok_all = True
        fails = []
        for nm, col in boundary.items():
            o = np.isfinite(col)
            r_ = np.corrcoef(np.stack([col[o], col[o]]))[0, 1] if o.sum() >= 2 else np.nan
            exp = 1.0 if np.isfinite(r_) else np.nan
            for kind, fn, make in (("stock", _sc, lambda c: c.reshape(-1, 1)),
                                   ("factor", _fc, lambda c: c.reshape(1, -1, 1))):
                if fn is None:
                    continue
                d = fn(make(col), None)[0, 0]
                good = (np.isnan(d) == np.isnan(exp)) and (np.isnan(exp) or abs(d - 1.0) < 1e-12)
                if not good:
                    ok_all = False
                    fails.append(f"{kind}.{nm}: diag={d} exp={exp}")
        if _sc is None and _fc is None:
            _rec("ext_corr_diag_boundary", False, "N/A", "oracle 同判", "QG 无 corr", na=True)
        else:
            _rec("ext_corr_diag_boundary", bool(ok_all), fails or "全部匹配 oracle",
                 "有限→1.0 / NaN→NaN", None if ok_all else str(fails))
    except NotImplementedError:
        _rec("ext_corr_diag_boundary", False, "N/A", "oracle 同判", "QG 无 corr", na=True)
    except Exception as e:
        _rec("ext_corr_diag_boundary", False, f"EXC:{type(e).__name__}", "oracle 同判", str(e)[:80])

    return checks


def _ord_float64(x: np.ndarray) -> np.ndarray:
    """独立 ordinal 整数秩（float64，1..K）。扩展锚点 oracle 用。"""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    return r


def _factor_corr_panel(backend, F3, mask):
    """适配 factor_corr panel 输入到 backend 接口。"""
    # 构造 (T,N,F) 三因子 panel，用共享层
    import backends as _B
    if backend.name == "numpy":
        return _B.np_factor_corr(F3, mask)
    elif backend.name == "cupy":
        return _B.cp_factor_corr(F3, mask)
    elif backend.name == "gpu":
        return np.asarray(backend.fc.factor_corr(F3, mask, backend="cuda"))
    raise NotImplementedError("QG 无 factor_corr")


def main() -> int:
    backend_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = BACKENDS if backend_arg == "all" else [backend_arg]
    if any(n not in BACKENDS for n in names):
        print(f"未知 backend {names}，可用: {BACKENDS}", file=sys.stderr)
        return 2

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "parity_report_v1.json"
    # 合并模式：读已有文件保留其他臂（审查修复，2026-08-03：原覆盖导致产物仅单臂）
    all_reports = {}
    if out_path.exists():
        try:
            all_reports = json.loads(out_path.read_text(encoding="utf-8"))
            # 移除旧的 _env（由本脚本统一生成）
            all_reports.pop("_env", None)
        except json.JSONDecodeError:
            all_reports = {}
    for name in names:
        print(f"--- backend: {name} ---")
        report = run_backend(name)
        all_reports[name] = report
        na_note = f" ({report['na']} N/A)" if report.get("na") else ""
        ext_note = f" | ext {report['extended_passed']}/{report['extended_total']}" if report.get("extended") else ""
        print(f"  {report['passed']}/{report['total']} PASS{na_note}{ext_note}")
        for r in report["cases"]:
            if r.get("na"):
                mark = "N/A "
                err = f" | {r.get('err', '')[:60]}"
            elif r["pass"]:
                mark = "PASS"
                err = ""
            else:
                mark = "FAIL"
                err = f" | {r.get('err', '')}" if r.get("err") else ""
            print(f"    [{mark}] {r['id']} (tol={r['tolerance']}){err}")
    # 环境指纹由脚本生成（审查修复：原手工附加不可复现）
    all_reports["_env"] = _collect_env()
    out_path.write_text(
        json.dumps(_json_safe(all_reports), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"\n报告: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
