# -*- coding: utf-8 -*-
"""factor-cuda contract parity 对抗语料生成器 v1 — 设计 §6（GPT-5.6-Sol 第三轮 R6 重构）。

每个 case 按冻结公共 API 真实可执行：operation、array_ref（映射 npz 数据键）、
inputs（(T,N) 面板或 1D 序列）、双侧 masks、params、expected（唯一 result 或
expected_exception）。case 可由单一输入+参数唯一推出 expected。
NPZ 只存数据数组；case schema 存于 manifest。
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
VERSION = "1.0.0"

# 契约冻结 API 的独立 oracle 计算（供 expected 生成与验证）
def _ordinal_rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    return r


def _spearman_ic(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    av, bv = a[ok], b[ok]
    if av.size < 2:
        return float("nan")
    with np.errstate(all="ignore"):
        return float(np.corrcoef(np.stack([_ordinal_rank(av), _ordinal_rank(bv)]))[0, 1])


def _corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        ok &= np.asarray(mask, dtype=bool)
    av, bv = a[ok], b[ok]
    if av.size < 2:
        return float("nan")
    with np.errstate(all="ignore"):
        return float(np.corrcoef(np.stack([av, bv]))[0, 1])


def build_data() -> dict:
    """NPZ 数据数组（键 = case.array_ref 目标）。"""
    # 单行 rolling_ic 面板（T=1, N=35）：用 NaN 控制有效数
    def row(n_valid: int, n: int = 35) -> np.ndarray:
        x = np.arange(n, dtype=np.float64)
        if n_valid < n:
            x[n_valid:] = np.nan
        return x

    return {
        # rank（1D）
        "rank_tie": np.array([3.0, 3.0, 1.0], dtype=np.float32),
        "rank_zero": np.array([0.0, -0.0, 1.0], dtype=np.float32),
        "rank_nan_inf": np.array([1.0, np.nan, np.inf, -np.inf, 2.0], dtype=np.float32),
        "rank_mask": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "rank_mask_m": np.array([True, False, True, True]),
        # rolling_ic（单行 (T=1,N=35) 面板）
        "ic_29_f": row(29).astype(np.float32),
        "ic_29_r": row(29),
        "ic_30_f": row(30).astype(np.float32),
        "ic_30_r": row(30),
        "ic_31_f": row(31).astype(np.float32),
        "ic_31_r": row(31),
        "ic_all_invalid_f": np.full(35, np.nan, dtype=np.float32),
        "ic_all_invalid_r": np.full(35, np.nan, dtype=np.float64),
        "ic_tie_f": np.array([1.0, 1.0, 2.0], dtype=np.float32),
        "ic_tie_r1": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        "ic_tie_r2": np.array([2.0, 1.0, 3.0], dtype=np.float64),
        "ic_nan_f": np.array([1.0, np.nan, 2.0, 3.0, 4.0], dtype=np.float32),
        "ic_nan_r": np.array([1.0, 2.0, np.nan, 3.0, 4.0], dtype=np.float64),
        "ic_inf_f": np.array([1.0, np.inf, 2.0, 3.0, 4.0], dtype=np.float32),
        "ic_neginf_r": np.array([1.0, 2.0, -np.inf, 3.0, 4.0], dtype=np.float64),
        "ic_tradable_nan_f": np.array([np.nan, 1.0, 2.0, 3.0], dtype=np.float32),
        "ic_tradable_nan_r": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        # correlation
        "corr_a": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        "corr_b": np.array([1.0, 2.0, np.nan], dtype=np.float64),
        "corr_c": np.array([0.0, 1.0, 2.0], dtype=np.float64),
        "corr_n2": np.array([1.0, 2.0], dtype=np.float64),
        "corr_n2_rev": np.array([2.0, 1.0], dtype=np.float64),
        "corr_const": np.array([1.0, 1.0, 1.0], dtype=np.float64),
        "corr_a32": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "corr_c32": np.array([0.0, 1.0, 2.0], dtype=np.float32),
        "corr_mask_false_finite": np.array([5.0, 5.0, 1.0, 2.0], dtype=np.float64),
        "corr_mask_false_finite_m": np.array([False, False, True, True]),
        "corr_no_common_a": np.array([np.nan, 1.0, 2.0], dtype=np.float64),
        "corr_no_common_b": np.array([3.0, np.nan, np.nan], dtype=np.float64),
        "corr_underflow": np.array([0.0, 5e-324], dtype=np.float64),
        # parameter_scan（(T,N) 二维输入）
        "scan_x": np.array([[3.0, 1.0, 2.0], [1.0, 3.0, 2.0], [2.0, 2.0, 1.0]], dtype=np.float32),
        "scan_mask": np.array([[True, True, True], [True, False, True], [True, True, True]]),
        # error
        "err_domain_x": np.array([1e151, 2e151, 3e151, 4e151], dtype=np.float64),
        "err_domain_y": np.array([4e151, 1e151, 3e151, 2e151], dtype=np.float64),
    }


def build_cases(data: dict) -> list:
    """显式 case 列表。expected 由冻结 API 语义唯一确定（本函数内用独立 oracle 计算）。"""
    D = data

    def sp(a, b, min_valid=30):
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < min_valid:
            return "nan"
        return _spearman_ic(a, b)

    cases = []

    # ---- rank 组 ----
    cases.append({"id": "rank_tie_asc", "operation": "cs_rank",
                  "array_ref": ["rank_tie"], "inputs": ["rank_tie"], "masks": {},
                  "params": {"descending": False}, "expected": [2.0, 3.0, 1.0],
                  "dtype": "float32", "tolerance": "exact"})
    cases.append({"id": "rank_tie_desc", "operation": "cs_rank",
                  "array_ref": ["rank_tie"], "inputs": ["rank_tie"], "masks": {},
                  "params": {"descending": True}, "expected": [1.0, 2.0, 3.0],
                  "dtype": "float32", "tolerance": "exact"})
    cases.append({"id": "rank_zero_asc", "operation": "cs_rank",
                  "array_ref": ["rank_zero"], "inputs": ["rank_zero"], "masks": {},
                  "params": {"descending": False}, "expected": [1.0, 2.0, 3.0],
                  "dtype": "float32", "tolerance": "exact"})
    cases.append({"id": "rank_nan_inf_asc", "operation": "cs_rank",
                  "array_ref": ["rank_nan_inf"], "inputs": ["rank_nan_inf"], "masks": {},
                  "params": {"descending": False}, "expected": [1.0, "nan", "nan", "nan", 2.0],
                  "dtype": "float32", "tolerance": "exact", "nan_payload": "0x7fc00000"})
    cases.append({"id": "rank_mask_mismatch_asc", "operation": "cs_rank",
                  "array_ref": ["rank_mask", "rank_mask_m"], "inputs": ["rank_mask"], "masks": {"mask": "rank_mask_m"},
                  "params": {"descending": False}, "expected": [1.0, "nan", 2.0, 3.0],
                  "dtype": "float32", "tolerance": "exact"})

    # ---- rolling_ic 组（min_valid=30）----
    cases.append({"id": "ic_valid_29", "operation": "rolling_ic",
                  "array_ref": ["ic_29_f", "ic_29_r"], "inputs": ["ic_29_f", "ic_29_r"], "masks": {},
                  "params": {"min_valid": 30}, "expected": "nan", "dtype": "float32", "tolerance": "exact"})
    cases.append({"id": "ic_valid_30", "operation": "rolling_ic",
                  "array_ref": ["ic_30_f", "ic_30_r"], "inputs": ["ic_30_f", "ic_30_r"], "masks": {},
                  "params": {"min_valid": 30}, "expected": sp(D["ic_30_f"], D["ic_30_r"], 30),
                  "dtype": "float32", "tolerance": "1e-12"})
    cases.append({"id": "ic_valid_31", "operation": "rolling_ic",
                  "array_ref": ["ic_31_f", "ic_31_r"], "inputs": ["ic_31_f", "ic_31_r"], "masks": {},
                  "params": {"min_valid": 30}, "expected": sp(D["ic_31_f"], D["ic_31_r"], 30),
                  "dtype": "float32", "tolerance": "1e-12"})
    cases.append({"id": "ic_all_invalid", "operation": "rolling_ic",
                  "array_ref": ["ic_all_invalid_f", "ic_all_invalid_r"],
                  "inputs": ["ic_all_invalid_f", "ic_all_invalid_r"], "masks": {},
                  "params": {"min_valid": 30}, "expected": "nan", "dtype": "float32", "tolerance": "exact"})
    cases.append({"id": "ic_two_sided_nan_inf_mask", "operation": "rolling_ic",
                  "array_ref": ["ic_inf_f", "ic_neginf_r"], "inputs": ["ic_inf_f", "ic_neginf_r"], "masks": {},
                  "params": {"min_valid": 2}, "expected": _spearman_ic(D["ic_inf_f"], D["ic_neginf_r"]),
                  "dtype": "float32", "tolerance": "1e-12"})
    cases.append({"id": "ic_factor_scatter_nan", "operation": "rolling_ic",
                  "array_ref": ["ic_nan_f", "ic_nan_r"], "inputs": ["ic_nan_f", "ic_nan_r"], "masks": {},
                  "params": {"min_valid": 2}, "expected": _spearman_ic(D["ic_nan_f"], D["ic_nan_r"]),
                  "dtype": "float32", "tolerance": "1e-12"})
    cases.append({"id": "ic_tie1", "operation": "rolling_ic",
                  "array_ref": ["ic_tie_f", "ic_tie_r1"], "inputs": ["ic_tie_f", "ic_tie_r1"], "masks": {},
                  "params": {"min_valid": 2}, "expected": _spearman_ic(D["ic_tie_f"], D["ic_tie_r1"]),
                  "dtype": "float32", "tolerance": "1e-12"})
    cases.append({"id": "ic_tie2", "operation": "rolling_ic",
                  "array_ref": ["ic_tie_f", "ic_tie_r2"], "inputs": ["ic_tie_f", "ic_tie_r2"], "masks": {},
                  "params": {"min_valid": 2}, "expected": _spearman_ic(D["ic_tie_f"], D["ic_tie_r2"]),
                  "dtype": "float32", "tolerance": "1e-12"})
    cases.append({"id": "ic_factor_tradable_nan", "operation": "rolling_ic",
                  "array_ref": ["ic_tradable_nan_f", "ic_tradable_nan_r"],
                  "inputs": ["ic_tradable_nan_f", "ic_tradable_nan_r"], "masks": {},
                  "params": {"min_valid": 2}, "expected": _spearman_ic(D["ic_tradable_nan_f"], D["ic_tradable_nan_r"]),
                  "dtype": "float32", "tolerance": "1e-12"})

    # ---- correlation 组 ----
    cases.append({"id": "corr_a_b", "operation": "factor_corr",
                  "array_ref": ["corr_a", "corr_b"], "inputs": ["corr_a", "corr_b"], "masks": {},
                  "params": {}, "expected": "nan", "dtype": "float64", "tolerance": "exact"})
    cases.append({"id": "corr_a_c", "operation": "factor_corr",
                  "array_ref": ["corr_a", "corr_c"], "inputs": ["corr_a", "corr_c"], "masks": {},
                  "params": {}, "expected": _corr(D["corr_a"], D["corr_c"]), "dtype": "float64", "tolerance": "1e-12"})
    cases.append({"id": "corr_n2_self", "operation": "factor_corr",
                  "array_ref": ["corr_n2", "corr_n2"], "inputs": ["corr_n2", "corr_n2"], "masks": {},
                  "params": {}, "expected": _corr(D["corr_n2"], D["corr_n2"]), "dtype": "float64", "tolerance": "1e-12"})
    cases.append({"id": "corr_n2_reverse", "operation": "factor_corr",
                  "array_ref": ["corr_n2", "corr_n2_rev"], "inputs": ["corr_n2", "corr_n2_rev"], "masks": {},
                  "params": {}, "expected": _corr(D["corr_n2"], D["corr_n2_rev"]), "dtype": "float64", "tolerance": "1e-12"})
    cases.append({"id": "corr_const_self", "operation": "factor_corr",
                  "array_ref": ["corr_const", "corr_const"], "inputs": ["corr_const", "corr_const"], "masks": {},
                  "params": {}, "expected": "nan", "dtype": "float64", "tolerance": "exact"})
    cases.append({"id": "corr_float32_ac", "operation": "factor_corr",
                  "array_ref": ["corr_a32", "corr_c32"], "inputs": ["corr_a32", "corr_c32"], "masks": {},
                  "params": {}, "expected": _corr(D["corr_a32"], D["corr_c32"]), "dtype": "float32", "tolerance": "1e-12"})
    cases.append({"id": "corr_mask_false_finite", "operation": "factor_corr",
                  "array_ref": ["corr_mask_false_finite", "corr_mask_false_finite_m"],
                  "inputs": ["corr_mask_false_finite"], "masks": {"mask": "corr_mask_false_finite_m"},
                  "params": {}, "expected": _corr(D["corr_mask_false_finite"], D["corr_mask_false_finite"],
                                                   D["corr_mask_false_finite_m"]), "dtype": "float64", "tolerance": "1e-12"})
    cases.append({"id": "corr_no_common_valid", "operation": "factor_corr",
                  "array_ref": ["corr_no_common_a", "corr_no_common_b"],
                  "inputs": ["corr_no_common_a", "corr_no_common_b"], "masks": {},
                  "params": {}, "expected": "nan", "dtype": "float64", "tolerance": "exact"})
    cases.append({"id": "corr_underflow_domain", "operation": "factor_corr",
                  "array_ref": ["corr_underflow", "corr_underflow"],
                  "inputs": ["corr_underflow", "corr_underflow"], "masks": {},
                  "params": {}, "expected": "ValueError", "dtype": "float64", "tolerance": "exception"})
    cases.append({"id": "corr_dtype_variant", "operation": "factor_corr",
                  "array_ref": ["corr_a", "corr_c"], "inputs": ["corr_a", "corr_c"], "masks": {},
                  "params": {"dtype": "float64"}, "expected": _corr(D["corr_a"], D["corr_c"]),
                  "dtype": "float64", "tolerance": "1e-12"})
    cases.append({"id": "corr_shape_error", "operation": "factor_corr",
                  "array_ref": ["corr_a"], "inputs": ["corr_a"], "masks": {},
                  "params": {}, "expected": "ValueError", "dtype": "float64", "tolerance": "exception"})

    # ---- error 组 ----
    cases.append({"id": "err_out_of_domain", "operation": "factor_corr",
                  "array_ref": ["err_domain_x", "err_domain_y"],
                  "inputs": ["err_domain_x", "err_domain_y"], "masks": {},
                  "params": {}, "expected": "ValueError", "dtype": "float64", "tolerance": "exception"})

    # ---- parameter_scan 容器锚点（(T,N) 二维输入 + mask + 具体 axes）----
    cases.append({"id": "scan_container", "operation": "parameter_scan",
                  "array_ref": ["scan_x", "scan_mask"], "inputs": ["scan_x"], "masks": {"mask": "scan_mask"},
                  "params": {"axes": [("direction", ["ascending", "descending"]),
                                      ("mask_mode", ["masked", "unmasked"])], "G": 4},
                  "expected": {"groups": ["(ascending,masked)", "(ascending,unmasked)",
                                          "(descending,masked)", "(descending,unmasked)"],
                               "result_shape": "(T,N) float32", "summary_fields": ["total_groups", "n_failed"]},
                  "dtype": "float32", "tolerance": "schema"})

    return cases


def main() -> None:
    data = build_data()
    cases = build_cases(data)
    out = HERE / "parity_anchors_v1.npz"
    np.savez(out, **data)
    digest = hashlib.sha256(out.read_bytes()).hexdigest().upper()
    manifest = {
        "corpus_id": "parity_anchors_v1",
        "family": "parity_anchors",
        "version": "v1",
        "protocol_version": "1",
        "shapes": {"T": 3, "N": 35, "F": 4},
        "arrays": [{"name": k, "dtype": str(v.dtype), "shape": list(v.shape)} for k, v in data.items()],
        "generation": {"script": str(pathlib.Path(__file__).resolve().relative_to(HERE.parent))},
        "generation_params": {"mode": "parity_anchors", "variant": "anchors"},
        "seeds": {"seeds_ref": "seeds.json"},
        "hash": {"data_sha256": digest, "algorithm": "SHA-256"},
        "stats": {},
        "labels": {"h": 5, "lag": 1, "W": 21,
                   "benchmark_row_range": {"start": 21, "stop": "T-(h+lag)", "stop_exclusive": True}},
        "env": {"python": "3.12.7", "numpy": "2.4.4",
                "env_fingerprint": "python-3.12.7_numpy-2.4.4"},
        "case_schema": cases,
        "groups": ["rank", "rolling_ic", "correlation", "error", "parameter_scan"],
    }
    (HERE / "parity_anchors_v1.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"generated {out} sha256={digest}")
    print(f"cases: {len(cases)}")


if __name__ == "__main__":
    main()
