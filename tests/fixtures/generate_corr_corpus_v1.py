# -*- coding: utf-8 -*-
"""生成 correlation 对抗 corpus v1.2（确定性 NPZ 与严格 manifest）。"""
from __future__ import annotations

import hashlib
import io
import json
import pathlib
import zipfile

import numpy as np

from corr_math_v1 import (
    canonical_ordinal_key_f32,
    canonical_ordinal_key_f64,
    safe_pearson,
    stable_ordinal_order,
)
from corr_oracle_v1 import corr_oracle

HERE = pathlib.Path(__file__).resolve().parent
SEED = 20260804
NPZ = HERE / "corr_corpus_v1.npz"
MANIFEST = HERE / "corr_corpus_v1.manifest.json"
VERSION = "1.2.0"
BIAS_THRESHOLD = 1e8


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _arr_sha(array: np.ndarray) -> str:
    return _sha256(np.ascontiguousarray(array).tobytes())


def _file_sha(path: pathlib.Path) -> str:
    return _sha256(path.read_bytes())


def _trigger(a, b, *, mask_a=None, mask_b=None):
    """冻结触发状态机；Pearson 只调用共享 ``safe_pearson``。"""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if mask_a is not None:
        valid &= np.asarray(mask_a, dtype=bool)
    if mask_b is not None:
        valid &= np.asarray(mask_b, dtype=bool)
    x, y = a[valid], b[valid]
    n = int(valid.sum())
    if n < 2:
        return "nan", {"n": n, "bias_metric": None, "ratio": None, "r": None}
    xm, ym = x.mean(), y.mean()
    dx, dy = x - xm, y - ym
    sxx = float((dx * dx).sum())
    syy = float((dy * dy).sum())
    if sxx == 0.0 or syy == 0.0:
        return "nan", {"n": n, "bias_metric": None, "ratio": None, "r": None}
    sxy = float((dx * dy).sum())
    r = safe_pearson(sxy, sxx, syy)
    bx = float(np.abs(x).max() / np.sqrt(sxx / n))
    by = float(np.abs(y).max() / np.sqrt(syy / n))
    bias = max(bx, by)
    with np.errstate(over="ignore", invalid="ignore"):
        ss = float((x.sum() * y.sum()) / n)
    ratio = float(abs(ss) / (abs(sxy) + 1e-300))
    metrics = {"n": n, "bias_metric": bias, "ratio": ratio, "r": float(r)}
    if bias > BIAS_THRESHOLD or abs(r) > 1.0 or not np.isfinite(r):
        return "kahan", metrics
    return "normal", metrics


def _stable_tie_metadata(array: np.ndarray, dtype: str) -> dict:
    values = [float(value) for value in array]
    key_fn = canonical_ordinal_key_f32 if dtype == "float32" else canonical_ordinal_key_f64
    keys = [int(key_fn(value)) for value in values]
    ascending = stable_ordinal_order(values, dtype=dtype, descending=False)
    descending = stable_ordinal_order(values, dtype=dtype, descending=True)
    assert keys[0] == keys[1]
    assert ascending == [0, 1] and descending == [0, 1]
    return {
        "sign_bits": [int(bit) for bit in np.signbit(array)],
        "canonical_keys": keys,
        "canonical_key_equal": True,
        "expected_stable_order": [0, 1],
        "ascending_order": ascending,
        "descending_order": descending,
    }


def _mk_case(case_id, a, b, *, mask_a=None, mask_b=None, desc="", dtype="float64", stable_tie=None):
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    mask_a_array = np.asarray(mask_a, dtype=bool) if mask_a is not None else None
    mask_b_array = np.asarray(mask_b, dtype=bool) if mask_b is not None else None
    a_saved = a64.astype(dtype)
    b_saved = b64.astype(dtype)
    a_test = a_saved.astype(np.float64)
    b_test = b_saved.astype(np.float64)
    expected = corr_oracle(a_test, b_test, mask_a=mask_a_array, mask_b=mask_b_array)
    branch, metrics = _trigger(a_test, b_test, mask_a=mask_a_array, mask_b=mask_b_array)
    if np.isnan(expected):
        branch = "nan"
    return {
        "id": case_id,
        "desc": desc,
        "dtype": dtype,
        "a": a_saved,
        "b": b_saved,
        "mask_a": mask_a_array,
        "mask_b": mask_b_array,
        "expected": expected,
        "branch": branch,
        "metrics": metrics,
        "stable_tie": stable_tie,
    }


def build_cases() -> list[dict]:
    rng = np.random.default_rng(SEED)
    sample_count = 100
    z = rng.normal(size=(sample_count, 2))
    a0 = z[:, 0]
    b0 = z[:, 1]
    near_plus = 0.9999999 * a0 + np.sqrt(1 - 0.9999999 ** 2) * b0
    near_minus = -0.99 * a0 + np.sqrt(1 - 0.99 ** 2) * b0
    sparse = rng.random(sample_count) > 0.7
    one_valid = np.zeros(sample_count, dtype=bool)
    one_valid[:1] = True

    cases = [
        _mk_case("bias_1e12", 1e12 + a0, 1e12 * (1 + 1e-6) + b0, desc="base=1e12 偏置触发 Kahan"),
        _mk_case("bias_1e15", 1e15 + a0, 1e15 * (1 + 1e-6) + b0, desc="base=1e15 偏置触发 Kahan"),
        _mk_case("near_plus1", a0, near_plus, desc="近 +1 相关"),
        _mk_case("near_minus1", a0, near_minus, desc="近 -1 相关"),
        _mk_case("near_zero_var", a0, 1.0 + 1e-9 * rng.normal(size=sample_count), desc="近零方差"),
        _mk_case("sparse_mask", a0, b0, mask_a=sparse, mask_b=sparse, desc="稀疏交错 mask"),
        _mk_case("f32_ulp_bias", 1e15 + a0, 1e15 + near_plus, desc="f32 大偏置量化坍缩", dtype="float32"),
        _mk_case("f64_ulp_bias", 1e15 + a0, 1e15 + near_plus, desc="f64 大偏置 ULP", dtype="float64"),
        _mk_case("constant_col", a0, np.full(sample_count, 5.0), desc="常量列"),
        _mk_case("all_invalid", a0, b0, mask_a=np.zeros(sample_count, dtype=bool), desc="全无效"),
        _mk_case("n_lt_2", a0, b0, mask_a=one_valid, desc="有效数小于 2"),
        _mk_case("f1_self", a0, a0.copy(), desc="单因子自相关"),
    ]

    overflow = np.array([-1e150, 1e150])
    adjacent = np.array([1.0, np.nextafter(1.0, 2.0)])
    cases.append(_mk_case("pearson_overflow", overflow, overflow, desc="根号乘积安全 Pearson"))
    cases.append(_mk_case("f64_adjacent_ulp", adjacent, adjacent, desc="f64 相邻 ULP extrema"))

    zero32 = np.array([-0.0, 0.0], dtype=np.float32)
    zero64 = np.array([-0.0, 0.0], dtype=np.float64)
    cases.append(
        _mk_case(
            "stable_zero_f32",
            zero32,
            np.array([-1.0, 1.0], dtype=np.float32),
            desc="f32 -0/+0 canonical key 稳定 tie",
            dtype="float32",
            stable_tie=_stable_tie_metadata(zero32, "float32"),
        )
    )
    cases.append(
        _mk_case(
            "stable_zero_f64",
            zero64,
            np.array([-1.0, 1.0], dtype=np.float64),
            desc="f64 -0/+0 canonical key 稳定 tie",
            dtype="float64",
            stable_tie=_stable_tie_metadata(zero64, "float64"),
        )
    )
    assert len(cases) == 16
    return cases


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def deterministic_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(arrays[name]), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return stream.getvalue()


def build_artifacts() -> tuple[bytes, dict]:
    cases = build_cases()
    arrays: dict[str, np.ndarray] = {}
    case_manifest = []
    for case in cases:
        case_id = case["id"]
        arrays[f"{case_id}_a"] = case["a"]
        arrays[f"{case_id}_b"] = case["b"]
        if case["mask_a"] is not None:
            arrays[f"{case_id}_mask_a"] = case["mask_a"]
        if case["mask_b"] is not None:
            arrays[f"{case_id}_mask_b"] = case["mask_b"]
        expected_is_nan = bool(np.isnan(case["expected"]))
        case_manifest.append(
            {
                "id": case_id,
                "desc": case["desc"],
                "dtype": case["dtype"],
                "shape": list(case["a"].shape),
                "a_sha256": _arr_sha(case["a"]),
                "b_sha256": _arr_sha(case["b"]),
                "mask_a_sha256": _arr_sha(case["mask_a"]) if case["mask_a"] is not None else None,
                "mask_b_sha256": _arr_sha(case["mask_b"]) if case["mask_b"] is not None else None,
                "valid_count": case["metrics"]["n"],
                "trigger": {
                    "bias_metric": case["metrics"]["bias_metric"],
                    "ratio": case["metrics"]["ratio"],
                    "r": case["metrics"]["r"],
                },
                "trigger_threshold": BIAS_THRESHOLD,
                "expected": None if expected_is_nan else float(case["expected"]),
                "expected_is_nan": expected_is_nan,
                "branch": case["branch"],
                "stable_tie": case["stable_tie"],
            }
        )
    npz_bytes = deterministic_npz_bytes(arrays)
    manifest = {
        "corpus_id": "corr_corpus_v1",
        "family": "adversarial",
        "version": VERSION,
        "seed": SEED,
        "case_count": len(case_manifest),
        "safe_pearson_source": "tests/fixtures/corr_math_v1.py::safe_pearson",
        "oracle": "corr_oracle_v1",
        "oracle_sha256": _file_sha(HERE / "corr_oracle_v1.py"),
        "math_source_sha256": _file_sha(HERE / "corr_math_v1.py"),
        "generator_sha256": _file_sha(pathlib.Path(__file__)),
        "npz_sha256": _sha256(npz_bytes),
        "npz_format": {
            "zip_entries_sorted": True,
            "zip_timestamp": "1980-01-01T00:00:00",
            "compression": "deflate-9",
        },
        "cases": case_manifest,
    }
    return npz_bytes, manifest


def main() -> int:
    npz_bytes, manifest = build_artifacts()
    NPZ.write_bytes(npz_bytes)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"cases: {manifest['case_count']}")
    for case in manifest["cases"]:
        expected = "NaN" if case["expected_is_nan"] else f"{case['expected']:.12g}"
        print(f"  {case['id']:<20} dtype={case['dtype']:<7} branch={case['branch']:<6} expected={expected}")
    print(f"npz_sha256: {manifest['npz_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
