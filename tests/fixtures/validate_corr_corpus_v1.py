# -*- coding: utf-8 -*-
"""独立校验 corr_corpus_v1 的字节、语义与稳定 tie。"""
from __future__ import annotations

import hashlib
import inspect
import io
import json
import math
import pathlib

import numpy as np

import generate_corr_corpus_v1 as generator
from corr_math_v1 import safe_pearson
from corr_oracle_v1 import corr_oracle

HERE = pathlib.Path(__file__).resolve().parent


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate() -> None:
    committed_npz = (HERE / "corr_corpus_v1.npz").read_bytes()
    committed_manifest = json.loads((HERE / "corr_corpus_v1.manifest.json").read_text(encoding="utf-8"))
    rebuilt_npz, rebuilt_manifest = generator.build_artifacts()
    assert committed_npz == rebuilt_npz, "NPZ 不是确定性重建结果"
    assert committed_manifest == rebuilt_manifest, "manifest 不是确定性重建结果"
    assert committed_manifest["version"] == "1.2.0"
    assert committed_manifest["case_count"] == 16
    assert len(committed_manifest["cases"]) == 16
    assert committed_manifest["npz_sha256"] == _sha(committed_npz)

    source = inspect.getsource(safe_pearson)
    assert "(sxy / math.sqrt(sxx)) / math.sqrt(syy)" in source
    with np.load(io.BytesIO(committed_npz), allow_pickle=False) as archive:
        case_by_id = {case["id"]: case for case in committed_manifest["cases"]}
        for case_id, case in case_by_id.items():
            a = archive[f"{case_id}_a"]
            b = archive[f"{case_id}_b"]
            mask_a_name = f"{case_id}_mask_a"
            mask_b_name = f"{case_id}_mask_b"
            mask_a = archive[mask_a_name] if mask_a_name in archive.files else None
            mask_b = archive[mask_b_name] if mask_b_name in archive.files else None
            assert hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() == case["a_sha256"]
            assert hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest() == case["b_sha256"]
            expected = corr_oracle(a.astype(np.float64), b.astype(np.float64), mask_a=mask_a, mask_b=mask_b)
            if case["expected_is_nan"]:
                assert math.isnan(expected)
                assert case["expected"] is None
            else:
                assert abs(expected - case["expected"]) <= 1e-15
            branch, metrics = generator._trigger(a, b, mask_a=mask_a, mask_b=mask_b)
            if math.isnan(expected):
                branch = "nan"
            assert branch == case["branch"]
            assert int(metrics["n"]) == int(case["valid_count"])

        for case_id, dtype in (("stable_zero_f32", "float32"), ("stable_zero_f64", "float64")):
            metadata = case_by_id[case_id]["stable_tie"]
            assert metadata["sign_bits"] == [1, 0]
            assert metadata["canonical_key_equal"] is True
            assert metadata["canonical_keys"][0] == metadata["canonical_keys"][1]
            assert metadata["expected_stable_order"] == [0, 1]
            assert metadata["ascending_order"] == [0, 1]
            assert metadata["descending_order"] == [0, 1]
            assert case_by_id[case_id]["dtype"] == dtype

    pearson = committed_manifest["cases"][[case["id"] for case in committed_manifest["cases"]].index("pearson_overflow")]
    adjacent = committed_manifest["cases"][[case["id"] for case in committed_manifest["cases"]].index("f64_adjacent_ulp")]
    assert abs(pearson["expected"] - 1.0) <= 1e-15
    assert abs(adjacent["expected"] - 1.0) <= 1e-15


def main() -> int:
    validate()
    print("corr_corpus_v1: PASS version=1.2.0 cases=16 byte_exact=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
