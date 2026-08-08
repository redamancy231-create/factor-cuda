# -*- coding: utf-8 -*-
"""生成 BinaryFrontier、补偿求和、排序键、alias 与 offset 机械 trace。"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import struct

from corr_math_v1 import (
    SAFE_PEARSON_EXPRESSION,
    INT32_MAX,
    SIZE_T_MAX,
    BinaryFrontier,
    CompensatedSum,
    KernelLayout,
    STRUCT_ABI,
    ViewSpec,
    canonical_ordinal_key_f32,
    canonical_ordinal_key_f64,
    checked_byte_offset,
    checked_global_element_offset,
    checked_scatter_out_base,
    expected_fixed_tree,
    factor_alias_failures,
    factor_is_aliasable,
    safe_pearson,
    select_factor_input_path,
    select_stock_input_path,
    stable_ordinal_order,
    stock_alias_failures,
    stock_is_aliasable,
)

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "corr_math_trace_v1.json"
VERSION = "1.0.0"


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _merge_text(left: str, right: str) -> str:
    return f"({left},{right})"


def _frontier_trace(count: int) -> dict:
    frontier = BinaryFrontier(_merge_text, max_levels=6)
    steps = []
    for index in range(count):
        frontier.ingest(index, str(index))
        steps.append({"leaf_index": index, "after": frontier.snapshot()})
    leaves = [str(index) for index in range(count)]
    expected = expected_fixed_tree(leaves, _merge_text)
    actual = frontier.finalize()
    assert actual == expected
    return {"leaf_count": count, "steps": steps, "final": actual, "expected": expected}


def _chunk_trace() -> dict:
    chunks = [list(range(0, 3)), list(range(3, 8)), list(range(8, 11)), list(range(11, 18))]
    frontier = BinaryFrontier(_merge_text, max_levels=6)
    records = []
    for chunk_index, chunk in enumerate(chunks):
        for leaf_index in chunk:
            frontier.ingest(leaf_index, str(leaf_index))
        records.append(
            {
                "chunk_index": chunk_index,
                "leaf_indices": chunk,
                "flush_snapshot": frontier.flush(),
            }
        )
    expected = expected_fixed_tree([str(index) for index in range(18)], _merge_text)
    actual = frontier.finalize()
    assert actual == expected
    return {"chunks": records, "final": actual, "expected": expected}


def _capacity_trace() -> dict:
    frontier = BinaryFrontier(_merge_text, max_levels=4)
    for index in range(16):
        frontier.ingest(index, str(index))
    full_tree = frontier.finalize()
    try:
        frontier.ingest(16, "16")
    except OverflowError as exc:
        overflow = {"type": type(exc).__name__, "message": str(exc)}
    else:
        raise AssertionError("capacity guard did not fire")

    sequence = BinaryFrontier(_merge_text, max_levels=4)
    try:
        sequence.ingest(1, "1")
    except ValueError as exc:
        sequence_error = {"type": type(exc).__name__, "message": str(exc)}
    else:
        raise AssertionError("sequence guard did not fire")
    return {"capacity": 16, "full_tree": full_tree, "overflow": overflow, "sequence": sequence_error}


def _compensated_trace() -> dict:
    left = CompensatedSum.from_values([-1e16])
    right = CompensatedSum.from_values([1e16, 1.0])
    correct = CompensatedSum(left.sum, left.c)
    correct.merge(right)
    wrong = CompensatedSum(left.sum, left.c)
    wrong.add(right.sum)
    wrong.add(right.c)
    assert correct.represented == 1.0
    assert wrong.represented == -1.0
    return {
        "left_values": [-1e16],
        "right_values": [1e16, 1.0],
        "left_state": {"sum": left.sum, "c": left.c, "represented": left.represented},
        "right_state": {"sum": right.sum, "c": right.c, "represented": right.represented},
        "correct_merge": {"replay": ["right.sum", "-right.c"], "represented": correct.represented},
        "wrong_sign_counterexample": {"replay": ["right.sum", "right.c"], "represented": wrong.represented},
    }


def _pearson_trace() -> dict:
    sxy = 8.163535225534784e199
    sxx = 4.794141230732863e198
    syy = 7.080783202392912e201
    sequential = safe_pearson(sxy, sxx, syy)
    root_product = sxy / (math.sqrt(sxx) * math.sqrt(syy))
    assert sequential.hex() == "0x1.c5b6c7cd447e1p-2"
    assert root_product.hex() == "0x1.c5b6c7cd447e0p-2"
    return {
        "inputs": {"Sxy": sxy, "Sxx": sxx, "Syy": syy},
        "expression": SAFE_PEARSON_EXPRESSION,
        "sequential_hex": sequential.hex(),
        "root_product_hex": root_product.hex(),
        "distinguishable": sequential != root_product,
    }


def _f32_nextafter_one() -> float:
    return struct.unpack("<f", struct.pack("<I", 0x3F800001))[0]


def _ordinal_trace() -> dict:
    records = {}
    for dtype, key_fn in (
        ("float32", canonical_ordinal_key_f32),
        ("float64", canonical_ordinal_key_f64),
    ):
        pair = [-0.0, 0.0]
        asc = stable_ordinal_order(pair, dtype=dtype, descending=False)
        desc = stable_ordinal_order(pair, dtype=dtype, descending=True)
        keys = [key_fn(value) for value in pair]
        assert keys[0] == keys[1]
        assert asc == [0, 1] and desc == [0, 1]
        large = [-0.0 if index % 2 == 0 else 0.0 for index in range(600)]
        assert stable_ordinal_order(large, dtype=dtype) == list(range(600))
        assert stable_ordinal_order(large, dtype=dtype, descending=True) == list(range(600))
        records[dtype] = {
            "sign_bits": [1, 0],
            "canonical_keys": keys,
            "ascending_order": asc,
            "descending_order": desc,
            "large_tie_count": len(large),
            "large_tie_ascending_stable": True,
            "large_tie_descending_stable": True,
        }

    f32_values = [2.0] * 258
    f32_values[255] = 1.0
    f32_values[256] = _f32_nextafter_one()
    f64_values = [2.0] * 258
    f64_values[255] = 1.0
    f64_values[256] = math.nextafter(1.0, 2.0)
    cross = {}
    for dtype, values in (("float32", f32_values), ("float64", f64_values)):
        order = stable_ordinal_order(values, dtype=dtype)
        pos_left = order.index(255)
        pos_right = order.index(256)
        assert pos_right == pos_left + 1
        cross[dtype] = {
            "left_index": 255,
            "right_index": 256,
            "left_value_hex": float(values[255]).hex(),
            "right_value_hex": float(values[256]).hex(),
            "sorted_positions": [pos_left, pos_right],
            "crosses_microtile_256": True,
        }
    return {"zero_ties": records, "adjacent_ulp_boundary": cross}


def _offset_trace() -> dict:
    exact_row = INT32_MAX // 5000
    scatter_exact = checked_scatter_out_base(exact_row, 5000)
    try:
        checked_scatter_out_base(exact_row + 1, 5000)
    except OverflowError as exc:
        scatter_overflow = type(exc).__name__
    else:
        raise AssertionError("scatter overflow guard did not fire")

    element_exact = SIZE_T_MAX // 8
    byte_exact = checked_byte_offset(element_exact, 8)
    try:
        checked_byte_offset(element_exact + 1, 8)
    except OverflowError as exc:
        byte_overflow = type(exc).__name__
    else:
        raise AssertionError("byte overflow guard did not fire")

    global_exact = checked_global_element_offset(10, 2, 5000, 4999)
    assert global_exact == 64999
    return {
        "scatter_out_base": {
            "row_base": exact_row,
            "N": 5000,
            "exact": scatter_exact,
            "overflow_next_row": scatter_overflow,
        },
        "global_element_offset": {
            "row_base": 10,
            "local_row": 2,
            "N": 5000,
            "local_column": 4999,
            "exact": global_exact,
        },
        "byte_offset": {
            "element_offset": element_exact,
            "item_size": 8,
            "exact": byte_exact,
            "overflow_next_element": byte_overflow,
        },
    }


def _alias_trace() -> dict:
    factor_layout = KernelLayout("float64", (2, 3, 4), (12, 4, 1), 16, "cuda:0", "factor-owner", 0)
    factor_full = ViewSpec("float64", (2, 3, 4), (12, 4, 1), 8, 0, 4096, 192, "cuda:0", "factor-owner", True, True)
    factor_sub = ViewSpec("float64", (2, 3, 2), (12, 4, 1), 8, 0, 4096, 192, "cuda:0", "factor-owner", True, True)
    factor_sub_layout = KernelLayout("float64", (2, 3, 2), (6, 2, 1), 16, "cuda:0", "factor-owner", 0)
    factor_f32 = ViewSpec("float32", (2, 3, 4), (12, 4, 1), 4, 0, 4096, 96, "cuda:0", "factor-owner", True, True)

    stock_layout = KernelLayout("float64", (3, 4), (4, 1), 16, "cuda:0", "stock-owner", 0)
    stock_full = ViewSpec("float64", (3, 4), (4, 1), 8, 0, 8192, 96, "cuda:0", "stock-owner", True, True)
    stock_sub = ViewSpec("float64", (3, 2), (4, 1), 8, 0, 8192, 96, "cuda:0", "stock-owner", True, True)
    stock_sub_layout = KernelLayout("float64", (3, 2), (2, 1), 16, "cuda:0", "stock-owner", 0)
    stock_f32 = ViewSpec("float32", (3, 4), (4, 1), 4, 0, 8192, 48, "cuda:0", "stock-owner", True, True)

    assert factor_is_aliasable(factor_full, factor_layout)
    assert not factor_is_aliasable(factor_sub, factor_sub_layout)
    assert stock_is_aliasable(stock_full, stock_layout)
    assert not stock_is_aliasable(stock_sub, stock_sub_layout)
    return {
        "factor": {
            "full_width_aliasable": factor_is_aliasable(factor_full, factor_layout),
            "full_width_path": select_factor_input_path(factor_full, factor_layout),
            "retained_base_stride_subblock_failures": factor_alias_failures(factor_sub, factor_sub_layout),
            "retained_base_stride_subblock_path": select_factor_input_path(factor_sub, factor_sub_layout),
            "f32_path": select_factor_input_path(factor_f32, factor_layout),
        },
        "stock": {
            "full_width_aliasable": stock_is_aliasable(stock_full, stock_layout),
            "full_width_path": select_stock_input_path(stock_full, stock_layout),
            "retained_base_stride_subblock_failures": stock_alias_failures(stock_sub, stock_sub_layout),
            "retained_base_stride_subblock_path": select_stock_input_path(stock_sub, stock_sub_layout),
            "f32_path": select_stock_input_path(stock_f32, stock_layout),
        },
        "predicate_dimensions": [
            "dtype", "shape", "strides", "alignment", "byte_offset", "device",
            "storage_bounds", "owner", "sync_lifetime"
        ],
    }


def build_trace() -> dict:
    return {
        "schema_version": VERSION,
        "generator": "tests/fixtures/generate_corr_math_trace_v1.py",
        "generator_sha256": _sha(pathlib.Path(__file__)),
        "math_source": "tests/fixtures/corr_math_v1.py",
        "math_source_sha256": _sha(HERE / "corr_math_v1.py"),
        "binary_frontier": {
            "leaf_counts": [_frontier_trace(count) for count in range(18)],
            "cross_chunk": _chunk_trace(),
            "guards": _capacity_trace(),
        },
        "compensated_sum": _compensated_trace(),
        "safe_pearson": _pearson_trace(),
        "ordinal": _ordinal_trace(),
        "checked_offsets": _offset_trace(),
        "aliasability": _alias_trace(),
        "struct_abi": STRUCT_ABI,
    }


def main() -> int:
    payload = build_trace()
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"generated: {OUT}")
    print("frontier leaf counts: 0..17")
    print("compensated counterexample: +1.0 (wrong sign: -1.0)")
    print(f"sha256: {_sha(OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
