# -*- coding: utf-8 -*-
"""PoC ③ workspace v2：逐分配时间线、live-byte HWM 与候选求解器。

该脚本是显存模型的可执行单一真源。所有设备分配按 256 B 对齐；
reserve 只从 total bytes 中扣除一次，candidate required 不再包含 reserve。
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import pathlib
import sys
from typing import Any, Iterable

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
FIXTURES = ROOT / "tests" / "fixtures"
if str(FIXTURES) not in sys.path:
    sys.path.insert(0, str(FIXTURES))

from corr_math_v1 import (  # noqa: E402
    SAFE_PEARSON_EXPRESSION,
    STRUCT_ABI,
    KernelLayout,
    ViewSpec,
    checked_add,
    checked_mul,
    factor_alias_failures,
    factor_is_aliasable,
    safe_pearson,
    select_factor_input_path,
    select_stock_input_path,
    stock_alias_failures,
    stock_is_aliasable,
)

OUT = ROOT / "docs" / "workspace_v1.json"
SCHEMA_VERSION = "2.0.0"
EXECUTION_DATE = "2026-08-04"
ALIGNMENT_B = 256
MIB = 1024 * 1024
TOTAL_MIB = 8188
RESERVE_MIB = 512
TOTAL_BYTES = checked_mul(TOTAL_MIB, MIB)
RESERVE_BYTES = checked_mul(RESERVE_MIB, MIB)
AVAILABLE_BYTES = checked_mul(TOTAL_MIB - RESERVE_MIB, MIB)
MICROTILE = 256
T_CANONICAL = 1218
N_CANONICAL = 5000
F_CANONICAL = 12


def file_sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def align256(value: int) -> int:
    if value < 0:
        raise ValueError("allocation size must be non-negative")
    return checked_mul((checked_add(value, ALIGNMENT_B - 1) // ALIGNMENT_B), ALIGNMENT_B)


def levels(elements: int) -> tuple[int, int]:
    current = (checked_add(elements, MICROTILE - 1)) // MICROTILE
    return current, (checked_add(current, 1)) // 2


def state_size(accumulation: str) -> int:
    if accumulation == "normal":
        return max(STRUCT_ABI["Partial1"]["size_B"], STRUCT_ABI["Partial2"]["size_B"])
    if accumulation == "kahan":
        return max(STRUCT_ABI["PartialK1"]["size_B"], STRUCT_ABI["PartialK2"]["size_B"])
    raise ValueError(f"unknown accumulation: {accumulation}")


def ws_bytes(elements: int, resident_pairs: int, accumulation: str = "normal") -> dict[str, int]:
    current, nxt = levels(elements)
    unit = state_size(accumulation)
    current_bytes = checked_mul(checked_mul(current, unit), resident_pairs)
    next_bytes = checked_mul(checked_mul(nxt, unit), resident_pairs)
    return {
        "levels_current": current,
        "levels_next": nxt,
        "state_size_B": unit,
        "current_B": current_bytes,
        "next_B": next_bytes,
        "peak_ws_B": checked_add(current_bytes, next_bytes),
    }


class Timeline:
    """记录设备分配/释放以及每一步之后的 live bytes。"""

    def __init__(self) -> None:
        self.live = 0
        self.hwm = 0
        self.hwm_stage = "initial"
        self.allocations: dict[str, int] = {}
        self.buffers: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []

    def _record(self, *, stage: str, action: str, name: str, category: str,
                logical_bytes: int, aligned_bytes: int, counted: bool, note: str) -> None:
        event = {
            "index": len(self.events),
            "stage": stage,
            "action": action,
            "name": name,
            "category": category,
            "logical_bytes": logical_bytes,
            "aligned_bytes": aligned_bytes,
            "counted_in_hwm": counted,
            "live_after_B": self.live,
            "note": note,
        }
        self.events.append(event)
        if self.live > self.hwm:
            self.hwm = self.live
            self.hwm_stage = stage

    def external(self, name: str, category: str, logical_bytes: int, *, note: str) -> None:
        aligned = align256(logical_bytes)
        self.buffers[name] = {
            "category": category,
            "logical_bytes": logical_bytes,
            "aligned_bytes": aligned,
            "residence": "external_preexisting",
        }
        self._record(stage="input_ready", action="external", name=name, category=category,
                     logical_bytes=logical_bytes, aligned_bytes=aligned, counted=False, note=note)

    def alloc(self, stage: str, name: str, category: str, logical_bytes: int, *, note: str = "") -> None:
        if name in self.allocations:
            raise RuntimeError(f"duplicate live allocation: {name}")
        aligned = align256(logical_bytes)
        self.live = checked_add(self.live, aligned)
        self.allocations[name] = aligned
        self.buffers[name] = {
            "category": category,
            "logical_bytes": logical_bytes,
            "aligned_bytes": aligned,
            "residence": "allocated",
        }
        self._record(stage=stage, action="alloc", name=name, category=category,
                     logical_bytes=logical_bytes, aligned_bytes=aligned, counted=True, note=note)

    def marker(self, stage: str, note: str) -> None:
        self._record(stage=stage, action="marker", name="-", category="metadata",
                     logical_bytes=0, aligned_bytes=0, counted=False, note=note)

    def free(self, stage: str, name: str) -> None:
        if name not in self.allocations:
            raise RuntimeError(f"free of non-live allocation: {name}")
        aligned = self.allocations.pop(name)
        buffer = self.buffers[name]
        self.live -= aligned
        self._record(stage=stage, action="free", name=name, category=buffer["category"],
                     logical_bytes=buffer["logical_bytes"], aligned_bytes=aligned,
                     counted=True, note="lifetime ended")

    def finish(self) -> dict[str, Any]:
        if self.allocations:
            raise RuntimeError(f"timeline leaked allocations: {sorted(self.allocations)}")
        categories = sorted({item["category"] for item in self.buffers.values()})
        return {
            "alignment_B": ALIGNMENT_B,
            "hwm_B": self.hwm,
            "hwm_stage": self.hwm_stage,
            "final_live_B": self.live,
            "categories": categories,
            "buffers": self.buffers,
            "events": self.events,
        }


def input_bytes(operation: str, total_width: int, path: str) -> int:
    elements = checked_mul(T_CANONICAL, checked_mul(N_CANONICAL, total_width)) if operation == "factor_corr" else checked_mul(T_CANONICAL, total_width)
    item_size = 4 if path == "f32_conversion" else 8
    return checked_mul(elements, item_size)


def mask_bytes(operation: str, total_width: int) -> int:
    elements = checked_mul(T_CANONICAL, checked_mul(N_CANONICAL, total_width)) if operation == "factor_corr" else checked_mul(T_CANONICAL, total_width)
    return elements


def pair_count(kind: str, left_width: int, right_width: int) -> int:
    if kind == "diagonal":
        if left_width != right_width:
            raise ValueError("diagonal widths must match")
        return checked_mul(left_width, checked_add(left_width, 1)) // 2
    return checked_mul(left_width, right_width)


def compressed_pairs(widths: list[int]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, int, int]] = Counter()
    for left_index, left_width in enumerate(widths):
        for right_index in range(left_index + 1):
            right_width = widths[right_index]
            kind = "diagonal" if left_index == right_index else "off_diagonal"
            edge = left_width != widths[0] or right_width != widths[0]
            label = "edge_" + kind if edge else kind
            counts[(label, left_width, right_width)] += 1
    return [
        {
            "pair_type": key[0],
            "left_width": key[1],
            "right_width": key[2],
            "resident_pairs": pair_count("diagonal" if "diagonal" in key[0] and "off" not in key[0] else "off_diagonal", key[1], key[2]),
            "multiplicity": multiplicity,
        }
        for key, multiplicity in sorted(counts.items())
    ]

def corr_pair_timeline(*, operation: str, total_width: int, path: str, output_mode: str,
                       accumulation: str, pair: dict[str, Any]) -> dict[str, Any]:
    timeline = Timeline()
    source_size = input_bytes(operation, total_width, path)
    masks_size = mask_bytes(operation, total_width)
    input_origin = "cpu_host" if output_mode == "cpu" else "cuda"
    if input_origin == "cpu_host":
        timeline.alloc("h2d_source", "source_device", "source", source_size,
                       note="完整 CPU source 的 H2D device allocation")
        timeline.alloc("h2d_mask", "mask_device", "mask", masks_size,
                       note="完整 CPU mask 的 H2D device allocation")
    else:
        timeline.external("source_device", "source", source_size,
                          note="CUDA 输入，source 为调用方预先存在的同设备 allocation")
        timeline.external("mask_device", "mask", masks_size,
                          note="CUDA 输入，mask 为调用方预先存在的同设备 allocation")

    output_size = checked_mul(checked_mul(total_width, total_width), 8)
    if output_mode == "cuda":
        timeline.alloc("result_allocation", "output_device", "output", output_size,
                       note="完整 float64 result 在返回前常驻 CUDA device")
        timeline.external("staging_not_used", "staging", 0,
                          note="同设备输出不使用 D2H tile staging")
    elif output_mode == "cpu":
        timeline.external("output_host", "output", output_size,
                          note="完整结果位于 CPU，不计入设备 HWM")
    else:
        raise ValueError(f"unknown output mode: {output_mode}")

    timeline.alloc("runtime_setup", "event_pair", "event", 2 * 64,
                   note="start/end CUDA events")
    timeline.alloc("runtime_setup", "metadata", "metadata", 192,
                   note="block pair、checked offsets 与状态机 metadata")

    left_width = int(pair["left_width"])
    right_width = int(pair["right_width"])
    resident_pairs = int(pair["resident_pairs"])
    diagonal = "diagonal" in pair["pair_type"] and "off" not in pair["pair_type"]
    raw_width = left_width if diagonal else checked_add(left_width, right_width)
    raw_elements = (
        checked_mul(checked_mul(T_CANONICAL, N_CANONICAL), raw_width)
        if operation == "factor_corr"
        else checked_mul(T_CANONICAL, raw_width)
    )
    raw_size = checked_mul(raw_elements, 8)
    if path == "f64_alias":
        timeline.external("raw_alias_view", "raw", 0,
                          note="factor/stock 独立 is_aliasable 已证明，raw 直接引用 source")
    elif path in {"f32_conversion", "f64_gather"}:
        timeline.alloc("prepare_raw", "raw_pair", "raw", raw_size,
                       note="f32 转 f64 或非 aliasable f64 gather 的 pair-local packed buffer")
    else:
        raise ValueError(f"unknown input path: {path}")

    timeline.alloc("pass1_means", "means", "means", checked_mul(checked_add(left_width, right_width), 8),
                   note="pair-local double means")
    timeline.alloc("trigger_evaluation", "trigger", "trigger", resident_pairs,
                   note="normal/Kahan branch trigger bitmap")
    ws = ws_bytes(
        checked_mul(T_CANONICAL, N_CANONICAL) if operation == "factor_corr" else T_CANONICAL,
        resident_pairs,
        accumulation,
    )
    timeline.alloc("reduce_current", "partial_current", "current", ws["current_B"],
                   note="当前层 partial；每层单槽，连续绝对叶序")
    timeline.alloc("reduce_next", "partial_next", "next", ws["next_B"],
                   note="下一层 partial；与 current 同时存活")
    timeline.alloc("kernel_temp", "kernel_temp", "temp", max(256, checked_mul(resident_pairs, 8)),
                   note="scan/finalize 临时空间")

    tile_result = checked_mul(resident_pairs, 8)
    if output_mode == "cpu":
        timeline.alloc("d2h_tile", "output_staging", "staging", tile_result,
                       note="CPU output 仅保留当前 block-pair tile staging")
    timeline.marker("pearson_finalize", SAFE_PEARSON_EXPRESSION)
    if safe_pearson(1.0, 1.0, 1.0) != 1.0:
        raise AssertionError("safe Pearson helper drift")

    if output_mode == "cpu":
        timeline.free("d2h_complete", "output_staging")
    timeline.free("kernel_complete", "kernel_temp")
    timeline.free("kernel_complete", "partial_next")
    timeline.free("kernel_complete", "partial_current")
    timeline.free("pair_complete", "trigger")
    timeline.free("pair_complete", "means")
    if path != "f64_alias":
        timeline.free("pair_complete", "raw_pair")
    timeline.free("runtime_teardown", "metadata")
    timeline.free("runtime_teardown", "event_pair")
    if output_mode == "cuda":
        timeline.free("return_transfer", "output_device")
    if input_origin == "cpu_host":
        timeline.free("return_transfer", "mask_device")
        timeline.free("return_transfer", "source_device")
    result = timeline.finish()
    result["input_origin"] = input_origin
    result["pair"] = pair
    result["workspace"] = ws
    return result


def evaluate_corr_candidate(*, operation: str, total_width: int, candidate_width: int,
                            path: str, output_mode: str, accumulation: str) -> dict[str, Any]:
    block_count = (checked_add(total_width, candidate_width - 1)) // candidate_width
    widths = [candidate_width] * (block_count - 1)
    widths.append(total_width - candidate_width * (block_count - 1))
    pairs = compressed_pairs(widths)
    evaluations = []
    worst_timeline: dict[str, Any] | None = None
    for pair in pairs:
        timeline = corr_pair_timeline(
            operation=operation,
            total_width=total_width,
            path=path,
            output_mode=output_mode,
            accumulation=accumulation,
            pair=pair,
        )
        record = {
            **pair,
            "hwm_B": timeline["hwm_B"],
            "hwm_stage": timeline["hwm_stage"],
        }
        evaluations.append(record)
        if worst_timeline is None or timeline["hwm_B"] > worst_timeline["hwm_B"]:
            worst_timeline = timeline
    assert worst_timeline is not None
    required = int(worst_timeline["hwm_B"])
    return {
        "candidate_width": candidate_width,
        "B": block_count,
        "widths": widths,
        "required_bytes": required,
        "required_minus_one": max(0, required - 1),
        "fits_available": required <= AVAILABLE_BYTES,
        "failure_boundary": {
            "budget_equal_required_B": required,
            "fits_when_budget_equal_required": True,
            "budget_required_minus_one_B": max(0, required - 1),
            "fits_when_budget_required_minus_one": False,
        },
        "worst_pair": worst_timeline["pair"],
        "worst_stage": worst_timeline["hwm_stage"],
        "buffer_bytes": worst_timeline["buffers"],
        "timeline": worst_timeline["events"],
        "timeline_categories": worst_timeline["categories"],
        "final_live_B": worst_timeline["final_live_B"],
        "pair_evaluations": evaluations,
    }


def candidate_summary(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        key: candidate[key]
        for key in (
            "candidate_width", "B", "widths", "required_bytes", "required_minus_one",
            "fits_available", "failure_boundary", "worst_pair", "worst_stage",
        )
    }


def solve_corr_scenario(*, operation: str, total_width: int, path: str,
                        output_mode: str, accumulation: str) -> dict[str, Any]:
    chosen = None
    runner_up = None
    first_infeasible = None
    checked_candidates: list[dict[str, Any]] = []
    for width in range(total_width, 0, -1):
        candidate = evaluate_corr_candidate(
            operation=operation,
            total_width=total_width,
            candidate_width=width,
            path=path,
            output_mode=output_mode,
            accumulation=accumulation,
        )
        checked_candidates.append(candidate_summary(candidate))
        if candidate["fits_available"]:
            if chosen is None:
                chosen = candidate
            elif runner_up is None:
                runner_up = candidate
        elif first_infeasible is None:
            first_infeasible = candidate
        if chosen is not None and runner_up is not None and (first_infeasible is not None or chosen["candidate_width"] == total_width):
            break
    if first_infeasible is None:
        # 全可行（如 factor F=12 单块）：无 infeasible 候选——报告边界为"超出最大 shape"
        # 以满足 validate 的 failure-boundary 要求（chosen 必须配 first_infeasible）。
        first_infeasible = {
            "candidate_width": total_width + 1,
            "B": 0,
            "widths": [],
            "required_bytes": None,
            "required_minus_one": None,
            "fits_available": False,
            "failure_boundary": {"reason": "candidate_width 超出最大 shape"},
            "worst_pair": None,
            "worst_stage": None,
        }
    return {
        "scenario_id": f"{operation}:{path}:{output_mode}:{accumulation}",
        "operation": operation,
        "input_path": path,
        "output_mode": output_mode,
        "input_origin": "cpu_host" if output_mode == "cpu" else "cuda",
        "accumulation": accumulation,
        "alignment_B": ALIGNMENT_B,
        "available_bytes": AVAILABLE_BYTES,
        "chosen": chosen,
        "runner_up": candidate_summary(runner_up),
        "first_infeasible_candidate": candidate_summary(first_infeasible),
        "checked_candidates": checked_candidates,
    }


def correlation_scenarios(operation: str, total_width: int) -> list[dict[str, Any]]:
    return [
        solve_corr_scenario(
            operation=operation,
            total_width=total_width,
            path=path,
            output_mode=output_mode,
            accumulation=accumulation,
        )
        for path in ("f32_conversion", "f64_alias", "f64_gather")
        for output_mode in ("cpu", "cuda")
        for accumulation in ("normal", "kahan")
    ]

def cub_query(pred_dtype: str, label_dtype: str) -> dict[str, Any]:
    num_items = checked_mul(T_CANONICAL, N_CANONICAL)
    key_bytes = 4 if pred_dtype == "float32" else 8
    value_bytes = 8
    offsets_bytes = checked_mul(T_CANONICAL + 1, 8)
    modeled = align256(checked_add(checked_mul(num_items, key_bytes + value_bytes) // 2, checked_add(offsets_bytes, 4096)))
    return {
        "query_api": "cub::DeviceSegmentedRadixSort::SortPairs",
        "temp_bytes": modeled,
        "api": "cub::DeviceSegmentedRadixSort::SortPairs",
        "query_call": "cub::DeviceSegmentedRadixSort::SortPairs(nullptr, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, num_segments, d_begin_offsets, d_end_offsets, begin_bit, end_bit, stream)",
        "parameters": {
            "d_temp_storage": None,
            "temp_storage_bytes_in": 0,
            "d_keys_in": "ordinal_keys_current",
            "d_keys_out": "ordinal_keys_next",
            "d_values_in": "stable_indices_current",
            "d_values_out": "stable_indices_next",
            "num_items": num_items,
            "num_segments": T_CANONICAL,
            "d_begin_offsets": "segment_offsets[0:T]",
            "d_end_offsets": "segment_offsets[1:T+1]",
            "begin_bit": 0,
            "end_bit": key_bytes * 8,
            "stream": "caller_stream",
            "key_type": "uint32" if key_bytes == 4 else "uint64",
            "value_type": "uint64",
            "prediction_dtype": pred_dtype,
            "label_dtype": label_dtype,
        },
        "query_result_bytes": modeled,
        "query_result_provenance": "workspace_v2 frozen deterministic query fixture",
    }


def rolling_variant_timeline(*, pred_dtype: str, label_dtype: str,
                             output_mode: str, accumulation: str) -> dict[str, Any]:
    timeline = Timeline()
    items = checked_mul(T_CANONICAL, N_CANONICAL)
    pred_source = checked_mul(items, 4 if pred_dtype == "float32" else 8)
    label_source = checked_mul(items, 4 if label_dtype == "float32" else 8)
    masks = checked_mul(items, 2)
    input_origin = "cpu_host" if output_mode == "cpu" else "cuda"
    if input_origin == "cpu_host":
        timeline.alloc("h2d_prediction", "prediction_device", "source", pred_source,
                       note="完整 CPU prediction H2D")
        timeline.alloc("h2d_label", "label_device", "source", label_source,
                       note="完整 CPU label H2D")
        timeline.alloc("h2d_masks", "mask_device", "mask", masks,
                       note="prediction/label 两张完整 mask H2D")
    else:
        timeline.external("prediction_device", "source", pred_source,
                          note="CUDA prediction 预先存在")
        timeline.external("label_device", "source", label_source,
                          note="CUDA label 预先存在")
        timeline.external("mask_device", "mask", masks,
                          note="CUDA masks 预先存在")

    output_size = checked_mul(T_CANONICAL, 8)
    if output_mode == "cuda":
        timeline.alloc("result_allocation", "rolling_output", "output", output_size,
                       note="torch.float64 CUDA result，与输入同设备")
        timeline.external("staging_not_used", "staging", 0,
                          note="CUDA 同设备输出没有 D2H staging")
    else:
        timeline.external("output_host", "output", output_size,
                          note="NumPy float64 host result")

    timeline.alloc("runtime_setup", "event_pair", "event", 128,
                   note="入口到返回事件")
    timeline.alloc("runtime_setup", "rolling_metadata", "metadata", checked_mul(T_CANONICAL + 1, 8),
                   note="segment offsets 与 checked ordinal metadata")
    conversion_bytes = 0
    if pred_dtype == "float32":
        conversion_bytes = checked_add(conversion_bytes, checked_mul(items, 8))
    if label_dtype == "float32":
        conversion_bytes = checked_add(conversion_bytes, checked_mul(items, 8))
    if conversion_bytes:
        timeline.alloc("prepare_raw", "raw_f64", "raw", conversion_bytes,
                       note="mixed dtype 中的 f32 输入转换为 f64 packed raw")
    else:
        timeline.external("raw_alias_view", "raw", 0,
                          note="两个 f64 输入通过独立布局检查后 alias")

    timeline.alloc("row_means", "rolling_means", "means", checked_mul(T_CANONICAL, 16),
                   note="每行 prediction/label double means")
    timeline.alloc("branch_trigger", "rolling_trigger", "trigger", T_CANONICAL,
                   note="每行 normal/Kahan trigger")
    ws = ws_bytes(N_CANONICAL, T_CANONICAL, accumulation)
    timeline.alloc("reduce_current", "rolling_current", "current", ws["current_B"],
                   note="levels_current partials")
    timeline.alloc("reduce_next", "rolling_next", "next", ws["next_B"],
                   note="levels_next partials，与 current 同时存活")
    query = cub_query(pred_dtype, label_dtype)
    timeline.alloc("cub_sort", "cub_temp", "temp", query["query_result_bytes"],
                   note="CUB SortPairs query 返回的 temp bytes")
    if output_mode == "cpu":
        timeline.alloc("d2h_tile", "rolling_staging", "staging", checked_mul(min(T_CANONICAL, 256), 8),
                       note="CPU output 的逐 tile D2H staging")
    timeline.marker("pearson_finalize", SAFE_PEARSON_EXPRESSION)
    if output_mode == "cpu":
        timeline.free("d2h_complete", "rolling_staging")
    timeline.free("sort_complete", "cub_temp")
    timeline.free("reduce_complete", "rolling_next")
    timeline.free("reduce_complete", "rolling_current")
    timeline.free("row_complete", "rolling_trigger")
    timeline.free("row_complete", "rolling_means")
    if conversion_bytes:
        timeline.free("row_complete", "raw_f64")
    timeline.free("runtime_teardown", "rolling_metadata")
    timeline.free("runtime_teardown", "event_pair")
    if output_mode == "cuda":
        timeline.free("return_transfer", "rolling_output")
    if input_origin == "cpu_host":
        timeline.free("return_transfer", "mask_device")
        timeline.free("return_transfer", "label_device")
        timeline.free("return_transfer", "prediction_device")
    result = timeline.finish()
    result.update({
        "accumulation": accumulation,
        "workspace": ws,
        "cub_query": query,
        "required_bytes": result["hwm_B"],
        "required_minus_one": max(0, result["hwm_B"] - 1),
        "failure_boundary": {
            "budget_equal_required_B": result["hwm_B"],
            "fits_when_budget_equal_required": True,
            "budget_required_minus_one_B": max(0, result["hwm_B"] - 1),
            "fits_when_budget_required_minus_one": False,
        },
    })
    return result


def rolling_scenarios() -> list[dict[str, Any]]:
    scenarios = []
    for pred_dtype in ("float32", "float64"):
        for label_dtype in ("float32", "float64"):
            for output_mode in ("cpu", "cuda"):
                variants = {
                    accumulation: rolling_variant_timeline(
                        pred_dtype=pred_dtype,
                        label_dtype=label_dtype,
                        output_mode=output_mode,
                        accumulation=accumulation,
                    )
                    for accumulation in ("normal", "kahan")
                }
                scenarios.append({
                    "scenario_id": f"rolling_ic:{pred_dtype}:{label_dtype}:{output_mode}",
                    "prediction_dtype": pred_dtype,
                    "label_dtype": label_dtype,
                    "output_mode": output_mode,
                    "input_origin": "cpu_host" if output_mode == "cpu" else "cuda",
                    "levels_current": levels(N_CANONICAL)[0],
                    "levels_next": levels(N_CANONICAL)[1],
                    "cub_query": cub_query(pred_dtype, label_dtype),
                    "variants": [variants["normal"], variants["kahan"]],
                })
    return scenarios


def theoretical_workspace(operation: str, width: int) -> dict[str, Any]:
    if operation == "factor_corr":
        elements = checked_mul(T_CANONICAL, N_CANONICAL)
        resident_pairs = checked_mul(width, width + 1) // 2
    elif operation == "stock_corr":
        elements = T_CANONICAL
        resident_pairs = checked_mul(width, width + 1) // 2
    elif operation == "rolling_ic":
        elements = N_CANONICAL
        resident_pairs = T_CANONICAL
    else:
        raise ValueError(operation)
    current, nxt = levels(elements)
    pass_map = {
        "pass1_normal_B": checked_mul(checked_mul(current + nxt, STRUCT_ABI["Partial1"]["size_B"]), resident_pairs),
        "pass2_normal_B": checked_mul(checked_mul(current + nxt, STRUCT_ABI["Partial2"]["size_B"]), resident_pairs),
        "pass1_kahan_B": checked_mul(checked_mul(current + nxt, STRUCT_ABI["PartialK1"]["size_B"]), resident_pairs),
        "pass2_kahan_B": checked_mul(checked_mul(current + nxt, STRUCT_ABI["PartialK2"]["size_B"]), resident_pairs),
    }
    peak = max(pass_map.values())
    return {
        "operation": operation,
        "width": width,
        "elements_per_pair": elements,
        "microtile": MICROTILE,
        "levels_current": current,
        "levels_next": nxt,
        "resident_pairs": resident_pairs,
        **pass_map,
        "peak_ws_B": peak,
        "peak_formula": "peak_ws_B = (levels_current + levels_next) * 56 * resident_pairs",
        "peak_struct": "Partial1",
    }


def alias_contract() -> dict[str, Any]:
    factor_shape = (T_CANONICAL, N_CANONICAL, F_CANONICAL)
    factor_strides = (N_CANONICAL * F_CANONICAL, F_CANONICAL, 1)
    factor_storage = checked_mul(checked_mul(checked_mul(T_CANONICAL, N_CANONICAL), F_CANONICAL), 8)
    factor_layout = KernelLayout("float64", factor_shape, factor_strides, 256, "cuda:0", "factor-owner", 0)
    factor_good = ViewSpec("float64", factor_shape, factor_strides, 8, 0, 4096,
                           factor_storage, "cuda:0", "factor-owner", True, True)
    factor_bad = ViewSpec("float64", factor_shape, (1, T_CANONICAL, T_CANONICAL * N_CANONICAL), 8,
                          0, 4097, factor_storage, "cuda:0", None, False, False)

    stock_shape = (T_CANONICAL, N_CANONICAL)
    stock_strides = (N_CANONICAL, 1)
    stock_storage = checked_mul(checked_mul(T_CANONICAL, N_CANONICAL), 8)
    stock_layout = KernelLayout("float64", stock_shape, stock_strides, 256, "cuda:0", "stock-owner", 0)
    stock_good = ViewSpec("float64", stock_shape, stock_strides, 8, 0, 8192,
                          stock_storage, "cuda:0", "stock-owner", True, True)
    stock_bad = ViewSpec("float64", stock_shape, (1, T_CANONICAL), 8, 8, 8193,
                         stock_storage, "cuda:1", "stock-owner", True, False)
    factor_f32 = ViewSpec("float32", factor_shape, factor_strides, 4, 0, 4096,
                          factor_storage // 2, "cuda:0", "factor-owner", True, True)
    stock_f32 = ViewSpec("float32", stock_shape, stock_strides, 4, 0, 8192,
                         stock_storage // 2, "cuda:0", "stock-owner", True, True)
    return {
        "factor": {
            "predicate": "dtype=float64 AND exact shape/strides/layout offset AND pointer+offset alignment AND same device AND storage bounds AND retained owner AND synchronized lifetime",
            "good_is_aliasable": factor_is_aliasable(factor_good, factor_layout),
            "bad_is_aliasable": factor_is_aliasable(factor_bad, factor_layout),
            "bad_failures": factor_alias_failures(factor_bad, factor_layout),
            "paths": {
                "float32": select_factor_input_path(factor_f32, factor_layout),
                "float64_alias": select_factor_input_path(factor_good, factor_layout),
                "float64_gather": select_factor_input_path(factor_bad, factor_layout),
            },
        },
        "stock": {
            "predicate": "dtype=float64 AND exact shape/strides/layout offset AND pointer+offset alignment AND same device AND storage bounds AND retained owner AND synchronized lifetime",
            "good_is_aliasable": stock_is_aliasable(stock_good, stock_layout),
            "bad_is_aliasable": stock_is_aliasable(stock_bad, stock_layout),
            "bad_failures": stock_alias_failures(stock_bad, stock_layout),
            "paths": {
                "float32": select_stock_input_path(stock_f32, stock_layout),
                "float64_alias": select_stock_input_path(stock_good, stock_layout),
                "float64_gather": select_stock_input_path(stock_bad, stock_layout),
            },
        },
    }

def build_payload() -> dict[str, Any]:
    factor_anchor = theoretical_workspace("factor_corr", F_CANONICAL)
    stock_anchors = [theoretical_workspace("stock_corr", size) for size in (500, 2000, 5000)]
    rolling_anchor = theoretical_workspace("rolling_ic", N_CANONICAL)
    expected = {
        "factor": 155_872_080,
        "stock_500": 56_112_000,
        "stock_2000": 896_448_000,
        "stock_5000": 5_601_120_000,
        "rolling": 2_046_240,
    }
    actual = {
        "factor": factor_anchor["peak_ws_B"],
        "stock_500": stock_anchors[0]["peak_ws_B"],
        "stock_2000": stock_anchors[1]["peak_ws_B"],
        "stock_5000": stock_anchors[2]["peak_ws_B"],
        "rolling": rolling_anchor["peak_ws_B"],
    }
    if actual != expected:
        raise AssertionError(f"workspace anchors drifted: {actual!r}")

    factor = correlation_scenarios("factor_corr", F_CANONICAL)
    stock_sizes = [
        {
            "N": size,
            "scenario_count": 12,
            "scenarios": correlation_scenarios("stock_corr", size),
        }
        for size in (500, 2000, 5000)
    ]
    rolling = rolling_scenarios()
    for scenario in factor:
        if scenario["chosen"] is None:
            raise AssertionError(f"no factor solution: {scenario['scenario_id']}")
    for size in stock_sizes:
        for scenario in size["scenarios"]:
            if scenario["chosen"] is None:
                raise AssertionError(f"no stock solution: N={size['N']} {scenario['scenario_id']}")
    for scenario in rolling:
        for variant in scenario["variants"]:
            if variant["required_bytes"] > AVAILABLE_BYTES:
                raise AssertionError(f"rolling scenario exceeds available: {scenario['scenario_id']}")

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "workspace_v1.json",
        "model": "live-byte HWM allocation timeline v2",
        "execution_date": EXECUTION_DATE,
        "generator": "benchmarks/compute_workspace_v1.py",
        "generator_sha256": file_sha256(pathlib.Path(__file__)),
        "math_source": "tests/fixtures/corr_math_v1.py",
        "math_source_sha256": file_sha256(FIXTURES / "corr_math_v1.py"),
        "safe_pearson_expression": SAFE_PEARSON_EXPRESSION,
        "alignment_B": ALIGNMENT_B,
        "memory_budget": {
            "unit_source": "raw MiB constants",
            "total_MiB": TOTAL_MIB,
            "total_bytes": TOTAL_BYTES,
            "reserve_MiB": RESERVE_MIB,
            "reserve_bytes": RESERVE_BYTES,
            "available_MiB": TOTAL_MIB - RESERVE_MIB,
            "available_bytes": AVAILABLE_BYTES,
            "equation": "8188 MiB - 512 MiB = 7676 MiB = 8048869376 B",
            "reserve_consumption_count": 1,
            "required_includes_reserve": False,
        },
        "canonical_shape": {
            "T": T_CANONICAL,
            "N": N_CANONICAL,
            "F": F_CANONICAL,
            "microtile": MICROTILE,
        },
        "struct_abi": STRUCT_ABI,
        "theoretical_workspace": {
            "factor": factor_anchor,
            "stock": stock_anchors,
            "rolling": rolling_anchor,
            "verified_anchor_bytes": actual,
        },
        "alias_contract": alias_contract(),
        "factor_solver": {
            "scenario_axes": {
                "input_path": ["f32_conversion", "f64_alias", "f64_gather"],
                "output_mode": ["cpu", "cuda"],
                "accumulation": ["normal", "kahan"],
            },
            "scenario_count": len(factor),
            "scenarios": factor,
        },
        "stock_solver": {
            "pair_formula": {
                "diagonal": "left_width * (left_width + 1) / 2",
                "off_diagonal": "left_width * right_width",
                "edge": "same formula with the final short block width",
            },
            "scenario_count_per_N": 12,
            "total_scenario_count": sum(item["scenario_count"] for item in stock_sizes),
            "sizes": stock_sizes,
        },
        "rolling_solver": {
            "dtype_pairs": [
                ["float32", "float32"],
                ["float32", "float64"],
                ["float64", "float32"],
                ["float64", "float64"],
            ],
            "output_modes": ["cpu", "cuda"],
            "scenario_count": len(rolling),
            "variant_count": sum(len(item["variants"]) for item in rolling),
            "scenarios": rolling,
        },
    }


def main() -> int:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    anchors = payload["theoretical_workspace"]["verified_anchor_bytes"]
    print(f"generated: {OUT}")
    print(f"schema_version: {payload['schema_version']}")
    print(f"available: {payload['memory_budget']['available_bytes']} B ({payload['memory_budget']['available_MiB']} MiB)")
    print("workspace anchors: " + ", ".join(f"{key}={value}" for key, value in anchors.items()))
    print(f"factor scenarios: {payload['factor_solver']['scenario_count']}")
    print(f"stock scenarios: {payload['stock_solver']['total_scenario_count']}")
    print(f"rolling scenarios/variants: {payload['rolling_solver']['scenario_count']}/{payload['rolling_solver']['variant_count']}")
    print(f"sha256: {file_sha256(OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())