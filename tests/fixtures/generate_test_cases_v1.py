# -*- coding: utf-8 -*-
"""生成 test_cases_v1 v1.1：wildcard template 与确定性展开实体。"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import pathlib
import re
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "test_cases_v1.json"
SCHEMA = HERE / "test_cases_v1.schema.json"
VERSION = "1.1.0"
EXECUTION_DATE = "2026-08-04"

RECOVERABLE = [
    "cudaErrorInvalidConfiguration",
    "cudaErrorLaunchOutOfResources",
]

FATAL_FAULTS = [
    {"id": "setup", "class": "fatal", "stage": "setup", "error": "cudaErrorInitializationError"},
    {"id": "allocator", "class": "fatal", "stage": "allocator", "error": "cudaErrorMemoryAllocation"},
    {"id": "h2d", "class": "fatal", "stage": "h2d", "error": "cudaErrorInvalidValue"},
    {"id": "event", "class": "fatal", "stage": "event", "error": "cudaErrorInvalidResourceHandle"},
    {"id": "launch", "class": "fatal", "stage": "launch", "error": "cudaErrorInvalidDeviceFunction"},
    {"id": "sync", "class": "fatal", "stage": "sync", "error": "cudaErrorUnknown"},
    {"id": "async", "class": "fatal", "stage": "async", "error": "cudaErrorUnknown"},
    {"id": "d2h", "class": "fatal", "stage": "d2h", "error": "cudaErrorECCUncorrectable"},
    {"id": "result_allocation", "class": "fatal", "stage": "result_allocation", "error": "MemoryError"},
    {"id": "context", "class": "fatal", "stage": "context", "error": "cudaErrorContextIsDestroyed"},
    {"id": "illegal_address", "class": "fatal", "stage": "illegal address", "error": "cudaErrorIllegalAddress"},
    {"id": "device_assert", "class": "fatal", "stage": "device assert", "error": "cudaErrorAssert"},
    {"id": "launch_failure", "class": "fatal", "stage": "launch failure", "error": "cudaErrorLaunchFailure"},
]

TEMPLATES: list[dict[str, Any]] = [
    {
        "template_id": "binary_frontier_leaf_count",
        "case_id_pattern": "binary_frontier_{leaf_count}",
        "target": "tests/fixtures/generate_corr_math_trace_v1.py",
        "axes": {"leaf_count": list(range(18))},
        "input": {"leaf_count": "{leaf_count}", "absolute_leaf_start": 0, "chunk_plan": [1, 3, 5, 8]},
        "injection": {"kind": "none", "continuous_absolute_leaf_order": True},
        "expect": {"fixed_tree_equal": True, "capacity_guard": True, "final_live_slots": "popcount(leaf_count)"},
    },
    {
        "template_id": "recoverable_launch_whitelist",
        "case_id_pattern": "recoverable_{error}",
        "target": "tests/fixtures/validate_test_cases_v1.py",
        "axes": {"error": RECOVERABLE},
        "input": {"operation": "parameter_scan", "group_index": 1},
        "injection": {"stage": "launch", "error": "{error}"},
        "expect": {"classification": "recoverable", "group_status": "failed", "scan_continues": True},
    },
    {
        "template_id": "fatal_fault_matrix",
        "case_id_pattern": "fatal_{fault}",
        "target": "tests/fixtures/validate_test_cases_v1.py",
        "axes": {"fault": [item["id"] for item in FATAL_FAULTS]},
        "input": {"operation": "parameter_scan", "group_index": 0},
        "injection": {"fault_id": "{fault}", "classification": "fatal"},
        "expect": {"raises": "RuntimeError", "no_partial_results": True, "raii_cleanup": True},
    },
    {
        "template_id": "rolling_mixed_dtype",
        "case_id_pattern": "rolling_{prediction_dtype}_{label_dtype}",
        "target": "benchmarks/compute_workspace_v1.py",
        "axes": {
            "prediction_dtype": ["float32", "float64"],
            "label_dtype": ["float32", "float64"]
        },
        "input": {"T": 1218, "N": 5000, "prediction_dtype": "{prediction_dtype}", "label_dtype": "{label_dtype}"},
        "injection": {"output_modes": ["cpu", "cuda_device"], "accumulation": ["normal", "kahan"]},
        "expect": {"levels_current": 20, "levels_next": 10, "cub_query_recorded": True, "timeline_complete": True},
    },
    {
        "template_id": "alias_path_matrix",
        "case_id_pattern": "alias_{operation}_{path}",
        "target": "tests/fixtures/corr_math_v1.py",
        "axes": {
            "operation": ["factor", "stock"],
            "path": ["f32_conversion", "f64_alias", "f64_gather"]
        },
        "input": {"operation": "{operation}", "dtype_path": "{path}"},
        "injection": {"layout_counterexamples": ["dtype", "strides", "alignment", "offset", "device", "owner", "lifetime"]},
        "expect": {"selected_path": "{path}", "independent_predicate": True},
    },
    {
        "template_id": "output_contract",
        "case_id_pattern": "output_{output_mode}",
        "target": "benchmarks/compute_workspace_v1.py",
        "axes": {"output_mode": ["cpu", "cuda_device"]},
        "input": {"backend": "cuda", "output_mode": "{output_mode}"},
        "injection": {"input_origin": "cpu_host"},
        "expect": {
            "cpu": {"container": "numpy.ndarray", "device_hwm": "tile_staging_only"},
            "cuda_device": {"container": "torch.Tensor", "dtype": "float64", "device": "mirror_input"}
        },
    },
    {
        "template_id": "artifact_validation",
        "case_id_pattern": "artifact_{fixture[id]}",
        "target": "{fixture[target]}",
        "axes": {
            "fixture": [
                {"id": "corr_corpus", "target": "tests/fixtures/validate_corr_corpus_v1.py"},
                {"id": "corr_math", "target": "tests/fixtures/generate_corr_math_trace_v1.py"},
                {"id": "workspace", "target": "benchmarks/compute_workspace_v1.py"},
                {"id": "calibration", "target": "tests/fixtures/generate_calibration_trace_v1.py"},
                {"id": "gate", "target": "benchmarks/generate_gate_config_v1.py"}
            ]
        },
        "input": {"artifact": "{fixture[id]}", "repo_relative_target": "{fixture[target]}"},
        "injection": {"kind": "none"},
        "expect": {"exit_code": 0, "deterministic_bytes": True}
    }
]


_FULL_TOKEN = re.compile(r"^\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def file_sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def substitute(value: Any, axes: dict[str, Any]) -> Any:
    if isinstance(value, str):
        match = _FULL_TOKEN.fullmatch(value)
        if match:
            return copy.deepcopy(axes[match.group(1)])
        return value.format_map(axes)
    if isinstance(value, list):
        return [substitute(item, axes) for item in value]
    if isinstance(value, dict):
        return {key: substitute(item, axes) for key, item in value.items()}
    return copy.deepcopy(value)


def expand_template(template: dict[str, Any]) -> list[dict[str, Any]]:
    names = list(template["axes"])
    values = [template["axes"][name] for name in names]
    expanded = []
    for combination in itertools.product(*values):
        axes = dict(zip(names, combination, strict=True))
        expanded.append({
            "id": substitute(template["case_id_pattern"], axes),
            "template_id": template["template_id"],
            "target": substitute(template["target"], axes),
            "axes": copy.deepcopy(axes),
            "input": substitute(template["input"], axes),
            "injection": substitute(template["injection"], axes),
            "expect": substitute(template["expect"], axes),
        })
    return expanded


def build_manifest() -> dict[str, Any]:
    expanded = [case for template in TEMPLATES for case in expand_template(template)]
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "manifest_version": VERSION,
        "execution_date": EXECUTION_DATE,
        "schema_path": "tests/fixtures/test_cases_v1.schema.json",
        "schema_sha256": file_sha256(SCHEMA),
        "generator_path": "tests/fixtures/generate_test_cases_v1.py",
        "generator_sha256": file_sha256(pathlib.Path(__file__)),
        "fixture_sha256_kind": "sha256(canonical-json-without-fixture_sha256)",
        "recoverable_launch_errors": RECOVERABLE,
        "fatal_faults": FATAL_FAULTS,
        "wildcard_templates": TEMPLATES,
        "expanded_case_count": len(expanded),
        "target_count": len({case["target"] for case in expanded}),
        "expanded_cases": expanded,
    }
    manifest["fixture_sha256"] = hashlib.sha256(canonical_bytes(manifest)).hexdigest()
    return manifest


def validate_targets(manifest: dict[str, Any]) -> None:
    for case in manifest["expanded_cases"]:
        target = case["target"]
        if "\\" in target or pathlib.PurePosixPath(target).is_absolute():
            raise ValueError(f"target is not repo-relative POSIX: {target}")
        resolved = (ROOT / pathlib.PurePosixPath(target)).resolve()
        resolved.relative_to(ROOT.resolve())
        if not resolved.is_file():
            raise FileNotFoundError(f"target does not exist: {target}")


def main(argv: list[str]) -> int:
    check = argv == ["--check"]
    if argv not in ([], ["--check"]):
        print("usage: generate_test_cases_v1.py [--check]", file=sys.stderr)
        return 2
    manifest = build_manifest()
    validate_targets(manifest)
    rendered = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if check:
        if not OUT.is_file() or OUT.read_text(encoding="utf-8") != rendered:
            print("test_cases_v1 check failed: committed fixture differs", file=sys.stderr)
            return 1
        print(f"test_cases_v1 check: PASS version={VERSION} cases={manifest['expanded_case_count']}")
        return 0
    OUT.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"generated: {OUT}")
    print(f"version: {VERSION}")
    print(f"templates: {len(TEMPLATES)} expanded_cases: {manifest['expanded_case_count']} targets: {manifest['target_count']}")
    print(f"fixture_sha256: {manifest['fixture_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))