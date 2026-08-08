# -*- coding: utf-8 -*-
"""独立验证 test_cases_v1 schema、展开、SHA、target 与错误分类。"""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import re
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST = HERE / "test_cases_v1.json"
SCHEMA = HERE / "test_cases_v1.schema.json"
GENERATOR = HERE / "generate_test_cases_v1.py"
EXPECTED_RECOVERABLE = {
    "cudaErrorInvalidConfiguration",
    "cudaErrorLaunchOutOfResources",
}
EXPECTED_FATAL_STAGES = {
    "setup",
    "allocator",
    "h2d",
    "event",
    "launch",
    "sync",
    "async",
    "d2h",
    "result_allocation",
    "context",
    "illegal address",
    "device assert",
    "launch failure",
}
_FULL_TOKEN = re.compile(r"^\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def fail(message: str) -> None:
    raise AssertionError(message)


def file_sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def independent_substitute(value: Any, environment: dict[str, Any]) -> Any:
    if isinstance(value, str):
        token = _FULL_TOKEN.fullmatch(value)
        if token:
            if token.group(1) not in environment:
                fail(f"unknown wildcard token: {value}")
            return copy.deepcopy(environment[token.group(1)])
        try:
            return value.format_map(environment)
        except (KeyError, ValueError) as exc:
            fail(f"invalid wildcard expression {value!r}: {exc}")
    if isinstance(value, list):
        return [independent_substitute(item, environment) for item in value]
    if isinstance(value, dict):
        return {key: independent_substitute(item, environment) for key, item in value.items()}
    return copy.deepcopy(value)


def independent_axis_product(axes: dict[str, list[Any]]) -> list[dict[str, Any]]:
    names = list(axes)
    results: list[dict[str, Any]] = []

    def visit(index: int, current: dict[str, Any]) -> None:
        if index == len(names):
            results.append(copy.deepcopy(current))
            return
        name = names[index]
        values = axes[name]
        if not isinstance(values, list) or not values:
            fail(f"axis {name} must be a non-empty list")
        for value in values:
            current[name] = value
            visit(index + 1, current)
        current.pop(name, None)

    visit(0, {})
    return results


def independently_expand(templates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for template in templates:
        for axes in independent_axis_product(template["axes"]):
            cases.append({
                "id": independent_substitute(template["case_id_pattern"], axes),
                "template_id": template["template_id"],
                "target": independent_substitute(template["target"], axes),
                "axes": copy.deepcopy(axes),
                "input": independent_substitute(template["input"], axes),
                "injection": independent_substitute(template["injection"], axes),
                "expect": independent_substitute(template["expect"], axes),
            })
    return cases


def validate_minimal_schema(schema: dict[str, Any], manifest: dict[str, Any]) -> None:
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        fail("schema draft is not 2020-12")
    required = schema.get("required")
    if not isinstance(required, list):
        fail("schema required must be a list")
    missing = [key for key in required if key not in manifest]
    if missing:
        fail(f"manifest misses schema-required keys: {missing}")
    try:
        import jsonschema  # type: ignore
    except ImportError:
        return
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(instance=manifest, schema=schema)


def validate_target(target: str) -> None:
    if not isinstance(target, str) or not target or "\\" in target:
        fail(f"target must be non-empty repo-relative POSIX path: {target!r}")
    pure = pathlib.PurePosixPath(target)
    if pure.is_absolute() or ".." in pure.parts:
        fail(f"target escapes repository: {target}")
    resolved = (ROOT / pure).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        fail(f"target resolves outside repository: {target}")
    if not resolved.is_file():
        fail(f"target does not exist: {target}")


def validate() -> tuple[int, int]:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    validate_minimal_schema(schema, manifest)
    if manifest["schema_version"] != "1.0.0" or manifest["manifest_version"] != "1.1.0":
        fail("manifest version drift")
    if manifest["execution_date"] != "2026-08-04":
        fail("execution date must be 2026-08-04")
    if manifest["schema_path"] != "tests/fixtures/test_cases_v1.schema.json":
        fail("schema_path drift")
    if manifest["generator_path"] != "tests/fixtures/generate_test_cases_v1.py":
        fail("generator_path drift")
    if manifest["schema_sha256"] != file_sha256(SCHEMA):
        fail("schema SHA mismatch")
    if manifest["generator_sha256"] != file_sha256(GENERATOR):
        fail("generator SHA mismatch")
    payload = dict(manifest)
    fixture_sha = payload.pop("fixture_sha256")
    if fixture_sha != hashlib.sha256(canonical_bytes(payload)).hexdigest():
        fail("fixture payload SHA mismatch")

    recoverable = manifest["recoverable_launch_errors"]
    if len(recoverable) != 2 or set(recoverable) != EXPECTED_RECOVERABLE:
        fail(f"recoverable whitelist must be exact: {sorted(EXPECTED_RECOVERABLE)}")
    fatal_faults = manifest["fatal_faults"]
    stages = {item["stage"] for item in fatal_faults}
    if stages != EXPECTED_FATAL_STAGES:
        fail(f"fatal stages mismatch: missing={sorted(EXPECTED_FATAL_STAGES - stages)} extra={sorted(stages - EXPECTED_FATAL_STAGES)}")
    if any(item.get("class") != "fatal" for item in fatal_faults):
        fail("all fatal fault records must be class=fatal")
    fatal_ids = {item["id"] for item in fatal_faults}
    if len(fatal_ids) != len(fatal_faults):
        fail("fatal fault ids are not unique")

    independent = independently_expand(manifest["wildcard_templates"])
    committed = manifest["expanded_cases"]
    if independent != committed:
        fail("committed expanded cases differ from independent wildcard expansion")
    if manifest["expanded_case_count"] != len(committed):
        fail("expanded_case_count mismatch")
    ids = [case["id"] for case in committed]
    if len(ids) != len(set(ids)):
        duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
        fail(f"expanded case ids are not unique: {duplicates}")
    for case in committed:
        if not isinstance(case["input"], dict) or not isinstance(case["injection"], dict) or not isinstance(case["expect"], dict):
            fail(f"case {case['id']} input/injection/expect must be objects")
        validate_target(case["target"])
    targets = {case["target"] for case in committed}
    if manifest["target_count"] != len(targets):
        fail("target_count mismatch")

    recoverable_cases = [case for case in committed if case["template_id"] == "recoverable_launch_whitelist"]
    if {case["injection"]["error"] for case in recoverable_cases} != EXPECTED_RECOVERABLE:
        fail("recoverable expanded cases mismatch")
    fatal_cases = [case for case in committed if case["template_id"] == "fatal_fault_matrix"]
    if {case["injection"]["fault_id"] for case in fatal_cases} != fatal_ids:
        fail("fatal expanded cases are not exhaustive")
    return len(committed), len(targets)


def main() -> int:
    try:
        cases, targets = validate()
    except Exception as exc:
        print(f"corr test manifest: FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"test_cases_v1: PASS version=1.1.0 cases={cases} targets={targets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())