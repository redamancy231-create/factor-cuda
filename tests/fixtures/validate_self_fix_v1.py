# -*- coding: utf-8 -*-
"""六审 15 项 finding 的单入口、静默子进程机械闭合验证器。"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPECTED_IDS = [*(f"F5-{index:02d}" for index in range(1, 14)), "N6-01", "N6-02"]


class SelfFixError(RuntimeError):
    """任一机械闭合检查失败。"""


def load_module(relative: str, name: str):
    path = ROOT / pathlib.PurePosixPath(relative)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SelfFixError(f"无法加载模块: {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_json(relative: str) -> dict[str, Any]:
    path = ROOT / pathlib.PurePosixPath(relative)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelfFixError(f"JSON 顶层必须为对象: {relative}")
    return value


def run_captured(arguments: list[str]) -> None:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "无输出"
        raise SelfFixError(f"子验证失败 ({' '.join(arguments)}): {detail}")


def compare_generated_payload(relative_json: str, module: Any) -> None:
    committed = read_json(relative_json)
    generated = module.build_trace() if hasattr(module, "build_trace") else module.build_payload()
    if committed != generated:
        raise SelfFixError(f"生成函数与提交 JSON 不一致: {relative_json}")


def validate_in_process() -> None:
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))

    math_source = load_module("tests/fixtures/corr_math_v1.py", "_self_fix_corr_math")
    merge = lambda left, right: f"({left},{right})"
    for count in range(18):
        frontier = math_source.BinaryFrontier(merge, max_levels=8)
        values = [str(index) for index in range(count)]
        for index, value in enumerate(values):
            frontier.ingest(index, value)
        frontier.flush()
        if frontier.finalize() != math_source.expected_fixed_tree(values, merge):
            raise SelfFixError(f"BinaryFrontier 固定树失败: leaf_count={count}")
    cross = math_source.BinaryFrontier(merge, max_levels=8)
    absolute = 0
    for chunk_size in (3, 5, 4, 6):
        for _ in range(chunk_size):
            cross.ingest(absolute, str(absolute))
            absolute += 1
        cross.flush()
    values = [str(index) for index in range(18)]
    if cross.finalize() != math_source.expected_fixed_tree(values, merge):
        raise SelfFixError("BinaryFrontier 跨 chunk 固定树失败")

    left = math_source.CompensatedSum.from_values([-1e16])
    right = math_source.CompensatedSum.from_values([1e16, 1.0])
    if left.merge(right).represented != 1.0:
        raise SelfFixError("CompensatedSum 正确符号反例失败")
    wrong = math_source.CompensatedSum.from_values([-1e16])
    wrong.add(right.sum)
    wrong.add(right.c)
    if wrong.represented != -1.0:
        raise SelfFixError("CompensatedSum 错误符号反例不可区分")

    math_trace_module = load_module(
        "tests/fixtures/generate_corr_math_trace_v1.py", "_self_fix_math_trace"
    )
    compare_generated_payload("tests/fixtures/corr_math_trace_v1.json", math_trace_module)
    calibration_module = load_module(
        "tests/fixtures/generate_calibration_trace_v1.py", "_self_fix_calibration_trace"
    )
    compare_generated_payload("tests/fixtures/calibration_trace_v1.json", calibration_module)
    workspace_module = load_module(
        "benchmarks/compute_workspace_v1.py", "_self_fix_workspace"
    )
    compare_generated_payload("docs/workspace_v1.json", workspace_module)

    trace = read_json("tests/fixtures/corr_math_trace_v1.json")
    if trace["safe_pearson"]["distinguishable"] is not True:
        raise SelfFixError("safe Pearson 可区分边界未闭合")
    if trace["checked_offsets"]["scatter_out_base"]["overflow_next_row"] != "OverflowError":
        raise SelfFixError("scatter_out_base checked boundary 未闭合")
    if trace["aliasability"]["factor"]["f32_path"] != "f32_conversion":
        raise SelfFixError("factor 三路径未闭合")
    if trace["aliasability"]["stock"]["f32_path"] != "f32_conversion":
        raise SelfFixError("stock 三路径未闭合")

    calibration = read_json("tests/fixtures/calibration_trace_v1.json")
    if calibration["execution_date"] != "2026-08-04":
        raise SelfFixError("calibration execution_date 漂移")
    if [item["ok"] for item in calibration["budget_boundaries"]] != [False, False, True]:
        raise SelfFixError("calibration budget 边界漂移")

    workspace = read_json("docs/workspace_v1.json")
    if workspace["execution_date"] != "2026-08-04":
        raise SelfFixError("workspace execution_date 漂移")
    expected_anchors = {
        "factor": 155_872_080,
        "stock_500": 56_112_000,
        "stock_2000": 896_448_000,
        "stock_5000": 5_601_120_000,
        "rolling": 2_046_240,
    }
    if workspace["theoretical_workspace"]["verified_anchor_bytes"] != expected_anchors:
        raise SelfFixError("workspace anchor 漂移")


def validate() -> None:
    # 子进程输出全部捕获，成功 stdout 仅由末尾 16 行构成。
    run_captured(["tests/fixtures/validate_corr_corpus_v1.py"])
    run_captured(["tests/fixtures/validate_test_cases_v1.py"])
    run_captured(["tests/fixtures/validate_implementation_v1.py"])
    run_captured([
        "benchmarks/generate_gate_config_v1.py",
        "--check",
        "poc2_baseline_20260804c",
    ])
    validate_in_process()


if __name__ == "__main__":
    try:
        validate()
    except (SelfFixError, OSError, ValueError, KeyError, AssertionError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
    for finding_id in EXPECTED_IDS:
        print(f"PASS {finding_id}")
    print("SUMMARY PASS 15/15")
