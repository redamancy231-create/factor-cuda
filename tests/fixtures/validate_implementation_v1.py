# -*- coding: utf-8 -*-
"""机械校验 IMPLEMENTATION v0.7 的状态、正文、自动块与实体资产。"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import re
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOCUMENT = ROOT / "docs" / "IMPLEMENTATION.md"
SELF_FIX_VALIDATOR = ROOT / "tests" / "fixtures" / "validate_self_fix_v1.py"
STATUS_BEGIN = "<!-- implementation-status-v1:begin -->"
STATUS_END = "<!-- implementation-status-v1:end -->"
GATE_BEGIN = "<!-- gate-config-v1:begin -->"
GATE_END = "<!-- gate-config-v1:end -->"
EXPECTED_IDS = [*(f"F5-{index:02d}" for index in range(1, 14)), "N6-01", "N6-02"]
EXPECTED_COMMAND = "python tests/fixtures/validate_self_fix_v1.py"
PROHIBITED_SUBSTRINGS = ("同 v0.5", "同上", "同旧版", "保持", "TODO", "待办")
SAFE_PEARSON = "corr = (Sxy/sqrt(Sxx)) / sqrt(Syy)"


class ValidationError(RuntimeError):
    """一致性校验失败。"""


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(relative: str) -> dict[str, Any]:
    path = ROOT / pathlib.PurePosixPath(relative)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"JSON 不可解析: {relative}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"JSON 顶层必须为对象: {relative}")
    return value


def load_module(relative: str, name: str):
    path = ROOT / pathlib.PurePosixPath(relative)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValidationError(f"无法加载模块: {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def repo_file(path_text: str) -> pathlib.Path:
    pure = pathlib.PurePosixPath(path_text)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValidationError(f"证据路径必须是仓库内相对路径: {path_text!r}")
    path = ROOT / pure
    if not path.is_file():
        raise ValidationError(f"证据文件不存在: {path_text}")
    return path


def parse_status(document: str) -> dict[str, Any]:
    if document.count(STATUS_BEGIN) != 1 or document.count(STATUS_END) != 1:
        raise ValidationError("implementation-status-v1 边界必须各出现一次")
    begin = document.index(STATUS_BEGIN) + len(STATUS_BEGIN)
    end = document.index(STATUS_END, begin)
    block = document[begin:end].strip()
    match = re.fullmatch(r"```json\n(?P<payload>.*)\n```", block, flags=re.DOTALL)
    if match is None:
        raise ValidationError("implementation-status-v1 必须包含单一 JSON fenced block")
    try:
        payload = json.loads(match.group("payload"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"implementation-status-v1 JSON 不可解析: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValidationError("implementation-status-v1 顶层必须为对象")
    return payload


def validate_status(document: str) -> None:
    status = parse_status(document)
    expected_header = {
        "schema_version": "1.0.0",
        "document_version": "0.7",
        "execution_date": "2026-08-04",
        "timezone": "Asia/Hong_Kong",
    }
    for key, expected in expected_header.items():
        if status.get(key) != expected:
            raise ValidationError(f"状态字段 {key} 期望 {expected!r}，实际 {status.get(key)!r}")
    findings = status.get("findings")
    if not isinstance(findings, list):
        raise ValidationError("状态 findings 必须为数组")
    ids = [item.get("id") if isinstance(item, dict) else None for item in findings]
    if ids != EXPECTED_IDS:
        raise ValidationError(f"finding 顺序错误: {ids!r}")
    validator_sha = sha256(SELF_FIX_VALIDATOR)
    for finding in findings:
        finding_id = finding["id"]
        if finding.get("status") != "closed":
            raise ValidationError(f"{finding_id} 未标记 closed")
        evidence = finding.get("evidence")
        if not isinstance(evidence, list) or len(evidence) < 2:
            raise ValidationError(f"{finding_id} 至少需要两个实体证据")
        bound_to_self_fix = False
        for record in evidence:
            if not isinstance(record, dict):
                raise ValidationError(f"{finding_id} evidence 必须为对象")
            if set(record) != {"path", "sha256", "command"}:
                raise ValidationError(f"{finding_id} evidence 字段必须精确为 path/sha256/command")
            path_text = record.get("path")
            digest = record.get("sha256")
            command = record.get("command")
            if not isinstance(path_text, str) or not isinstance(digest, str):
                raise ValidationError(f"{finding_id} evidence path/sha256 类型错误")
            path = repo_file(path_text)
            actual_sha = sha256(path)
            if digest != actual_sha:
                raise ValidationError(f"{finding_id} 证据 SHA 漂移: {path_text}: {digest} != {actual_sha}")
            if command != EXPECTED_COMMAND:
                raise ValidationError(f"{finding_id} evidence command 不一致")
            if path.resolve() == SELF_FIX_VALIDATOR.resolve() and digest == validator_sha:
                bound_to_self_fix = True
            if path.resolve() == DOCUMENT.resolve():
                raise ValidationError(f"{finding_id} 不得使用 IMPLEMENTATION 自身 SHA 自证")
        if not bound_to_self_fix:
            raise ValidationError(f"{finding_id} 未绑定 validate_self_fix_v1.py 当前 SHA")


def require_text(document: str, tokens: list[str], *, context: str) -> None:
    missing = [token for token in tokens if token not in document]
    if missing:
        raise ValidationError(f"{context} 缺少正文锚点: {missing}")


def function_span(document: str, name: str, next_name: str | None) -> str:
    needle = f"def {name}("
    if document.count(needle) != 1:
        raise ValidationError(f"参考函数 {name} 必须且只能定义一次")
    start = document.index(needle)
    end = len(document) if next_name is None else document.index(f"def {next_name}(", start + len(needle))
    return document[start:end]


def validate_document_text(document: str) -> None:
    for forbidden in PROHIBITED_SUBSTRINGS:
        if forbidden in document:
            raise ValidationError(f"正文出现禁用占位子串: {forbidden}")
    require_text(
        document,
        [
            "# factor-cuda 实现设计（Implementation Design）v0.7",
            "2026-08-04，星期二，Asia/Hong_Kong",
            "backend='cpu'",
            "backend='cuda'",
            "NumPy float64",
            "torch float64",
            "mirror 输入 device",
            "CPU 输入走 CUDA 后结果回 CPU",
            "CUDA 输入结果留在原 CUDA device",
            "NumPy-only API 子集",
            "不能证明完整 solver",
            SAFE_PEARSON,
            "slot = (leaf_index >> level) & 1",
            "sum - c",
            "-right.c",
            "scatter_out_base = checked_mul(row_base, N)",
            "checked_global_element_offset",
            "checked_byte_offset",
            "full-N sentinel",
            "canonicalize ±0",
            "stable ordinal",
            "factor_is_aliasable(view, kernel_layout)",
            "stock_is_aliasable(view, kernel_layout)",
            "f32_conversion",
            "f64_alias",
            "f64_gather",
            "Partial1",
            "Partial2",
            "PartialK1",
            "PartialK2",
            "cudaErrorInvalidConfiguration",
            "cudaErrorLaunchOutOfResources",
            "reserve 唯一消费点",
            "nearest-rank p99",
            "tmp + flush + fsync + close + replace",
            "N6-01",
            "N6-02",
        ],
        context="v0.7 自包含正文",
    )
    if document.count(SAFE_PEARSON) < 3:
        raise ValidationError("safe Pearson 必须在规范正文/伪代码/验证映射中逐字出现至少三次")

    functions = ["factor_corr_reference", "stock_corr_reference", "rolling_ic_reference"]
    spans = {
        functions[0]: function_span(document, functions[0], functions[1]),
        functions[1]: function_span(document, functions[1], functions[2]),
        functions[2]: function_span(document, functions[2], None),
    }
    common = [
        "validate_input",
        "for chunk in deterministic_chunks",
        "first_pass_state",
        "second_pass_state",
        "fallback",
        "writeback",
        "RuntimeError",
        SAFE_PEARSON,
    ]
    for name, span in spans.items():
        missing = [token for token in common if token not in span]
        if missing:
            raise ValidationError(f"{name} 不是完整独立参考函数，缺少 {missing}")
        for other in functions:
            if other != name and f"{other}(" in span:
                raise ValidationError(f"{name} 不得调用另一个参考函数 {other}")

    abi_tokens = [
        "Partial1 | 56 | 8 | count=0, sum_x=8, sum_y=16, min_x=24, max_x=32, min_y=40, max_y=48",
        "Partial2 | 24 | 8 | sxx=0, syy=8, sxy=16",
        "PartialK1 | 40 | 8 | count=0, sum_x=8, c_x=16, sum_y=24, c_y=32",
        "PartialK2 | 48 | 8 | sxx=0, c_xx=8, syy=16, c_yy=24, sxy=32, c_xy=40",
    ]
    require_text(document, abi_tokens, context="struct ABI")


def validate_corr_assets() -> None:
    manifest = load_json("tests/fixtures/corr_corpus_v1.manifest.json")
    if manifest.get("version") != "1.2.0" or manifest.get("case_count") != 16:
        raise ValidationError("corr corpus 必须为 v1.2.0 / 16 cases")
    ids = {case.get("id") for case in manifest.get("cases", []) if isinstance(case, dict)}
    required_ids = {"pearson_overflow", "f64_adjacent_ulp", "stable_zero_f32", "stable_zero_f64"}
    if not required_ids.issubset(ids):
        raise ValidationError(f"corr corpus 缺少反例: {sorted(required_ids - ids)}")
    if manifest.get("generator_sha256") != sha256(ROOT / "tests/fixtures/generate_corr_corpus_v1.py"):
        raise ValidationError("corr corpus generator SHA 不一致")
    if manifest.get("math_source_sha256") != sha256(ROOT / "tests/fixtures/corr_math_v1.py"):
        raise ValidationError("corr corpus math source SHA 不一致")
    if manifest.get("npz_sha256") != sha256(ROOT / "tests/fixtures/corr_corpus_v1.npz"):
        raise ValidationError("corr corpus NPZ SHA 不一致")

    trace = load_json("tests/fixtures/corr_math_trace_v1.json")
    if trace.get("schema_version") != "1.0.0":
        raise ValidationError("corr math trace schema 版本错误")
    frontier = trace.get("binary_frontier", {})
    if len(frontier.get("leaf_counts", [])) != 18:
        raise ValidationError("BinaryFrontier trace 必须覆盖 0..17 叶")
    cross = frontier.get("cross_chunk", {})
    if cross.get("final") != cross.get("expected"):
        raise ValidationError("BinaryFrontier 跨 chunk 树形不一致")
    comp = trace.get("compensated_sum", {})
    if comp.get("correct_merge", {}).get("represented") != 1.0:
        raise ValidationError("CompensatedSum 正确 merge 反例未得到 +1")
    if comp.get("wrong_sign_counterexample", {}).get("represented") != -1.0:
        raise ValidationError("CompensatedSum 错误符号反例不可区分")
    pearson = trace.get("safe_pearson", {})
    if pearson.get("expression") != SAFE_PEARSON or not pearson.get("distinguishable"):
        raise ValidationError("safe Pearson trace 未区分逐次除法")
    abi = trace.get("struct_abi", {})
    sizes = {name: record.get("size_B") for name, record in abi.items() if isinstance(record, dict)}
    if sizes != {"Partial1": 56, "Partial2": 24, "PartialK1": 40, "PartialK2": 48}:
        raise ValidationError(f"struct ABI trace 错误: {sizes}")
    if trace.get("generator_sha256") != sha256(ROOT / "tests/fixtures/generate_corr_math_trace_v1.py"):
        raise ValidationError("corr math trace generator SHA 不一致")
    if trace.get("math_source_sha256") != sha256(ROOT / "tests/fixtures/corr_math_v1.py"):
        raise ValidationError("corr math trace source SHA 不一致")


def validate_test_manifest() -> None:
    manifest = load_json("tests/fixtures/test_cases_v1.json")
    if manifest.get("manifest_version") != "1.1.0":
        raise ValidationError("test manifest version 必须为 1.1.0")
    if manifest.get("execution_date") != "2026-08-04":
        raise ValidationError("test manifest execution_date 必须为 2026-08-04")
    if "future_date_provenance" in manifest:
        raise ValidationError("test manifest 不得包含 future_date_provenance")
    if manifest.get("expanded_case_count") != 50 or manifest.get("target_count") != 7:
        raise ValidationError("test manifest 必须为 50 cases / 7 targets")
    if manifest.get("schema_sha256") != sha256(ROOT / manifest["schema_path"]):
        raise ValidationError("test manifest schema SHA 不一致")
    if manifest.get("generator_sha256") != sha256(ROOT / manifest["generator_path"]):
        raise ValidationError("test manifest generator SHA 不一致")
    recoverable = manifest.get("recoverable_launch_errors")
    if recoverable != ["cudaErrorInvalidConfiguration", "cudaErrorLaunchOutOfResources"]:
        raise ValidationError("recoverable launch 白名单必须精确为两个错误码")
    fatal_stages = {item.get("stage") for item in manifest.get("fatal_faults", []) if isinstance(item, dict)}
    expected_stages = {"setup", "allocator", "h2d", "event", "launch", "sync", "async", "d2h", "result_allocation", "context"}
    if not expected_stages.issubset(fatal_stages):
        raise ValidationError(f"fatal fault 矩阵缺少 stage: {sorted(expected_stages - fatal_stages)}")


def validate_workspace() -> None:
    workspace = load_json("docs/workspace_v1.json")
    if workspace.get("schema_version") != "2.0.0" or workspace.get("execution_date") != "2026-08-04":
        raise ValidationError("workspace 必须为 schema 2.0.0 / execution_date 2026-08-04")
    if "future_date_provenance" in workspace:
        raise ValidationError("workspace 不得包含 future_date_provenance")
    if workspace.get("generator_sha256") != sha256(ROOT / "benchmarks/compute_workspace_v1.py"):
        raise ValidationError("workspace generator SHA 不一致")
    if workspace.get("math_source_sha256") != sha256(ROOT / "tests/fixtures/corr_math_v1.py"):
        raise ValidationError("workspace math source SHA 不一致")
    if workspace.get("safe_pearson_expression") != SAFE_PEARSON:
        raise ValidationError("workspace safe Pearson 单一真源漂移")
    budget = workspace.get("memory_budget", {})
    if budget.get("available_bytes") != 8_048_869_376 or budget.get("available_MiB") != 7676:
        raise ValidationError("available byte/MiB 计算错误")
    if budget.get("reserve_consumption_count") != 1 or budget.get("required_includes_reserve") is not False:
        raise ValidationError("reserve 必须只消费一次")
    theoretical = workspace.get("theoretical_workspace", {})
    expected_anchors = {
        "factor": 155_872_080,
        "stock_500": 56_112_000,
        "stock_2000": 896_448_000,
        "stock_5000": 5_601_120_000,
        "rolling": 2_046_240,
    }
    if theoretical.get("verified_anchor_bytes") != expected_anchors:
        raise ValidationError("workspace 理论锚点漂移")
    factor = workspace.get("factor_solver", {})
    stock = workspace.get("stock_solver", {})
    rolling = workspace.get("rolling_solver", {})
    if factor.get("scenario_count") != 12 or stock.get("total_scenario_count") != 36:
        raise ValidationError("factor/stock scenario 数必须为 12/36")
    if rolling.get("scenario_count") != 8 or rolling.get("variant_count") != 16:
        raise ValidationError("rolling scenario/variant 数必须为 8/16")
    if {item.get("N") for item in stock.get("sizes", [])} != {500, 2000, 5000}:
        raise ValidationError("stock solver 必须覆盖 N=500/2000/5000")
    factor_modes = {item.get("output_mode") for item in factor.get("scenarios", [])}
    stock_modes = {item.get("output_mode") for size in stock.get("sizes", []) for item in size.get("scenarios", [])}
    rolling_modes = {item.get("output_mode") for item in rolling.get("scenarios", [])}
    if factor_modes != {"cpu", "cuda"} or stock_modes != {"cpu", "cuda"} or rolling_modes != {"cpu", "cuda"}:
        raise ValidationError("三个 solver 均须覆盖 CPU/CUDA output mode")
    for scenario in factor.get("scenarios", []):
        if scenario.get("chosen") is None or scenario.get("runner_up") is None or scenario.get("first_infeasible_candidate") is None:
            raise ValidationError(f"factor scenario 缺 chosen/runner-up/failure: {scenario.get('scenario_id')}")
    for size in stock.get("sizes", []):
        for scenario in size.get("scenarios", []):
            if scenario.get("chosen") is None or scenario.get("first_infeasible_candidate") is None:
                raise ValidationError(f"stock scenario 缺 chosen/failure: {scenario.get('scenario_id')}")
    for scenario in rolling.get("scenarios", []):
        query = scenario.get("cub_query", {})
        if not isinstance(query, dict) or not query.get("query_api") or query.get("temp_bytes") is None:
            raise ValidationError(f"rolling scenario 缺 CUB query: {scenario.get('scenario_id')}")
        variants = scenario.get("variants")
        if not isinstance(variants, list) or len(variants) != 2:
            raise ValidationError(f"rolling scenario 必须有 normal/Kahan 两个 variant: {scenario.get('scenario_id')}")


def validate_calibration() -> None:
    schema = load_json("tests/fixtures/calibration_v1.schema.json")
    trace = load_json("tests/fixtures/calibration_trace_v1.json")
    if trace.get("trace_version") != "1.0.0" or trace.get("schema_version") != "1.0.0":
        raise ValidationError("calibration trace/schema 版本错误")
    if trace.get("execution_date") != "2026-08-04":
        raise ValidationError("calibration execution_date 必须为 2026-08-04")
    if trace.get("schema_sha256") != sha256(ROOT / trace["schema"]):
        raise ValidationError("calibration schema SHA 不一致")
    if trace.get("source_sha256") != sha256(ROOT / trace["source"]):
        raise ValidationError("calibration source SHA 不一致")
    if trace.get("generator_sha256") != sha256(ROOT / trace["generator"]):
        raise ValidationError("calibration generator SHA 不一致")
    if schema.get("$id") != "calibration_v1.schema.json":
        raise ValidationError("calibration schema $id 错误")
    sampling = trace.get("sampling", {})
    if sampling.get("formula") != "max(0, free_before - min_free_during)":
        raise ValidationError("calibration 采样公式错误")
    if len(sampling.get("samples_bytes", [])) != 7 or sampling.get("calibrated_p99_bytes") != 650_000_000:
        raise ValidationError("calibration nearest-rank p99 锚点错误")
    if len(trace.get("cache_key_fields", [])) != 6:
        raise ValidationError("calibration cache key 必须有六字段")
    if [item.get("ok") for item in trace.get("budget_boundaries", [])] != [False, False, True]:
        raise ValidationError("calibration budget 边界必须为 fail/fail/pass")
    atomic = trace.get("atomic_write", {})
    if atomic.get("utf8_bom") is not False or atomic.get("lf_only") is not True:
        raise ValidationError("calibration 原子文件编码错误")
    if atomic.get("last_complete_rename_wins") is not True or atomic.get("temp_leftovers") != []:
        raise ValidationError("calibration 原子 replace 语义错误")


def validate_gate(document: str) -> None:
    if document.count(GATE_BEGIN) != 1 or document.count(GATE_END) != 1:
        raise ValidationError("Gate Markdown 边界必须各出现一次")
    gate_module = load_module("benchmarks/generate_gate_config_v1.py", "_gate_validator_module")
    expected = gate_module.build_config(gate_module.DEFAULT_RUN_ID)
    actual_bytes = (ROOT / "docs" / "gate_config_v1.json").read_bytes()
    if actual_bytes != gate_module.render_json_bytes(expected):
        raise ValidationError("gate_config_v1.json 不是精确生成结果")
    begin = document.index(GATE_BEGIN)
    end = document.index(GATE_END, begin) + len(GATE_END)
    if document[begin:end] != gate_module.render_markdown_block(expected):
        raise ValidationError("IMPLEMENTATION Gate 自动块漂移")
    if actual_bytes.startswith(b"\xef\xbb\xbf") or not b"\r\n" in actual_bytes or actual_bytes.endswith((b"\n", b"\r")):
        raise ValidationError("gate_config_v1.json 必须 CRLF、UTF-8 无 BOM、无末尾换行")


def validate_line_endings(document_bytes: bytes) -> None:
    if document_bytes.startswith(b"\xef\xbb\xbf") or b"\r\n" in document_bytes:
        raise ValidationError("IMPLEMENTATION.md 必须 LF、UTF-8 无 BOM")
    workspace_bytes = (ROOT / "docs/workspace_v1.json").read_bytes()
    if workspace_bytes.startswith(b"\xef\xbb\xbf") or b"\r\n" in workspace_bytes or not workspace_bytes.endswith(b"\n"):
        raise ValidationError("workspace_v1.json 必须 LF、UTF-8 无 BOM、末尾单 LF")


def validate() -> None:
    document_bytes = DOCUMENT.read_bytes()
    try:
        document = document_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError(f"IMPLEMENTATION 不是 UTF-8: {exc}") from exc
    validate_line_endings(document_bytes)
    validate_status(document)
    validate_document_text(document)
    validate_corr_assets()
    validate_test_manifest()
    validate_workspace()
    validate_calibration()
    validate_gate(document)


if __name__ == "__main__":
    try:
        validate()
    except ValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print("PASS implementation v0.7 findings=15 assets=6")
