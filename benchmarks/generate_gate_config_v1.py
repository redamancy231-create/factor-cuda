# -*- coding: utf-8 -*-
"""PoC ④ Gate 配置与 IMPLEMENTATION §3.5 自动块生成器。

正常模式从三臂 canonical run 生成：
- docs/gate_config_v1.json
- docs/IMPLEMENTATION.md 中 gate-config-v1 边界块

``--check`` 模式只读且 fail-closed，精确校验 JSON、canonical source SHA、
generator SHA、payload SHA 与 Markdown 自动块；机器判定只使用 exact_half，display
仅为向负无穷取整到两位小数的展示值。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
RUNS_DIR = HERE / "results" / "runs"
OUT = REPO_ROOT / "docs" / "gate_config_v1.json"
IMPLEMENTATION = REPO_ROOT / "docs" / "IMPLEMENTATION.md"
DEFAULT_RUN_ID = "poc2_baseline_20260804c"
BEGIN_MARKER = "<!-- gate-config-v1:begin -->"
END_MARKER = "<!-- gate-config-v1:end -->"
JSON_NEWLINE = "\r\n"  # 保持既有 gate_config_v1.json 的 CRLF 风格。
BACKENDS = ["numpy", "cupy", "qgplearn"]
SAME_SEMANTICS = {
    "cs_rank": "qgplearn",
    "cs_rank_desc": "qgplearn",
    "parameter_scan(G=4)": "qgplearn",
    "factor_corr": "cupy",
    "rolling_ic": "cupy",  # QG float32 known-deviation 不具同语义资格。
}
FORMAL_STOCK = {"stock_corr(N=500)"}
EXT_STOCK = {"stock_corr(N=2000)", "stock_corr(N=5000)"}


class GateConfigError(RuntimeError):
    """Gate 构建或一致性检查失败。"""


def _file_sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_path(path: pathlib.Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _floor2(value: float) -> float:
    """向负无穷取整到 2 位小数，保证展示值不放宽机器阈值。"""
    return math.floor(value * 100.0) / 100.0


def _canonical_payload(config_without_payload_sha: dict[str, Any]) -> bytes:
    return json.dumps(
        config_without_payload_sha,
        ensure_ascii=False,
        indent=1,
        separators=(",", ": "),
    ).encode("utf-8")


def _load_reports(run_id: str) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    run_dir = RUNS_DIR / run_id
    if not run_dir.is_dir():
        raise GateConfigError(f"run 目录不存在: {_repo_path(run_dir)}")

    reports: dict[str, Any] = {}
    sources: dict[str, dict[str, str]] = {}
    for backend in BACKENDS:
        source = run_dir / f"{backend}.json"
        if not source.is_file():
            raise GateConfigError(f"缺少 canonical source: {_repo_path(source)}")
        try:
            reports[backend] = json.loads(source.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GateConfigError(f"canonical source 不可解析: {_repo_path(source)}: {exc}") from exc
        sources[backend] = {
            "path": _repo_path(source),
            "sha256": _file_sha(source),
        }
    return reports, sources


def build_config(run_id: str) -> dict[str, Any]:
    reports, sources = _load_reports(run_id)

    operation_labels: list[str] = []
    for backend in BACKENDS:
        operations = reports[backend].get("operations")
        if not isinstance(operations, dict):
            raise GateConfigError(f"{backend}.json 缺少对象 operations")
        for label in operations:
            if label not in operation_labels:
                operation_labels.append(label)

    config: dict[str, Any] = {
        "schema_version": "1.1.0",
        "run_id": run_id,
        "generator": _repo_path(pathlib.Path(__file__)),
        "generator_sha256": _file_sha(pathlib.Path(__file__)),
        "canonical_sources": sources,
        "gates": {},
    }

    for label in operation_labels:
        if label in SAME_SEMANTICS:
            best_backend = SAME_SEMANTICS[label]
            scope = "formal"
        elif label in FORMAL_STOCK:
            best_backend = "cupy"
            scope = "formal"
        elif label in EXT_STOCK:
            best_backend = "cupy"
            scope = "extension"
        else:
            continue

        operation = reports[best_backend]["operations"].get(label)
        if not isinstance(operation, dict) or operation.get("wall_ms") is None:
            raise GateConfigError(f"{best_backend}.json 缺少 gate operation: {label}")
        raw = operation["wall_ms"]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)) or raw < 0:
            raise GateConfigError(f"非法 wall_ms: backend={best_backend} operation={label} value={raw!r}")
        exact_half = float(raw) / 2.0
        display = _floor2(exact_half)
        gate = {
            "scope": scope,
            "best_backend": best_backend,
            "source_path": sources[best_backend]["path"],
            "source_sha256": sources[best_backend]["sha256"],
            "source_operation_key": label,
            "raw_wall_ms": raw,
            "exact_half": exact_half,
            "display": display,
            "display_le_exact_half": display <= exact_half,
        }
        if not gate["display_le_exact_half"]:
            raise GateConfigError(f"display 放宽 exact_half: {label}")
        config["gates"][label] = gate

    expected = set(SAME_SEMANTICS) | FORMAL_STOCK | EXT_STOCK
    missing = sorted(expected - set(config["gates"]))
    if missing:
        raise GateConfigError(f"Gate 未完整生成: {missing}")

    config["config_payload_sha256"] = hashlib.sha256(_canonical_payload(config)).hexdigest()
    return config


def render_json_bytes(config: dict[str, Any]) -> bytes:
    text = json.dumps(config, ensure_ascii=False, indent=1, separators=(",", ": "))
    return text.replace("\n", JSON_NEWLINE).encode("utf-8")


def _json_number(value: int | float) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def render_markdown_block(config: dict[str, Any]) -> str:
    lines = [
        BEGIN_MARKER,
        "本块由 `python benchmarks/generate_gate_config_v1.py <run-id>` 自动生成；禁止手改。",
        f"canonical run：`{config['run_id']}`。run-id 中的日期片段是冻结 provenance 标签，不是本文执行日期。",
        f"配置 payload SHA-256：`{config['config_payload_sha256']}`；generator SHA-256：`{config['generator_sha256']}`。",
        "机器 Gate 只比较全精度 `exact_half = raw_wall_ms / 2`；`display` 是向负无穷取整两位的小数，且必须满足 `display <= exact_half`。",
        "",
        "| scope | operation | backend | raw_wall_ms | exact_half | display | canonical source |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for label, gate in config["gates"].items():
        lines.append(
            "| {scope} | `{label}` | `{backend}` | {raw} | {exact} | {display:.2f} | `{source}` |".format(
                scope=gate["scope"],
                label=label,
                backend=gate["best_backend"],
                raw=_json_number(gate["raw_wall_ms"]),
                exact=_json_number(gate["exact_half"]),
                display=gate["display"],
                source=gate["source_path"],
            )
        )
    lines.extend(
        [
            "",
            "canonical source SHA-256：",
        ]
    )
    for backend in BACKENDS:
        source = config["canonical_sources"][backend]
        lines.append(f"- `{backend}`：`{source['path']}` → `{source['sha256']}`")
    lines.append(END_MARKER)
    return "\n".join(lines)


def _document_with_block(document: str, block: str, *, allow_insert: bool) -> str:
    begin_count = document.count(BEGIN_MARKER)
    end_count = document.count(END_MARKER)
    if begin_count == 1 and end_count == 1:
        begin = document.index(BEGIN_MARKER)
        end = document.index(END_MARKER, begin) + len(END_MARKER)
        if end < begin:
            raise GateConfigError("Gate Markdown 边界顺序错误")
        return document[:begin] + block + document[end:]
    if begin_count or end_count:
        raise GateConfigError(
            f"Gate Markdown 边界不完整: begin={begin_count} end={end_count}"
        )
    if not allow_insert:
        raise GateConfigError("IMPLEMENTATION 缺少 gate-config-v1 边界块")

    heading_prefix = "### 3.5 "
    heading_start = document.find(heading_prefix)
    if heading_start < 0 or document.find(heading_prefix, heading_start + 1) >= 0:
        raise GateConfigError("IMPLEMENTATION 必须恰有一个 §3.5 标题")
    heading_end = document.find("\n", heading_start)
    if heading_end < 0:
        raise GateConfigError("IMPLEMENTATION §3.5 标题缺少换行")
    next_heading = document.find("\n### ", heading_end)
    if next_heading < 0:
        raise GateConfigError("IMPLEMENTATION §3.5 后缺少下一小节")
    return document[: heading_end + 1] + "\n" + block + "\n" + document[next_heading:]


def _actual_markdown_block(document: str) -> str:
    if document.count(BEGIN_MARKER) != 1 or document.count(END_MARKER) != 1:
        raise GateConfigError("IMPLEMENTATION 的 Gate Markdown 边界必须各出现一次")
    begin = document.index(BEGIN_MARKER)
    end = document.index(END_MARKER, begin) + len(END_MARKER)
    return document[begin:end]


def _validate_embedded_hashes(actual: dict[str, Any]) -> None:
    if actual.get("generator") != _repo_path(pathlib.Path(__file__)):
        raise GateConfigError("gate config generator 路径不一致")
    if actual.get("generator_sha256") != _file_sha(pathlib.Path(__file__)):
        raise GateConfigError("gate config generator SHA-256 不一致")

    payload_sha = actual.get("config_payload_sha256")
    if not isinstance(payload_sha, str):
        raise GateConfigError("gate config 缺少 config_payload_sha256")
    payload = dict(actual)
    del payload["config_payload_sha256"]
    if hashlib.sha256(_canonical_payload(payload)).hexdigest() != payload_sha:
        raise GateConfigError("gate config payload SHA-256 不一致")

    sources = actual.get("canonical_sources")
    if not isinstance(sources, dict):
        raise GateConfigError("gate config 缺少 canonical_sources")
    for backend in BACKENDS:
        source = sources.get(backend)
        if not isinstance(source, dict):
            raise GateConfigError(f"gate config 缺少 canonical source: {backend}")
        path_text = source.get("path")
        if not isinstance(path_text, str):
            raise GateConfigError(f"canonical source path 非字符串: {backend}")
        path = REPO_ROOT / pathlib.PurePosixPath(path_text)
        if not path.is_file():
            raise GateConfigError(f"canonical source 不存在: {path_text}")
        if source.get("sha256") != _file_sha(path):
            raise GateConfigError(f"canonical source SHA-256 不一致: {path_text}")


def check(run_id: str) -> None:
    expected = build_config(run_id)
    expected_bytes = render_json_bytes(expected)
    if not OUT.is_file():
        raise GateConfigError(f"缺少 Gate 配置: {_repo_path(OUT)}")
    actual_bytes = OUT.read_bytes()
    if actual_bytes != expected_bytes:
        raise GateConfigError("docs/gate_config_v1.json 与 canonical 输入的精确生成结果不一致")
    try:
        actual = json.loads(actual_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateConfigError(f"Gate 配置不可解析: {exc}") from exc
    _validate_embedded_hashes(actual)

    if not IMPLEMENTATION.is_file():
        raise GateConfigError(f"缺少 IMPLEMENTATION: {_repo_path(IMPLEMENTATION)}")
    document = IMPLEMENTATION.read_text(encoding="utf-8")
    actual_block = _actual_markdown_block(document)
    expected_block = render_markdown_block(expected)
    if actual_block != expected_block:
        raise GateConfigError("IMPLEMENTATION §3.5 Gate 自动块与配置不一致")


def generate(run_id: str) -> dict[str, Any]:
    config = build_config(run_id)
    OUT.write_bytes(render_json_bytes(config))

    document = IMPLEMENTATION.read_text(encoding="utf-8")
    newline = "\r\n" if "\r\n" in document else "\n"
    normalized = document.replace("\r\n", "\n")
    updated = _document_with_block(
        normalized,
        render_markdown_block(config),
        allow_insert=True,
    )
    IMPLEMENTATION.write_bytes(updated.replace("\n", newline).encode("utf-8"))
    return config


def _print_summary(config: dict[str, Any], *, mode: str) -> None:
    print(f"{mode}: {_repo_path(OUT)}")
    print(
        f"run_id={config['run_id']} gates={len(config['gates'])} "
        f"config_payload_sha256={config['config_payload_sha256']}"
    )
    for label, gate in config["gates"].items():
        print(
            f"  [{gate['scope']:>9}] {label:<22} backend={gate['best_backend']:<8} "
            f"raw={gate['raw_wall_ms']:.6f} exact_half={gate['exact_half']:.6f} "
            f"display={gate['display']:.2f}"
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只读校验，不写文件")
    parser.add_argument("run_id", nargs="?", default=DEFAULT_RUN_ID)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.check:
            check(args.run_id)
            config = build_config(args.run_id)
            _print_summary(config, mode="check PASS")
        else:
            config = generate(args.run_id)
            _print_summary(config, mode="generated")
    except GateConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())