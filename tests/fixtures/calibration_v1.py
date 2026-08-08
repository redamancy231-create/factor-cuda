# -*- coding: utf-8 -*-
"""PoC ③ 显存 reserve 校准契约与可执行状态机。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
import pathlib
import tempfile
from typing import Any, Iterable

SCHEMA_VERSION = "1.0.0"
SOLVER_VERSION = "2.0.0"
DEFAULT_RESERVE_BYTES = 512 * 1024 * 1024
MIN_SAMPLES = 6
MAX_AGE_DAYS = 30


def sample_bytes(free_before: int, min_free_during: int) -> int:
    """单轮占用样本：负值按 0 截断。"""
    if free_before < 0 or min_free_during < 0:
        raise ValueError("free byte counters must be non-negative")
    return max(0, int(free_before) - int(min_free_during))


def nearest_rank_p99(samples: Iterable[int]) -> int:
    """nearest-rank p99；0-based 索引为 ceil(0.99*n)-1。"""
    values = sorted(int(value) for value in samples)
    if not values:
        raise ValueError("at least one sample is required")
    if any(value < 0 for value in values):
        raise ValueError("samples must be non-negative")
    index = math.ceil(0.99 * len(values)) - 1
    return values[index]


def reserve_bytes(samples: Iterable[int]) -> int:
    values = list(samples)
    if len(values) < MIN_SAMPLES:
        raise ValueError(f"at least {MIN_SAMPLES} samples are required")
    return max(DEFAULT_RESERVE_BYTES, nearest_rank_p99(values))


def make_cache_key(
    *,
    gpu_uuid: str,
    driver_version: str,
    cuda_runtime: str,
    total_device_bytes: int,
    schema_version: str = SCHEMA_VERSION,
    solver_version: str = SOLVER_VERSION,
) -> dict[str, Any]:
    if not gpu_uuid or not driver_version or not cuda_runtime:
        raise ValueError("cache key strings must be non-empty")
    if total_device_bytes <= 0:
        raise ValueError("total_device_bytes must be positive")
    return {
        "gpu_uuid": gpu_uuid,
        "driver_version": driver_version,
        "cuda_runtime": cuda_runtime,
        "total_device_bytes": int(total_device_bytes),
        "schema_version": schema_version,
        "solver_version": solver_version,
    }


def _parse_utc(timestamp: str) -> datetime:
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include timezone")
    return parsed.astimezone(timezone.utc)


def validate_cache(
    payload: Any,
    *,
    expected_key: dict[str, Any],
    now_utc: datetime,
) -> tuple[bool, list[str]]:
    """Fail closed；返回稳定失效原因集合。"""
    reasons: list[str] = []
    if not isinstance(payload, dict):
        return False, ["corrupt"]
    if payload.get("schema_version") != SCHEMA_VERSION:
        reasons.append("version_changed")
    if payload.get("solver_version") != SOLVER_VERSION:
        reasons.append("version_changed")
    if payload.get("cache_key") != expected_key:
        reasons.append("key_or_device_changed")
    samples = payload.get("samples_bytes")
    if not isinstance(samples, list) or any(not isinstance(value, int) or value < 0 for value in samples):
        reasons.append("corrupt")
    elif len(samples) < MIN_SAMPLES:
        reasons.append("samples_lt_6")
    try:
        measured_at = _parse_utc(payload["measured_at_utc"])
        age = now_utc.astimezone(timezone.utc) - measured_at
        if age.total_seconds() < 0 or age.total_seconds() > MAX_AGE_DAYS * 86400:
            reasons.append("age_gt_30_days")
    except (KeyError, TypeError, ValueError):
        reasons.append("corrupt")
    if isinstance(samples, list) and len(samples) >= MIN_SAMPLES and all(
        isinstance(value, int) and value >= 0 for value in samples
    ):
        expected_p99 = nearest_rank_p99(samples)
        expected_reserve = max(DEFAULT_RESERVE_BYTES, expected_p99)
        if payload.get("calibrated_p99_bytes") != expected_p99:
            reasons.append("corrupt")
        if payload.get("reserve_bytes") != expected_reserve:
            reasons.append("corrupt")
    return not reasons, list(dict.fromkeys(reasons))


def load_cache(
    path: pathlib.Path,
    *,
    expected_key: dict[str, Any],
    now_utc: datetime,
) -> tuple[str, dict[str, Any] | None, list[str]]:
    if not path.exists():
        return "missing", None, ["missing"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return "invalid", None, ["corrupt"]
    valid, reasons = validate_cache(payload, expected_key=expected_key, now_utc=now_utc)
    return ("ready", payload, []) if valid else ("invalid", None, reasons)


def calibration_state(
    *,
    cache_status: str,
    sample_count: int,
    write_succeeded: bool | None = None,
) -> str:
    """状态机：missing/invalid → collecting → ready；写失败可重试。"""
    if cache_status == "ready":
        return "ready"
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    if sample_count < MIN_SAMPLES:
        return "collecting"
    if write_succeeded is False:
        return "retryable_write_failure"
    return "ready"


def build_payload(
    *,
    cache_key: dict[str, Any],
    samples_bytes: Iterable[int],
    measured_at_utc: str,
) -> dict[str, Any]:
    samples = [int(value) for value in samples_bytes]
    p99 = nearest_rank_p99(samples)
    reserve = max(DEFAULT_RESERVE_BYTES, p99)
    return {
        "schema_version": SCHEMA_VERSION,
        "solver_version": SOLVER_VERSION,
        "cache_key": cache_key,
        "measured_at_utc": measured_at_utc,
        "sample_formula": "max(0, free_before - min_free_during)",
        "p99_method": "nearest-rank: ceil(0.99*n)-1",
        "samples_bytes": samples,
        "calibrated_p99_bytes": p99,
        "default_reserve_bytes": DEFAULT_RESERVE_BYTES,
        "reserve_bytes": reserve,
    }


def atomic_write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    """同目录唯一临时文件，flush+fsync 后 close，再原子 replace。"""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = pathlib.Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        try:
            temp_path.unlink(missing_ok=True)
        finally:
            raise


@dataclass(frozen=True)
class BudgetDecision:
    ok: bool
    free_bytes: int
    reserve_bytes: int
    base_required_bytes: int
    available_after_reserve_bytes: int
    reason: str


def budget_decision(*, free_bytes: int, reserve_bytes: int, base_required_bytes: int) -> BudgetDecision:
    """reserve 只在此处消费；要求 available 严格大于 base。"""
    if min(free_bytes, reserve_bytes, base_required_bytes) < 0:
        raise ValueError("budget values must be non-negative")
    available = max(0, free_bytes - reserve_bytes)
    ok = free_bytes > reserve_bytes + base_required_bytes
    reason = "pass" if ok else "insufficient_after_reserve"
    return BudgetDecision(ok, free_bytes, reserve_bytes, base_required_bytes, available, reason)
