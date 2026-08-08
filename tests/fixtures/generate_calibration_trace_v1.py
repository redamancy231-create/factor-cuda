# -*- coding: utf-8 -*-
"""生成 reserve 校准状态机、缓存和预算边界的机械 trace。"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import pathlib
import tempfile

from calibration_v1 import (
    DEFAULT_RESERVE_BYTES,
    MAX_AGE_DAYS,
    MIN_SAMPLES,
    SCHEMA_VERSION,
    SOLVER_VERSION,
    atomic_write_json,
    budget_decision,
    build_payload,
    calibration_state,
    load_cache,
    make_cache_key,
    nearest_rank_p99,
    reserve_bytes,
    sample_bytes,
    validate_cache,
)

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "calibration_trace_v1.json"
TRACE_VERSION = "1.0.0"
NOW = datetime(2026, 8, 4, 12, 0, 0, tzinfo=timezone.utc)


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_trace() -> dict:
    key = make_cache_key(
        gpu_uuid="GPU-RTX4060-LAPTOP-POC",
        driver_version="poc-driver-v1",
        cuda_runtime="13.3",
        total_device_bytes=8188 * 1024 * 1024,
    )
    raw_pairs = [
        (8_000_000_000, 7_900_000_000),
        (8_000_000_000, 7_850_000_000),
        (8_000_000_000, 7_700_000_000),
        (8_000_000_000, 7_600_000_000),
        (8_000_000_000, 7_400_000_000),
        (8_000_000_000, 7_350_000_000),
        (100, 120),
    ]
    samples = [sample_bytes(before, minimum) for before, minimum in raw_pairs]
    p99 = nearest_rank_p99(samples)
    calibrated_reserve = reserve_bytes(samples)
    payload = build_payload(
        cache_key=key,
        samples_bytes=samples,
        measured_at_utc="2026-08-04T11:30:00Z",
    )
    valid, valid_reasons = validate_cache(payload, expected_key=key, now_utc=NOW)
    assert valid and valid_reasons == []

    short_payload = dict(payload)
    short_payload["samples_bytes"] = samples[:5]
    short_payload["calibrated_p99_bytes"] = nearest_rank_p99(samples[:5])
    short_payload["reserve_bytes"] = max(DEFAULT_RESERVE_BYTES, short_payload["calibrated_p99_bytes"])
    short_valid, short_reasons = validate_cache(short_payload, expected_key=key, now_utc=NOW)
    changed_key = dict(key)
    changed_key["gpu_uuid"] = "GPU-OTHER"
    changed_valid, changed_reasons = validate_cache(payload, expected_key=changed_key, now_utc=NOW)
    stale_payload = dict(payload)
    stale_payload["measured_at_utc"] = "2026-06-01T00:00:00Z"
    stale_valid, stale_reasons = validate_cache(stale_payload, expected_key=key, now_utc=NOW)

    base = 4096
    reserve = DEFAULT_RESERVE_BYTES
    boundaries = [
        budget_decision(free_bytes=reserve, reserve_bytes=reserve, base_required_bytes=base),
        budget_decision(free_bytes=reserve + base, reserve_bytes=reserve, base_required_bytes=base),
        budget_decision(free_bytes=reserve + base + 1, reserve_bytes=reserve, base_required_bytes=base),
    ]
    assert [item.ok for item in boundaries] == [False, False, True]

    with tempfile.TemporaryDirectory() as directory:
        cache_path = pathlib.Path(directory) / "reserve-cache.json"
        missing_status, _, missing_reasons = load_cache(cache_path, expected_key=key, now_utc=NOW)
        cache_path.write_bytes(b"{not-json")
        corrupt_status, _, corrupt_reasons = load_cache(cache_path, expected_key=key, now_utc=NOW)
        atomic_write_json(cache_path, payload)
        first_bytes = cache_path.read_bytes()
        first_status, _, first_reasons = load_cache(cache_path, expected_key=key, now_utc=NOW)
        later = dict(payload)
        later["measured_at_utc"] = "2026-08-04T11:45:00Z"
        atomic_write_json(cache_path, later)
        last_payload = json.loads(cache_path.read_text(encoding="utf-8"))
        temp_leftovers = sorted(path.name for path in cache_path.parent.glob(f".{cache_path.name}.*.tmp"))
        atomic = {
            "utf8_bom": first_bytes.startswith(b"\xef\xbb\xbf"),
            "lf_only": b"\r\n" not in first_bytes,
            "missing": {"status": missing_status, "reasons": missing_reasons},
            "corrupt": {"status": corrupt_status, "reasons": corrupt_reasons},
            "first_complete_write": {"status": first_status, "reasons": first_reasons},
            "last_complete_rename_wins": last_payload["measured_at_utc"] == later["measured_at_utc"],
            "temp_leftovers": temp_leftovers,
        }

    return {
        "trace_version": TRACE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "solver_version": SOLVER_VERSION,
        "execution_date": "2026-08-04",
        "generator": "tests/fixtures/generate_calibration_trace_v1.py",
        "generator_sha256": _sha(pathlib.Path(__file__)),
        "source": "tests/fixtures/calibration_v1.py",
        "source_sha256": _sha(HERE / "calibration_v1.py"),
        "schema": "tests/fixtures/calibration_v1.schema.json",
        "schema_sha256": _sha(HERE / "calibration_v1.schema.json"),
        "constants": {
            "minimum_samples": MIN_SAMPLES,
            "max_age_days": MAX_AGE_DAYS,
            "default_reserve_bytes": DEFAULT_RESERVE_BYTES,
        },
        "sampling": {
            "formula": "max(0, free_before - min_free_during)",
            "raw_pairs": [{"free_before": a, "min_free_during": b} for a, b in raw_pairs],
            "samples_bytes": samples,
            "nearest_rank_index": __import__("math").ceil(0.99 * len(samples)) - 1,
            "calibrated_p99_bytes": p99,
            "reserve_rule": "max(512 MiB, calibrated_p99)",
            "reserve_bytes": calibrated_reserve,
        },
        "cache_key_fields": list(key.keys()),
        "invalidation": {
            "missing": ["missing"],
            "corrupt": ["corrupt"],
            "samples_lt_6": {"valid": short_valid, "reasons": short_reasons},
            "key_or_device_changed": {"valid": changed_valid, "reasons": changed_reasons},
            "age_gt_30_days": {"valid": stale_valid, "reasons": stale_reasons},
        },
        "state_machine": [
            {"from": "missing", "sample_count": 0, "to": calibration_state(cache_status="missing", sample_count=0)},
            {"from": "collecting", "sample_count": 5, "to": calibration_state(cache_status="invalid", sample_count=5)},
            {"from": "collecting", "sample_count": 6, "to": calibration_state(cache_status="invalid", sample_count=6)},
            {"from": "persist", "sample_count": 6, "write_succeeded": False, "to": calibration_state(cache_status="invalid", sample_count=6, write_succeeded=False)},
            {"from": "cache", "sample_count": 0, "to": calibration_state(cache_status="ready", sample_count=0)},
        ],
        "budget_boundaries": [item.__dict__ for item in boundaries],
        "atomic_write": atomic,
        "valid_payload": payload,
    }


def main() -> int:
    trace = build_trace()
    OUT.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"generated: {OUT}")
    print(f"samples={len(trace['sampling']['samples_bytes'])} p99={trace['sampling']['calibrated_p99_bytes']}")
    print("budget boundaries: fail, fail, pass")
    print(f"sha256: {_sha(OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
