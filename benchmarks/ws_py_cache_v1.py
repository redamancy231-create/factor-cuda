# -*- coding: utf-8 -*-
"""[P3] 适配层自动缓存——Python 端性能证据（无缓存 vs 有缓存，同面板）。

测量对象：fc.rolling_ic / fc.cross_sectional_rank 端到端（corpus 面板
1218x5000）。关键口径：**同一面板、同一输入、同一 mask，唯一变量 = 缓存
开关**（methodology-perf-same-panel-discipline）；11 次调用取中位（镜像
C++ perf methodology）。

fail-closed 链（写盘前全部 gate 通过，否则 exit 3 + 删旧产物）：
  1. bitwise_identical -- 无缓存 vs 有缓存输出逐位一致（equal_nan），缓存
     绝不能改变任何结果（P3 核心正确性）。
  2. cache_reused -- 同形状第二次调用不新增 cache 条目（防"缓存未生效但
     收益是噪声"假绿）。
  3. direction -- 有缓存中位 < 无缓存中位（缓存必须更快，方向反则拒绝）。
  4. speedup >= THRESHOLD（1.2x，P3 gate）。
  5. provenance（git_head/git_dirty/env/generated_at）+ capture_sha256
     绑定原始测量数据；原子写（.tmp + os.replace）。

复现：PYTHONIOENCODING=utf-8 python benchmarks/ws_py_cache_v1.py
输出：benchmarks/results/ws_py_cache_v1.{json,md}
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS_DIR = HERE / "results"
OUT_JSON = RESULTS_DIR / "ws_py_cache_v1.json"
OUT_MD = RESULTS_DIR / "ws_py_cache_v1.md"

VERSION = "1.0.0"
GENERATOR = "ws_py_cache_v1.py"
T, N = 1218, 5000
SEED = 42
REPS = 11          # median of 11 (mirror C++ perf methodology)
THRESHOLD = 1.2    # P3 gate: >= 1.2x cached-vs-uncached speedup


def git_head() -> str:
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else "n/a"
    except Exception:
        return "n/a"


def git_dirty() -> bool:
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                           capture_output=True, text=True, timeout=30)
        return bool(r.stdout.strip()) if r.returncode == 0 else False
    except Exception:
        return False


def _env_fingerprint() -> dict:
    """CPU/GPU/binding identity. ASCII-safe (no machine paths)."""
    try:
        import torch
        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else None
        dev = torch.cuda.get_device_properties(0) if cuda else None
        cc = f"{dev.major}.{dev.minor}" if dev else None
    except Exception:
        gpu = cc = None
    return {
        "python": sys.version.split()[0],
        "torch": (None if sys.modules.get("torch") is None else
                  getattr(sys.modules.get("torch"), "__version__", None)),
        "cuda_available": bool(gpu),
        "gpu": gpu,
        "compute_capability": cc,
    }


def _sha(obj) -> str:
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def _measure_op(name, cached_fn, uncached_fn, cache_obj, out_kind):
    """Measure ONE op on the SAME panel with the cache disabled vs enabled.

    Returns the measurement dict consumed by _validate + the payload. cache_obj
    is the op's fc._workspace._WorkspaceCache instance (entry-count probe for
    the cache_reused gate)."""
    from fc import _workspace as _ws
    import fc

    # ---- uncached path (cache disabled) ---------------------------------
    fc.clear_workspaces()
    for c in _ws._CACHES:
        c.set_enabled(False)
    out_u = uncached_fn()                          # bitwise probe output
    uncached_fn()                                  # warmup the timed region
    uncached_times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        uncached_fn()
        uncached_times.append((time.perf_counter() - t0) * 1000.0)
    uncached_times.sort()
    uncached_ms = uncached_times[REPS // 2]

    # ---- cached path (enabled, fresh caches) ----------------------------
    for c in _ws._CACHES:
        c.set_enabled(True)
    fc.clear_workspaces()
    out_c = cached_fn()                            # bitwise probe output
    n_after_first = len(cache_obj._items)          # 1 entry created
    cached_fn()                                    # same shape again
    n_after_second = len(cache_obj._items)         # must NOT grow
    cached_times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        cached_fn()
        cached_times.append((time.perf_counter() - t0) * 1000.0)
    cached_times.sort()
    cached_ms = cached_times[REPS // 2]

    identical = bool(np.array_equal(out_u, out_c, equal_nan=True))
    reused = n_after_second == n_after_first and n_after_first >= 1
    speedup = uncached_ms / cached_ms if cached_ms > 0 else 0.0

    return {
        "op": name,
        "uncached_ms": round(uncached_ms, 4),
        "cached_ms": round(cached_ms, 4),
        "speedup_x": round(speedup, 4),
        "bitwise_identical": identical,
        "cache_reused": reused,
        "cache_entries_after_first": n_after_first,
        "cache_entries_after_second": n_after_second,
        "output_kind": out_kind,
        "output_sha256": _sha(out_u.tolist()),
        "uncached_raw_ms": [round(v, 4) for v in uncached_times],
        "cached_raw_ms": [round(v, 4) for v in cached_times],
    }


def _validate(ops) -> list:
    """Fail-closed validation. Returns a list of problems; non-empty -> reject."""
    problems = []
    for m in ops:
        if not m["bitwise_identical"]:
            problems.append(
                f"{m['op']}: cached vs uncached NOT bitwise identical")
        if not m["cache_reused"]:
            problems.append(
                f"{m['op']}: cache not reused across same-shape calls "
                f"(entries {m['cache_entries_after_first']}->"
                f"{m['cache_entries_after_second']})")
        if not (m["uncached_ms"] > m["cached_ms"]):
            problems.append(
                f"{m['op']}: direction wrong (uncached {m['uncached_ms']} ms <= "
                f"cached {m['cached_ms']} ms)")
        if m["speedup_x"] < THRESHOLD:
            problems.append(
                f"{m['op']}: speedup {m['speedup_x']}x < {THRESHOLD}x gate")
    return problems


def render_md(payload: dict) -> str:
    lines = [
        "# ws_py_cache_v1 — 适配层自动缓存 Python 端证据",
        "",
        f"- generator: `{payload['generator']}` (schema {payload['schema_version']})",
        f"- generated_at: {payload['generated_at']}",
        f"- closure_status: `{payload['closure_status']}`",
        f"- git_head: `{payload['provenance']['git_head']}`",
        f"- git_dirty: {payload['provenance']['git_dirty']}",
        f"- capture_sha256: `{payload['provenance']['capture_sha256']}`",
        f"- env: python {payload['env'].get('python')}, "
        f"{payload['env'].get('gpu')} (CC {payload['env'].get('compute_capability')})",
        "",
        f"面板 `(T,N)=({payload['panel']['T']},{payload['panel']['N']})`, "
        f"seed {payload['panel']['seed']}。同一面板/输入/mask 仅缓存开关不同；"
        f"11 次取中位。gate = speedup ≥ {payload['threshold']}x。",
        "",
        "| op | 无缓存 (ms) | 有缓存 (ms) | speedup | 位级一致 | 缓存复用 | 判定 |",
        "|----|------------|------------|---------|---------|---------|------|",
    ]
    for m in payload["ops"]:
        ok = m["speedup_x"] >= payload["threshold"] and m["bitwise_identical"] \
            and m["cache_reused"]
        lines.append(
            f"| {m['op']} | {m['uncached_ms']} | {m['cached_ms']} | "
            f"{m['speedup_x']}x | {'PASS' if m['bitwise_identical'] else 'FAIL'} | "
            f"{'PASS' if m['cache_reused'] else 'FAIL'} | "
            f"{'BEATS' if ok else 'FAIL'} |")
    lines.append("")
    lines.append("## 判定")
    if all(m["speedup_x"] >= payload["threshold"] for m in payload["ops"]):
        lines.append(
            "**BEATS gate**：全部 op 收益 ≥ 1.2x，缓存路径结果位级一致且缓存"
            "实际复用 → 自动缓存收益成立。")
    else:
        lines.append("**FAIL**：至少一个 op 未达 gate（详见上表）。")
    lines.append("")
    lines.append("## 备注")
    lines.append(payload["disclosure"])
    return "\n".join(lines) + "\n"


def main() -> int:
    sys.path.insert(0, str(ROOT))
    import fc
    from fc import _workspace as _ws
    from fc.cross_sectional_rank import _RANK_CACHE
    from fc.rolling_ic import _ROLL_CACHE

    rng = np.random.default_rng(SEED)
    f = rng.standard_normal((T, N)).astype(np.float64)
    r = rng.standard_normal((T, N)).astype(np.float64)
    x = rng.standard_normal((T, N)).astype(np.float32)
    fm = rng.random((T, N)) < 0.95
    rm = rng.random((T, N)) < 0.97

    # warmup: CUDA context + binding load + first workspace allocs
    fc.rolling_ic(f, r, fm, rm, min_valid=30, device="cuda")
    fc.cross_sectional_rank(x)
    fc.clear_workspaces()

    def _cached_roll():
        return fc.rolling_ic(f, r, fm, rm, min_valid=30, device="cuda")

    def _uncached_roll():
        return fc.rolling_ic(f, r, fm, rm, min_valid=30, device="cuda")

    def _cached_rank():
        return fc.cross_sectional_rank(x)

    def _uncached_rank():
        return fc.cross_sectional_rank(x)

    ops = [
        _measure_op("rolling_ic", _cached_roll, _uncached_roll, _ROLL_CACHE,
                    "ic"),
        _measure_op("cs_rank", _cached_rank, _uncached_rank, _RANK_CACHE,
                    "rank"),
    ]

    # restore a clean cached state for any later process
    for c in _ws._CACHES:
        c.set_enabled(True)

    problems = _validate(ops)
    if problems:
        print("FAIL-CLOSED: evidence rejected, nothing written:")
        for p in problems:
            print(f"  - {p}")
        for p in (OUT_JSON, OUT_MD):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        return 3

    env = _env_fingerprint()
    payload = {
        "schema_version": VERSION,
        "artifact": "ws_py_cache_v1.json",
        "generator": GENERATOR,
        "generated_at": datetime.datetime.now().astimezone().isoformat(
            timespec="seconds"),
        "env": env,
        "panel": {"T": T, "N": N, "seed": SEED},
        "method": ("same-panel cached vs uncached (only cache on/off differs); "
                   "median of 11 wall-clock calls; warmup before timing"),
        "threshold": THRESHOLD,
        "judgement": ("speedup >= 1.2x AND bitwise_identical AND cache_reused "
                      "for every op; P3 gate"),
        "closure_status": "OK",
        "provenance": {
            "source": "live",
            "git_head": git_head(),
            "git_dirty": git_dirty(),
            "capture_sha256": _sha([m["uncached_raw_ms"] for m in ops]
                                   + [m["cached_raw_ms"] for m in ops]),
        },
        "disclosure": ("Python 适配层 + 绑定层开销≈0（上会话实测）；缓存收益来自"
                       "消除 per-call 设备分配（C++ workspace 先例：cs_rank "
                       "16.4→9.17ms、rolling_ic 48.98→33.0ms）。"),
        "ops": ops,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    try:
        tmp_json = OUT_JSON.with_suffix(".json.tmp")
        tmp_md = OUT_MD.with_suffix(".md.tmp")
        tmp_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2)
                            + "\n", encoding="utf-8", newline="\n")
        tmp_md.write_text(render_md(payload), encoding="utf-8", newline="\n")
        os.replace(tmp_json, OUT_JSON)
        os.replace(tmp_md, OUT_MD)
    except Exception as e:
        print(f"FAIL-CLOSED: write error ({e}); stale artifacts invalidated")
        for p in (OUT_JSON, OUT_MD):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        return 3

    print("evidence written: ws_py_cache_v1.{json,md}")
    for m in ops:
        print(f"  {m['op']}: uncached {m['uncached_ms']} ms -> cached "
              f"{m['cached_ms']} ms = {m['speedup_x']}x  "
              f"bitwise={m['bitwise_identical']} reuse={m['cache_reused']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
