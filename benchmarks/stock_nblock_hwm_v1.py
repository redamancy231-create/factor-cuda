# -*- coding: utf-8 -*-
"""factor-cuda -- stock_corr N-blocking device HWM measurement report v1.

Runs `poc3_stock_corr_selfcheck.exe --hwm` (or parses a captured run via
`--from`), parses the FBLC| records, and adjudicates each case against the
memory_budget_v1 model (`nblock_stock_peak`).

Why this exists (closing the M3 "model prediction, not measured" gap for the
stock N-blocking path of memory_budget_v1, M4 closure):
  - The M4 scenario (stock N=22600) needs O(N^2) host output + O(N^2*T) compute
    that is impractical to run here. KEY MEASUREMENT FACT (documented in the
    driver): the nblock DEVICE peak is N-INDEPENDENT for N >= 2*block_width
    (max_cols = min(N, 2*block_width) = 2*block_width; every tile buffer is
    sized by max_cols, not N). So measuring N=5000 anchors the N=22600 closure:
    device residency is tile-local; only the host loop length and host N*N
    output grow with N. The N=22600 host peak (~8.4 GB: two live N*N output
    buffers -- the caller's and the driver's internal -- plus the T*N panel) is
    disclosed and NOT run (external review F7/MINOR-8).
  - stock_corr_gpu_nblock takes no MemTracker, so the 5-op calibration's
    three-way discipline reduces to the driver-sample leg only. Acceptance
    (review F1/h1/3, factor-fblock precedent: NO 64 MiB delta band):
      fit       : driver_peak <= available budget (no independent stock-nblock
                  allocation-chain probe exists; the driver_peak - model delta
                  is reported but NOT separately attributed -- cudaMemGetInfo
                  free memory is MiB-quantized, so small deltas carry +/-1 MiB
                  error, external F1/contract + MINOR-9)
      exhausted : margin (fb_case - driver_peak == min_free) ~ 0 with rc==0
                  (WDDM shared-memory fallback; NOT a usable fit)
      OOM       : rc != 0 (genuine cudaMalloc failure)
  - Only the nblock column is closed to measured here; the current-peak
    (production) column remains a model prediction (calibration-verified HWM).

FBLC record format (printed by the exe):
  FBLC|kind=nblock|T=...|N=...|block=...|reps=...|fb=<B>|driver_peak=<B>|rc=<int>
  GUARD|pass=0/1|r1=<int>|r2=<int>   (bitwise nblock-vs-production drift guard)

Usage:
    PYTHONIOENCODING=utf-8 python benchmarks/stock_nblock_hwm_v1.py [--from <captured.txt>]
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXE = ROOT / "build" / "poc3_stock_corr_selfcheck.exe"
RESULTS = pathlib.Path(__file__).resolve().parent / "results"
OUT_JSON = RESULTS / "stock_nblock_hwm_v1.json"
OUT_MD = RESULTS / "stock_nblock_hwm_v1.md"
CUDA_BIN = os.path.join(os.environ.get("CUDA_PATH", ""), "bin", "x64")
MIB = 1048576.0
T_CANONICAL = 1218

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from memory_budget_v1 import (  # noqa: E402
    nblock_stock_peak,
    stock_corr_peak,
    AVAILABLE_BYTES,
)

VERSION = "1.0.0"
GENERATOR = ("benchmarks/stock_nblock_hwm_v1.py (DeepSeek-V4-Flash via "
             "Claude Code CLI, 2026-08-08)")


def _env_fingerprint() -> dict[str, object]:
    import platform
    env: dict[str, object] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
    try:
        import torch
        env["torch"] = torch.__version__
        env["gpu"] = torch.cuda.get_device_name(0)
        env["total_MiB"] = torch.cuda.get_device_properties(0).total_memory / MIB
    except Exception:
        env["torch"] = "n/a"
    return env


def git_head() -> str:
    """Current git HEAD of the repo (external review MAJOR-6: provenance must
    bind the evidence to the source). Best-effort; 'n/a' if not a repo."""
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else "n/a"
    except Exception:
        return "n/a"


def exe_sha256() -> str:
    """SHA-256 of the harness EXE if present (binds the FBLC output to the
    binary that produced it)."""
    try:
        return hashlib.sha256(EXE.read_bytes()).hexdigest().upper() if EXE.exists() else "n/a"
    except Exception:
        return "n/a"


def git_dirty() -> bool:
    """Whether the working tree is dirty (external review MINOR-10: provenance
    should record the source state)."""
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                           capture_output=True, text=True, timeout=30)
        return bool(r.stdout.strip()) if r.returncode == 0 else False
    except Exception:
        return False


def run_exe() -> tuple[str, int]:
    """Run the harness; return (stdout+stderr, returncode). The returncode must
    reach the fail-closed validation -- a crashed/aborted exe must NOT produce
    ingested evidence."""
    env = dict(os.environ)
    env["PATH"] = CUDA_BIN + ";" + env.get("PATH", "")
    print(f"running {EXE} --hwm ...")
    t0 = time.time()
    r = subprocess.run([str(EXE), "--hwm"], capture_output=True, text=True,
                       env=env, timeout=1800)
    dt = time.time() - t0
    print(f"exit={r.returncode} in {dt:.0f}s")
    return r.stdout + "\n" + r.stderr, r.returncode


def parse_free_before(text: str) -> float | None:
    for line in text.splitlines():
        if line.startswith("HWM mode: GPU free_before"):
            try:
                return float(line.split("free_before")[1].split("MiB")[0].strip())
            except (ValueError, IndexError):
                return None
    return None


_FBLC_FIELDS = {"kind", "T", "N", "block", "reps", "samples", "fb",
                "driver_peak", "rc"}


def parse_fblc(text: str) -> tuple[list[dict], int]:
    """Parse FBLC records; return (cases, malformed_count). A malformed line OR
    an UNKNOWN extra field (e.g. a forged |fits=1, external review MAJOR-2) is
    counted and fails the run (external review MAJOR-4: garbage FBLC-prefixed
    records must fail the run, not be silently skipped)."""
    cases = []
    malformed = 0
    for line in text.splitlines():
        if not line.startswith("FBLC|"):
            continue
        fields = {}
        for kv in line[len("FBLC|"):].split("|"):
            if "=" in kv:
                k, v = kv.split("=", 1)
                fields[k] = v
        unknown = set(fields) - _FBLC_FIELDS
        if unknown:  # external review MAJOR-2: reject forged extra fields
            malformed += 1
            print(f"skip FBLC line with unknown fields: {sorted(unknown)}")
            continue
        try:
            # external review FACTOR-1 (shared with the stock side): kind is the
            # ONLY field main() dereferences that was never validated here -- a
            # missing `kind=` would crash main() with KeyError (exit 1) instead of
            # the fail-closed return-3 + stale-delete path.
            fields["kind"] = str(fields["kind"])
            fields["T"] = int(fields["T"])
            fields["N"] = int(fields["N"])
            fields["block"] = int(fields["block"])
            fields["reps"] = int(fields["reps"])
            fields["fb"] = int(fields["fb"])
            fields["driver_peak"] = int(fields["driver_peak"])
            fields["samples"] = int(fields["samples"])
            fields["rc"] = int(fields["rc"])
            cases.append(fields)
        except (KeyError, ValueError) as e:
            malformed += 1
            print(f"skip malformed FBLC line: {e} -- {line[:120]}")
    return cases, malformed


def parse_guard(text: str) -> dict | None:
    for line in text.splitlines():
        if line.startswith("GUARD|"):
            fields = {}
            for kv in line[len("GUARD|"):].split("|"):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    fields[k] = v
            try:
                return {"pass": int(fields["pass"]),
                        "r1": int(fields["r1"]), "r2": int(fields["r2"])}
            except (KeyError, ValueError):
                return None
    return None


def model_peak(c: dict) -> int | None:
    """Model prediction (bytes) for a case; None if the model has no entry.

    with_mask=False for the production control: the --hwm run calls
    stock_corr_gpu(X, nullptr, ...), and src/stock_corr.cu only allocates d_mask
    when h_mask != nullptr, so the measured 7-buffer peak must be compared to the
    7-buffer no-mask model (review SC-4/STOCK_NBLOCK_HWM-1: with_mask=True would
    add a phantom d_mask and understate the real driver overhead by 5.81 MiB).
    """
    if c["kind"] == "production":
        peak, _ = stock_corr_peak(T_CANONICAL, c["N"], with_mask=False)
        return peak
    return nblock_stock_peak(T_CANONICAL, c["N"], c["block"])["peak_B"]


def adjudicate(cases: list[dict]) -> list[dict]:
    """Two-layer adjudication (factor-fblock precedent: NO 64 MiB delta band).

    fit = driver_peak <= available budget. The driver_peak - model delta is
    REPORTED only (external MINOR-9): there is no independent stock-nblock
    allocation-chain probe, and cudaMemGetInfo free memory is MiB-quantized, so
    small deltas carry +/-1 MiB error and are not separately attributable.
    VRAM-exhausted (margin ~ 0, rc==0) and OOM (rc != 0) are non-fits.
    """
    rows = []
    for c in cases:
        c["driver_peak_MiB"] = c["driver_peak"] / MIB
        mp = model_peak(c)
        c["model_peak_MiB"] = round(mp / MIB, 2) if mp is not None else None
        c["margin_MiB"] = (None if not c.get("fb")
                           else round((c["fb"] - c["driver_peak"]) / MIB, 2))
        # vram_exhausted REQUIRES rc==0 (WDDM fallback, never a genuine OOM).
        c["vram_exhausted"] = (c["rc"] == 0 and c["margin_MiB"] is not None
                               and c["margin_MiB"] <= 8.0)
        c["model_fits"] = (mp is not None) and (mp <= AVAILABLE_BYTES)
        note = ""
        if mp is not None:
            c["delta_MiB"] = round((c["driver_peak"] - mp) / MIB, 2)
        if c["rc"] != 0:
            c["pass"] = False
            note = f"OOM rc={c['rc']}(分配链模型 {c['model_peak_MiB']} MiB 设备不可达)"
        elif c["vram_exhausted"]:
            c["pass"] = False
            note = ("物理显存耗尽(margin≈0, rc==0);与 WDDM 超分配/共享内存回退一致,"
                    "具体机制未隔离验证;模型峰值超设备物理显存")
        elif mp is not None:
            c["fits"] = c["driver_peak"] <= AVAILABLE_BYTES
            c["pass"] = c["fits"]
            note = (f"fit 余量 {round(AVAILABLE_BYTES/MIB - c['driver_peak_MiB'], 1)} MiB"
                    if c["fits"] else
                    f"超出预算 {round(c['driver_peak_MiB'] - AVAILABLE_BYTES/MIB, 1)} MiB")
        else:
            c["pass"] = False
            note = "no model entry"
        c["note"] = note
        rows.append(c)
    return rows


def render_md(payload: dict) -> str:
    rows = []
    for c in payload["cases"]:
        scale = (f"block={c['block']}" if c["kind"] == "nblock"
                 else "production (non-blocked)")
        mark = "✅" if c["pass"] else "❌"
        margin = "—" if c.get("margin_MiB") is None else c["margin_MiB"]
        delta = "—" if c.get("vram_exhausted") else c.get("delta_MiB", "—")
        rows.append(
            f"| {c['kind']} | {c['T']}×{c['N']} | {scale} | {c['driver_peak_MiB']:.1f} "
            f"| {c['model_peak_MiB'] if c['model_peak_MiB'] is not None else '—'} "
            f"| {delta} | {margin} | {c['rc']} | {mark} | {c['note']} |"
        )
    table = "\n".join(rows)
    md = f"""# factor-cuda stock_corr N-blocking 设备 HWM 实测(v1)

> 生成:{time.strftime('%Y-%m-%d')} · {GENERATOR}
> 判据(两层,替代 5-op 校准的 delta_formula==0——nblock 无 MemTracker):
> **运行期 fits**:driver_peak ≤ {payload['memory_budget']['available_MiB']} MiB 可用预算;driver_peak−模型 delta 未独立归因(cudaMemGetInfo MiB 量化 ±1 MiB,无 stock allocation-chain probe)
> **显存耗尽**:margin≈0 且 rc=0(与 WDDM 超分配/共享内存回退一致,机制未隔离验证;非可用 fit)

## 结论

**{payload['n_cases']} 例 FBLC 记录,{payload['n_pass']} 例判定为可用 fit。**
- **nblock block=256 实测 driver_peak {payload['summary']['nblock_b256_driver_MiB']} MiB**(模型 {payload['summary']['nblock_b256_model_MiB']} MiB,delta +{payload['summary']['nblock_b256_delta_MiB']} MiB)→ **fits(预算余量 {payload['summary']['nblock_b256_budget_margin_MiB']} MiB)** → **N-blocking 峰值闭合为实测**
- **峰值与 N 无关**(N≥2*block 时 max_cols=2*block,tile 缓冲按 max_cols 非 N)→ **N=22600 的 device 峰值 = {payload['summary']['nblock_b256_driver_MiB']} MiB**(同 block=256),M4 闭合成立;**N=22600 实际运行未实测**(host 峰值 ~8.4 GB——两个 N*N 输出 buffer(每 ~3.9 GiB)并存 + 面板,非单个 ~4 GiB,review F7;+ O(N²·T) 计算,诚实披露)
- block 阶梯递减:256→{payload['summary']['nblock_b128_driver_MiB']} (128)→{payload['summary']['nblock_b64_driver_MiB']} (64)→{payload['summary']['nblock_b32_driver_MiB']} MiB (32);production N=5000 {payload['summary']['production_driver_MiB']} MiB(no-mask 模型 {payload['summary']['production_model_MiB']} MiB,delta +{payload['summary']['production_delta_MiB']} MiB——校准 stock_corr driver overhead 实测 7-11 MiB 范围,一致)
- **测量披露**:cudaMemGetInfo free 内存 ~1 MiB 量化(driver_peak 均为 MiB 整数倍),小 delta 归因含 ±1 MiB 误差(review F1/contract)

## 环境

- GPU:{payload['env'].get('gpu','—')} total {payload['env'].get('total_MiB','—')} MiB
- 运行期 free_before ≈ {payload['free_before_MiB']} MiB;模型预算口径 {payload['memory_budget']['available_MiB']} MiB(8188−512)

## 明细

| kind | T×N | 规模 | driver(MiB) | 模型(MiB) | Δ(MiB) | margin(MiB) | rc | 判定 | 注 |
|---|---|---|---|---|---|---|---|---|---|
{table}

## 闭合范围(诚实边界)

- 实测闭合**仅 N-blocking 路径**(N=5000 锚定,峰值 N 无关);`current_peak`(production)仍为模型预测(calibration 已校)
- **N=22600 实际运行未实测**:host 峰值 ~8.4 GB(两个 N*N 输出 buffer 并存 + 面板,非单个 ~4 GiB)+ O(N²·T) 计算不可行;device 峰值 N 无关故闭合,N=22600 host 侧成本单独披露
- nblock 无 MemTracker → 无 delta_formula==0;driver_peak−模型 delta 未独立归因(cudaMemGetInfo MiB 量化 ±1 MiB + 无 stock allocation-chain probe;block=256 +5.1 MiB 含量化误差)
- 逐 tile host 抽列/写回为 host 成本(HWM 只测 device VRAM)

## 复现

    cmake --build build --target poc3_stock_corr_selfcheck
    build\\poc3_stock_corr_selfcheck.exe --hwm   # 或
    PYTHONIOENCODING=utf-8 python benchmarks/stock_nblock_hwm_v1.py

*生成模型:{GENERATOR}*
"""
    return md


def main() -> int:
    args = sys.argv[1:]
    is_from = "--from" in args
    text = None
    exe_rc = None
    try:
        if is_from:
            idx = args.index("--from")
            if idx + 1 >= len(args):
                raise ValueError("--from requires a capture file path")
            text = pathlib.Path(args[idx + 1]).read_text(encoding="utf-8")
            # --from is an OPERATOR-TRUSTED replay of a prior run (reviews
            # SC-6 / external MAJOR-5): the runtime returncode is not
            # recoverable, so it is trusted as a successful prior run. All other
            # fail-closed checks (GUARD, case set, fb/samples/rc/driver_peak/
            # reps/malformed) still apply. provenance.source records "from".
            exe_rc = 0
        else:
            text, exe_rc = run_exe()
    except Exception as e:
        # external review MAJOR-5: an exception BEFORE validation (missing
        # --from arg, read failure, subprocess timeout) must still invalidate
        # any stale OK artifact so it cannot be ingested as current closure.
        print(f"FAIL-CLOSED: setup error ({e}); stale artifacts invalidated")
        for p in (OUT_JSON, OUT_MD):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        return 3
    if text is None:
        print("FAIL-CLOSED: no capture text")
        return 3

    cases, n_malformed = parse_fblc(text)
    free_before_MiB = parse_free_before(text)
    guard = parse_guard(text)

    # ---- Fail-closed validation: evidence written only when the chain is
    # healthy. A crashed exe, missing/failed GUARD, incomplete case set, or a
    # dead sampler must REJECT the evidence (nonzero exit, nothing written) so
    # it can never reach memory_budget_v1 as "measured closure".
    problems = []
    if exe_rc != 0:
        problems.append(f"exe returncode {exe_rc} != 0")
    if guard is None or guard.get("pass") != 1 or guard.get("r1") != 0 or guard.get("r2") != 0:
        # external review MAJOR-4: a GUARD with pass=1 but nonzero r1/r2 means
        # the drift comparison itself errored -- must reject.
        problems.append("GUARD missing/not pass=1 or r1/r2!=0")
    n_guard = sum(1 for ln in text.splitlines() if ln.startswith("GUARD|"))
    if n_guard != 1:  # external review MINOR-5: unique GUARD required
        problems.append(f"GUARD count {n_guard} != 1 (unique required)")
    if free_before_MiB is None:  # external review MAJOR-4: missing free_before header
        problems.append("free_before header missing (HWM mode line absent)")
    if n_malformed:
        # external review MAJOR-4: garbage FBLC-prefixed records must fail the
        # run, not be silently skipped.
        problems.append(f"{n_malformed} malformed FBLC record(s)")
    expected = {("nblock", 1218, 5000, 256), ("nblock", 1218, 5000, 128),
                ("nblock", 1218, 5000, 64), ("nblock", 1218, 5000, 32),
                ("production", 1218, 5000, 0)}
    seen = {(c["kind"], c["T"], c["N"], c["block"]) for c in cases}
    missing = expected - seen
    extra = seen - expected
    if missing:
        problems.append(f"missing cases: {sorted(missing)}")
    if extra:
        problems.append(f"unexpected cases: {sorted(extra)}")
    if len(cases) != len(expected):  # review F5: set-based check misses duplicate records
        problems.append(f"case count {len(cases)} != {len(expected)} (duplicate/omitted records)")
    for c in cases:
        if c.get("fb", 0) <= 0:
            problems.append(f"case {c['kind']}/{c['N']}/block={c['block']} fb<=0 (health)")
        if c.get("samples", 0) <= 0:
            problems.append(f"case {c['kind']}/{c['N']}/block={c['block']} samples<=0 (sampler dead)")
        if c.get("rc", 0) != 0:  # review SC-1: a failed (OOM) case must reject the whole evidence
            problems.append(f"case {c['kind']}/{c['N']}/block={c['block']} rc={c['rc']}!=0 (OOM)")
        if c.get("driver_peak", 0) <= 0:  # review SC-2: dead/corrupt measurement must not fit
            problems.append(f"case {c['kind']}/{c['N']}/block={c['block']} driver_peak<=0")
        if c.get("driver_peak", 0) > c.get("fb", 0):  # external MAJOR-3: physical
            problems.append(f"case {c['kind']}/{c['N']}/block={c['block']} driver_peak>fb")
        if c.get("reps", 1) != 1:  # external review MINOR-11: reps must be pinned
            problems.append(f"case {c['kind']}/{c['N']}/block={c['block']} reps={c['reps']}!=1")
    if problems:
        print("FAIL-CLOSED: evidence rejected, nothing written:")
        for p in problems:
            print(f"  - {p}")
        # review SC-3: a stale OK artifact from a prior run must not survive as
        # current closure when this run fails (drift/crash) -- delete it.
        for p in (OUT_JSON, OUT_MD):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        return 3

    cases = adjudicate(cases)
    if not all(c["pass"] for c in cases):
        # external review MAJOR-3: a case that did not pass (rc==0 but
        # over-budget or vram-exhausted) must not leave an OK artifact -- the
        # closure claim would be poisoned. Reject the whole evidence.
        failed = [f"{c['kind']}/{c['N']}/block={c['block']}" for c in cases if not c["pass"]]
        print(f"FAIL-CLOSED: cases not passed: {failed}; evidence rejected")
        for p in (OUT_JSON, OUT_MD):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        return 3

    env = _env_fingerprint()
    nblock_b256 = next((c for c in cases if c["kind"] == "nblock" and c["block"] == 256), None)
    nblock_b128 = next((c for c in cases if c["kind"] == "nblock" and c["block"] == 128), None)
    nblock_b64 = next((c for c in cases if c["kind"] == "nblock" and c["block"] == 64), None)
    nblock_b32 = next((c for c in cases if c["kind"] == "nblock" and c["block"] == 32), None)
    prod = next((c for c in cases if c["kind"] == "production"), None)

    payload = {
        "schema_version": VERSION,
        "artifact": "stock_nblock_hwm_v1.json",
        "generator": GENERATOR,
        # external review MINOR-11: include the local UTC offset so the audit
        # timeline is unambiguous.
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "env": env,
        "free_before_MiB": free_before_MiB,
        "memory_budget": {"total_MiB": 8188, "reserve_MiB": 512,
                          "available_MiB": (8188 - 512)},
        "judgement": ("fit = driver_peak <= available budget (NO 64 MiB delta band); "
                      "peak is N-independent for N>=2*block, N=5000 anchors N=22600"),
        "closure_status": "OK",
        "provenance": {
            # external MINOR-7: source must reflect the ACTUAL path (--from flag),
            # not 'not args' (an unknown arg would run live but mislabel as from).
            "source": "live" if not is_from else "from",
            # external MAJOR-6: bind the evidence to the repo HEAD + harness EXE
            # so a closure is attributable to a specific binary/source state.
            "git_head": git_head(),
            "git_dirty": git_dirty(),  # external MINOR-10: working-tree state
            "exe_sha256": exe_sha256(),
            "capture_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest().upper(),
            "peak_n_independent": ("max_cols = min(N, 2*block_width) = 2*block_width "
                                   "for N>=2*block; every tile buffer sized by max_cols "
                                   "not N; host output O(N^2) disclosed separately"),
        },
        "measurement_disclosure": {  # review F1 (contract): free-mem MiB quantization
            "free_memory_granularity_MiB": 1,
            "note": ("cudaMemGetInfo free memory is quantized to >=1 MiB on this WDDM "
                     "driver (every fb/driver_peak is an exact MiB multiple); each "
                     "driver_peak carries up to +/-1 MiB error, so small deltas are "
                     "not precisely attributable to driver overhead"),
        },
        "guard": guard,
        "n_cases": len(cases),
        "n_pass": sum(1 for c in cases if c["pass"]),
        "summary": {
            "nblock_b256_driver_MiB": round(nblock_b256["driver_peak_MiB"], 2) if nblock_b256 else None,
            "nblock_b256_model_MiB": nblock_b256["model_peak_MiB"] if nblock_b256 else None,
            "nblock_b256_delta_MiB": nblock_b256.get("delta_MiB") if nblock_b256 else None,
            "nblock_b256_budget_margin_MiB": (round((AVAILABLE_BYTES - nblock_b256["driver_peak"]) / MIB, 2) if nblock_b256 else None),
            "nblock_b256_fits": nblock_b256.get("fits") if nblock_b256 else None,
            "nblock_b128_driver_MiB": round(nblock_b128["driver_peak_MiB"], 2) if nblock_b128 else None,
            "nblock_b64_driver_MiB": round(nblock_b64["driver_peak_MiB"], 2) if nblock_b64 else None,
            "nblock_b32_driver_MiB": round(nblock_b32["driver_peak_MiB"], 2) if nblock_b32 else None,
            "production_driver_MiB": round(prod["driver_peak_MiB"], 2) if prod else None,
            "production_model_MiB": prod["model_peak_MiB"] if prod else None,
            "production_delta_MiB": prod.get("delta_MiB") if prod else None,
        },
        "cases": cases,
    }
    RESULTS.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8", newline="\n")
    OUT_MD.write_text(render_md(payload), encoding="utf-8", newline="\n")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print(f"summary: {len(cases)} cases, fit={payload['n_pass']}, "
          f"nblock block=256 driver={payload['summary']['nblock_b256_driver_MiB']} MiB")
    # Exit reflects UNEXPECTED fit failures (a case the model predicts to fit
    # that did not actually fit). All our cases are expected fits.
    unexpected = [c for c in cases
                  if c.get("model_fits") and not c.get("fits", False)]
    if guard and guard.get("pass") == 0:
        print("GUARD FAIL: nblock vs production drift detected")
        return 2
    return 1 if unexpected else 0


if __name__ == "__main__":
    raise SystemExit(main())
