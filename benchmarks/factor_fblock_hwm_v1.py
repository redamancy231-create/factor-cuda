# -*- coding: utf-8 -*-
"""factor-cuda -- F=128 fblock device HWM measurement report v1.

Runs `poc3_factor_corr_selfcheck.exe --hwm-f128` (or parses a captured run via
`--from`), parses the FBLC| records, and adjudicates each case against the
memory_budget_v1 model (`fblock_factor_peak`).

Why this exists (closing the M3 "model prediction, not measured" gap for the
factor fblock path of memory_budget_v1):
  - The F=128 (N=5000, T=1218) scenario's current implementation needs
    12645 MiB > 8188 MiB device memory, so only the F-blocking path can run.
  - factor_corr_gpu_fblock takes no MemTracker, so the 5-op calibration's
    three-way (formula / tracker HWM / driver sample) discipline reduces to the
    driver-sample leg only. Acceptance (review F1/h1/3: there is NO 64 MiB delta
    band -- the 5-op calibration's band applies to small allocations and does not
    hold for multi-GiB F-blocking allocations):
      fit       : driver_peak <= available budget (allocation-chain accuracy is
                  probe-verified separately; the runtime driver_peak - chain
                  delta grows at low free margin -- PHENOMENON verified by
                  2026-08-07 isolation experiment, WDDM mechanism NOT isolated)
      exhausted : margin (fb_case - driver_peak == min_free) ~ 0 with rc==0
                  (WDDM shared-memory fallback; NOT a usable fit)
      OOM       : rc != 0 (genuine cudaMalloc failure; not seen on this WDDM
                  device -- shared-memory fallback preempts it)
  - Only the fblock (项③) column is closed to measured here; the current-peak
    and streaming (项②) columns remain model predictions.

FBLC record format (printed by the exe):
  FBLC|kind=fblock|T=...|N=...|F=...|block=...|reps=...|fb=<B>|driver_peak=<B>|rc=<int>
  GUARD|pass=0/1|r1=<int>|r2=<int>   (bitwise fblock-vs-production drift guard)

Usage:
    PYTHONIOENCODING=utf-8 python benchmarks/factor_fblock_hwm_v1.py [--from <captured.txt>]
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
EXE = ROOT / "build" / "poc3_factor_corr_selfcheck.exe"
RESULTS = pathlib.Path(__file__).resolve().parent / "results"
OUT_JSON = RESULTS / "factor_fblock_hwm_v1.json"
OUT_MD = RESULTS / "factor_fblock_hwm_v1.md"
CUDA_BIN = os.path.join(os.environ.get("CUDA_PATH", ""), "bin", "x64")
MIB = 1048576.0
T_CANONICAL = 1218

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from memory_budget_v1 import (  # noqa: E402
    fblock_factor_peak,
    factor_corr_peak,
    AVAILABLE_BYTES,
)

VERSION = "1.0.0"
GENERATOR = ("benchmarks/factor_fblock_hwm_v1.py (DeepSeek-V4-Flash via "
             "Claude Code CLI, 2026-08-07)")


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
    """Current git HEAD of the repo (synced to the stock-side external MAJOR-6:
    provenance must bind the evidence to the source). Best-effort; 'n/a'."""
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
    should record the source state; a dirty tree means the evidence may not
    correspond to a committed source state)."""
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                           capture_output=True, text=True, timeout=30)
        return bool(r.stdout.strip()) if r.returncode == 0 else False
    except Exception:
        return False


def run_exe() -> tuple[str, int]:
    """Run the harness; return (stdout+stderr, returncode).

    Review F1 (external): the returncode must reach the fail-closed validation --
    a crashed/aborted exe must NOT produce ingested evidence.
    """
    env = dict(os.environ)
    env["PATH"] = CUDA_BIN + ";" + env.get("PATH", "")
    print(f"running {EXE} --hwm-f128 ...")
    t0 = time.time()
    # The F=128 panel construction + 8256-pair computation takes several minutes.
    r = subprocess.run([str(EXE), "--hwm-f128"], capture_output=True, text=True,
                       env=env, timeout=1800)
    dt = time.time() - t0
    print(f"exit={r.returncode} in {dt:.0f}s")
    return r.stdout + "\n" + r.stderr, r.returncode


def parse_free_before(text: str) -> float | None:
    """The exe prints its own free_before snapshot in the HWM mode header."""
    for line in text.splitlines():
        if line.startswith("HWM mode: GPU free_before"):
            try:
                return float(line.split("free_before")[1].split("MiB")[0].strip())
            except (ValueError, IndexError):
                return None
    return None


_FBLC_FIELDS = {"kind", "T", "N", "F", "block", "reps", "samples", "fb",
                "driver_peak", "rc"}


def parse_fblc(text: str) -> tuple[list[dict], int]:
    """Parse FBLC records; return (cases, malformed_count). A malformed line OR
    an UNKNOWN extra field (e.g. a forged |fits=1 that could bypass the
    expected-fit gate -- external review MAJOR-2) is counted and fails the run.
    Synced to the stock-side external MAJOR-4."""
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
            # external review FACTOR-1: kind is the ONLY field the rest of main
            # dereferences that was never validated here -- a missing `kind=`
            # would crash main() with KeyError (exit 1) instead of the fail-closed
            # return-3 + stale-delete path. Validate it (count as malformed).
            fields["kind"] = str(fields["kind"])
            fields["T"] = int(fields["T"])
            fields["N"] = int(fields["N"])
            fields["F"] = int(fields["F"])
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
    """Parse the fblock-vs-production drift guard record (GUARD|)."""
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
    """Model prediction (bytes) for a case; None if the model has no entry."""
    if c["kind"] == "production":
        peak, _ = factor_corr_peak(T_CANONICAL * c["N"], c["F"], with_mask=True)
        return peak
    return fblock_factor_peak(T_CANONICAL * c["N"], c["F"], c["block"])["peak_B"]


def adjudicate(cases: list[dict]) -> list[dict]:
    """Two-layer adjudication (review F1/h1/3: NO 64 MiB delta band).

    The model's ALLOCATION CHAIN is probe-verified separately (scratch/alloc_probe
    measured 6328 MiB vs model 6325.11 MiB, +3 MiB; not re-derived here). This
    function decides each case's RUNTIME driver_peak against the available budget:
      - fit cases: driver_peak <= available (the model's fit claim). The delta
        vs the allocation-chain model grows at low free margin (PHENOMENON verified
        by 2026-08-07 isolation experiment, same-path low-margin control; WDDM
        mechanism not isolated -- review b32-delta-iso M3).
      - VRAM-exhausted cases (per-case margin = fb_case - driver_peak == min_free
        ~ 0, rc still 0): the allocation exceeded physical VRAM and WDDM fell
        back to shared GPU memory -- NOT a usable fit.
      - OOM cases (rc != 0): genuine cudaMalloc failure (not seen on this device;
        WDDM shared-memory fallback preempts it).
    """
    rows = []
    for c in cases:
        c["driver_peak_MiB"] = c["driver_peak"] / MIB
        mp = model_peak(c)
        c["model_peak_MiB"] = round(mp / MIB, 2) if mp is not None else None
        # Per-case margin (review F2): the exhausted gate must not depend on a
        # single header free_before that can drift between cases.
        c["margin_MiB"] = (None if not c.get("fb")
                           else round((c["fb"] - c["driver_peak"]) / MIB, 2))
        # vram_exhausted REQUIRES rc==0 (review F3): a nonzero rc is a genuine
        # cudaMalloc failure (OOM), never the WDDM shared-memory fallback.
        c["vram_exhausted"] = (c["rc"] == 0 and c["margin_MiB"] is not None
                               and c["margin_MiB"] <= 8.0)
        # model_fits drives the exit-code "unexpected" signal (review F4): a case
        # the model predicts to fit but that cannot run (OOM) must fail CI.
        c["model_fits"] = (mp is not None) and (mp <= AVAILABLE_BYTES)
        note = ""
        if mp is not None:
            c["delta_MiB"] = round((c["driver_peak"] - mp) / MIB, 2)
        if c["rc"] != 0:
            # OOM branch first (review F3): a real allocation failure is unambiguous.
            c["pass"] = False
            note = f"OOM rc={c['rc']}(分配链模型 {c['model_peak_MiB']} MiB 设备不可达)"
        elif c["vram_exhausted"]:
            c["pass"] = False
            note = ("物理显存耗尽(margin≈0, rc==0);与 WDDM 超分配/共享内存回退一致,"
                    "具体机制未隔离验证(review F4);模型峰值超设备物理显存")
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
        if c["kind"] == "production":
            scale = f"{c['F']} (production)"
        else:
            scale = f"{c['F']} block={c['block']}"
        mark = "✅" if c["pass"] else "❌"
        margin = "—" if c.get("margin_MiB") is None else c["margin_MiB"]
        # VRAM-exhausted cases saturate driver_peak at free_before, so their
        # delta vs the model is meaningless -- show a dash.
        delta = "—" if c.get("vram_exhausted") else c.get("delta_MiB", "—")
        rows.append(
            f"| {c['kind']} | {c['T']}×{c['N']} | {scale} | {c['driver_peak_MiB']:.1f} "
            f"| {c['model_peak_MiB'] if c['model_peak_MiB'] is not None else '—'} "
            f"| {delta} | {margin} | {c['rc']} | {mark} | {c['note']} |"
        )
    table = "\n".join(rows)
    md = f"""# factor-cuda F=128 fblock 设备 HWM 实测(v1)

> 生成:{time.strftime('%Y-%m-%d')} · {GENERATOR}
> 判据(两层,替代 5-op 校准的 delta_formula==0——fblock 无 MemTracker):
> **分配链验证**(probe `scratch/alloc_probe.cu`):实测 drop 6328 MiB vs 模型 6325.11 MiB,delta +3 MiB → 模型分配链精确
> **运行期 fits**:driver_peak ≤ {payload['memory_budget']['available_MiB']} MiB 可用预算;driver_peak−分配链为运行期 delta(低 free 余量时现象已验证增大,WDDM 机制未隔离验证)
> **显存耗尽**:margin≈0 且 rc=0(与 WDDM 超分配/共享内存回退一致,机制未隔离验证;非可用 fit)
> **测量披露**(synced to stock external F1/contract, F2-hardened):{payload['measurement_disclosure']['note']}

## 结论

**{payload['n_cases']} 例 FBLC 记录,{payload['n_pass']} 例判定为可用 fit。**
- F=128 (N=5000) block=32 实测 driver_peak {payload['summary']['f128_block32_driver_MiB']} MiB(**fits**:预算口径余量 {payload['summary']['f128_block32_budget_margin_MiB']} MiB,实际运行余量 {payload['summary']['f128_block32_margin_MiB']} MiB)→ **fblock(项③) 峰值闭合为实测**;分配链模型 6325.11 MiB(probe 6328,delta 3 MiB);总 delta +{payload['summary']['f128_block32_delta_MiB']} MiB(driver−model)。**运行期 delta 现象已验证**(2026-08-07 隔离验证:同路径低余量对照 block=8 +pad 压余量 delta 从 +4.4 跳 +223;纯分配恒 ~+3-5),**WDDM 具体机制未隔离验证**(reviews/b32_delta_iso_verification_2026-08-07.md)
- block=64 / production F=128 实测**显存耗尽(margin≈0),非 OOM**——与 WDDM 超分配/共享内存回退一致(具体机制未隔离验证,review F4);模型 12645 MiB 超物理显存被实证;两者**不可用(非 fit)**
- F=12 block=6 交叉验证锚点:driver_peak ≈ 模型 ≈ 校准 current 1191/2381(fblock 模型在 block≤F/2 区间钉死)

## 环境

- GPU:{payload['env'].get('gpu','—')} total {payload['env'].get('total_MiB','—')} MiB
- 运行期 free_before ≈ {payload['free_before_MiB']} MiB(total 8188,实际占用 ~{8188 - payload['free_before_MiB']:.0f} MiB);模型预算口径 {payload['memory_budget']['available_MiB']} MiB(8188−512)→ 实际可用比口径少 {payload['memory_budget']['available_MiB'] - payload['free_before_MiB']:.0f} MiB,block=32 实际余量 {payload['summary']['f128_block32_margin_MiB']} MiB
- host staging(**估算,非实测**——仅 device VRAM 被 cudaMemGetInfo 采样,review h4):面板 F3 全量 {payload['host_panel_MiB']:.1f} MiB + 逐 tile 临时最大 {payload['host_tile_max_MiB']:.1f} MiB(M6 注记)

## 明细

| kind | T×N | 规模 | driver(MiB) | 模型(MiB) | Δ(MiB) | margin(MiB) | rc | 判定 | 注 |
|---|---|---|---|---|---|---|---|---|---|
{table}

## 闭合范围(诚实边界)

- 实测闭合**仅 fblock(项③)** 路径;`current_peak`(production)仍为模型预测(12645 MiB,已实测显存耗尽确认超预算)、`streaming_peak`(项②)无实现仍为预测
- fblock 无 MemTracker → 无 delta_formula==0;分配链用 probe 单独验证(delta 3 MiB),运行期 driver_peak 用预算判 fit
- block=64/production 报"显存耗尽 margin≈0(与 WDDM 超分配/共享内存回退一致,机制未隔离验证)"而非"12.6 GiB 实测";模型峰值 12645 MiB 为设备不可达的预算断言
- block=1 场景省略(8256-tile host 循环 ~30min,无决策价值;block=8/16/32 递减曲线已覆盖)

## 复现

    cmake --build build --target poc3_factor_corr_selfcheck
    build\\poc3_factor_corr_selfcheck.exe --hwm-f128   # 或
    PYTHONIOENCODING=utf-8 python benchmarks/factor_fblock_hwm_v1.py

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
            # --from is an OPERATOR-TRUSTED replay of a prior run (synced to the
            # stock-side external MAJOR-5): the runtime returncode is not
            # recoverable, so it is trusted as a successful prior run. All other
            # fail-closed checks (GUARD, case set, fb/samples/rc/driver_peak/
            # reps/malformed) still apply. provenance.source records "from".
            exe_rc = 0
        else:
            text, exe_rc = run_exe()
    except Exception as e:
        # synced to the stock-side external MAJOR-5: an exception BEFORE
        # validation (missing --from arg, read failure, subprocess timeout) must
        # still invalidate any stale OK artifact.
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

    # ---- Fail-closed validation (review F1, external + synced to the stock-side
    # external BLOCKER/MAJOR depth): the evidence is written only when the whole
    # chain is healthy. A crashed exe, a missing/failed GUARD (r1/r2 must be 0),
    # malformed FBLC records, an incomplete/duplicate case set, a dead sampler,
    # a non-zero rc or a pinned-reps violation must REJECT the evidence (nonzero
    # exit, nothing written, stale artifacts deleted) so it can never reach
    # memory_budget_v1 as "measured closure".
    problems = []
    if exe_rc != 0:
        problems.append(f"exe returncode {exe_rc} != 0")
    if guard is None or guard.get("pass") != 1 or guard.get("r1") != 0 or guard.get("r2") != 0:
        problems.append("GUARD missing/not pass=1 or r1/r2!=0")
    n_guard = sum(1 for ln in text.splitlines() if ln.startswith("GUARD|"))
    if n_guard != 1:  # external review MINOR-5: a unique GUARD is required (a
        # stale leading pass=1 record must not mask a later failed one)
        problems.append(f"GUARD count {n_guard} != 1 (unique required)")
    if free_before_MiB is None:  # external review MAJOR-4: a missing free_before
        # header would leave a closure_status=OK JSON whose render crashes on None
        problems.append("free_before header missing (HWM mode line absent)")
    if n_malformed:
        problems.append(f"{n_malformed} malformed FBLC record(s)")
    expected = {("fblock", 1218, 5000, 12, 6), ("fblock", 1218, 10000, 12, 6),
                ("fblock", 1218, 5000, 128, 8), ("fblock", 1218, 5000, 128, 16),
                ("fblock", 1218, 5000, 128, 32), ("fblock", 1218, 5000, 128, 64),
                ("production", 1218, 5000, 128, 0)}
    seen = {(c["kind"], c["T"], c["N"], c["F"], c["block"]) for c in cases}
    missing = expected - seen
    extra = seen - expected
    if missing:
        problems.append(f"missing cases: {sorted(missing)}")
    if extra:
        problems.append(f"unexpected cases: {sorted(extra)}")
    if len(cases) != len(expected):
        problems.append(f"case count {len(cases)} != {len(expected)} (duplicate/omitted records)")
    for c in cases:
        if c.get("fb", 0) <= 0:
            problems.append(f"case {c['kind']}/{c['N']}/{c['F']} fb<=0 (health)")
        if c.get("samples", 0) <= 0:
            problems.append(f"case {c['kind']}/{c['N']}/{c['F']} samples<=0 (sampler dead)")
        if c.get("rc", 0) != 0:
            problems.append(f"case {c['kind']}/{c['N']}/{c['F']} rc={c['rc']}!=0 (OOM)")
        if c.get("driver_peak", 0) <= 0:
            problems.append(f"case {c['kind']}/{c['N']}/{c['F']} driver_peak<=0")
        if c.get("driver_peak", 0) > c.get("fb", 0):  # external MAJOR-3: physical
            problems.append(f"case {c['kind']}/{c['N']}/{c['F']} driver_peak>fb")
        # external MINOR-11: reps must be pinned to the harness's per-case value
        # (F=12 cross-check anchors run reps=2; F=128 blocks / production reps=1).
        expected_reps = 2 if (c["kind"] == "fblock" and c["F"] == 12) else 1
        if c.get("reps", expected_reps) != expected_reps:
            problems.append(f"case {c['kind']}/{c['N']}/{c['F']} reps={c['reps']}!={expected_reps}")
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

    cases = adjudicate(cases)
    # synced to the stock-side external MAJOR-3, hardened by this port review's
    # external FACTOR-3: reject before the payload write any rc==0 case that is
    # NEITHER a fit NOR a vram-exhausted non-fit (a true anomaly: over-budget
    # with margin>8 the model does not predict to fit), PLUS any expected-fit
    # case (model_fits) that did not actually fit. An OK artifact must never
    # contain such a case.
    bad = [c for c in cases
           if (c.get("rc", 0) == 0 and not c.get("fits", False)
               and not c.get("vram_exhausted", False))
           or (c.get("model_fits") and not c.get("fits", False))]
    if bad:
        failed = [f"{c['kind']}/{c['N']}/F={c['F']}/block={c['block']}" for c in bad]
        print(f"FAIL-CLOSED: anomalous/expected-fit-failed cases: {failed}; evidence rejected")
        for p in (OUT_JSON, OUT_MD):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        return 3

    env = _env_fingerprint()
    f128_b32 = next((c for c in cases if c["kind"] == "fblock" and c["F"] == 128
                     and c["block"] == 32), None)
    f128_b64 = next((c for c in cases if c["kind"] == "fblock" and c["F"] == 128
                     and c["block"] == 64), None)
    # external review F2/SEM-2 + external MAJOR/MINOR: the disclosure's fb and
    # driver_peak counts AND the block=32 is/not-MiB-multiple assertion are ALL
    # DERIVED from the current cases (never hardcoded -- a stale literal from a
    # prior run was previously baked in).
    _mi_multiples = sum(1 for c in cases if c["driver_peak"] % MIB == 0)
    _fb_multiples = sum(1 for c in cases if c["fb"] % MIB == 0)
    _b32_not_mi = f128_b32 is not None and f128_b32["driver_peak"] % MIB != 0
    _disclosure_note = (
        "free-memory sampling granularity is driver/time-dependent on this WDDM: "
        f"{_fb_multiples}/{len(cases)} fb snapshots and "
        f"{_mi_multiples}/{len(cases)} driver_peak values are exact MiB multiples, "
        "block=32 ("
        + (f"{f128_b32['driver_peak_MiB']:.3f} MiB" if f128_b32 else "n/a")
        + (") is NOT a MiB multiple" if _b32_not_mi else ") is a MiB multiple")
        + " -- so small deltas carry sampling-granularity error and are not "
        "precisely attributable to driver overhead"
    )

    payload = {
        "schema_version": VERSION,
        "artifact": "factor_fblock_hwm_v1.json",
        "generator": GENERATOR,
        # external review MINOR-11: include the local UTC offset so the audit
        # timeline is unambiguous (plain %Y-%m-%dT%H:%M:%S has no timezone).
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "env": env,
        "free_before_MiB": free_before_MiB,
        "memory_budget": {"total_MiB": 8188, "reserve_MiB": 512,
                          "available_MiB": (8188 - 512)},
        "judgement": ("fit = driver_peak <= available budget (NO 64 MiB delta band; "
                      "allocation-chain accuracy probe-verified separately)"),
        "closure_status": "OK",  # set ONLY after the fail-closed validation passed
        "measurement_disclosure": {  # synced to stock external F1/contract, F2-hardened
            "note": _disclosure_note,
        },
        "provenance": {
            "source": "live" if not is_from else "from",  # synced to stock MINOR-7
            "git_head": git_head(),  # synced to stock external MAJOR-6
            "git_dirty": git_dirty(),  # external MINOR-10: working-tree state
            "exe_sha256": exe_sha256(),
            "capture_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest().upper(),
            "probe_note": ("allocation-chain probe (scratch/alloc_probe.cu, gitignored) "
                           "原始字节未保存;6328.0/6325.11/3.0 为历史观察硬编码,"
                           "fresh clone 不能精确复验(review F5)"),
        },
        "guard": guard,
        "n_cases": len(cases),
        "n_pass": sum(1 for c in cases if c["pass"]),
        "host_panel_MiB": round(1218 * 5000 * 128 * 8 / MIB, 2),
        "host_tile_max_MiB": round(2 * 32 * 1218 * 5000 * 8 / MIB, 2),
        "allocation_chain_validation": {
            "probe": "scratch/alloc_probe.cu (temp, gitignored)",
            "measured_drop_MiB": 6328.0,
            "model_MiB": 6325.11,
            "delta_MiB": 3.0,
            "conclusion": ("模型分配链 probe 精确(+3 MiB);block=32 总 delta "
                           f"(driver−model {f128_b32.get('delta_MiB') if f128_b32 else '—'} MiB,"
                           "含分配链 +3 MiB)。运行期 delta 现象已验证(2026-08-07 "
                           "隔离验证,同路径低余量对照),WDDM 具体机制未隔离验证"
                           "(reviews/b32_delta_iso_verification_2026-08-07.md)"),
        },
        "summary": {
            "f128_block32_driver_MiB": round(f128_b32["driver_peak_MiB"], 2) if f128_b32 else None,
            "f128_block32_model_MiB": f128_b32["model_peak_MiB"] if f128_b32 else None,
            "f128_block32_delta_MiB": f128_b32.get("delta_MiB") if f128_b32 else None,
            "f128_block32_margin_MiB": f128_b32.get("margin_MiB") if f128_b32 else None,
            "f128_block32_budget_margin_MiB": (round((AVAILABLE_BYTES - f128_b32["driver_peak"]) / MIB, 2) if f128_b32 else None),
            "f128_block32_fits": f128_b32.get("fits") if f128_b32 else None,
            "f128_block64_driver_MiB": round(f128_b64["driver_peak_MiB"], 2) if f128_b64 else None,
            "f128_block64_margin_MiB": f128_b64.get("margin_MiB") if f128_b64 else None,
            "f128_block64_vram_exhausted": f128_b64.get("vram_exhausted") if f128_b64 else None,
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
          f"f128 block=32 driver={payload['summary']['f128_block32_driver_MiB']} MiB")
    # unexpected (model_fits but not fits) is gated BEFORE the payload write
    # (synced to the stock-side external MAJOR-3), so a clean exit here means
    # every expected-fit case fit. block=64 / production F=128 are EXPECTED
    # over-budget (model_fits=False) and are valid non-fit evidence.
    if guard and guard.get("pass") == 0:
        print("GUARD FAIL: fblock vs production drift detected")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
