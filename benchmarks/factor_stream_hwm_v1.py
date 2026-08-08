# -*- coding: utf-8 -*-
"""factor-cuda -- factor_corr streaming (item 2) device HWM measurement report v1.

Runs `poc3_factor_corr_selfcheck.exe --hwm-stream` (or parses a captured run via
`--from`), parses the FBLC| records, and adjudicates each case against the
memory_budget_v1 model (`streaming_factor_peak`).

Why this exists (closing the M3 "model prediction, not measured" gap for the
streaming (item 2) column of memory_budget_v1):
  - The streaming path removes the pre-transpose d_F overlap: the full (T*N,F)
    d_F never exists on device; d_F_chunk (max_transpose_rows rows) is uploaded
    + range-transposed into the fully-resident d_Xt/d_valid in slices. Decision A
    keeps d_Xt/d_valid in full (Kahan re-run reads them). Model peak =
    d_Xt+d_valid+d_pp1+d_pp2+d_F_chunk+pair workspace (~6.9 GiB for F=128) vs the
    current ~12.6 GiB (d_F+d_Xt+d_valid overlap). This closes the streaming
    column to measured.
  - factor_corr_gpu_stream takes no MemTracker, so the calibration three-way
    discipline reduces to the driver-sample leg only. Acceptance:
      fit       : driver_peak <= available budget AND margin (fb - driver_peak)
                  > 8 MiB (a WDDM vram-exhausted margin~0 is NOT a fit).
      exhausted : margin ~ 0 with rc==0 (WDDM shared-memory fallback) -- expected
                  ONLY for the production F=128 control, never for a stream case.
      OOM       : rc != 0 (genuine cudaMalloc failure).
  - Only the streaming column is closed to measured here; the current-peak
    (production) column remains a model prediction (calibration-verified HWM).
    The production F=128 control confirms the model's over-budget claim is real.

FBLC record format (printed by the exe):
  FBLC|kind=stream|T=...|N=...|F=...|block=<max_transpose_rows>|reps=...|fb=<B>|driver_peak=<B>|samples=<N>|rc=<int>
  FBLC|kind=production|T=...|N=...|F=...|block=0|reps=...|fb=<B>|driver_peak=<B>|samples=<N>|rc=<int>
  GUARD|pass=0/1|r1=<int>|r2=<int>   (bitwise stream-vs-production drift guard)

Usage:
    PYTHONIOENCODING=utf-8 python benchmarks/factor_stream_hwm_v1.py [--from <captured.txt>]
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
OUT_JSON = RESULTS / "factor_stream_hwm_v1.json"
OUT_MD = RESULTS / "factor_stream_hwm_v1.md"
CUDA_BIN = os.path.join(os.environ.get("CUDA_PATH", ""), "bin", "x64")
MIB = 1048576.0
T_CANONICAL = 1218
STREAM_TT = 4096  # max_transpose_rows (report config; matches memory_budget STREAM_TT)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from memory_budget_v1 import (  # noqa: E402
    streaming_factor_peak,
    factor_corr_peak,
    AVAILABLE_BYTES,
)

VERSION = "1.0.0"
GENERATOR = ("benchmarks/factor_stream_hwm_v1.py (DeepSeek-V4-Flash via "
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
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else "n/a"
    except Exception:
        return "n/a"


def exe_sha256() -> str:
    try:
        return hashlib.sha256(EXE.read_bytes()).hexdigest().upper() if EXE.exists() else "n/a"
    except Exception:
        return "n/a"


def git_dirty() -> bool:
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
    print(f"running {EXE} --hwm-stream ...")
    t0 = time.time()
    r = subprocess.run([str(EXE), "--hwm-stream"], capture_output=True, text=True,
                       env=env, timeout=3600)
    dt = time.time() - t0
    print(f"exit={r.returncode} in {dt:.0f}s")
    return r.stdout + "\n" + r.stderr, r.returncode


def parse_hwm_header(text: str) -> tuple[float | None, float | None]:
    """Parse 'HWM mode: GPU free_before X MiB / total Y MiB' -> (free_before, total).
    F-06 (external MAJOR): return BOTH so the chain can validate per-case fb
    against the GPU physical total (a forged fb > total must not pass)."""
    for line in text.splitlines():
        if line.startswith("HWM mode: GPU free_before"):
            try:
                fb = float(line.split("free_before")[1].split("MiB")[0].strip())
                tot = float(line.split("total")[1].split("MiB")[0].strip())
                return fb, tot
            except (ValueError, IndexError):
                return None, None
    return None, None


_FBLC_FIELDS = {"kind", "T", "N", "F", "block", "reps", "samples", "fb",
                "driver_peak", "rc"}


def parse_fblc(text: str) -> tuple[list[dict], int]:
    """Parse FBLC records; return (cases, malformed_count). A malformed line OR
    an UNKNOWN extra field (forged e.g. |fits=1) is counted and fails the run.
    F-09 (external MINOR): duplicate keys, bare tokens and tokens without '='
    are malformed (a silent last-value-wins duplicate-key parse is fail-open)."""
    cases = []
    malformed = 0
    for line in text.splitlines():
        if not line.startswith("FBLC|"):
            continue
        fields = {}
        dup = False
        for kv in line[len("FBLC|"):].split("|"):
            if "=" not in kv or not kv.split("=", 1)[0]:
                malformed += 1
                print(f"skip FBLC line with bare/empty token: {kv!r}")
                dup = True
                break
            k, v = kv.split("=", 1)
            if k in fields:
                malformed += 1
                print(f"skip FBLC line with duplicate key: {k}")
                dup = True
                break
            fields[k] = v
        if dup:
            continue
        unknown = set(fields) - _FBLC_FIELDS
        if unknown:
            malformed += 1
            print(f"skip FBLC line with unknown fields: {sorted(unknown)}")
            continue
        try:
            # kind is the ONLY field main() dereferences that was never validated
            # in the stock precedent -- a missing kind= would KeyError (exit 1)
            # instead of the fail-closed return-3 + stale-delete path.
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


_GUARD_FIELDS = {"pass", "r1", "r2"}


def parse_guard(text: str) -> dict | None:
    """F-09 (external MINOR): a GUARD with duplicate keys, bare tokens or an
    UNKNOWN field (e.g. |junk=x) is malformed and must fail the run -- a
    pass=0|pass=1 duplicate-key line must not parse as a valid unique guard."""
    for line in text.splitlines():
        if line.startswith("GUARD|"):
            fields = {}
            bad = False
            for kv in line[len("GUARD|"):].split("|"):
                if "=" not in kv or not kv.split("=", 1)[0]:
                    bad = True
                    break
                k, v = kv.split("=", 1)
                if k in fields or k not in _GUARD_FIELDS:
                    bad = True
                    break
                fields[k] = v
            if bad:
                return None
            try:
                return {"pass": int(fields["pass"]),
                        "r1": int(fields["r1"]), "r2": int(fields["r2"])}
            except (KeyError, ValueError):
                return None
    return None


def model_peak(c: dict) -> int | None:
    """Model prediction (bytes) for a case; None if the model has no entry."""
    R = T_CANONICAL * c["N"]
    if c["kind"] == "stream":
        return streaming_factor_peak(R, c["F"], c["block"])
    return factor_corr_peak(R, c["F"], with_mask=True)[0]  # production control


def adjudicate(cases: list[dict]) -> list[dict]:
    """Two-layer adjudication (fblock/nblock precedent: NO 64 MiB delta band).

    stream: fit = driver_peak <= available budget AND margin > 8 MiB (a real
    margin; WDDM vram-exhausted margin~0 is never a fit). production control:
    expected vram-exhausted (margin ~ 0, rc==0). OOM (rc != 0) is never a fit.
    """
    rows = []
    for c in cases:
        c["driver_peak_MiB"] = c["driver_peak"] / MIB
        mp = model_peak(c)
        c["model_peak_MiB"] = round(mp / MIB, 2) if mp is not None else None
        c["margin_MiB"] = (None if not c.get("fb")
                           else round((c["fb"] - c["driver_peak"]) / MIB, 2))
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
            # legitimate ONLY for the production control (model predicts non-fit)
            c["pass"] = False
            note = ("物理显存耗尽(margin≈0, rc==0);与 WDDM 超分配/共享内存回退一致,"
                    "机制未隔离验证;模型峰值超设备物理显存")
        elif c["kind"] == "production":
            # F-03 (external MAJOR): the production control MUST be exhausted
            # (vram-exhausted, handled above) or OOM -- the current path is
            # over-budget (model_fits=False). A would-be-fit production record
            # (margin>8) is forged / not a genuine exhaustion; it must not close.
            c["pass"] = False
            c["fits"] = c["driver_peak"] <= AVAILABLE_BYTES
            note = ("production 对照不应 fit(当前路径模型超预算,model_fits=False);"
                    "margin>8 → 伪造或非真实耗尽,证据拒")
        elif mp is not None:
            # F-01 (external BLOCKER): the explicit allocation-chain model is a
            # LOWER bound on any real driver peak (actual allocs >= model; the
            # driver adds overhead). A driver_peak far below model means a forged
            # or dead-sampler record -- reject. Wide tolerance (0.5x) so MiB
            # quantization / driver overhead cannot false-reject.
            lower_bound = int(mp * 0.5)
            if c["driver_peak"] < lower_bound:
                c["pass"] = False
                note = (f"driver_peak {round(c['driver_peak']/MIB, 1)} MiB 远低于"
                        f"分配链模型 {round(mp/MIB, 1)} MiB(<0.5×)——伪造或采样失效")
            else:
                c["fits"] = c["driver_peak"] <= AVAILABLE_BYTES
                c["pass"] = c["fits"]
                if c["fits"]:
                    # STREAM-HWM-1/stream-01: disclose BOTH the nominal budget margin
                    # (AVAILABLE 7676 - driver) and the physical margin
                    # (fb free_before - driver). The fb-vs-AVAILABLE gap (570 MiB on
                    # this box) is real: the physical margin is the honest headroom.
                    note = (f"fit: 预算余量 {round(AVAILABLE_BYTES/MIB - c['driver_peak_MiB'], 1)} MiB,"
                            f" 物理余量 {round((c['fb']-c['driver_peak'])/MIB, 1)} MiB")
                else:
                    note = f"超出预算 {round(c['driver_peak_MiB'] - AVAILABLE_BYTES/MIB, 1)} MiB"
        else:
            c["pass"] = False
            note = "no model entry"
        c["note"] = note
        rows.append(c)
    return rows


def render_md(payload: dict) -> str:
    rows = []
    for c in payload["cases"]:
        scale = (f"tt={c['block']} (max_transpose_rows)" if c["kind"] == "stream"
                 else "production (non-streamed)")
        mark = "✅" if c["pass"] else "❌"
        margin = "—" if c.get("margin_MiB") is None else c["margin_MiB"]
        delta = "—" if c.get("vram_exhausted") else c.get("delta_MiB", "—")
        rows.append(
            f"| {c['kind']} | {c['T']}×{c['N']}×{c['F']} | {scale} | {c['driver_peak_MiB']:.2f} "
            f"| {c['model_peak_MiB'] if c['model_peak_MiB'] is not None else '—'} "
            f"| {delta} | {margin} | {c['rc']} | {mark} | {c['note']} |"
        )
    table = "\n".join(rows)
    md = f"""# factor-cuda factor_corr streaming(项②) 设备 HWM 实测(v1)

> 生成:{time.strftime('%Y-%m-%d')} · {GENERATOR}
> 判据(两层,替代 5-op 校准的 delta_formula==0——streaming 无 MemTracker):
> **运行期 fits**:driver_peak ≤ {payload['memory_budget']['available_MiB']} MiB 可用预算 且 margin>8 MiB(WDDM margin≈0 非 fit);driver_peak−模型 delta 未独立归因(cudaMemGetInfo MiB 量化 ±1 MiB)
> **显存耗尽**:margin≈0 且 rc=0(与 WDDM 超分配/共享内存回退一致,机制未隔离验证;非可用 fit——仅 production 对照合法)

## 结论

**{payload['n_cases']} 例 FBLC 记录,{payload['n_pass']} 例判定为可用 fit。**
- **streaming F=128 (N=5000, tt=4096) 实测 driver_peak {payload['summary']['stream_f128_driver_MiB']} MiB**(模型 {payload['summary']['stream_f128_model_MiB']} MiB,delta +{payload['summary']['stream_f128_delta_MiB']} MiB)→ **fits**。**余量双口径**:预算余量 {payload['summary']['stream_f128_budget_margin_MiB']} MiB(相对 7676 名义预算)/ **物理余量 {payload['summary']['stream_f128_physical_margin_MiB']} MiB**(相对运行期 free_before {payload['free_before_MiB']} MiB)——free_before 7106 vs 预算 7676 存在 570 MiB 缺口,物理余量才是本机真实空间 → **streaming(项②) 列闭合为实测(本机 RTX 4060, fb=7106 MiB)**
- **对比口径(诚实标注)**:模型 current 12.6 GiB(未实测,calibration-校) vs 实测 streaming {payload['summary']['stream_f128_driver_MiB']} MiB——5.7 GiB 节省为**模型-实测**比较(本机未物化 12.6 GiB);**measured-vs-measured**:production 对照 {payload['summary']['prod_driver_MiB']} MiB(vram-exhausted)→ streaming {payload['summary']['stream_f128_driver_MiB']} MiB,物理差 {payload['summary']['mv_measured_gap_MiB']} MiB
- **F=12 锚点**:N=5000 {payload['summary']['f12_n5000_driver_MiB']} MiB(模型 {payload['summary']['f12_n5000_model_MiB']} MiB,delta {payload['summary']['f12_n5000_delta_MiB']} MiB)、N=10000 {payload['summary']['f12_n10000_driver_MiB']} MiB(模型 {payload['summary']['f12_n10000_model_MiB']} MiB,delta {payload['summary']['f12_n10000_delta_MiB']} MiB)——模型-实现一致性校准
- **production F=128 对照** {payload['summary']['prod_driver_MiB']} MiB(margin≈0,**vram-exhausted** 非 fit)——确认模型 current 峰值 12.6 GiB 超预算声明真实
- **测量披露**:{payload['disclosure']}

## 环境

- GPU:{payload['env'].get('gpu','—')} total {payload['env'].get('total_MiB','—')} MiB
- 运行期 free_before ≈ {payload['free_before_MiB']} MiB;模型预算口径 {payload['memory_budget']['available_MiB']} MiB(8188−512)

## 明细

| kind | T×N×F | 规模 | driver(MiB) | 模型(MiB) | Δ(MiB) | margin(MiB) | rc | 判定 | 注 |
|---|---|---|---|---|---|---|---|---|---|
{table}

## 闭合范围(诚实边界)

- 实测闭合**仅 streaming(项②) 列**(F=128 N=5000 锚定);`current_peak`(production)仍为模型预测(calibration 已校)
- **streaming 路径 device 峰值随 F 线性**(d_Xt+d_valid+d_pp 全驻留),无"峰值 N 无关"简化——N=10000 F=12 锚点实测校准,N=10000 F=128 未实测(host 面板 11.9 GiB 不可行)
- streaming 无 MemTracker → 无 delta_formula==0;driver_peak−模型 delta 未独立归因(cudaMemGetInfo MiB 量化 ±1 MiB;F=128 +123 MiB 与 fblock B32 低余量 WDDM 波动同量级)
- 逐 sub-chunk host 上传/range 转置为 host 成本(HWM 只测 device VRAM)
- d_pp 延续状态(F=128 ~161 MiB)是 streaming/chunked 路径固有成本,计入模型

## 复现

    cmake --build build --target poc3_factor_corr_selfcheck
    build\\poc3_factor_corr_selfcheck.exe --hwm-stream   # 或
    PYTHONIOENCODING=utf-8 python benchmarks/factor_stream_hwm_v1.py

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
            # --from is an OPERATOR-TRUSTED replay of a prior run: the runtime
            # returncode is not recoverable, so it is trusted as a successful
            # prior run. All other fail-closed checks still apply.
            exe_rc = 0
        else:
            text, exe_rc = run_exe()
    except Exception as e:
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
    free_before_MiB, total_MiB = parse_hwm_header(text)
    guard = parse_guard(text)

    # ---- Fail-closed validation: evidence written only when the chain is
    # healthy. A crashed exe, missing/failed GUARD, incomplete case set, or a
    # dead sampler must REJECT the evidence (nonzero exit, nothing written).
    problems = []
    if exe_rc != 0:
        problems.append(f"exe returncode {exe_rc} != 0")
    if guard is None or guard.get("pass") != 1 or guard.get("r1") != 0 or guard.get("r2") != 0:
        problems.append("GUARD missing/not pass=1 or r1/r2!=0")
    n_guard = sum(1 for ln in text.splitlines() if ln.startswith("GUARD|"))
    if n_guard != 1:
        problems.append(f"GUARD count {n_guard} != 1 (unique required)")
    if free_before_MiB is None or total_MiB is None:
        # F-06 (external MAJOR): total is REQUIRED -- without it a forged fb can
        # not be checked against the physical GPU ceiling.
        problems.append("HWM header missing/free_before or total absent")
    if n_malformed:
        problems.append(f"{n_malformed} malformed FBLC record(s)")
    expected = {("stream", 1218, 5000, 12, 4096), ("stream", 1218, 10000, 12, 4096),
                ("stream", 1218, 5000, 128, 4096), ("production", 1218, 5000, 128, 0)}
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
            problems.append(f"case {c['kind']}/{c['N']}/F={c['F']} fb<=0 (health)")
        # F-06 (external MAJOR): fb must be physically possible -- <= GPU total
        # and consistent with the header free_before baseline (per-case fb is
        # sampled just before each case, so it can drift down but not up).
        if total_MiB and c.get("fb", 0) > total_MiB * MIB:
            problems.append(f"case {c['kind']}/{c['N']}/F={c['F']} fb={c['fb']} > GPU total "
                            f"({total_MiB:.0f} MiB, 物理不可能)")
        if free_before_MiB and c.get("fb", 0) > free_before_MiB * MIB + 512 * MIB:
            problems.append(f"case {c['kind']}/{c['N']}/F={c['F']} fb > header "
                            f"free_before+512MiB (baseline 漂移)")
        if c.get("samples", 0) <= 0:
            problems.append(f"case {c['kind']}/{c['N']}/F={c['F']} samples<=0 (sampler dead)")
        if c.get("driver_peak", 0) <= 0:
            problems.append(f"case {c['kind']}/{c['N']}/F={c['F']} driver_peak<=0")
        if c.get("driver_peak", 0) > c.get("fb", 0):
            problems.append(f"case {c['kind']}/{c['N']}/F={c['F']} driver_peak>fb")
        if c["kind"] == "production":
            # STREAM-HWM-5: a genuine production OOM (rc!=0) is a LEGITIMATE
            # control (cudaMalloc failure on the over-budget current path) -- it
            # must NOT reject the evidence; adjudicate records it as an OOM
            # non-fit. Only reps is pinned for the production control.
            if c.get("reps", 0) != 1:
                problems.append(f"case production reps={c['reps']}!=1")
            continue
        # stream cases MUST run clean (rc==0); a failed stream case rejects.
        if c.get("rc", 0) != 0:
            problems.append(f"case {c['kind']}/{c['N']}/F={c['F']} rc={c['rc']}!=0 (OOM)")
        expected_reps = 2 if c["F"] == 12 else 1
        if c.get("reps", 0) != expected_reps:
            problems.append(f"case {c['kind']}/{c['N']}/F={c['F']} reps={c['reps']}!={expected_reps}")
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
    # All stream cases must pass (fit); the production control is a legitimate
    # non-fit (vram-exhausted) and must NOT poison the evidence. A stream case
    # that did not pass rejects the whole evidence.
    failed = [f"{c['kind']}/{c['N']}/F={c['F']}" for c in cases
              if c["kind"] == "stream" and not c["pass"]]
    if failed:
        print(f"FAIL-CLOSED: stream cases not passed: {failed}; evidence rejected")
        for p in (OUT_JSON, OUT_MD):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        return 3
    # F-03 (external MAJOR): the production control must be a genuine
    # vram-exhausted/OOM non-fit -- a forged fit (driver<=budget, margin>8)
    # falsifies the "production over-budget" premise and must reject.
    prod = next((c for c in cases if c["kind"] == "production"), None)
    if prod is not None and prod.get("fits"):
        print("FAIL-CLOSED: production control forged-fit (driver<=budget); evidence rejected")
        for p in (OUT_JSON, OUT_MD):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        return 3

    env = _env_fingerprint()
    sf128 = next((c for c in cases if c["kind"] == "stream" and c["F"] == 128), None)
    f12_n5000 = next((c for c in cases if c["kind"] == "stream" and c["F"] == 12 and c["N"] == 5000), None)
    f12_n10000 = next((c for c in cases if c["kind"] == "stream" and c["F"] == 12 and c["N"] == 10000), None)
    prod = next((c for c in cases if c["kind"] == "production"), None)
    # ---- measurement disclosure (STREAM-HWM-3 + granularity-disclosure):
    # derived from the LIVE cases, never hardcoded -- a stale literal (+~123 MiB
    # / "all MiB multiples") drifts the moment a run differs.
    f128_delta = sf128.get("delta_MiB") if sf128 else None
    mi_multiples = all(c["driver_peak"] % MIB == 0 for c in cases)
    if mi_multiples:
        gran = "本 run 全部 driver_peak 为 MiB 整数倍(±1 MiB 量化)"
    else:
        gran = (f"本 run 存在非 MiB 整数倍 driver_peak(如 F128 "
                f"{round(sf128['driver_peak']/MIB, 2) if sf128 else '?'} MiB)——"
                f"free 内存量化粒度随驱动/时刻变化,非恒定 1 MiB")
    disclosure_note = (f"{gran};F=128 delta +{f128_delta} MiB 远大于 F=12 锚点"
                       f"(delta ≤2 MiB),与 fblock B32 低余量 WDDM 运行期波动同量级,"
                       f"未独立归因;本机 F=128 物理余量 "
                       f"{round((sf128['fb']-sf128['driver_peak'])/MIB, 1) if sf128 else '?'} MiB"
                       f"(相对 free_before {round(free_before_MiB, 0) if free_before_MiB else '?'} MiB,"
                       f"接近耗尽档,delta 含分配粒度/驱动开销)")

    payload = {
        "schema_version": VERSION,
        "artifact": "factor_stream_hwm_v1.json",
        "generator": GENERATOR,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "env": env,
        "free_before_MiB": free_before_MiB,
        "memory_budget": {"total_MiB": 8188, "reserve_MiB": 512,
                          "available_MiB": (8188 - 512)},
        "judgement": ("fit = driver_peak <= available budget AND margin > 8 MiB "
                      "(NO 64 MiB delta band); production F=128 control is a "
                      "legitimate vram-exhausted non-fit; stream cases must fit"),
        "streaming_config": {"max_transpose_rows": STREAM_TT,
                             "chunks": "37x32 + 34 rows (non-final 32*5000=160000=625*256)"},
        "closure_status": "OK",
        "provenance": {
            "source": "live" if not is_from else "from",
            "git_head": git_head(),
            "git_dirty": git_dirty(),
            "exe_sha256": exe_sha256(),
            "capture_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest().upper(),
        },
        "measurement_disclosure": {
            "free_memory_granularity_MiB": 1,
            "all_driver_peaks_mi_multiple": mi_multiples,
            "note": disclosure_note,
        },
        "disclosure": disclosure_note,
        "guard": guard,
        "n_cases": len(cases),
        "n_pass": sum(1 for c in cases if c["pass"]),
        "summary": {
            "stream_f128_driver_MiB": round(sf128["driver_peak_MiB"], 2) if sf128 else None,
            "stream_f128_model_MiB": sf128["model_peak_MiB"] if sf128 else None,
            "stream_f128_delta_MiB": sf128.get("delta_MiB") if sf128 else None,
            "stream_f128_budget_margin_MiB": (round((AVAILABLE_BYTES - sf128["driver_peak"]) / MIB, 2) if sf128 else None),
            "stream_f128_physical_margin_MiB": (round((sf128["fb"] - sf128["driver_peak"]) / MIB, 2) if sf128 else None),
            "mv_measured_gap_MiB": (round((prod["driver_peak"] - sf128["driver_peak"]) / MIB, 2)
                                    if prod and sf128 else None),
            "f12_n5000_driver_MiB": round(f12_n5000["driver_peak_MiB"], 2) if f12_n5000 else None,
            "f12_n5000_model_MiB": f12_n5000["model_peak_MiB"] if f12_n5000 else None,
            "f12_n5000_delta_MiB": f12_n5000.get("delta_MiB") if f12_n5000 else None,
            "f12_n10000_driver_MiB": round(f12_n10000["driver_peak_MiB"], 2) if f12_n10000 else None,
            "f12_n10000_model_MiB": f12_n10000["model_peak_MiB"] if f12_n10000 else None,
            "f12_n10000_delta_MiB": f12_n10000.get("delta_MiB") if f12_n10000 else None,
            "prod_driver_MiB": round(prod["driver_peak_MiB"], 2) if prod else None,
        },
        "cases": cases,
    }
    RESULTS.mkdir(exist_ok=True)
    # F-04 (external MAJOR): atomic write (temp + os.replace) so a partial or
    # failed write never leaves a corrupt-but-present artifact; and any render/
    # write exception (incl. an enormous forged integer overflowing a MiB
    # conversion) invalidates stale artifacts so a later run cannot ingest a
    # leftover closure_status=OK.
    try:
        tmp_json = OUT_JSON.with_suffix(".json.tmp")
        tmp_md = OUT_MD.with_suffix(".md.tmp")
        tmp_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8", newline="\n")
        tmp_md.write_text(render_md(payload), encoding="utf-8", newline="\n")
        os.replace(tmp_json, OUT_JSON)
        os.replace(tmp_md, OUT_MD)
    except Exception as e:
        print(f"FAIL-CLOSED: render/write error ({e}); stale artifacts invalidated")
        for p in (OUT_JSON, OUT_MD):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        return 3

    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print(f"summary: {len(cases)} cases, fit={payload['n_pass']}, "
          f"stream F=128 driver={payload['summary']['stream_f128_driver_MiB']} MiB")
    if guard and guard.get("pass") == 0:
        print("GUARD FAIL: stream drift detected")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
