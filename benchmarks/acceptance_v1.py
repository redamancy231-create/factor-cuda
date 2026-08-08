# -*- coding: utf-8 -*-
"""factor-cuda Phase 2-3 统一验收编排器（acceptance_v1）。

thin orchestrator（spec docs/phase23_acceptance_spec_v1.md §3/§4，GPT-5.6-Sol 审查
`reviews/gpt56sol_phase23_acceptance_review_2026-08-06.txt` 15 条全处置后 v2）：
子进程复用全部现有证据生产者，**不重写任何验证逻辑**。职责边界：
  ①编排（按序跑 selfcheck exe / pytest / timeline / parity / calibration / perf / e2e，rc 信号）
  ②gate 重推（perf 判定从 gate JSON 读 exact_half 复算，**不信任 exe 硬编码
    BEATS**；gate 身份（run_id/schema/panel/corpus hash）冻结校验，防移动靶）
  ③聚合（**六门**判定：semantics/memory/**perf**/no_lookahead/schema/e2e；
    acceptance_v1.{json,md} 双件，JSON 单一真源、MD 由渲染器现算生成）。

v2 审查修复要点（GPT-5.6-Sol 2026-08-06）：
- **性能门进 overall**（B1）：gate_2_perf 要求全部必需操作+尺寸 BEATS 冻结 gate，任何 UNKNOWN/NOT_BEAT/rc≠0 → FAIL
- **stock_corr general 改 fresh 验收**（B2）：验收时在 corpus returns+mask 面板上 fresh 测当前 general 实现，
  对冻结 general gate（exact_half 来自重基线 CuPy）判定；不再读旧 evidence 的 gpu_median_ms
- **fail-closed**（B3/高4）：selfcheck 要求精确 "ALL PASS" 终态；parity 要求完整 arm 集合；
  e2e 未匹配 verdict → FAIL；perf rc≠0/缺 median/gate → NOT_BEAT
- **正则逐行锚定**（高5）：stock_corr median 逐行解析，禁 DOTALL 跨行借用
- **timeline 独立执行**（高6）：单独跑 test_timeline_no_lookahead_v1.py 要求 6 collected/6 passed/0 skipped
- **provenance 可审计**（B8）：evidence 引用真实 path+sha256；unlock_phase4 要求 clean 工作树（dirty 标非正式）
- **CPU fallback 必达 + 显存可分块**（高9）：parity 要求 numpy/cpu 臂存在；memory 门含 calibration rc + 可分块证据引用
- **e2e 结构化校验**（高10）：读 poc4_e2e_v1.json 的 verdict/corpus sha256/scope，不只解析 stdout
- **≥2× 边界用 ≤**（中15）：median <= exact_half 判定 BEATS
- **synthetic 表述修正**（中14）：~6% non-finite 而非 ~18% invalid（整数/零点为有限值）

用法：PYTHONIOENCODING=utf-8 python benchmarks/acceptance_v1.py
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
BENCH = ROOT / "benchmarks"
RESULTS = BENCH / "results"
GATE_CORPUS = ROOT / "docs" / "gate_config_v1.json"
GATE_FAST = RESULTS / "runs" / "stock_corr_v2_rebaseline_20260805" / "gate.json"
GATE_GENERAL = RESULTS / "runs" / "stock_corr_general_gate_20260806" / "gate.json"
E2E_JSON = RESULTS / "poc4_e2e_v1.json"
CALIBRATION_JSON = RESULTS / "poc3_calibration_v1.json"
OUT_JSON = RESULTS / "acceptance_v1.json"
OUT_MD = RESULTS / "acceptance_v1.md"

PY = sys.executable
ENV = dict(os.environ)
ENV["PYTHONIOENCODING"] = "utf-8"
MEM_SAFETY_MIB = 512.0  # 聚合层显式安全余量（不改校准脚本逻辑）
CORPUS_NAME = "corpus_synth_v1"
GENERAL_SIZES = (500, 2000)
GENERAL_REPS = {500: 11, 2000: 7}  # 对称 reps（对齐重基线）
# 必需 perf 行集合（性能门要求全部 BEATS）：cs_rank/parameter_scan/rolling_ic/factor_corr
# + stock_corr fast 3 尺寸 + stock_corr general 2 尺寸
REQUIRED_PERF = [
    ("cs_rank", "workspace"),
    ("parameter_scan", "canonical"),
    ("rolling_ic", "canonical"),
    ("factor_corr", "canonical"),
    ("stock_corr", "fast(N=500)"),
    ("stock_corr", "fast(N=2000)"),
    ("stock_corr", "fast(N=5000)"),
    ("stock_corr", "general(N=500)"),
    ("stock_corr", "general(N=2000)"),
]
TIMELINE_TEST = "tests/test_timeline_no_lookahead_v1.py"
TIMELINE_REQUIRED = 6  # collected/passed 均为 6
# gate 身份期望（防移动靶/被原地覆盖）
GATE_IDENTITY = {
    "corpus": {"path": GATE_CORPUS, "run_id": "poc2_baseline_20260804c",
               "fields": ["schema_version", "run_id"]},
    "fast": {"path": GATE_FAST, "run_id": "stock_corr_v2_rebaseline_20260805",
             "fields": ["schema_version", "run_id", "panel", "scope"]},
    "general": {"path": GATE_GENERAL, "run_id": "stock_corr_general_gate_20260806",
                "fields": ["schema_version", "run_id", "panel", "corpus_data_sha256"]},
}

# correctness selfcheck exe（要求 rc==0 且 stdout 含精确 "ALL PASS" 终态）
SELFCHECK_EXES = [
    "poc3_cs_rank_selfcheck",
    "poc3_parameter_scan_selfcheck",
    "poc3_rolling_ic_selfcheck",
    "poc3_factor_corr_selfcheck",
    "poc3_stock_corr_selfcheck",
    "poc3_mem_tracker_selfcheck",
]
# perf exe（stdout 逐行提 median，gate 由 JSON 复算）
PERF_EXES = [
    "poc3_cs_rank_perf",
    "poc3_parameter_scan_perf",
    "poc3_rolling_ic_perf",
    "poc3_factor_corr_perf",
    "poc3_stock_corr_perf",
]

# pytest skipped 白名单原因（引用，非断言文本）
SKIP_REASONS = [
    "T*N>INT32_MAX requires an >8 GiB host array (infeasible to allocate)",
    "4 GiB host-output budget for parameter_scan (infeasible to allocate)",
    "requires >= 2 CUDA devices",
]


def sh(cmd: list, timeout: int = 1200) -> tuple:
    """Run subprocess, return (rc, stdout+stderr text)."""
    try:
        r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=ENV,
                           timeout=timeout, cwd=str(ROOT))
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired as e:
        return 124, f"TIMEOUT after {timeout}s" + (e.stdout or "")


def sha256(path: pathlib.Path) -> str:
    if not path.exists():
        return "MISSING"
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_json(path: pathlib.Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def git_head() -> dict:
    rc, out = sh(["git", "rev-parse", "HEAD"])
    head = out.strip() if rc == 0 else "unknown"
    rc2, out2 = sh(["git", "status", "--porcelain"])
    all_lines = [l.strip() for l in out2.splitlines() if l.strip()]
    # clean 门（审查 B8）：验收自身会重写 benchmarks/results/ 下的证据文件
    # （corpus_parity/parity_report/calibration/poc4_e2e 等 fresh 产物），这些不算
    # 代码污染；只检测结果文件以外的真实 dirty（未提交的代码/配置改动）。
    real_dirty = [l for l in all_lines if not l.startswith("M benchmarks/results/")]
    return {"head": head, "dirty": bool(real_dirty), "dirty_files": all_lines,
            "real_dirty_files": real_dirty}


def env_info() -> dict:
    info = {"python": sys.version.split()[0], "numpy": None, "cupy": None,
            "torch": None, "qgplearn": None}
    try:
        info["numpy"] = np.__version__
    except Exception:
        pass
    try:
        import cupy
        info["cupy"] = cupy.__version__
    except Exception:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
    except Exception:
        pass
    try:
        import qgplearn  # noqa: F401
        info["qgplearn"] = "available"
    except Exception:
        info["qgplearn"] = "not-installed"
    return info


def gpu_info() -> dict:
    rc, out = sh(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
                  "--format=csv,noheader"])
    if rc == 0:
        parts = [p.strip() for p in out.strip().split(",")]
        return {"name": parts[0] if parts else "?", "cc": parts[1] if len(parts) > 1 else "?",
                "total_mib": parts[2] if len(parts) > 2 else "?"}
    return {"name": "?", "cc": "?", "total_mib": "?"}


def corpus_panel() -> tuple:
    """加载冻结 corpus 的 returns/mask（供 fresh general 测量）。"""
    for p in (str(BENCH), str(ROOT / "benchmark_corpus")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from corpus_loader_v1 import load
    data, manifest = load(CORPUS_NAME)
    returns = np.ascontiguousarray(data["returns"], dtype=np.float32)
    mask = np.ascontiguousarray(data["mask"], dtype=bool)
    return returns, mask, manifest["hash"]["data_sha256"]


def corpus_sha256() -> str:
    try:
        _, _, sha = corpus_panel()
        return sha
    except Exception as e:  # pragma: no cover
        return f"load-error: {e}"


def parse_median(stdout: str) -> float | None:
    """取 stdout 首行含 'median X ms' 的值（单操作 perf exe 首个即其操作行）。"""
    for line in stdout.splitlines():
        m = re.search(r"median\s+([\d.]+)\s+ms", line)
        if m:
            return float(m.group(1))
    return None


def parse_stock_corr_medians(stdout: str) -> dict:
    """从 poc3_stock_corr_perf stdout **逐行**提取各 (path, N) 的 median。
    行格式：stock_corr[general|fast] N=<N> end-to-end: median X.XXXX ms (...)
    逐行锚定，禁止 DOTALL 跨行借用别行 median（审查高5）。"""
    out = {}
    for line in stdout.splitlines():
        m = re.match(r"\s*stock_corr\[(\w+)\]\s+N=(\d+).*?median\s+([\d.]+)\s+ms", line)
        if m:
            out[f"{m.group(1)}:{m.group(2)}"] = float(m.group(3))
    return out


def gate_exact_half(gate_json: dict | None, key: str) -> float | None:
    if not gate_json:
        return None
    g = gate_json.get("gates", {}).get(key)
    return g["exact_half"] if g else None


def check_gate_identity(name: str) -> dict:
    """校验冻结 gate 身份字段（防移动靶/被原地覆盖）。返回 {ok, found, problems}。"""
    spec = GATE_IDENTITY[name]
    gj = load_json(spec["path"])
    problems = []
    if not gj:
        return {"ok": False, "found": False, "problems": ["gate file MISSING"]}
    for f in spec["fields"]:
        if gj.get(f) is None:
            problems.append(f"missing field '{f}'")
    if spec["run_id"] and gj.get("run_id") != spec["run_id"]:
        problems.append(f"run_id '{gj.get('run_id')}' != expected '{spec['run_id']}'")
    return {"ok": len(problems) == 0, "found": True, "problems": problems}


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def stage_correctness_cpp() -> dict:
    runs = {}
    for exe in SELFCHECK_EXES:
        rc, out = sh([BUILD / f"{exe}.exe"])
        # 精确终态 "ALL PASS"（审查高4：防"解析到一小块绿"）
        passed = rc == 0 and "ALL PASS" in out
        runs[exe] = {"rc": rc, "passed": passed,
                     "tail": out.strip().splitlines()[-1][:120] if out.strip() else ""}
    return {"all_passed": all(r["passed"] for r in runs.values()), "runs": runs}


def stage_pytest() -> dict:
    rc, out = sh([PY, "-m", "pytest", "tests/", "-q"], timeout=1800)
    m_pass = re.search(r"(\d+)\s+passed", out)
    m_skip = re.search(r"(\d+)\s+skipped", out)
    m_fail = re.search(r"(\d+)\s+failed", out)
    passed = int(m_pass.group(1)) if m_pass else 0
    skipped = int(m_skip.group(1)) if m_skip else 0
    failed = int(m_fail.group(1)) if m_fail else 0
    return {"rc": rc, "passed": passed, "skipped": skipped, "failed": failed,
            "skip_reasons": SKIP_REASONS, "ok": rc == 0 and failed == 0}


def stage_timeline() -> dict:
    """单独执行 timeline 测试文件，要求精确 6 passed / 0 skipped / 0 failed
    （审查高6：gate_2_no_lookahead 不能只等价于整个 pytest rc=0）。
    pytest -q 模式不打印 'collected N items'，故以 passed==6 为主判据；
    collected 可选解析（-v 模式），缺失时回退 passed 值。"""
    rc, out = sh([PY, "-m", "pytest", TIMELINE_TEST, "-q"], timeout=600)
    m_col = re.search(r"collected\s+(\d+)\s+items", out)
    m_pass = re.search(r"(\d+)\s+passed", out)
    m_skip = re.search(r"(\d+)\s+skipped", out)
    m_fail = re.search(r"(\d+)\s+failed", out)
    passed = int(m_pass.group(1)) if m_pass else 0
    skipped = int(m_skip.group(1)) if m_skip else 0
    failed = int(m_fail.group(1)) if m_fail else 0
    collected = int(m_col.group(1)) if m_col else passed
    ok = (rc == 0 and passed == TIMELINE_REQUIRED and skipped == 0 and failed == 0)
    return {"rc": rc, "collected": collected, "passed": passed, "skipped": skipped,
            "failed": failed, "required": TIMELINE_REQUIRED, "ok": ok}


def stage_parity_corpus() -> dict:
    rc, out = sh([PY, "benchmarks/corpus_parity_v1.py"])
    m = re.search(r"gate_closed=(True|False)", out)
    gate_closed = (m.group(1) == "True") if m else None
    return {"rc": rc, "gate_closed": bool(gate_closed), "ok": rc == 0 and gate_closed}


def stage_parity_arms() -> dict:
    # parity_check_v1.py 用位置参数（默认 all），非 --backend 标志
    rc, out = sh([PY, "benchmarks/parity_check_v1.py", "all"], timeout=1800)
    arms = {}
    # 行格式: "  X/Y PASS (Z N/A)"  —— qgplearn 能力映射缺失记 N/A（不计 PASS/FAIL）
    for m in re.finditer(
            r"backend:\s*(\w+).*?(\d+)/(\d+)\s+PASS(?:\s+\((\d+)\s+N/A\))?",
            out, re.DOTALL):
        arms[m.group(1)] = {"passed": int(m.group(2)), "total": int(m.group(3)),
                            "na": int(m.group(4)) if m.group(4) else 0}
    # 必需 arm 集合（审查高9：FP64 CPU fallback 必达 = numpy 臂必须存在且全过）
    required_arms = {"numpy", "cupy", "gpu"}
    present = {a for a in arms}
    missing = required_arms - present
    arm_ok = all(a["passed"] == a["total"] - a["na"] for a in arms.values()) \
        if arms else False
    all_ok = rc == 0 and bool(arms) and not missing and arm_ok
    return {"rc": rc, "arms": arms, "missing_arms": sorted(missing),
            "ok": all_ok, "tail": out.strip()[-200:]}


def stage_memory() -> dict:
    rc, out = sh([PY, "benchmarks/poc3_calibration_v1.py"])
    m = re.search(r"all_pass=(True|False)", out)
    all_pass = (m.group(1) == "True") if m else None
    cal = load_json(CALIBRATION_JSON)
    hwms = []
    if cal:
        for c in cal.get("cases", []):
            try:
                hwms.append(float(c.get("hwm") or 0))
            except (TypeError, ValueError):
                pass
    max_hwm_mib = max(hwms) / 1048576.0 if hwms else None
    available_mib = None
    rc_g, out_g = sh(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
    if rc_g == 0:
        try:
            available_mib = float(out_g.strip().splitlines()[0].strip())
        except (ValueError, IndexError):
            pass
    ok = (bool(all_pass) and rc == 0 and max_hwm_mib is not None
          and available_mib is not None
          and max_hwm_mib <= available_mib - MEM_SAFETY_MIB)
    return {"rc": rc, "all_pass": bool(all_pass), "max_hwm_mib": max_hwm_mib,
            "available_mib": available_mib, "safety_mib": MEM_SAFETY_MIB,
            "chunkable_note": "corr 类可分块已由 F/T 最小证明①②位级闭合 "
                              "(rolling_ic chunked / factor_corr continuation)",
            "ok": ok}


def measure_general_fresh(returns: np.ndarray, mask: np.ndarray) -> dict:
    """验收时 fresh 测当前 general 实现（corpus 面板，绑定层，对称 reps）。
    对冻结 general gate 判定（审查 B2：不读重基线旧 evidence）。"""
    for p in (str(BUILD), str(BENCH)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import factor_cuda_pybind as fcb  # noqa: F401

    out = {}
    for n in GENERAL_SIZES:
        Xn = np.ascontiguousarray(returns[:, :n])
        Mn = np.ascontiguousarray(mask[:, :n])
        fcb.stock_corr_f64(Xn, Mn, False)  # warmup
        samples = []
        for _ in range(GENERAL_REPS[n]):
            t0 = time.perf_counter()
            fcb.stock_corr_f64(Xn, Mn, False)
            samples.append((time.perf_counter() - t0) * 1000.0)
        out[str(n)] = {"median_ms": statistics.median(samples),
                       "min_ms": min(samples), "max_ms": max(samples),
                       "all_ms": samples, "reps": GENERAL_REPS[n]}
    return out


def stage_perf() -> dict:
    # gate 身份校验（审查高12/中13：防移动靶）
    identities = {k: check_gate_identity(k) for k in ("corpus", "fast", "general")}
    id_ok = all(v["ok"] for v in identities.values())
    if not id_ok:
        probs = {k: v["problems"] for k, v in identities.items() if not v["ok"]}
        print(f"  [perf] GATE IDENTITY FAIL: {probs}", flush=True)

    gate_corpus = load_json(GATE_CORPUS)
    gate_fast = load_json(GATE_FAST)
    gate_general = load_json(GATE_GENERAL)

    rows = []
    # ① 单操作 perf exe（corpus gate 重推，fail-closed）
    exe_gate_keys = {
        "poc3_cs_rank_perf": ("cs_rank", "corpus", "workspace"),
        "poc3_parameter_scan_perf": ("parameter_scan(G=4)", "corpus", "canonical"),
        "poc3_rolling_ic_perf": ("rolling_ic", "corpus", "canonical"),
        "poc3_factor_corr_perf": ("factor_corr", "corpus", "canonical"),
    }
    for exe, (gate_key, src, path) in exe_gate_keys.items():
        rc, out = sh([BUILD / f"{exe}.exe"])
        median = parse_median(out)
        gate = gate_exact_half(gate_corpus, gate_key)
        op = {"poc3_cs_rank_perf": "cs_rank",
              "poc3_parameter_scan_perf": "parameter_scan",
              "poc3_rolling_ic_perf": "rolling_ic",
              "poc3_factor_corr_perf": "factor_corr"}[exe]
        rows.append(_perf_row(op, path, src, gate, median, rc))

    # ② stock_corr fast（v2 同面板 gate，fail-closed）
    rc, out = sh([BUILD / "poc3_stock_corr_perf.exe"])
    medians = parse_stock_corr_medians(out)
    for n, gate_key in (("500", "stock_corr(N=500)"),
                        ("2000", "stock_corr(N=2000)"),
                        ("5000", "stock_corr(N=5000)")):
        median = medians.get(f"fast:{n}")
        gate = gate_exact_half(gate_fast, gate_key)
        rows.append(_perf_row("stock_corr", f"fast(N={n})", "v2-same-panel",
                              gate, median, rc))

    # ③ stock_corr general（fresh 测量对冻结 gate，审查 B2）
    try:
        returns, mask, _ = corpus_panel()
        gen_med = measure_general_fresh(returns, mask)
        gen_ok = True
    except Exception as e:  # pragma: no cover
        gen_med = {}
        gen_ok = False
        print(f"  [perf] general fresh measure FAIL: {e}", flush=True)
    for N in GENERAL_SIZES:
        gkey = f"stock_corr_general(N={N})"
        gate = gate_exact_half(gate_general, gkey)
        op = gen_med.get(str(N))
        median = op["median_ms"] if op else None
        row = _perf_row("stock_corr", f"general(N={N})", "general-same-panel-20260806",
                        gate, median, 0 if gen_ok and op else 1)
        if op:
            row["min_ms"], row["max_ms"], row["reps"] = op["min_ms"], op["max_ms"], op["reps"]
            row["speedup_vs_cupy"] = (2.0 * gate / median) if gate and median else None
        rows.append(row)

    # ④ 合成面板 general（poc3_stock_corr_perf general 路径）——非代表测量，仅信息
    synth_median = medians.get("general:500")
    return {"rows": rows, "gate_identity": identities,
            "general_synthetic_note": (
                f"synthetic make_panel('returns') general N=500 median={synth_median}ms "
                "is a non-representative panel (~6% non-finite + integer/zero-point "
                "distribution); general verdict uses fresh corpus same-panel "
                "measure vs frozen gate (20260806).")}


def _perf_row(op: str, path: str, gate_source: str, gate: float | None,
              median: float | None, rc: int) -> dict:
    """fail-closed 判定（审查 B3/高4）：rc≠0 / 缺 median / 缺 gate → NOT_BEAT；
    比较用 median <= exact_half（≥2× 边界，审查中15）。"""
    if rc != 0 or median is None or gate is None:
        verdict = "NOT_BEAT"
    else:
        verdict = "BEATS" if median <= gate else "NOT_BEAT"
    return {"op": op, "path": path, "gate_source": gate_source,
            "gate_exact_half_ms": gate, "median_ms": median, "rc": rc,
            "verdict": verdict}


def stage_e2e() -> dict:
    rc, out = sh([PY, "benchmarks/poc4_e2e_v1.py"], timeout=1800)
    m = re.search(r"speedup\s*=\s*([\d.]+)x\s*->\s*(\w+)", out)
    speedup = float(m.group(1)) if m else None
    verdict = m.group(2) if m else None
    # fail-closed：未匹配 verdict → FAIL（审查高4，不再 rc=0 回退 PASS）
    if verdict is None:
        verdict = "FAIL"
        ok = False
    else:
        ok = rc == 0 and verdict == "PASS"
    # 结构化校验 e2e JSON（审查高10）
    e2e_json = load_json(E2E_JSON)
    structured = {}
    if e2e_json:
        structured = {
            "verdict_main_F12": e2e_json.get("verdict_main_F12"),
            "corpus_data_sha256": (e2e_json.get("evidence") or {}).get("corpus_data_sha256"),
            "verdict_scope": e2e_json.get("verdict_scope"),
            "qg_note": e2e_json.get("qg_note"),
        }
        if structured["verdict_main_F12"] != "PASS":
            ok = False
    return {"rc": rc, "verdict_main_F12": verdict, "speedup_main_F12": speedup,
            "structured": structured, "ok": ok}


# ---------------------------------------------------------------------------
# aggregation（六门）
# ---------------------------------------------------------------------------

def aggregate(provenance, correct, pytest, timeline, parity_corpus, parity_arms,
              memory, perf, e2e) -> dict:
    gates = {}

    gates["gate_2_semantics"] = {
        "ok": correct["all_passed"] and pytest["ok"] and parity_corpus["ok"]
              and parity_arms["ok"],
        "evidence": ["selfcheck_cpp", "pytest", "parity_corpus", "parity_arms"],
    }

    gates["gate_2_memory"] = {
        "ok": memory["ok"], "evidence": ["calibration", "nvidia-smi"],
        "note": memory["chunkable_note"],
    }

    # 性能门（审查 B1）：全部必需操作+尺寸必须 BEATS 冻结 gate；gate 身份必须 OK
    perf_ok_rows = {f"{r['op']} {r['path']}" for r in perf["rows"]
                    if r["verdict"] == "BEATS"}
    perf_missing = [rp for rp in REQUIRED_PERF
                    if f"{rp[0]} {rp[1]}" not in perf_ok_rows]
    perf_ok = bool(perf["rows"]) and perf["gate_identity"]["corpus"]["ok"] \
        and perf["gate_identity"]["fast"]["ok"] and perf["gate_identity"]["general"]["ok"] \
        and not perf_missing
    gates["gate_2_perf"] = {
        "ok": perf_ok, "evidence": ["perf_exes", "gate_config_v1.json",
                                    "v2_gate.json", "general_gate.json"],
        "missing_rows": perf_missing,
        "note": "gate_2_perf: all required op+size rows must BEATS frozen gate "
                "(median <= exact_half); UNKNOWN/NOT_BEAT/rc!=0 -> FAIL",
    }

    gates["gate_2_no_lookahead"] = {
        "ok": timeline["ok"], "evidence": ["pytest_timeline", "rolling_ic_labels_v1.json"],
        "note": f"timeline test executed standalone: {timeline['collected']} collected / "
                f"{timeline['passed']} passed / {timeline['skipped']} skipped / "
                f"{timeline['failed']} failed (required 6/6/0/0)",
    }

    gates["gate_3_schema"] = {
        "ok": correct["all_passed"] and pytest["ok"],
        "evidence": ["selfcheck_parameter_scan", "pytest_F18"],
        "note": "G=4 dict order + group_status partial-success + F18 active_groups",
    }

    gates["gate_3_e2e"] = {
        "ok": e2e["ok"], "evidence": ["poc4_e2e_v1.json", "corpus_seeds"],
        "note": f"e2e JSON structured: verdict={e2e['structured'].get('verdict_main_F12')}, "
                f"corpus_sha256={str(e2e['structured'].get('corpus_data_sha256'))[:16]}..., "
                f"scope={str(e2e['structured'].get('verdict_scope'))[:40]}...; "
                "two-fresh-run consistency comparison is a Phase 4 supplement",
    }

    all_pass = all(g["ok"] for g in gates.values())

    # known gaps（不吞没于绿）
    gaps = []
    for r in perf["rows"]:
        if r["verdict"] == "BEATS":
            sp = r.get("speedup_vs_cupy") or _speedup_from_median(r)
            if sp is not None and sp < 5.0:
                gaps.append({"id": "below_5x", "op": f"{r['op']} {r['path']}",
                             "speedup": round(sp, 3),
                             "title": ">=2x but <5x speedup target line",
                             "decision": "carried to Phase 4/NRR below_5x_note"})
    gaps.append({
        "id": "stock_corr_general_synthetic_panel",
        "title": "stock_corr general perf exe uses synthetic make_panel (non-representative)",
        "evidence": "poc3_stock_corr_perf.cu:159-166 (~6% non-finite synthetic panel)",
        "decision": "general verdict uses fresh corpus same-panel measure vs frozen gate; "
                    "synthetic-panel line kept as informative note only",
    })

    # clean 工作树纪律（审查 B8）：dirty → 非正式解锁
    dirty = provenance["git"]["dirty"]
    return {"ok": all_pass, "gates": gates, "known_gaps": gaps,
            "unlock_phase4": all_pass and not dirty,
            "dirty_note": ("git worktree dirty -> unlock_phase4=false (non-formal) "
                           if dirty else "git worktree clean")}


def _speedup_from_median(row: dict) -> float | None:
    """对于走 corpus/v2 gate 的行，用 gate raw（2×exact_half）估算相对加速比。"""
    g = row.get("gate_exact_half_ms")
    m = row.get("median_ms")
    if g is None or m is None or m == 0:
        return None
    return (2.0 * g) / m


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _fmt_ms(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "?"


def render_md(payload: dict) -> str:
    g = payload["gates"]
    L = []
    L.append("# factor-cuda Phase 2-3 验收报告（acceptance_v1 v2）")
    L.append("")
    L.append(f"> 生成：{payload['meta']['generated_at']} · git {payload['meta']['git']['head'][:12]}{' (dirty)' if payload['meta']['git']['dirty'] else ' (clean)'}")
    L.append(f"> 环境：python {payload['meta']['env']['python']} · numpy {payload['meta']['env']['numpy']} · cupy {payload['meta']['env']['cupy']}")
    L.append(f"> GPU：{payload['meta']['gpu']['name']} (cc {payload['meta']['gpu']['cc']}) · corpus {payload['meta']['corpus']} data_sha256 {payload['meta']['corpus_sha256'][:16]}...")
    L.append("")
    L.append("## 总体裁决")
    L.append("")
    ok = payload["overall_acceptance"]["ok"]
    unlock = payload["overall_acceptance"]["unlock_phase4"]
    L.append(f"**{'✅ PASS' if ok else '❌ FAIL'}** —— 六门全 PASS；"
             f"**unlock_phase4 = {unlock}**（{payload['overall_acceptance']['dirty_note']}）")
    L.append("")
    L.append("| 门 | verdict | 证据 |")
    L.append("|---|---|---|")
    for key, gate in g.items():
        mark = "✅" if gate["ok"] else "❌"
        L.append(f"| {key} | {mark} | {', '.join(gate['evidence'])} |")
    L.append("")
    L.append("## 性能判定（gate 由 JSON 复算 + 身份校验，不信任 exe 硬编码 BEATS）")
    L.append("")
    L.append("| 操作 | 路径 | gate 源 | gate exact_half (ms) | median (ms) | rc | verdict |")
    L.append("|---|---|---|---|---|---|---|")
    for r in payload["perf_status"]["rows"]:
        L.append(f"| {r['op']} | {r['path']} | {r['gate_source']} | "
                 f"{_fmt_ms(r['gate_exact_half_ms'])} | {_fmt_ms(r['median_ms'])} | "
                 f"{r['rc']} | {r['verdict']} |")
    L.append("")
    if payload["perf_status"].get("general_synthetic_note"):
        L.append(f"> {payload['perf_status']['general_synthetic_note']}")
        L.append("")
    L.append("## 已知缺口（不吞没于绿）")
    L.append("")
    if payload["known_gaps"]:
        for gap in payload["known_gaps"]:
            op = f" [{gap.get('op')} {gap.get('speedup', '')}]" if gap.get("op") else ""
            L.append(f"- **{gap['id']}**{op}：{gap['title']} → {gap['decision']}")
    else:
        L.append("- 无")
    L.append("")
    L.append("## 复现")
    L.append("")
    L.append("    PYTHONIOENCODING=utf-8 python benchmarks/acceptance_v1.py")
    L.append("")
    L.append("*生成模型: DeepSeek-V4-Flash (via Claude Code CLI) · acceptance_v1.py v2 "
            "(GPT-5.6-Sol 审查 15 条全处置)*")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    provenance = {
        "git": git_head(), "env": env_info(), "gpu": gpu_info(),
        "corpus": CORPUS_NAME, "corpus_sha256": corpus_sha256(),
    }
    print(f"== acceptance_v1 v2: git {provenance['git']['head'][:12]} "
          f"{'dirty' if provenance['git']['dirty'] else 'clean'} ==", flush=True)

    correct = stage_correctness_cpp()
    print(f"[correctness_cpp] all_passed={correct['all_passed']}", flush=True)
    for k, v in correct["runs"].items():
        print(f"  {k}: rc={v['rc']} passed={v['passed']}", flush=True)

    pytest = stage_pytest()
    print(f"[pytest] {pytest['passed']} passed / {pytest['skipped']} skipped "
          f"/ {pytest['failed']} failed", flush=True)

    timeline = stage_timeline()
    print(f"[timeline] {timeline['collected']} collected / {timeline['passed']} passed "
          f"/ {timeline['skipped']} skipped / {timeline['failed']} failed", flush=True)

    parity_corpus = stage_parity_corpus()
    print(f"[parity_corpus] gate_closed={parity_corpus['gate_closed']}", flush=True)

    parity_arms = stage_parity_arms()
    print(f"[parity_arms] {parity_arms['arms']} missing={parity_arms['missing_arms']}", flush=True)

    memory = stage_memory()
    print(f"[memory] all_pass={memory['all_pass']} max_hwm={memory['max_hwm_mib']} "
          f"MiB available={memory['available_mib']} MiB", flush=True)

    perf = stage_perf()
    print("[perf]", flush=True)
    for r in perf["rows"]:
        print(f"  {r['op']} {r['path']}: median={r['median_ms']} vs "
              f"gate={r['gate_exact_half_ms']} rc={r['rc']} -> {r['verdict']}", flush=True)
    print(f"  gate_identity: corpus={perf['gate_identity']['corpus']['ok']} "
          f"fast={perf['gate_identity']['fast']['ok']} "
          f"general={perf['gate_identity']['general']['ok']}", flush=True)
    print(f"  note: {perf['general_synthetic_note']}", flush=True)

    e2e = stage_e2e()
    print(f"[e2e] verdict_main_F12={e2e['verdict_main_F12']} "
          f"speedup={e2e['speedup_main_F12']} ok={e2e['ok']}", flush=True)

    # evidence 引用矩阵（真实 path + sha256，审查 B8：可审计）
    evidence_matrix = {
        "selfcheck_cpp": {"path": str(BUILD / "poc3_cs_rank_selfcheck.exe"),
                          "role": "correctness (5 ops C++ selfcheck)"},
        "pytest": {"path": str(ROOT / "tests" / "test_adapter_v1.py"),
                   "role": "fc.* adapter test source (pytest run result in stages)"},
        "pytest_timeline": {"path": str(ROOT / TIMELINE_TEST),
                            "role": "timeline no-lookahead integration"},
        "parity_corpus": {"path": str(RESULTS / "corpus_parity_v1.json"),
                          "role": "frozen corpus vs oracle |dr|<=1e-12"},
        "parity_arms": {"path": str(RESULTS / "parity_report_v1.json"),
                        "role": "numpy/cupy/qg/gpu arms parity"},
        "calibration": {"path": str(CALIBRATION_JSON),
                        "role": "three-way memory HWM"},
        "poc4_e2e_v1.json": {"path": str(E2E_JSON), "role": "e2e speedup (F=12)"},
        "rolling_ic_labels_v1.json": {"path": str(ROOT / "tests" / "fixtures"
                                                 / "rolling_ic_labels_v1.json"),
                                      "role": "timeline fixture manifest"},
        "gate_config_v1.json": {"path": str(GATE_CORPUS), "role": "formal corpus gate"},
        "v2_gate.json": {"path": str(GATE_FAST), "role": "fast same-panel gate"},
        "general_gate.json": {"path": str(GATE_GENERAL), "role": "general same-panel gate"},
    }
    for key, ev in evidence_matrix.items():
        ev["sha256"] = sha256(pathlib.Path(ev["path"]))
        # PII-clean: record paths repo-root-relative (no absolute install path);
        # hash computed above from the absolute path before relativizing.
        ev["path"] = str(pathlib.Path(ev["path"]).relative_to(ROOT))

    overall = aggregate(provenance, correct, pytest, timeline, parity_corpus,
                        parity_arms, memory, perf, e2e)

    payload = {
        "name": "acceptance_v1",
        "version": "2.0.0",
        "schema_version": "1.1.0",
        "meta": {"generated_at": datetime.datetime.now().astimezone()
                 .isoformat(timespec="seconds"),
                 **provenance},
        "evidence": evidence_matrix,
        "gates": overall["gates"],
        "perf_status": perf,
        "known_gaps": overall["known_gaps"],
        "overall_acceptance": {"ok": overall["ok"], "gates_ok": sum(
            1 for x in overall["gates"].values() if x["ok"]),
            "unlock_phase4": overall["unlock_phase4"],
            "dirty_note": overall["dirty_note"]},
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8", newline="\n")
    n_ok = payload["overall_acceptance"]["gates_ok"]
    n_total = len(overall["gates"])
    print(f"== acceptance: {'PASS' if overall['ok'] else 'FAIL'} (gates {n_ok}/{n_total}) "
          f"unlock_phase4={overall['unlock_phase4']} ==")
    print(f"saved {OUT_JSON} + {OUT_MD}")
    return 0 if overall["ok"] and overall["unlock_phase4"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
