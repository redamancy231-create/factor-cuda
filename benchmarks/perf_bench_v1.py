# -*- coding: utf-8 -*-
"""PoC ② 公平基线——性能基准 v2.0（R2/R4/R6 重构）。

设计依据：FAIR_BASELINE_PROTOCOL_v1.md §4 + GPT-5.6-Sol 二轮审查修复。

R2（证据链可复算）：每次执行写独立不可变 run-id 目录；禁读旧 JSON 合并；
   JSON/Markdown 由同一生成链（--render 从 run JSON 渲染）；speedup 从 JSON 复算。
R4（三口径标签污染）：cold 在 GPU 预上传前测（首个 op 含 CUDA context 初始化）；
   resident 用纯设备 API（cs_rank/rolling_ic device→device）；corr 无纯设备 →
   标注 pure_device=False（实际口径）；upload 测全输入 H2D；native event per backend。
R6（N 规模结构化）：--stock-sizes 500,2000,5000 每规模独立记录（耗时/CI/显存/状态/命令）。

用法：
    python perf_bench_v1.py [--backend numpy|cupy|qgplearn|all] [--subset smoke|full]
                            [--stock-sizes 500,2000,5000] [--runs N] [--run-id <id>]
    python perf_bench_v1.py --render <run_id>     # 从 run JSON 生成 Markdown + 一致性校验
输出：results/runs/<run_id>/<backend>.json（+ --render 生成 results/perf_report_v1.md）
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import threading
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
CORPUS_DIR = HERE.parent / "benchmark_corpus"
RESULTS_DIR = HERE / "results"
RUNS_DIR = RESULTS_DIR / "runs"

VERSION = "2.0.0"
BACKENDS = ["numpy", "cupy", "qgplearn"]
MIN_VALID = 30
SAMPLES = 20  # 每操作 warm 样本数
DEFAULT_STOCK_SIZES = [500, 2000, 5000]

from backends import (  # noqa: E402
    np_cs_rank, np_factor_corr, np_stock_corr, np_rolling_ic, np_parameter_scan,
    cp_cs_rank, cp_factor_corr, cp_stock_corr, cp_rolling_ic, cp_parameter_scan,
    cp_cs_rank_gpu, cp_rolling_ic_gpu,
    qg_cs_rank, qg_rolling_ic, qg_parameter_scan,
    qg_cs_rank_device, qg_rolling_ic_device,
)


def _json_safe(obj):
    """递归规范化 NumPy 标量/数组（S1 同款——避免 default=str 静默变字符串）。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# corpus / env
# ---------------------------------------------------------------------------

def _load_corpus(subset: bool):
    sys.path.insert(0, str(CORPUS_DIR))
    from corpus_loader_v1 import load
    cid = "corpus_synth_smoke_v1" if subset else "corpus_synth_v1"
    d, m = load(cid)
    return d, m


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, cwd=str(HERE))
        return out.stdout.strip()
    except Exception:
        return "n/a"


def _collect_env() -> dict:
    """完整环境指纹：CPU/GPU/驱动/CUDA/BLAS/git commit。"""
    import platform
    env: dict[str, object] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    for mod, attr in (("pandas", "__version__"), ("scipy", "__version__"),
                      ("torch", "__version__"), ("cupy", "__version__")):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, attr)
        except Exception:
            env[mod] = "n/a"
    try:
        import torch
        env["torch_cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["cuda_version"] = torch.version.cuda
    except Exception:
        env["torch_cuda"] = False
    try:
        import cupy as cp
        env["cupy_runtime"] = cp.cuda.runtime.runtimeGetVersion()
    except Exception:
        env["cupy_runtime"] = "n/a"
    try:
        out = subprocess.run(["wmic", "cpu", "get", "name"], capture_output=True, text=True)
        cpu = [l.strip() for l in out.stdout.splitlines() if l.strip() and "Name" not in l]
        env["cpu"] = cpu[0] if cpu else "n/a"
    except Exception:
        env["cpu"] = "n/a"
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True)
        env["driver"] = out.stdout.strip()
        out2 = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True)
        fields = [f.strip() for f in out2.stdout.split(",")] if out2.stdout.strip() else []
        if len(fields) >= 4:
            env["gpu_temp_c"] = float(fields[0])
            env["gpu_power_w"] = float(fields[1])
            env["gpu_clocks_sm_mhz"] = float(fields[2])
            env["gpu_clocks_mem_mhz"] = float(fields[3])
    except Exception:
        env["driver"] = "n/a"
    try:
        cfg = np.show_config(mode="dicts")
        blas = cfg.get("Build", {}).get("blas_info", {}) if isinstance(cfg, dict) else {}
        env["blas"] = blas.get("libraries", "n/a")
    except Exception:
        env["blas"] = "n/a"
    env["git_commit"] = _git_commit()
    return env


# ---------------------------------------------------------------------------
# 上传 / 计时辅助
# ---------------------------------------------------------------------------

def _upload(name: str, arr):
    """numpy → GPU 设备数组（cupy blocking=True / torch CUDA）。numpy 臂原样返回。

    按 dtype 选择设备 dtype：bool mask → bool（避免 torch 位运算对 Float 失败）。
    """
    if name == "cupy":
        import cupy as cp
        dt = cp.bool_ if arr.dtype == bool else None
        return cp.asarray(arr, dtype=dt, blocking=True)
    if name == "qgplearn":
        import torch
        dt = torch.bool if arr.dtype == bool else torch.float32
        return torch.tensor(arr, device="cuda", dtype=dt)
    return arr


def _upload_args(name: str, args):
    """只上传数组参数（descending bool / min_valid int 保持）。"""
    return [_upload(name, a) if isinstance(a, np.ndarray) else a for a in args]


def _event_helpers(name):
    """native event per backend（审查 B3/R4：CuPy/Torch 各自原生）。返回 (make, elapsed_ms)。"""
    if name == "cupy":
        import cupy as cp
        def make():
            s, e = cp.cuda.Event(), cp.cuda.Event()
            s.record()
            return s, e
        def elapsed(ev):
            s, e = ev
            e.record(); e.synchronize()
            return cp.cuda.get_elapsed_time(s, e)
        return make, elapsed
    if name == "qgplearn":
        import torch
        def make():
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            return s, e
        def elapsed(ev):
            s, e = ev
            e.record(); e.synchronize()
            return s.elapsed_time(e)
        return make, elapsed
    return None, None


def _stats(times):
    arr = np.array(times)
    med = float(np.median(arr))
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    cv = float(arr.std() / arr.mean()) if arr.mean() > 0 else 0.0
    rng = np.random.default_rng(0)
    boot = np.array([np.median(rng.choice(arr, len(arr), replace=True)) for _ in range(1000)])
    return {
        "wall_ms": med, "q1_ms": q1, "q3_ms": q3, "cv": cv,
        "ci95_lo_ms": float(np.percentile(boot, 2.5)),
        "ci95_hi_ms": float(np.percentile(boot, 97.5)),
        "n_samples": len(arr),
    }


def _time_operation(fn, name: str) -> dict:
    """warm e2e：预热 3 + 20 样本 + wall 统计 + native cudaEvent（GPU 臂）。"""
    is_gpu = name in ("cupy", "qgplearn")
    make_ev, elapsed_ev = _event_helpers(name)
    for _ in range(3):
        fn()
    times, ev_times = [], []
    for _ in range(SAMPLES):
        t0 = time.perf_counter()
        if is_gpu:
            ev = make_ev()
            fn()
            ev_times.append(elapsed_ev(ev))
        else:
            fn()
        times.append((time.perf_counter() - t0) * 1000)
    r = _stats(times)
    if is_gpu and ev_times:
        eva = np.array(ev_times)
        # M-04：event 包整个 host 入口（含 H2D/D2H/同步），非纯 kernel → 改名 gpu_inclusive_ms
        r["gpu_inclusive_ms"] = float(np.median(eva))
        r["gpu_inclusive_ci95_lo_ms"] = float(np.percentile(eva, 2.5))
        r["gpu_inclusive_ci95_hi_ms"] = float(np.percentile(eva, 97.5))
    return r


def _measure_cold(fn) -> float:
    """cold/first：fn 首次调用（GPU 臂在预上传前 → 首个 op 含 CUDA context 初始化）。"""
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000


def _time_upload(name: str, inputs) -> float | None:
    """测该操作**全部输入**一次性 H2D 总耗时（n=5 中位数）。numpy 臂无上传。

    M-04：用对应 backend 的同步（CuPy null stream / Torch cuda synchronize），
    不能以 CuPy 同步断言 Torch stream 完成。
    """
    if name == "numpy":
        return None
    if name == "cupy":
        import cupy as cp
        sync = cp.cuda.Stream.null.synchronize
    else:
        import torch
        sync = torch.cuda.synchronize
    for _ in range(2):
        for a in inputs:
            _upload(name, a)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        for a in inputs:
            _upload(name, a)
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def _time_resident(dev_fn, name: str) -> dict:
    """resident：纯设备调用（输入已 GPU、输出 GPU），只测 device 计算（native event）。"""
    if name == "cupy":
        import cupy as cp
        sync = cp.cuda.Stream.null.synchronize
    else:
        import torch
        sync = torch.cuda.synchronize
    make_ev, elapsed_ev = _event_helpers(name)
    for _ in range(3):
        dev_fn()
    ev_times, wall_times = [], []
    for _ in range(SAMPLES):
        t0 = time.perf_counter()
        ev = make_ev()
        dev_fn()
        ev_times.append(elapsed_ev(ev))
        wall_times.append((time.perf_counter() - t0) * 1000)
    sync()
    eva = np.array(ev_times)
    return {
        "resident_wall_ms": float(np.median(wall_times)),
        "resident_gpu_ms": float(np.median(eva)),
        "resident_gpu_ci95_lo_ms": float(np.percentile(eva, 2.5)),
        "resident_gpu_ci95_hi_ms": float(np.percentile(eva, 97.5)),
        "n_samples": len(ev_times),
    }


def _gpu_memory_peak_mb(dev_fn, interval=0.01):
    """stock_corr 每规模的显存峰值：后台线程高频采样 cudaMemGetInfo。

    M-02：前后净差在 allocator pool 复用时会稳定报 0——改为运行期采样
    （捕获分配瞬间的 used 峰值，即使释放后也能记录）。方法局限：小 N（<interval
    时间量级）可能漏采；输出标 `method: memgetinfo_sampling`。
    """
    import cupy as cp
    free_idle = cp.cuda.runtime.memGetInfo()[0]
    used_peak = [0.0]
    stop = threading.Event()

    def _sample():
        while not stop.is_set():
            try:
                used = free_idle - cp.cuda.runtime.memGetInfo()[0]
                if used > used_peak[0]:
                    used_peak[0] = used
            except Exception:
                pass
            time.sleep(interval)

    th = threading.Thread(target=_sample, daemon=True)
    th.start()
    try:
        dev_fn()
    finally:
        cp.cuda.Stream.null.synchronize()
        stop.set()
        th.join(timeout=0.5)
    return max(0.0, used_peak[0] / 1e6)


# ---------------------------------------------------------------------------
# op 规格与分派
# ---------------------------------------------------------------------------

def _build_op_specs(X, mask, F3, fwd, returns, stock_sizes):
    """每 op：label / op / args（numpy 输入，数组将被上传）/ upload_inputs（全输入）。

    M-03：stock_corr 记录 requested_N/actual_N；请求超 corpus 列数 → skip
    （smoke 切片超界静默截断，不得以请求值冒充实际值）。
    """
    specs = []
    def add(label, op, args, upload_inputs, **extra):
        specs.append({"label": label, "op": op, "args": args, "upload_inputs": upload_inputs, **extra})
    add("cs_rank", "cs_rank", (X, mask, False), (X, mask))
    add("cs_rank_desc", "cs_rank", (X, mask, True), (X, mask))
    add("factor_corr", "factor_corr", (F3, mask), (F3, mask))
    add("rolling_ic", "rolling_ic", (X, fwd, mask, mask, MIN_VALID), (X, fwd, mask, mask))
    add("parameter_scan(G=4)", "parameter_scan", (X, mask), (X, mask))
    ncols = returns.shape[1]
    for sz in stock_sizes:
        actual = min(sz, ncols)
        add(f"stock_corr(N={sz})", "stock_corr",
            (returns[:, :actual], mask[:, :actual]),
            (returns[:, :actual], mask[:, :actual]),
            requested_N=sz, actual_N=actual, skip=(sz > actual))
    return specs


def _dispatch(name: str, op: str, *args):
    if name == "numpy":
        if op == "cs_rank":
            return np_cs_rank(*args)
        if op == "factor_corr":
            return np_factor_corr(*args)
        if op == "stock_corr":
            return np_stock_corr(*args)
        if op == "rolling_ic":
            return np_rolling_ic(*args)
        if op == "parameter_scan":
            return np_parameter_scan(*args)
    elif name == "cupy":
        if op == "cs_rank":
            return cp_cs_rank(*args)
        if op == "factor_corr":
            return cp_factor_corr(*args)
        if op == "stock_corr":
            return cp_stock_corr(*args)
        if op == "rolling_ic":
            return cp_rolling_ic(*args)
        if op == "parameter_scan":
            return cp_parameter_scan(*args)
    elif name == "qgplearn":
        if op == "cs_rank":
            return qg_cs_rank(*args)
        if op == "factor_corr":
            raise NotImplementedError("QuantGplearn 无 factor_corr 原生算子")
        if op == "stock_corr":
            raise NotImplementedError("QuantGplearn 无 stock_corr 原生算子")
        if op == "rolling_ic":
            return qg_rolling_ic(*args)
        if op == "parameter_scan":
            return qg_parameter_scan(*args)
    raise NotImplementedError(f"backend {name} 无 {op}")


def _make_dev_fn(name: str, sp: dict, gpu_inputs: dict):
    """纯设备 callable：输入 GPU、返回 GPU（R4 resident）。无纯设备变体 → None。

    仅 cs_rank/rolling_ic 有纯设备变体；corr/scan/stock_corr 返回 None 且**不访问**
    gpu_inputs（stock_corr 独立生命周期不预上传，M-04——否则 KeyError）。
    """
    if name not in ("cupy", "qgplearn"):
        return None
    if sp["op"] not in ("cs_rank", "rolling_ic"):
        return None
    gargs = gpu_inputs[sp["label"]]
    if name == "cupy":
        if sp["op"] == "cs_rank":
            Xg, Mg, desc = gargs
            return lambda: cp_cs_rank_gpu(Xg, Mg, desc)
        if sp["op"] == "rolling_ic":
            fg, rg, fmg, rmg, mv = gargs
            return lambda: cp_rolling_ic_gpu(fg, rg, fmg, rmg, mv)
        return None
    # qgplearn
    if sp["op"] == "cs_rank":
        Xg, Mg, desc = gargs
        return lambda: qg_cs_rank_device(Xg, Mg, desc)
    if sp["op"] == "rolling_ic":
        fg, rg, fmg, rmg, mv = gargs
        return lambda: qg_rolling_ic_device(fg, rg, fmg, rmg, mv)
    return None


# ---------------------------------------------------------------------------
# 基准执行
# ---------------------------------------------------------------------------

def run_backend(name: str, subset: bool, stock_sizes: list[int]) -> dict:
    d, m = _load_corpus(subset)
    X = d["factor_a"]
    mask = d["mask"]
    F3 = d["factors"]
    fwd = d["forward_returns"]
    returns = d["returns"]
    T, N = X.shape
    is_gpu = name in ("cupy", "qgplearn")
    specs = _build_op_specs(X, mask, F3, fwd, returns, stock_sizes)

    results: dict = {}

    # ---- cold/first：GPU 臂在预上传前测（首个 op 含 CUDA context 初始化）----
    if is_gpu:
        for sp in specs:
            try:
                results[sp["label"]] = {
                    "cold_first_ms": _measure_cold(lambda s=sp: _dispatch(name, s["op"], *s["args"])),
                    "status": "ok",
                }
            except NotImplementedError:
                results[sp["label"]] = {"cold_first_ms": None, "status": "na"}
            except Exception as e:
                results[sp["label"]] = {"cold_first_ms": None, "status": "error",
                                        "error": f"{type(e).__name__}: {str(e)[:120]}"}

    # ---- 预上传 GPU 输入（resident/upload 用）----
    # M-04：stock_corr 无纯设备 resident、独立生命周期（按规模上传/执行/释放），不预上传；
    # skip 不预上传。避免大 N 上传 OOM 时整个 backend 在逐规模记录前终止。
    gpu_inputs = {}
    if is_gpu:
        for sp in specs:
            if sp["op"] == "stock_corr" or sp.get("skip"):
                continue
            gpu_inputs[sp["label"]] = _upload_args(name, sp["args"])

    # ---- warm e2e + upload + resident ----
    for sp in specs:
        label = sp["label"]
        if sp.get("skip"):
            # M-03：请求 N 超 corpus 列数 → 明确 skipped，不得以请求值冒充实际值
            results[label] = {"status": "skipped", "requested_N": sp["requested_N"],
                              "actual_N": sp["actual_N"],
                              "skip_reason": "请求 N 超 corpus 列数（smoke）"}
            print(f"    {label:<22} SKIPPED（N 超 corpus 列数，actual_N={sp['actual_N']}）")
            continue
        rec = results.get(label, {"status": "ok"})
        if sp.get("requested_N") is not None:
            rec["requested_N"] = sp["requested_N"]
            rec["actual_N"] = sp["actual_N"]
        e2e_fn = lambda s=sp: _dispatch(name, s["op"], *s["args"])
        dev_fn = _make_dev_fn(name, sp, gpu_inputs)

        # e2e（判据口径）
        try:
            rec.update(_time_operation(e2e_fn, name))
            # R6：stock_corr 每规模记录显存峰值（GPU 臂，cudaMemGetInfo 前后差）
            if sp["op"] == "stock_corr" and is_gpu:
                try:
                    rec["memory_peak_mb"] = _gpu_memory_peak_mb(e2e_fn)
                except Exception as me:
                    rec["memory_peak_mb"] = None
                    rec.setdefault("warnings", []).append(f"显存采样失败: {str(me)[:80]}")
        except NotImplementedError:
            rec.update({"status": "na"})
            results[label] = rec
            continue
        except Exception as e:
            is_oom = "out of memory" in str(e).lower() or "OutOfMemory" in type(e).__name__
            rec.update({"status": "oom" if is_oom else "error",
                        "error": f"{type(e).__name__}: {str(e)[:120]}"})
            results[label] = rec
            continue

        # GPU 臂：upload（全输入 H2D）+ resident（纯设备 / 诚实标注）
        if is_gpu:
            try:
                rec["upload_ms"] = _time_upload(name, [a for a in sp["upload_inputs"]])
            except Exception as e:
                rec["upload_ms"] = None
                rec.setdefault("warnings", []).append(f"upload 失败: {str(e)[:80]}")
            if dev_fn is not None:
                try:
                    res = _time_resident(dev_fn, name)
                    res["pure_device"] = True
                    rec["resident"] = res
                except Exception as e:
                    rec["resident"] = {"status": "error", "pure_device": True,
                                       "error": f"{type(e).__name__}: {str(e)[:100]}"}
            else:
                # corr / parameter_scan：无纯设备 API → 诚实标注实际口径（含 CPU 回拷）
                rec["resident"] = {"pure_device": False,
                                   "note": "无纯设备 API（内部回拷 CPU 构造/重上传），resident 为实际口径"}
        results[label] = rec

        # 打印进度
        note = f"{rec.get('wall_ms', 0):>8.1f} ms"
        if rec.get("gpu_inclusive_ms"):
            note += f" gpu={rec['gpu_inclusive_ms']:.1f}"
        if rec.get("resident"):
            note += f" res={rec['resident'].get('resident_gpu_ms', 0):.1f}"
        if rec.get("upload_ms") is not None:
            note += f" up={rec['upload_ms']:.1f}"
        if rec.get("cold_first_ms") is not None:
            note += f" cold={rec['cold_first_ms']:.1f}"
        print(f"    {label:<22} {note}  CV={rec.get('cv', 0):.2%}")

    return {
        "backend": name,
        "version": VERSION,
        "protocol": {
            "samples": SAMPLES,
            "min_valid": MIN_VALID,
            "stock_sizes": stock_sizes,
            "timing": ["warm e2e", "cold first-use", "upload all-operands", "resident pure-device"],
        },
        "corpus": {
            "name": "corpus_synth_smoke_v1" if subset else "corpus_synth_v1",
            "data_sha256": m.get("hash", {}).get("data_sha256", ""),
            "T": T, "N": N, "F": F3.shape[-1],
        },
        "operations": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# 多进程 / 合并
# ---------------------------------------------------------------------------

def _merge_median(reps):
    """跨 run 合并：嵌套数值字段取中位数，非数值取首个（R4：不只 wall_ms）。"""
    if not reps:
        return None
    head = reps[0]
    if isinstance(head, dict):
        out = {}
        for k in head:
            vals = [r.get(k) for r in reps if isinstance(r, dict)]
            num = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None]
            if num and len(num) == len(vals) - vals.count(None):
                out[k] = float(np.median(num)) if num else None
            elif all(isinstance(v, dict) for v in vals if v is not None):
                out[k] = _merge_median([v for v in vals if v is not None])
            else:
                out[k] = head[k]
        return out
    return head


def _run_backend_process(args) -> dict:
    name, subset, stock_sizes = args
    try:
        return run_backend(name, subset, stock_sizes)
    except Exception as e:  # noqa: BLE001
        return {"backend": name, "error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Markdown 渲染（R2：JSON → MD 同生成链）
# ---------------------------------------------------------------------------

_SAME_SEMANTICS_PREFIX = {
    "cs_rank": ["numpy", "cupy", "qgplearn"],
    "cs_rank_desc": ["numpy", "cupy", "qgplearn"],
    "factor_corr": ["numpy", "cupy"],
    "rolling_ic": ["numpy", "cupy"],  # QG float32 known-deviation 不具同语义资格
    "parameter_scan(G=4)": ["numpy", "cupy", "qgplearn"],
    "stock_corr": ["numpy", "cupy"],
}


def _render_consistency_check(reports) -> list[str]:
    """R2：corpus 名/hash/shape/commit 不一致 → 拒绝汇总并列出。"""
    probs = []
    items = [r for r in reports.values() if "corpus" in r and "operations" in r]
    if not items:
        return ["无有效 run 记录"]
    base = items[0]
    for r in items[1:]:
        for k in ("name", "data_sha256", "T", "N", "F"):
            if r["corpus"].get(k) != base["corpus"].get(k):
                probs.append(f"{r['backend']} corpus.{k} {r['corpus'].get(k)} != {base['corpus'].get(k)}")
        if r.get("env", {}).get("git_commit") != base.get("env", {}).get("git_commit"):
            probs.append(f"{r['backend']} git_commit {r['env'].get('git_commit')[:8]} != {base['env'].get('git_commit')[:8]}")
    return probs


def render_markdown(run_id: str) -> int:
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        print(f"run 目录不存在: {run_dir}", file=sys.stderr)
        return 2
    # H-03：只读精确 canonical 文件名（raw/ 子目录与 __rN 原始重复不参与 backend 映射）
    reports = {}
    for be in BACKENDS:
        p = run_dir / f"{be}.json"
        if p.exists():
            rep = json.loads(p.read_text(encoding="utf-8"))
            reports[rep["backend"]] = rep
    # H-04：完整校验——run-id 匹配 / 三臂齐全唯一 / 无顶层 error / version 一致 / 无操作级 error-oom
    probs = []
    for be in BACKENDS:
        if be not in reports:
            probs.append(f"缺臂 {be}")
    if len(reports) != len(BACKENDS):
        probs.append(f"canonical backend 数 {len(reports)} != {len(BACKENDS)}（不唯一）")
    for be, rep in reports.items():
        if rep.get("error"):
            probs.append(f"{be} 顶层 error: {str(rep['error'])[:100]}")
        if rep.get("run_id") != run_id:
            probs.append(f"{be} run_id {rep.get('run_id')} != 目录 {run_id}（异源）")
        if rep.get("version") != VERSION:
            probs.append(f"{be} version {rep.get('version')} != {VERSION}")
        op_errs = [lab for lab, op in rep.get("operations", {}).items()
                   if isinstance(op, dict) and op.get("status") in ("error", "oom")]
        if op_errs:
            probs.append(f"{be} 操作级 error/oom: {op_errs}")
    if not probs:
        probs = _render_consistency_check(reports)
    if probs:
        print("一致性校验失败，拒绝生成汇总（R2）:")
        for pr in probs:
            print(f"  - {pr}")
        (RESULTS_DIR / f"perf_report_v1__{run_id}__INCONSISTENT.txt").write_text(
            "\n".join(probs), encoding="utf-8")
        return 3

    env = next(iter(reports.values())).get("env", {})
    corpus = next(iter(reports.values())).get("corpus", {})
    # 收集所有 op label（保序）
    op_labels = []
    for r in reports.values():
        for lab in r.get("operations", {}):
            if lab not in op_labels:
                op_labels.append(lab)

    def _same_semantic_arms(label: str):
        for prefix, arms in _SAME_SEMANTICS_PREFIX.items():
            if label.startswith(prefix):
                return arms
        return []

    lines = []
    lines.append(f"# PoC ② 公平基线性能报告（run `{run_id}`）")
    lines.append("")
    lines.append(f"- 生成命令: `python perf_bench_v1.py --render {run_id}`")
    lines.append(f"- corpus: `{corpus.get('name')}` sha256={corpus.get('data_sha256', '')[:16]}… "
                 f"T×N×F={corpus.get('T')}×{corpus.get('N')}×{corpus.get('F')}")
    lines.append(f"- git_commit: `{env.get('git_commit', 'n/a')}`")
    lines.append(f"- 环境: {env.get('platform', '')} | {env.get('gpu', '')} | driver {env.get('driver', '')} "
                 f"| python {env.get('python', '')} numpy {env.get('numpy', '')} "
                 f"cupy {env.get('cupy', '')} torch {env.get('torch', '')}")
    lines.append(f"- 三口径: warm e2e（判据）/ cold first-use / upload 全输入 H2D / resident（纯设备，corr 除外）")
    lines.append("")

    # 主表：e2e wall_ms + speedup
    lines.append("## 端到端 wall（ms）+ speedup（相对同语义最佳免费替代）")
    lines.append("")
    lines.append("| 操作 | numpy | cupy | qgplearn | 同语义最佳 |")
    lines.append("|---|---:|---:|---:|---:|")
    for lab in op_labels:
        cells = {}
        for be, rep in reports.items():
            op = rep.get("operations", {}).get(lab)
            if op and op.get("wall_ms") is not None:
                cells[be] = op["wall_ms"]
            else:
                cells[be] = None
        arms = _same_semantic_arms(lab)
        valid = {be: v for be, v in cells.items() if v is not None and be in arms}
        best = min(valid.values()) if valid else None
        row = [lab]
        for be in ("numpy", "cupy", "qgplearn"):
            v = cells.get(be)
            if v is None:
                row.append("N/A")
            elif best and be in arms:
                row.append(f"{v:.1f} ({best / v:.2f}×)")
            else:
                row.append(f"{v:.1f} (非同语义)")
        row.append(f"{best:.1f}" if best else "N/A")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # GPU 臂三口径表
    lines.append("## GPU 臂三口径（ms）")
    lines.append("")
    lines.append("| 操作 | backend | cold | upload | resident(gpu) | pure_device |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for lab in op_labels:
        for be in ("cupy", "qgplearn"):
            rep = reports.get(be)
            op = (rep or {}).get("operations", {}).get(lab)
            if not op:
                continue
            cold = op.get("cold_first_ms")
            up = op.get("upload_ms")
            res = op.get("resident", {})
            rg = res.get("resident_gpu_ms")
            pd = res.get("pure_device")
            pd_s = "是" if pd else ("否" if pd is False else "-")
            lines.append(f"| {lab} | {be} | {f'{cold:.1f}' if cold else 'N/A'} | "
                         f"{f'{up:.1f}' if up is not None else 'N/A'} | "
                         f"{f'{rg:.1f}' if rg else 'N/A'} | {pd_s} |")
    lines.append("")

    out_path = RESULTS_DIR / "perf_report_v1.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"已渲染: {out_path}")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="all")
    ap.add_argument("--subset", action="store_true", help="用 smoke corpus 快速冒烟")
    ap.add_argument("--runs", type=int, default=1, help="多进程重复次数（每 run 独立记录 + 全字段中位数合并）")
    ap.add_argument("--stock-sizes", default=",".join(map(str, DEFAULT_STOCK_SIZES)),
                    help="stock_corr 规模列表，如 500,2000,5000")
    ap.add_argument("--run-id", default=None, help="指定 run-id（默认时间戳）")
    ap.add_argument("--force", action="store_true", help="run-id 已存在时允许覆盖（默认拒绝，R2/H-02）")
    ap.add_argument("--render", default=None, help="run-id → 生成 Markdown + 一致性校验")
    args = ap.parse_args()

    if args.render:
        return render_markdown(args.render)

    names = BACKENDS if args.backend == "all" else [args.backend]
    if any(n not in BACKENDS for n in names):
        print(f"未知 backend {names}，可用: {BACKENDS}", file=sys.stderr)
        return 2
    stock_sizes = [int(s.strip()) for s in args.stock_sizes.split(",") if s.strip()]

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / run_id
    if run_dir.exists() and not args.force:
        # R2/H-02：run-id 须不可变——已存在即拒绝，首次证据不得被覆盖
        print(f"run-id 已存在，拒绝覆盖（R2/H-02，run-id 不可变）: {run_dir}", file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run-id: {run_id} → {run_dir}")

    env = _collect_env()
    for name in names:
        print(f"--- backend: {name} ---")
        if args.runs > 1:
            tasks = [(name, args.subset, stock_sizes)] * args.runs
            with mp.Pool(args.runs) as pool:
                results = pool.starmap(_run_backend_process, tasks)
            ok = [r for r in results if "error" not in r]
            # H-03：raw 重复写 raw/ 子目录，不污染 canonical（renderer 只读 canonical 文件名）
            raw_dir = run_dir / "raw"
            raw_dir.mkdir(exist_ok=True)
            for i, r in enumerate(results):
                (raw_dir / f"{name}__r{i + 1}.json").write_text(
                    json.dumps(_json_safe(r), ensure_ascii=False, indent=1), encoding="utf-8")
            if not ok:
                print(f"  ERROR: {results[0].get('error', '?')}")
                continue
            merged = _merge_median(ok)
            merged["backend"] = name
            merged["n_runs"] = len(ok)
            rep = merged
        else:
            try:
                rep = run_backend(name, args.subset, stock_sizes)
            except Exception as e:  # noqa: BLE001
                print(f"  ERROR: {type(e).__name__}: {e}")
                rep = {"backend": name, "error": f"{type(e).__name__}: {e}"}
        rep["run_id"] = run_id
        rep["command"] = " ".join(sys.argv)
        rep["env"] = env
        (run_dir / f"{name}.json").write_text(
            json.dumps(_json_safe(rep), ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  ✓ 已写入 {name}.json")

    # R2/H-02：completion manifest——全部预期臂完成后写，记录创建时间/命令/backend 集
    manifest = {
        "run_id": run_id,
        "command": " ".join(sys.argv),
        "backends": names,
        "completed": [n for n in names if (run_dir / f"{n}.json").exists()],
        "stock_sizes": stock_sizes,
        "subset": args.subset,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\nrun 目录: {run_dir}")
    print(f"渲染 Markdown: python perf_bench_v1.py --render {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
