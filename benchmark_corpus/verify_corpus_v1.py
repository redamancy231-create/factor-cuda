# -*- coding: utf-8 -*-
"""factor-cuda benchmark corpus 校验脚本 v1 — 设计 §4.4 断言清单。

用法：
  python benchmark_corpus/verify_corpus_v1.py --npz benchmark_corpus/corpus_synth_v1.npz \
      --manifest benchmark_corpus/corpus_synth_v1.manifest.json
  python benchmark_corpus/verify_corpus_v1.py --npz ... --manifest ... --regenerate  # synth 位级复现
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

MIN_VALID = 30
H, LAG = 5, 1
W = 21
CORR_DOMAIN_MAX = 1e150
CORR_DOMAIN_MIN = 1e-150
# GPT-5.6-Sol #8 修复：NaN 载荷必须从 uint32 位模式 view 得到。
# np.float32(0x7FC00000) 会得到 2.14e9（非 NaN）——应 reinterpret 位模式。
NAN_PAYLOAD = np.array([0x7FC00000], dtype=np.uint32).view(np.float32)[0]


def sha256_file(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest().upper()


def _finalize(errors: list[str], npz_name: str = "") -> None:
    if errors:
        print(f"FAIL ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print(f"PASS: {npz_name} 校验通过")


def verify(npz_path: pathlib.Path, manifest_path: pathlib.Path, regenerate: bool) -> None:
    errors: list[str] = []

    # ⓪ Draft-07 schema 校验（GPT-5.6-Sol #7 修复：verify 第一步必须过 schema）
    try:
        import jsonschema
    except ImportError:
        errors.append("⓪ jsonschema 未安装——无法校验 manifest schema")
        return _finalize(errors, npz_path.name)
    # schema 路径：优先 manifest 同目录，回退到 verify 脚本所在目录（支持临时目录 regenerate）
    schema_path = manifest_path.parent / "manifest_schema_v1.json"
    if not schema_path.exists():
        schema_path = pathlib.Path(__file__).resolve().parent / "manifest_schema_v1.json"
    if schema_path.exists():
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        v = list(jsonschema.Draft7Validator(schema).iter_errors(manifest))
        if v:
            errors.append(f"⓪ manifest 不通过 schema: {len(v)} 错误（首条: {v[0].message}）")
    else:
        errors.append("⓪ manifest_schema_v1.json 缺失，无法校验 schema")

    # ① npz SHA-256 == manifest data_sha256
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    got = sha256_file(npz_path)
    exp = manifest["hash"]["data_sha256"]
    if got != exp:
        errors.append(f"① npz sha256 mismatch: got {got} exp {exp}")

    # ③④⑤ 数据校验
    d = np.load(npz_path, allow_pickle=False)
    factors = d["factors"]
    t, n, f = factors.shape

    if not np.array_equal(d["factor_a"], factors[:, :, 0]):
        errors.append("③ factor_a != factors[...,0]")

    if factors.dtype != np.float32:
        errors.append(f"④ factors dtype {factors.dtype} != float32")
    if d["mask"].dtype != np.bool_:
        errors.append(f"④ mask dtype {d['mask'].dtype} != bool")
    if d["forward_returns"].dtype != np.float64:
        errors.append(f"④ forward_returns dtype {d['forward_returns'].dtype} != float64")
    if not factors.flags["C_CONTIGUOUS"]:
        errors.append("④ factors not C-contiguous")

    if int(d["h"][0]) != H or int(d["lag"][0]) != LAG:
        errors.append(f"⑤ h/lag {d['h'][0]}/{d['lag'][0]} != {H}/{LAG}")

    # ⑥ 基准行有效数 ≥ min_valid（半开区间 [W, T-6) 即 t≤T-7）
    mask = d["mask"]
    fwd = d["forward_returns"]
    for day in range(W, t - (H + LAG)):
        valid = mask[day] & np.isfinite(factors[day, :, 0]) & np.isfinite(fwd[day, :])
        if valid.sum() < MIN_VALID:
            errors.append(f"⑥ row {day} valid {valid.sum()} < {MIN_VALID}")

    # ⑦ 末尾 h+lag 行 forward_returns 全 NaN
    if not np.all(np.isnan(fwd[t - (H + LAG):, :])):
        errors.append("⑦ trailing forward_returns not all NaN")

    # ⑧ 数值域（float64 price/forward_returns）
    # GPT-5.6-Sol #8 修复：绝对值检查（原 finite.max() 漏负数、min() 得布尔）。
    price = d["price"]
    for arr, name in ((price, "price"), (fwd, "forward_returns")):
        finite = arr[np.isfinite(arr)]
        if finite.size:
            abs_finite = np.abs(finite)
            abs_nonzero = abs_finite[abs_finite != 0]
            over_max = abs_finite.max() > CORR_DOMAIN_MAX
            under_min = abs_nonzero.size > 0 and abs_nonzero.min() < CORR_DOMAIN_MIN
            if over_max or under_min:
                errors.append(f"⑧ {name} out of corr domain (max_abs={abs_finite.max()!r})")

    # ⑨ regenerate：在临时目录重生成 + 比较 bytes 与 array_sha256（GPT-5.6-Sol #5 修复）
    if regenerate:
        import tempfile
        import subprocess
        import os
        # 从 manifest 读取生成参数
        fam = manifest.get("family")
        if fam != "synthetic":
            errors.append(f"⑨ regenerate 仅支持 synthetic；family={fam}")
        else:
            # P4 修复：从 manifest generation_params 读 variant/T/N/F（不猜尺寸）
            gp = manifest.get("generation_params", {})
            t, n, f = gp.get("T"), gp.get("N"), gp.get("F")
            variant = gp.get("variant", "canonical")
            if not (t and n and f):
                errors.append("⑨ manifest 缺 generation_params.T/N/F，无法重生成")
            else:
                with tempfile.TemporaryDirectory() as td:
                    p = subprocess.run(
                        [sys.executable, str(pathlib.Path(__file__).resolve().parent / "generate_corpus_v1.py"),
                         "--mode", "synth", "--variant", variant,
                         "--T", str(t), "--N", str(n), "--F", str(f), "--out-dir", td],
                        capture_output=True, text=True, encoding="utf-8", errors="replace",
                        env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                    )
                    if p.returncode != 0:
                        errors.append(f"⑨ 重生成失败: {p.stderr.strip()[:200]}")
                    else:
                        suffix = {"canonical": "v1", "smoke": "smoke_v1", "sweep": "sweep_v1"}.get(variant, "v1")
                        regen_npz = pathlib.Path(td) / f"corpus_synth_{suffix}.npz"
                        if regen_npz.exists():
                            regen_bytes = regen_npz.read_bytes()
                            orig_bytes = npz_path.read_bytes()
                            if regen_bytes != orig_bytes:
                                errors.append(f"⑨ regenerate bytes 不一致（重生成 ≠ 归档）")
                            else:
                                print(f"⑨ regenerate OK: {npz_path.name} 位级复现一致")

    _finalize(errors, npz_path.name)


def main() -> None:
    ap = argparse.ArgumentParser(description="verify factor-cuda benchmark corpus v1")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--regenerate", action="store_true")
    args = ap.parse_args()
    verify(pathlib.Path(args.npz), pathlib.Path(args.manifest), args.regenerate)


if __name__ == "__main__":
    main()
