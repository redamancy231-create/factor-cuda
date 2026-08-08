# -*- coding: utf-8 -*-
"""factor-cuda benchmark corpus 运行时读路径 v1 — 基准唯一读入口。

设计 §4.5：加载时校验 data_sha256，不匹配抛 RuntimeError；本地完整 npz 缺失 → 明确错误提示
按生成协议重生成，不静默降级；基准请求超 full 尺寸 → ValueError；基准运行时不得 import
生成/统计脚本、不得调用 RNG、不得写 npz；子集参数化由 loader 按冻结规则确定性派生
（perf=前导前缀；parity=seed 派生固定索引）。
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Optional, Tuple

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


class CorpusError(RuntimeError):
    """corpus 加载/校验失败。"""


def _sha256(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest().upper()


def _freeze_readonly(a: np.ndarray) -> np.ndarray:
    """物化拷贝并设为只读（P7：防 NpzFile 懒加载新数组破坏写保护）。"""
    a = np.array(a, copy=True) if not a.flags.owndata else a
    a = np.ascontiguousarray(a)
    a.flags.writeable = False
    return a


def load(corpus_id: str, data_dir: Optional[pathlib.Path] = None) -> Tuple[dict, dict]:
    """加载 corpus 数据 + manifest，校验 data_sha256。

    corpus_id 如 "corpus_synth_v1" / "corpus_synth_smoke_v1" / "corpus_real_v1"。
    返回 (npz 数据 dict, manifest dict)。
    """
    data_dir = data_dir or HERE
    npz_path = data_dir / f"{corpus_id}.npz"
    manifest_path = data_dir / f"{corpus_id}.manifest.json"

    if not npz_path.exists():
        raise CorpusError(
            f"corpus {corpus_id}.npz 不存在于 {data_dir}——按生成协议重新运行 "
            f"generate_corpus_v1.py（synth 确定性复现；real 需外部快照）"
        )
    if not manifest_path.exists():
        raise CorpusError(f"manifest {corpus_id}.manifest.json 缺失")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    got = _sha256(npz_path)
    exp = manifest.get("hash", {}).get("data_sha256")
    if exp is None:
        raise CorpusError(f"manifest {corpus_id}.manifest.json 缺 hash.data_sha256")
    if got != exp:
        raise CorpusError(
            f"corpus {corpus_id}.npz sha256 {got} != manifest {exp}——数据被篡改或版本漂移，禁止使用"
        )

    # P7 修复：一次性物化为普通 dict，全部数组只读（NpzFile 懒加载每次解压新数组
    # 导致写保护无效——必须物化拷贝后才可保证只读）
    with np.load(npz_path, allow_pickle=False) as nz:
        d = {k: _freeze_readonly(nz[k]) for k in nz.files}
    return d, manifest


def subset_prefix(d: dict, t: Optional[int] = None, n: Optional[int] = None, f: Optional[int] = None) -> dict:
    """perf 子集：前导前缀（设计 §2.3，确定性、保时间线/标签有效性）。"""
    full_t, full_n, full_f = d["factors"].shape
    if t is not None and (t < 1 or t > full_t):
        raise ValueError(f"T'={t} 超全尺寸 {full_t}")
    if n is not None and (n < 1 or n > full_n):
        raise ValueError(f"N'={n} 超全尺寸 {full_n}")
    if f is not None and (f < 1 or f > full_f):
        raise ValueError(f"F'={f} 超全尺寸 {full_f}")
    t = t or full_t
    n = n or full_n
    f = f or full_f

    return {
        "dates": _freeze_readonly(d["dates"][:t]),
        "ids": _freeze_readonly(d["ids"][:n]),
        "names": _freeze_readonly(d["names"][:f]),
        "factors": _freeze_readonly(d["factors"][:t, :n, :f]),
        "factor_a": _freeze_readonly(d["factor_a"][:t, :n]),
        "returns": _freeze_readonly(d["returns"][:t, :n]),
        "price": _freeze_readonly(d["price"][:t, :n]),
        "mask": _freeze_readonly(d["mask"][:t, :n]),
        "forward_returns": _freeze_readonly(d["forward_returns"][:t, :n]),
        "h": _freeze_readonly(d["h"]), "lag": _freeze_readonly(d["lag"]),
    }
