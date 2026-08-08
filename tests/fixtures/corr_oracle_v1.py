# -*- coding: utf-8 -*-
"""correlation oracle wrapper v1 — factor-cuda PoC ① 冻结契约的唯一 correlation oracle。

引用：reviews/_draft_contract.md §2 uncentered_gram（唯一 oracle = 本 wrapper）。
锁定环境：Python 3.12.7 / NumPy 2.4.4 / 平台与 BLAS 构建指纹见 np.show_config()。
语义：对每个输出条目，按其共同有效子集、以规范 (t,i) 行主序切片后，
      直接执行 np.corrcoef(np.stack([xv, yv]))[0,1]。本 wrapper 是唯一裁决者：
      - 返回有限值 → 数学快捷规则（n=2→|r|=1、对角线→1.0、精确零方差→NaN）可适用；
      - 返回 NaN（如平方溢出） → 所有快捷规则让位，输出保持 NaN。
测试只约束本 wrapper 的最终值/NaN 与绝对误差 |Δr|≤1e-12；
不声称 GPU 内部复现 numpy 的私有归约树（mean/dot 为 numpy 内部实现）。
"""
from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

__all__ = ["corr_oracle", "pair_valid_mask", "VERSION", "ENV_FINGERPRINT"]

VERSION = "1.0.0"
ENV_FINGERPRINT = "python-3.12.7_numpy-2.4.4"

# 输入可接受 numpy 数组或任意数值序列；内部统一转为 float64 ndarray
VectorLike = Union[NDArray[np.float64], "np.generic", list, tuple]


def pair_valid_mask(
    a: VectorLike,
    b: VectorLike,
    mask_a: Union[NDArray[np.bool_], list, tuple, None] = None,
    mask_b: Union[NDArray[np.bool_], list, tuple, None] = None,
) -> NDArray[np.bool_]:
    """计算一对序列的共同有效观测 mask（bool 数组）。

    语义（与契约 §2 aggregation/mask 一致）：
      - 单元有效 iff 两侧均有限（isfinite）且两侧 mask 均为 True（缺省视为全 True）；
      - 有效观测 = mask 与数值有限性取交集；
      - 按输入原始顺序（规范 (t,i) 行主序）返回，不排序。
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"corr_oracle: a.shape={a.shape} != b.shape={b.shape}")
    valid: NDArray[np.bool_] = np.isfinite(a) & np.isfinite(b)
    if mask_a is not None:
        valid &= np.asarray(mask_a, dtype=bool)
    if mask_b is not None:
        valid &= np.asarray(mask_b, dtype=bool)
    return valid


def corr_oracle(
    a: VectorLike,
    b: VectorLike,
    *,
    mask_a: Union[NDArray[np.bool_], list, tuple, None] = None,
    mask_b: Union[NDArray[np.bool_], list, tuple, None] = None,
    _direct: bool = True,
) -> float:
    """冻结 correlation oracle：对共同有效子集直接执行 np.corrcoef。

    参数：
      a, b      — 两条输入序列（等长，numpy 数组或数值序列）。
      mask_a/mask_b — 可选的有效性 mask（True=有效），缺省全 True。
      _direct   — 内部标记，恒为 True；保留以防未来扩展 wrapper 版本。

    返回：
      float，或 NaN（如 numpy 因平方溢出/下溢返回 NaN）。

    说明：
      - 有效子集为空或样本数 n<2 时，np.corrcoef 自然返回 NaN（与契约
        empty_or_degenerate / ddof 条款一致），本 wrapper 不额外拦截。
      - 本 wrapper 只读，不修改输入；输入按原顺序切片，不做任何重排。
    """
    valid = pair_valid_mask(a, b, mask_a, mask_b)
    a = np.asarray(a, dtype=np.float64)[valid]
    b = np.asarray(b, dtype=np.float64)[valid]
    if a.size < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_ = np.corrcoef(np.stack([a, b]))
    # corrcoef 对 2×m 输入返回 2×2 相关矩阵；取 [0,1] 即 a 与 b 的 Pearson 相关
    r = float(np.asarray(r_)[0, 1])
    return r


def self_check() -> bool:
    """冒烟自检：验证 wrapper 与契约锚点反例一致。

    断言：
      - 有限 n=2 → |r|=1（0.9999999999999999）；
      - 典型量化值域 |x|∈[1e-3,1e3] → 有限正常；
      - 域外（max|x|>1e150）输入由调用方 API 前置条件拒绝（本 wrapper 不校验域）。
    """
    r = corr_oracle([1.0, 2.0], [2.0, 5.0])
    assert abs(abs(r) - 1.0) < 1e-12, f"n=2 finite should be |r|=1, got {r}"
    r_typ = corr_oracle([1e-3, 2e-2, 0.5, 10.0, 1e3], [2e-3, 1e-2, 0.7, 8.0, 9e2])
    assert np.isfinite(r_typ), f"typical range should be finite, got {r_typ}"
    return True


if __name__ == "__main__":
    ok = self_check()
    print(f"corr_oracle_v1 self_check: {'PASS' if ok else 'FAIL'}")
