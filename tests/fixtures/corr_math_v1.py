# -*- coding: utf-8 -*-
"""六审自修复使用的确定性数学与布局单一真源。"""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Any, Callable, Iterable, Sequence

UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1
SIZE_T_MAX = UINT64_MAX
INT32_MAX = (1 << 31) - 1
SAFE_PEARSON_EXPRESSION = "corr = (Sxy/sqrt(Sxx)) / sqrt(Syy)"


def safe_pearson(sxy: float, sxx: float, syy: float) -> float:
    """按冻结顺序逐次归一化，避免先形成 ``sxx * syy``。"""
    if not (sxx > 0.0 and syy > 0.0):
        return float("nan")
    return (sxy / math.sqrt(sxx)) / math.sqrt(syy)


@dataclass
class CompensatedSum:
    """Kahan 状态；真实表示值恒为 ``sum - c``。"""

    sum: float = 0.0
    c: float = 0.0

    def add(self, x: float) -> None:
        y = float(x) - self.c
        t = self.sum + y
        self.c = (t - self.sum) - y
        self.sum = t

    @property
    def represented(self) -> float:
        return self.sum - self.c

    def merge(self, right: "CompensatedSum") -> "CompensatedSum":
        """按固定左右顺序重放右状态，补偿项必须取负。"""
        self.add(right.sum)
        self.add(-right.c)
        return self

    @classmethod
    def from_values(cls, values: Iterable[float]) -> "CompensatedSum":
        state = cls()
        for value in values:
            state.add(float(value))
        return state


class BinaryFrontier:
    """连续绝对叶序的单槽 binary-carry frontier。"""

    def __init__(self, merge: Callable[[Any, Any], Any], max_levels: int = 63):
        if max_levels < 1:
            raise ValueError("max_levels must be >= 1")
        self.merge = merge
        self.max_levels = int(max_levels)
        self.slots: list[Any | None] = [None] * (self.max_levels + 1)
        self.next_expected_index = 0

    def ingest(self, leaf_index: int, node: Any) -> None:
        if leaf_index != self.next_expected_index:
            raise ValueError(
                f"leaf_index={leaf_index} != next_expected_index={self.next_expected_index}"
            )
        if leaf_index < 0 or leaf_index >= (1 << self.max_levels):
            raise OverflowError("BinaryFrontier leaf capacity exceeded")

        carry = node
        for level in range(self.max_levels + 1):
            bit = (leaf_index >> level) & 1
            if bit == 0:
                if self.slots[level] is not None:
                    raise RuntimeError(f"frontier invariant: level {level} must be empty")
                self.slots[level] = carry
                self.next_expected_index += 1
                return
            left = self.slots[level]
            if left is None:
                raise RuntimeError(f"frontier invariant: level {level} lacks left node")
            carry = self.merge(left, carry)
            self.slots[level] = None
        raise OverflowError("BinaryFrontier carry exceeded allocated slots")

    def snapshot(self) -> dict[str, Any]:
        return {
            "next_expected_index": self.next_expected_index,
            "occupied": [
                {"level": level, "node": node}
                for level, node in enumerate(self.slots)
                if node is not None
            ],
        }

    def flush(self) -> dict[str, Any]:
        """chunk 边界只验不变量，不改变树形。"""
        occupied_leaf_count = sum(
            (1 << level) for level, node in enumerate(self.slots) if node is not None
        )
        if occupied_leaf_count != self.next_expected_index:
            raise RuntimeError("frontier invariant: occupied capacity mismatch")
        return self.snapshot()

    def finalize(self) -> Any | None:
        """从高 occupied level 向低 level 合并，保留全局叶序。"""
        result = None
        for level in range(self.max_levels, -1, -1):
            node = self.slots[level]
            if node is None:
                continue
            result = node if result is None else self.merge(result, node)
        return result


def _perfect_tree(values: Sequence[Any], merge: Callable[[Any, Any], Any]) -> Any:
    if len(values) == 1:
        return values[0]
    midpoint = len(values) // 2
    return merge(
        _perfect_tree(values[:midpoint], merge),
        _perfect_tree(values[midpoint:], merge),
    )


def expected_fixed_tree(values: Sequence[Any], merge: Callable[[Any, Any], Any]) -> Any | None:
    """按 n 的高位到低位分解连续区间，再从左向右拼接。"""
    if not values:
        return None
    offset = 0
    components: list[Any] = []
    for level in range((len(values)).bit_length() - 1, -1, -1):
        width = 1 << level
        if len(values) & width:
            components.append(_perfect_tree(values[offset : offset + width], merge))
            offset += width
    result = components[0]
    for component in components[1:]:
        result = merge(result, component)
    return result


def checked_add(a: int, b: int, *, limit: int = SIZE_T_MAX) -> int:
    if a < 0 or b < 0 or a > limit or b > limit - a:
        raise OverflowError(f"checked_add overflow: {a} + {b} > {limit}")
    return a + b


def checked_mul(a: int, b: int, *, limit: int = SIZE_T_MAX) -> int:
    if a < 0 or b < 0 or (b != 0 and a > limit // b):
        raise OverflowError(f"checked_mul overflow: {a} * {b} > {limit}")
    return a * b


def checked_scatter_out_base(row_base: int, n_columns: int, *, limit: int = INT32_MAX) -> int:
    return checked_mul(row_base, n_columns, limit=limit)


def checked_global_element_offset(
    row_base: int,
    local_row: int,
    n_columns: int,
    local_column: int,
    *,
    limit: int = SIZE_T_MAX,
) -> int:
    global_row = checked_add(row_base, local_row, limit=limit)
    row_offset = checked_mul(global_row, n_columns, limit=limit)
    return checked_add(row_offset, local_column, limit=limit)


def checked_byte_offset(element_offset: int, item_size: int, *, limit: int = SIZE_T_MAX) -> int:
    return checked_mul(element_offset, item_size, limit=limit)


def _f32_bits(value: float) -> int:
    quantized = struct.unpack("<f", struct.pack("<f", float(value)))[0]
    return struct.unpack("<I", struct.pack("<f", quantized))[0]


def _f64_bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def canonical_ordinal_key_f32(value: float) -> int:
    quantized = struct.unpack("<f", struct.pack("<f", float(value)))[0]
    if not math.isfinite(quantized):
        return UINT32_MAX
    bits = 0 if quantized == 0.0 else _f32_bits(quantized)
    return ((~bits) & UINT32_MAX) if (bits & (1 << 31)) else (bits | (1 << 31))


def canonical_ordinal_key_f64(value: float) -> int:
    value = float(value)
    if not math.isfinite(value):
        return UINT64_MAX
    bits = 0 if value == 0.0 else _f64_bits(value)
    return ((~bits) & UINT64_MAX) if (bits & (1 << 63)) else (bits | (1 << 63))


def stable_ordinal_order(
    values: Sequence[float], *, dtype: str, descending: bool = False
) -> list[int]:
    if dtype == "float32":
        key_fn = canonical_ordinal_key_f32
        invalid = UINT32_MAX
    elif dtype == "float64":
        key_fn = canonical_ordinal_key_f64
        invalid = UINT64_MAX
    else:
        raise ValueError("dtype must be float32 or float64")
    valid = [index for index, value in enumerate(values) if key_fn(value) != invalid]
    return sorted(valid, key=lambda index: key_fn(values[index]), reverse=descending)


@dataclass(frozen=True)
class ViewSpec:
    dtype: str
    shape: tuple[int, ...]
    strides_elements: tuple[int, ...]
    item_size: int
    byte_offset: int
    pointer_address: int
    storage_bytes: int
    device: str
    owner_id: str | None
    owner_retained: bool
    lifetime_synchronized: bool


@dataclass(frozen=True)
class KernelLayout:
    dtype: str
    shape: tuple[int, ...]
    strides_elements: tuple[int, ...]
    alignment: int
    device: str
    required_owner_id: str | None = None
    expected_byte_offset: int | None = None


def _required_storage_end(view: ViewSpec) -> int:
    if any(dimension < 0 for dimension in view.shape):
        raise ValueError("negative shape")
    if any(stride < 0 for stride in view.strides_elements):
        raise ValueError("negative stride")
    if not view.shape or any(dimension == 0 for dimension in view.shape):
        return view.byte_offset
    last_element = 0
    for dimension, stride in zip(view.shape, view.strides_elements, strict=True):
        last_element = checked_add(
            last_element,
            checked_mul(dimension - 1, stride),
        )
    return checked_add(
        view.byte_offset,
        checked_mul(checked_add(last_element, 1), view.item_size),
    )


def factor_alias_failures(view: ViewSpec, layout: KernelLayout) -> list[str]:
    """factor `(T,N,w)` packed 视图的独立判据。"""
    failures: list[str] = []
    if view.dtype != "float64" or view.dtype != layout.dtype or view.item_size != 8:
        failures.append("dtype")
    if view.shape != layout.shape or len(view.shape) != 3:
        failures.append("shape")
    if view.strides_elements != layout.strides_elements:
        failures.append("strides")
    if view.byte_offset < 0:
        failures.append("byte_offset")
    if layout.expected_byte_offset is not None and view.byte_offset != layout.expected_byte_offset:
        failures.append("byte_offset")
    if layout.alignment <= 0 or (view.pointer_address + view.byte_offset) % layout.alignment != 0:
        failures.append("alignment")
    if view.device != layout.device:
        failures.append("device")
    try:
        if _required_storage_end(view) > view.storage_bytes:
            failures.append("storage_bounds")
    except (OverflowError, ValueError):
        failures.append("storage_bounds")
    if not view.owner_retained or view.owner_id is None:
        failures.append("owner")
    if layout.required_owner_id is not None and view.owner_id != layout.required_owner_id:
        failures.append("owner")
    if not view.lifetime_synchronized:
        failures.append("sync_lifetime")
    return list(dict.fromkeys(failures))


def stock_alias_failures(view: ViewSpec, layout: KernelLayout) -> list[str]:
    """stock `(T,w)` packed 视图的独立判据；不调用 factor 判据。"""
    failures: list[str] = []
    if view.dtype != "float64" or view.dtype != layout.dtype or view.item_size != 8:
        failures.append("dtype")
    if view.shape != layout.shape or len(view.shape) != 2:
        failures.append("shape")
    if view.strides_elements != layout.strides_elements:
        failures.append("strides")
    if view.byte_offset < 0:
        failures.append("byte_offset")
    if layout.expected_byte_offset is not None and view.byte_offset != layout.expected_byte_offset:
        failures.append("byte_offset")
    if layout.alignment <= 0 or (view.pointer_address + view.byte_offset) % layout.alignment != 0:
        failures.append("alignment")
    if view.device != layout.device:
        failures.append("device")
    try:
        if _required_storage_end(view) > view.storage_bytes:
            failures.append("storage_bounds")
    except (OverflowError, ValueError):
        failures.append("storage_bounds")
    if not view.owner_retained or view.owner_id is None:
        failures.append("owner")
    if layout.required_owner_id is not None and view.owner_id != layout.required_owner_id:
        failures.append("owner")
    if not view.lifetime_synchronized:
        failures.append("sync_lifetime")
    return list(dict.fromkeys(failures))


def factor_is_aliasable(view: ViewSpec, layout: KernelLayout) -> bool:
    return not factor_alias_failures(view, layout)


def stock_is_aliasable(view: ViewSpec, layout: KernelLayout) -> bool:
    return not stock_alias_failures(view, layout)


def select_factor_input_path(view: ViewSpec, layout: KernelLayout) -> str:
    if view.dtype == "float32":
        return "f32_conversion"
    if view.dtype != "float64":
        raise ValueError("factor input dtype must be float32 or float64")
    return "f64_alias" if factor_is_aliasable(view, layout) else "f64_gather"


def select_stock_input_path(view: ViewSpec, layout: KernelLayout) -> str:
    if view.dtype == "float32":
        return "f32_conversion"
    if view.dtype != "float64":
        raise ValueError("stock input dtype must be float32 or float64")
    return "f64_alias" if stock_is_aliasable(view, layout) else "f64_gather"


STRUCT_ABI = {
    "Partial1": {
        "size_B": 56,
        "alignment_B": 8,
        "offsets_B": {
            "count": 0,
            "sum_x": 8,
            "sum_y": 16,
            "min_x": 24,
            "max_x": 32,
            "min_y": 40,
            "max_y": 48,
        },
    },
    "Partial2": {
        "size_B": 24,
        "alignment_B": 8,
        "offsets_B": {"sxx": 0, "syy": 8, "sxy": 16},
    },
    "PartialK1": {
        "size_B": 40,
        "alignment_B": 8,
        "offsets_B": {"count": 0, "sum_x": 8, "c_x": 16, "sum_y": 24, "c_y": 32},
    },
    "PartialK2": {
        "size_B": 48,
        "alignment_B": 8,
        "offsets_B": {"sxx": 0, "c_xx": 8, "syy": 16, "c_yy": 24, "sxy": 32, "c_xy": 40},
    },
}


def validate_struct_abi() -> None:
    expected = {"Partial1": 56, "Partial2": 24, "PartialK1": 40, "PartialK2": 48}
    for name, size in expected.items():
        record = STRUCT_ABI[name]
        assert record["size_B"] == size
        assert record["alignment_B"] == 8
        assert max(record["offsets_B"].values()) < size


validate_struct_abi()
