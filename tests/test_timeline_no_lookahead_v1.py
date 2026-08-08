# -*- coding: utf-8 -*-
"""Phase 2 无未来函数（timeline）集成测试。

覆盖 CLAUDE.md §timeline_info_constraint / §label_ownership 的机械验收链：
  1. manifest data_sha256 与冻结 npz 匹配（防篡改）；
  2. 生成脚本重跑产出与冻结 npz 逐元素一致（脚本为唯一权威、可复现）；
  3. forward_returns[t] = price[t+1+h]/price[t+1] - 1（h=5, lag=1，无未来函数语义）；
  4. 末尾 h+lag 行及停牌单元为 NaN（时间线边界）；
  5. fc.rolling_ic 能消费该标签（行对齐冒烟：输出 (T,) 截面 IC 序列，抽查逐行值与独立参考一致）。

这是 PLAN.md Phase 2「无未来函数测试」门槛的执行证据（spec phase23_acceptance_spec_v1.md §6 P0-1）。
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
import pytest

import generate_rolling_ic_labels_v1 as gen

FIX = pathlib.Path(__file__).resolve().parent / "fixtures"
MANIFEST = FIX / "rolling_ic_labels_v1.json"
NPZ = FIX / "rolling_ic_labels_v1.npz"
GEN = FIX / "generate_rolling_ic_labels_v1.py"


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _independent_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """独立 Spearman 参考：stable argsort ordinal 秩 + np.corrcoef（Pearson on ranks）。

    与 fc/_cpu_core.py 的 _ordinal_rank_1d + _two_pass_corr 非同源实现路径，
    用作行对齐冒烟的外部核对。
    """
    oa = np.empty(len(a), dtype=np.int64)
    oa[np.argsort(a, kind="stable")] = np.arange(len(a))
    ob = np.empty(len(b), dtype=np.int64)
    ob[np.argsort(b, kind="stable")] = np.arange(len(b))
    return float(np.corrcoef(oa, ob)[0, 1])


class TestTimelineNoLookahead:
    """冻结 fixture 完整性 + 无未来函数语义 + fc.rolling_ic 消费冒烟。"""

    def test_manifest_npz_hash_match(self):
        m = _load_manifest()
        digest = hashlib.sha256(NPZ.read_bytes()).hexdigest().upper()
        assert digest == m["data_sha256"], "冻结 npz 与 manifest data_sha256 不符"

    def test_script_sha256_matches_manifest(self):
        m = _load_manifest()
        digest = hashlib.sha256(GEN.read_bytes()).hexdigest().upper()
        assert digest == m["script_sha256"], "生成脚本与 manifest script_sha256 不符"

    def test_regenerate_matches_frozen(self, tmp_path, monkeypatch):
        """重跑生成脚本到临时目录，产出与冻结 npz 逐元素一致（脚本可复现）。"""
        tmp_npz = tmp_path / "rolling_ic_labels_v1.npz"
        tmp_manifest = tmp_path / "rolling_ic_labels_v1.json"
        monkeypatch.setattr(gen, "NPZ_PATH", tmp_npz)
        monkeypatch.setattr(gen, "MANIFEST_PATH", tmp_manifest)
        gen.generate()

        regenerated = np.load(tmp_npz)
        frozen = np.load(NPZ)
        assert sorted(regenerated.files) == sorted(frozen.files)
        for key in frozen.files:
            r, f = regenerated[key], frozen[key]
            if f.dtype.kind in "fi":
                assert np.array_equal(r, f, equal_nan=True), key
            else:
                assert np.array_equal(r, f), key

        # 逐元素一致 => 重生成 npz 的 hash 亦等于 manifest data_sha256
        digest = hashlib.sha256(tmp_npz.read_bytes()).hexdigest().upper()
        assert digest == _load_manifest()["data_sha256"]

    def test_forward_returns_formula(self):
        """无未来函数语义：forward_returns[t] = price[t+1+h]/price[t+1] - 1。"""
        z = np.load(NPZ)
        price = z["price"]
        fwd = z["forward_returns"]
        h = int(z["h"][0])
        lag = int(z["lag"][0])
        # 契约冻结值硬断言（审查高7：防 fixture/生成器/manifest 集体漂移仍全绿）
        assert h == 5 and lag == 1, f"frozen contract h={h} lag={lag} != 5/1"
        T, N = price.shape

        expected = np.full((T, N), np.nan)
        for t in range(T - (h + lag)):
            entry = price[t + lag]
            exit_ = price[t + h + lag]
            valid = np.isfinite(entry) & np.isfinite(exit_)
            with np.errstate(all="ignore"):
                expected[t, valid] = exit_[valid] / entry[valid] - 1.0
        assert np.array_equal(fwd, expected, equal_nan=True)

    def test_tail_and_halt_nan(self):
        """末尾 h+lag 行（无完整窗口）与停牌单元（入场价 NaN）为 NaN。"""
        z = np.load(NPZ)
        price = z["price"]
        fwd = z["forward_returns"]
        h = int(z["h"][0])
        lag = int(z["lag"][0])

        # 末尾 h+lag 行全 NaN
        assert np.all(np.isnan(fwd[-(h + lag):, :]))

        # 停牌单元：入场价 price[t+lag] 非有限 → fwd[t] NaN
        for t in range(fwd.shape[0] - (h + lag)):
            halt = ~np.isfinite(price[t + lag, :])
            if halt.any():
                assert np.all(np.isnan(fwd[t, halt])), t

        # 至少存在有限标签（非空洞断言）
        assert np.any(np.isfinite(fwd))

    def test_fc_rolling_ic_consumes_labels(self):
        """行对齐冒烟：fc.rolling_ic 消费冻结标签，输出 (T,) 截面 IC 序列，
        且有效行逐值与独立 Spearman 参考一致。"""
        import fc

        z = np.load(NPZ)
        price = z["price"]
        fwd = z["forward_returns"]
        T = price.shape[0]

        out = fc.rolling_ic(price, fwd, min_valid=3, device="cpu")
        assert isinstance(out, np.ndarray)
        assert out.shape == (T,)

        # 抽查每个有效截面：min_valid=3 且两侧秩非退化
        for t in range(T):
            o = np.isfinite(price[t]) & np.isfinite(fwd[t])
            if o.sum() >= 3 and np.ptp(price[t][o]) > 0 and np.ptp(fwd[t][o]) > 0:
                ref = _independent_spearman(price[t][o], fwd[t][o])
                assert out[t] == pytest.approx(ref, abs=1e-12), (t, out[t], ref)
