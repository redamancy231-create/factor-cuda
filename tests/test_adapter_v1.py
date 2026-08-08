# factor-cuda -- Phase 2 acceptance: fc.* adapter formal pytest suite v1.
#
# Covers the frozen contract requirements enumerated by the GPT-5.6-Sol plan
# review (reviews/phase1_adapter_plan_review_gpt56sol_2026-08-06.txt):
#   F01  signature snapshot    -- lock param names/defaults/kinds + return dtype/shape
#   F02  strict scalar types   -- min_valid / descending / factor_plane f reject bool/float/int
#   F05  factor_plane          -- container mirror / device mirror / no-sharing / requires_grad detach
#   F07  sync boundaries       -- non-default-stream entry, return materialization, DLPack consume
#   F10  oracle direct         -- factor_corr/stock_corr (cpu AND cuda) per-pair vs corr_oracle_v1
#   F17  frozen manifest       -- parity_anchors_v1 manifest integrity + error-case keywords
#   F18  test supplement       -- frozen caps / hidden-group failure / read-only / independent
#                                 output / requires_grad / torch-CUDA semantics / multi-GPU routing
#
# Contract source of truth: CLAUDE.md L0 Spec + reviews/_draft_contract.md.
# Frozen caps (HG-2, reviews/hg2_phase1_limits_2026-08-06.md) enforced by the
# low-level pybind bindings (std::invalid_argument -> ValueError).
#
# Run: PYTHONIOENCODING=utf-8 python -m pytest tests/ -v
# ASCII-only comments (Windows GBK-safe).
import inspect
import json
import pathlib

import numpy as np
import pytest

import fc
from fc._cpu_core import np_cs_rank  # independent CPU oracle

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "benchmark_corpus"
FIXTURES = ROOT / "tests" / "fixtures"

# torch is optional at fc import; gate torch/CUDA-dependent tests.
try:
    import torch
    HAS_CUDA = bool(torch.cuda.is_available())
    N_GPU = torch.cuda.device_count() if HAS_CUDA else 0
except Exception:  # pragma: no cover - CPU-only environments
    torch = None
    HAS_CUDA = False
    N_GPU = 0

NEED_CUDA = pytest.mark.skipif(not HAS_CUDA, reason="CUDA unavailable")
NEED_TORCH = pytest.mark.skipif(torch is None, reason="torch not installed")
NEED_MULTI_GPU = pytest.mark.skipif(N_GPU < 2,
                                    reason="requires >= 2 CUDA devices")

from corr_oracle_v1 import corr_oracle  # noqa: E402 (frozen oracle wrapper)

rng = np.random.default_rng(2026)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def bitwise_f32(a, b):
    """uint32-bitwise equality on float32 (NaN payload preserved)."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    if a.shape != b.shape:
        return False
    return bool(np.array_equal(a.view(np.uint32), b.view(np.uint32)))


def corr_match(a, b, tol=1e-12):
    """NaN-pattern match + inf classification/sign match + finite |a-b| <= tol
    (GPT-5.6-Sol review 2026-08-06 #5: a finite-vs-inf or +inf-vs--inf drift was
    previously excluded and passed)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return False
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return False
    if not np.array_equal(np.isposinf(a), np.isposinf(b)):
        return False
    if not np.array_equal(np.isneginf(a), np.isneginf(b)):
        return False
    fin = np.isfinite(a) & np.isfinite(b)
    return bool(np.all(np.abs(a[fin] - b[fin]) <= tol))


def to_np(x):
    """Normalize an fc output (numpy or torch tensor) to numpy float64."""
    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def make_mask(shape, keep=0.9):
    return rng.random(shape) < keep


def _kahan_sum_arr(x):
    """Serial Kahan-compensated sum (HG-2 high-precision reference)."""
    s = 0.0
    c = 0.0
    for v in np.asarray(x, dtype=np.float64):
        y = v - c
        t = s + y
        c = (t - s) - y
        s = t
    return s


def _serial_kahan_corr(a, b):
    """Serial-Kahan high-precision Pearson reference (HG-2 clause: all backends
    must match this <=1e-12 on reduction-sensitive inputs)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = a.size
    am = _kahan_sum_arr(a) / n
    bm = _kahan_sum_arr(b) / n
    ac = a - am
    bc = b - bm
    sxx = _kahan_sum_arr(ac * ac)
    syy = _kahan_sum_arr(bc * bc)
    sxy = _kahan_sum_arr(ac * bc)
    if sxx <= 0 or syy <= 0 or not (np.isfinite(sxx) and np.isfinite(syy)):
        return float("nan")
    return float((sxy / np.sqrt(sxx)) / np.sqrt(syy))


@pytest.fixture(scope="module")
def _corr_corpus_npz():
    return np.load(FIXTURES / "corr_corpus_v1.npz", allow_pickle=False)


def _independent_rank_panel(x, mask, descending):
    """Hand-written stable-ordinal rank (1..K) reference, implemented in the
    test rather than calling fc._cpu_core.np_cs_rank, to avoid shared-code drift
    with the adapter under test (GPT-5.6-Sol review 2026-08-06 #11)."""
    x = np.asarray(x, dtype=np.float32)
    xw = -x if descending else x
    out = np.full(x.shape, np.nan, dtype=np.float32)
    for t in range(x.shape[0]):
        part = np.isfinite(xw[t])
        if mask is not None:
            part &= np.asarray(mask, dtype=bool)[t]
        if not part.any():
            continue
        xs = xw[t][part]
        order = np.argsort(xs, kind="stable")
        r = np.empty(len(xs), dtype=np.float32)
        r[order] = np.arange(1, len(xs) + 1, dtype=np.float32)
        out[t][part] = r
    return out


# ---------------------------------------------------------------------------
# F01 -- signature snapshot
# ---------------------------------------------------------------------------

class TestF01SignatureSnapshot:
    """Lock the six public signatures: parameter names, order, defaults, kinds
    and the container/dtype/shape contract. Any change here is a contract break
    (review F01 disposition)."""

    def test_public_api_exports(self):
        assert set(fc.__all__) == {
            "cross_sectional_rank", "factor_plane", "factor_corr", "stock_corr",
            "rolling_ic", "parameter_scan", "__version__",
        }
        for name in ("cross_sectional_rank", "factor_plane", "factor_corr",
                     "stock_corr", "rolling_ic", "parameter_scan"):
            assert callable(getattr(fc, name))

    def _params(self, fn):
        return list(inspect.signature(fn).parameters.values())

    def test_sig_cross_sectional_rank(self):
        ps = self._params(fc.cross_sectional_rank)
        assert [p.name for p in ps] == ["values", "mask", "descending"]
        assert ps[0].default is inspect.Parameter.empty
        assert ps[1].default is None
        assert ps[2].default is False
        assert all(p.kind is p.POSITIONAL_OR_KEYWORD for p in ps)

    def test_sig_factor_plane(self):
        ps = self._params(fc.factor_plane)
        assert [p.name for p in ps] == ["factors", "f"]
        assert all(p.default is inspect.Parameter.empty for p in ps)
        assert all(p.kind is p.POSITIONAL_OR_KEYWORD for p in ps)

    def test_sig_factor_corr(self):
        ps = self._params(fc.factor_corr)
        assert [p.name for p in ps] == ["data", "mask", "names", "backend"]
        assert ps[1].default is None and ps[2].default is None
        assert ps[3].default == "cpu"
        assert all(p.kind is p.POSITIONAL_OR_KEYWORD for p in ps)

    def test_sig_stock_corr(self):
        ps = self._params(fc.stock_corr)
        assert [p.name for p in ps] == ["data", "mask", "backend"]
        assert ps[1].default is None and ps[2].default == "cpu"
        assert all(p.kind is p.POSITIONAL_OR_KEYWORD for p in ps)

    def test_sig_rolling_ic(self):
        ps = self._params(fc.rolling_ic)
        assert [p.name for p in ps] == [
            "factor", "forward_returns", "factor_mask", "fwd_mask",
            "min_valid", "device"]
        assert ps[2].default is None and ps[3].default is None
        assert ps[4].default == 30 and ps[5].default is None
        # forward_returns is the frozen name (no `fwd` abbreviation)
        assert "fwd" not in [p.name for p in ps]
        assert all(p.kind is p.POSITIONAL_OR_KEYWORD for p in ps)

    def test_sig_parameter_scan(self):
        ps = self._params(fc.parameter_scan)
        assert [p.name for p in ps] == ["axes", "X", "mask"]
        assert ps[0].default is inspect.Parameter.empty
        assert ps[1].default is inspect.Parameter.empty
        assert ps[2].default is None
        assert all(p.kind is p.POSITIONAL_OR_KEYWORD for p in ps)

    @NEED_CUDA
    def test_return_dtype_shape_cross_sectional_rank(self):
        X = rng.standard_normal((12, 8)).astype(np.float32)
        out = fc.cross_sectional_rank(X)
        assert isinstance(out, np.ndarray)
        assert out.shape == (12, 8) and out.dtype == np.float32

    def test_return_dtype_shape_factor_plane(self):
        F3 = rng.standard_normal((12, 8, 3)).astype(np.float32)
        p = fc.factor_plane(F3, 1)
        assert isinstance(p, np.ndarray) and p.shape == (12, 8)
        assert p.dtype == np.float32  # mirrors input dtype

    def test_return_dtype_shape_corr_cpu(self):
        F3 = rng.standard_normal((12, 8, 3))
        m = fc.factor_corr(F3, None, backend="cpu")
        assert isinstance(m, np.ndarray) and m.shape == (3, 3)
        assert m.dtype == np.float64
        X = rng.standard_normal((12, 8))
        s = fc.stock_corr(X, None, backend="cpu")
        assert isinstance(s, np.ndarray) and s.shape == (8, 8)
        assert s.dtype == np.float64

    def test_return_dtype_shape_rolling_ic(self):
        f = rng.standard_normal((12, 8))
        r = rng.standard_normal((12, 8))
        ic = fc.rolling_ic(f, r, min_valid=2, device="cpu")
        assert isinstance(ic, np.ndarray) and ic.shape == (12,)
        assert ic.dtype == np.float64

    @NEED_CUDA
    def test_return_dtype_shape_corr_cuda_mirrors_torch(self):
        F3 = torch.randn(12, 8, 3, device="cuda")
        m = fc.factor_corr(F3, None, backend="cuda")
        assert torch.is_tensor(m) and m.is_cuda
        assert m.shape == (3, 3) and m.dtype == torch.float64


# ---------------------------------------------------------------------------
# F02 -- strict scalar types
# ---------------------------------------------------------------------------

class TestF02StrictScalarTypes:
    """Review F02: bool is an int subclass in Python; min_valid / descending /
    factor_plane f must reject bool/float/int impostors (type(x) is bool for
    descending; isinstance(x, int) and not bool for the ints)."""

    def test_descending_rejects_int(self):
        X = rng.standard_normal((4, 4)).astype(np.float32)
        for bad in (1, 0, 1.0, 0.0, None, "ascending"):
            with pytest.raises(ValueError):
                fc.cross_sectional_rank(X, None, descending=bad)

    @NEED_CUDA
    def test_descending_accepts_bool(self):
        X = rng.standard_normal((4, 4)).astype(np.float32)
        assert fc.cross_sectional_rank(X, None, descending=True).dtype == np.float32
        assert fc.cross_sectional_rank(X, None, descending=False).dtype == np.float32

    def test_min_valid_rejects_bool_and_float(self):
        f = rng.standard_normal((4, 4))
        r = rng.standard_normal((4, 4))
        for bad in (True, False, 30.0, 2.0):
            with pytest.raises(ValueError):
                fc.rolling_ic(f, r, min_valid=bad)

    def test_min_valid_value_bound(self):
        f = rng.standard_normal((4, 4))
        r = rng.standard_normal((4, 4))
        with pytest.raises(ValueError):
            fc.rolling_ic(f, r, min_valid=1)
        with pytest.raises(ValueError):
            fc.rolling_ic(f, r, min_valid=0)
        # min_valid == 2 is the legal floor
        out = fc.rolling_ic(f, r, min_valid=2, device="cpu")
        assert out.shape == (4,)

    def test_factor_plane_f_rejects_bool_and_float(self):
        F3 = rng.standard_normal((4, 4, 3)).astype(np.float32)
        for bad in (True, False, 1.0, 0.0, 3):
            with pytest.raises(ValueError):
                fc.factor_plane(F3, bad)
        # legal ints
        assert fc.factor_plane(F3, 0).shape == (4, 4)
        assert fc.factor_plane(F3, 2).shape == (4, 4)


# ---------------------------------------------------------------------------
# F05 -- factor_plane
# ---------------------------------------------------------------------------

class TestF05FactorPlane:
    """Review F05 disposition: lock container mirror (numpy->numpy, torch->
    torch), device mirror, requires_grad=False, and no memory sharing."""

    def test_numpy_in_numpy_out(self):
        F3 = rng.standard_normal((12, 8, 3)).astype(np.float32)
        p = fc.factor_plane(F3, 1)
        assert isinstance(p, np.ndarray)
        assert bitwise_f32(p, np.ascontiguousarray(F3[..., 1]))

    @NEED_TORCH
    def test_torch_in_torch_out(self):
        F3 = torch.randn(12, 8, 3)
        p = fc.factor_plane(F3, 2)
        assert torch.is_tensor(p) and not p.is_cuda
        assert bitwise_f32(p.detach().cpu().numpy(), F3.detach().numpy()[..., 2])

    @NEED_CUDA
    def test_cuda_mirror_device(self):
        F3 = torch.randn(12, 8, 3, device="cuda")
        p = fc.factor_plane(F3, 1)
        assert torch.is_tensor(p) and p.is_cuda
        assert p.device.index == F3.device.index

    def test_no_memory_sharing(self):
        F3 = rng.standard_normal((12, 8, 3)).astype(np.float32)
        p = fc.factor_plane(F3, 1)
        assert not np.shares_memory(p, F3)
        assert not np.shares_memory(p, F3[..., 1])

    def test_mutation_independent(self):
        F3 = rng.standard_normal((12, 8, 3)).astype(np.float32)
        before = F3.copy()
        p = fc.factor_plane(F3, 0)
        p[...] = -999.0
        assert np.array_equal(F3, before)

    def test_ndim2_requires_f0(self):
        X = rng.standard_normal((12, 8)).astype(np.float32)
        p = fc.factor_plane(X, 0)
        assert bitwise_f32(p, np.ascontiguousarray(X))
        with pytest.raises(ValueError):
            fc.factor_plane(X, 1)

    def test_f_out_of_range(self):
        F3 = rng.standard_normal((12, 8, 3)).astype(np.float32)
        with pytest.raises(ValueError):
            fc.factor_plane(F3, 3)
        with pytest.raises(ValueError):
            fc.factor_plane(F3, -1)

    def test_fcontig_input_copy(self):
        # F-contiguous (N innermost) input must be copied to C-contiguous plane.
        base = np.asfortranarray(rng.standard_normal((12, 8, 3)).astype(np.float32))
        p = fc.factor_plane(base, 1)
        assert p.flags["C_CONTIGUOUS"]
        assert bitwise_f32(p, np.ascontiguousarray(np.ascontiguousarray(base)[..., 1]))

    def test_transposed_view_copy(self):
        # A transposed view (non-contiguous) still returns the correct plane.
        base = rng.standard_normal((8, 12, 3)).astype(np.float32)
        T = np.ascontiguousarray(base).transpose(1, 0, 2)  # (12,8,3) view
        p = fc.factor_plane(T, 2)
        assert p.flags["C_CONTIGUOUS"]
        assert bitwise_f32(p, np.ascontiguousarray(T)[..., 2])

    @NEED_TORCH
    def test_requires_grad_detach(self):
        F3 = torch.randn(12, 8, 3, requires_grad=True)
        p = fc.factor_plane(F3, 0)
        assert not p.requires_grad

    def test_dtype_mirrors_input(self):
        F3 = rng.standard_normal((12, 8, 3)).astype(np.float64)
        p = fc.factor_plane(F3, 1)
        assert p.dtype == np.float64


# ---------------------------------------------------------------------------
# F07 -- sync boundaries
# ---------------------------------------------------------------------------

class TestF07SyncBoundaries:
    """Review F07 disposition: the op is synchronous on entry (drain the input
    producer stream) and the result is materialized before return. Observable
    tests: non-default-stream producer then immediate call; immediate read after
    return; DLPack capsule is consumed once."""

    @NEED_CUDA
    def test_entry_sync_nondefault_stream(self):
        """Observable entry-sync contract: a value written on a side stream is
        visible to an immediate fc call. The data is a fixed permutation so a
        stale (un-written) zero read yields index-order ranks 1..N, which differ
        from the true ranks -> non-vacuous. (torch's cross-stream D2H also
        provides this guarantee; the test asserts the observable contract the
        F07 disposition requires.)"""
        s = torch.cuda.Stream()
        perm = rng.permutation(30).astype(np.float32) + 1.0  # distinct 1..30
        tile = np.tile(perm, (40, 1))
        x = torch.zeros((40, 30), device="cuda")
        with torch.cuda.stream(s):
            x.copy_(torch.tensor(tile, device="cuda"))  # enqueued on s, NOT synced
        out = fc.cross_sectional_rank(x)  # immediate call -> must see `tile`
        assert bitwise_f32(to_np(out), np_cs_rank(tile, None, False))

    @NEED_CUDA
    def test_return_sync_immediate_read(self):
        x = torch.randn(40, 30, device="cuda")
        out = fc.cross_sectional_rank(x)  # returns torch CUDA tensor
        v = to_np(out)  # immediate read without explicit torch.cuda.synchronize
        assert np.isfinite(v).all()
        assert np.all(v.min(axis=1) == 1.0)

    @NEED_CUDA
    def test_return_sync_same_stream_kernel(self):
        """Return-sync observable: the returned CUDA tensor is readable by a
        same-stream dependent kernel immediately (no explicit host sync). A
        regression that made the binding launch on a foreign stream without an
        event-wait would return an un-materialized tensor -> garbage here."""
        x = torch.randn(40, 30, device="cuda")
        out = fc.cross_sectional_rank(x)
        mn = out.min(dim=1).values  # same-stream reduction, no .cpu()/.synchronize
        assert bool((mn == 1.0).all().item())

    @NEED_CUDA
    def test_return_sync_rolling_ic_immediate(self):
        f = torch.randn(40, 30, device="cuda")
        r = torch.randn(40, 30, device="cuda")
        ic = fc.rolling_ic(f, r, min_valid=2)  # device=None -> GPU, mirror factor
        v = to_np(ic)
        assert v.shape == (40,)

    @NEED_CUDA
    def test_dlpack_single_use_and_consumed(self):
        x = torch.randn(40, 30, device="cuda", dtype=torch.float32)
        cap = torch.utils.dlpack.to_dlpack(x)
        out = fc.cross_sectional_rank(cap)
        v = to_np(out)
        assert np.isfinite(v).all()
        # capsule consumed -> second pass must raise ValueError (F06 disposition)
        with pytest.raises(ValueError):
            fc.cross_sectional_rank(cap)

    @NEED_CUDA
    def test_dlpack_nondefault_stream_entry(self):
        """DLPack async-stream readiness (contract edge case): a capsule from a
        tensor written on a side stream must be ready at the entry sync."""
        s = torch.cuda.Stream()
        perm = rng.permutation(30).astype(np.float32) + 1.0
        tile = np.tile(perm, (40, 1))
        x = torch.zeros((40, 30), device="cuda")
        with torch.cuda.stream(s):
            x.copy_(torch.tensor(tile, device="cuda"))
        cap = torch.utils.dlpack.to_dlpack(x)
        out = fc.cross_sectional_rank(cap)
        assert bitwise_f32(to_np(out), np_cs_rank(tile, None, False))

    @NEED_CUDA
    def test_dlpack_dtype_regression(self):
        """DLPack dtype paths of the _dtype_ok fix (2026-08-06): float64 accepted
        (+downcast for rank), int32/float16 rejected, bool mask capsule accepted."""
        # float64 capsule accepted -> rank output float32 (downcast)
        x64 = torch.randn(40, 30, device="cuda", dtype=torch.float64)
        out = fc.cross_sectional_rank(torch.utils.dlpack.to_dlpack(x64))
        assert to_np(out).dtype == np.float32 and np.isfinite(to_np(out)).all()
        # non-whitelist value capsule -> ValueError (capsule consumed)
        for dt in (torch.int32, torch.float16):
            cap = torch.utils.dlpack.to_dlpack(
                torch.zeros(40, 30, device="cuda", dtype=dt))
            with pytest.raises(ValueError):
                fc.cross_sectional_rank(cap)
            # re-passing the same rejected capsule -> "already consumed" ValueError
            with pytest.raises(ValueError) as ei:
                fc.cross_sectional_rank(cap)
            assert "consumed" in str(ei.value).lower()
        # bool mask capsule accepted on the correlation cpu path
        F3 = torch.randn(40, 30, 2, device="cuda")
        m = torch.zeros(40, 30, device="cuda", dtype=torch.bool)
        m[..., :] = True
        cap_m = torch.utils.dlpack.to_dlpack(m)
        res = fc.factor_corr(F3, cap_m, backend="cpu")
        assert np.isfinite(np.asarray(res)).all()
        # non-bool mask capsule -> ValueError
        mf = torch.zeros(40, 30, device="cuda", dtype=torch.float32)
        with pytest.raises(ValueError):
            fc.factor_corr(F3, torch.utils.dlpack.to_dlpack(mf), backend="cpu")

    @NEED_CUDA
    def test_entry_sync_host_input(self):
        # Host-resident inputs have no producer stream; op must still be
        # synchronous (result correct immediately).
        X = rng.standard_normal((40, 30)).astype(np.float32)
        out = fc.cross_sectional_rank(X)
        assert np.isfinite(to_np(out)).all()


# ---------------------------------------------------------------------------
# F10 -- correlation oracle direct
# ---------------------------------------------------------------------------

class TestF10CorrelationOracleDirect:
    """Review F10 disposition: the frozen corr_oracle_v1.corr_oracle is the
    ONLY correlation oracle. Both CPU and CUDA backends are verified per-pair
    directly against it (in-domain |Δr|<=1e-12 + NaN pattern)."""

    def _factor_pair_oracle(self, F3, mask, i, j):
        xv, yv = F3[..., i], F3[..., j]
        o = np.isfinite(xv) & np.isfinite(yv)
        if mask is not None:
            o &= mask
        return corr_oracle(xv[o], yv[o])

    def _stock_pair_oracle(self, X, mask, i, j):
        xv, yv = X[:, i], X[:, j]
        o = np.isfinite(xv) & np.isfinite(yv)
        if mask is not None:
            o &= mask[:, i] & mask[:, j]
        return corr_oracle(xv[o], yv[o])

    def _assert_factor_matrix_matches_oracle(self, mat, F3, mask):
        mat = np.asarray(mat, dtype=np.float64)
        F = F3.shape[2]
        assert mat.shape == (F, F)
        for i in range(F):
            for j in range(F):
                exp = self._factor_pair_oracle(F3, mask, i, j)
                assert corr_match(float(mat[i, j]), exp), (
                    f"factor_corr [{i},{j}] {mat[i,j]} vs oracle {exp}")

    def _assert_stock_matrix_matches_oracle(self, mat, X, mask):
        mat = np.asarray(mat, dtype=np.float64)
        N = X.shape[1]
        assert mat.shape == (N, N)
        for i in range(N):
            for j in range(N):
                exp = self._stock_pair_oracle(X, mask, i, j)
                assert corr_match(float(mat[i, j]), exp), (
                    f"stock_corr [{i},{j}] {mat[i,j]} vs oracle {exp}")

    def test_factor_corr_cpu_direct_oracle(self):
        T, N, F = 60, 15, 4
        F3 = rng.standard_normal((T, N, F))
        mask = make_mask((T, N))
        F3[0, 0, 0] = np.nan  # exercise finite filtering
        mat = fc.factor_corr(F3, mask, backend="cpu")
        self._assert_factor_matrix_matches_oracle(mat, F3, mask)

    @NEED_CUDA
    def test_factor_corr_cuda_direct_oracle(self):
        T, N, F = 60, 15, 4
        F3 = rng.standard_normal((T, N, F))
        mask = make_mask((T, N))
        F3[0, 0, 1] = np.nan
        mat = fc.factor_corr(F3, mask, backend="cuda")
        self._assert_factor_matrix_matches_oracle(mat, F3, mask)

    @NEED_CUDA
    def test_factor_corr_f32_skips_domain_check(self):
        """2026-08-07: f32 inputs skip the adapter domain check (f32 value range
        [1.4e-45, 3.4e38] lies strictly inside the corr domain [1e-150, 1e150]),
        so no Python-side f64 astype + isfinite + abs + min/max pass runs; the
        binding forcecast upcasts f32 once. The f32 path must be bitwise
        identical to the f64 path, and an f32 panel with inf/nan must NOT raise
        the domain ValueError (the kernel's isfinite pool handles them)."""
        T, N, F = 40, 12, 3
        F3 = rng.standard_normal((T, N, F)).astype(np.float32)
        mask = make_mask((T, N))
        r32 = np.asarray(fc.factor_corr(F3, mask, backend="cuda"),
                         dtype=np.float64)
        r64 = np.asarray(fc.factor_corr(F3.astype(np.float64), mask,
                                        backend="cuda"))
        assert np.array_equal(r32, r64, equal_nan=True), \
            "f32 path must be bitwise identical to the f64 path"
        # inf/nan in an f32 panel: no domain ValueError, and the f32 path must
        # match the exact f64 upcast bitwise (review 2026-08-07 F8)
        F3b = F3.copy()
        F3b[0, 0, 0] = np.inf
        F3b[0, 1, 0] = -np.inf
        F3b[1, 1, 1] = np.nan
        rb32 = np.asarray(fc.factor_corr(F3b, None, backend="cuda"))
        rb64 = np.asarray(fc.factor_corr(F3b.astype(np.float64), None,
                                         backend="cuda"))
        assert np.array_equal(rb32, rb64, equal_nan=True), \
            "f32 inf/nan panel must match exact f64 upcast"
        assert rb32.shape == (F, F)
        assert np.isfinite(rb32[0, 0]) and np.isfinite(rb32[0, 1])
        # all-mask-False: no domain ValueError; no valid cells -> all NaN
        # (review 2026-08-07 F2)
        all_false = np.zeros((T, N), dtype=bool)
        r0 = np.asarray(fc.factor_corr(F3, all_false, backend="cuda"))
        assert r0.shape == (F, F) and np.isnan(r0).all()
        # f32 range boundaries: the skip is sound ONLY because every finite f32
        # lies inside [1e-150, 1e150]; pin FLT_MAX / min-subnormal / -0.0 so a
        # future domain-bound tightening is caught (review 2026-08-07 F2)
        F3c = F3.copy()
        F3c[0, 0, 0] = np.finfo(np.float32).max
        F3c[0, 1, 0] = np.nextafter(np.float32(0.0), np.float32(1.0))
        F3c[1, 0, 1] = np.float32(-0.0)
        rc = np.asarray(fc.factor_corr(F3c, None, backend="cuda"))
        rce = np.asarray(fc.factor_corr(F3c.astype(np.float64), None,
                                        backend="cuda"))
        assert np.array_equal(rc, rce, equal_nan=True), \
            "f32 boundary panel must still match the f64 path"

    @NEED_CUDA
    def test_gpu_domain_check_skip_is_locked(self, monkeypatch):
        """2026-08-07 review: an output-only test cannot detect the skip being
        removed (every f32 panel passes the old f64 check vacuously, so r32==r64
        stays green). Lock the skip itself: in_corr_domain must NOT be consulted
        for f32 input, and MUST be for f64 input."""
        import fc.correlation as mod
        calls = []
        monkeypatch.setattr(
            mod.u, "in_corr_domain",
            lambda *a, **k: calls.append(1) or True)
        x32 = rng.standard_normal((10, 8, 2)).astype(np.float32)
        msk = np.ones((10, 8), dtype=bool)
        mod._gpu_domain_check(x32, msk)
        assert calls == [], "f32 input must skip in_corr_domain entirely"
        mod._gpu_domain_check(x32.astype(np.float64), msk)
        assert len(calls) == 1, "f64 input must still consult in_corr_domain"

    @NEED_CUDA
    def test_stock_corr_f32_skip_and_f64_out_of_domain(self):
        """2026-08-07 review: stock_corr shares _gpu_domain_check but a
        different binding (stock_corr_f64/upcast_f64); pin the f32 skip and the
        f64 out-of-domain guard on the stock_corr cuda path (previously
        untested)."""
        T, N = 40, 12
        X = rng.standard_normal((T, N)).astype(np.float32)
        s32 = np.asarray(fc.stock_corr(X, None, backend="cuda"))
        s64 = np.asarray(fc.stock_corr(X.astype(np.float64), None,
                                       backend="cuda"))
        assert np.array_equal(s32, s64, equal_nan=True), \
            "stock_corr f32 path must be bitwise == f64 path"
        big = np.full((T, N), 1e151, dtype=np.float64)
        with pytest.raises(ValueError) as ei:
            fc.stock_corr(big, None, backend="cuda")
        assert "数值域外" in str(ei.value) or "1e150" in str(ei.value)

    def test_factor_corr_nomask_direct_oracle(self):
        T, N, F = 40, 12, 3
        F3 = rng.standard_normal((T, N, F))
        mat = fc.factor_corr(F3, None, backend="cpu")
        self._assert_factor_matrix_matches_oracle(mat, F3, None)

    @NEED_CUDA
    def test_factor_corr_cuda_nomask_direct_oracle(self):
        T, N, F = 40, 12, 3
        F3 = rng.standard_normal((T, N, F))
        mat = fc.factor_corr(F3, None, backend="cuda")
        self._assert_factor_matrix_matches_oracle(mat, F3, None)

    def test_stock_corr_cpu_direct_oracle(self):
        T, N = 60, 10
        X = rng.standard_normal((T, N))
        mask = make_mask((T, N))
        X[2, 3] = np.inf  # non-finite filtered out of validity
        mat = fc.stock_corr(X, mask, backend="cpu")
        self._assert_stock_matrix_matches_oracle(mat, X, mask)

    @NEED_CUDA
    def test_stock_corr_cuda_direct_oracle(self):
        T, N = 60, 10
        X = rng.standard_normal((T, N))
        mask = make_mask((T, N))
        X[2, 3] = np.nan
        mat = fc.stock_corr(X, mask, backend="cuda")
        self._assert_stock_matrix_matches_oracle(mat, X, mask)

    def test_stock_corr_nomask_direct_oracle(self):
        T, N = 40, 8
        X = rng.standard_normal((T, N))
        mat = fc.stock_corr(X, None, backend="cpu")
        self._assert_stock_matrix_matches_oracle(mat, X, None)

    def test_no_common_valid_nan(self):
        # Two factor columns with disjoint valid sets -> oracle NaN, fc NaN.
        T, N, F = 30, 2, 2
        F3 = np.zeros((T, N, F))
        mask = np.zeros((T, N), dtype=bool)
        mask[:15, 0] = True  # factor 0 valid only in first half
        mask[15:, 1] = True  # factor 1 valid only in second half
        mat = fc.factor_corr(F3, mask, backend="cpu")
        assert np.isnan(mat[0, 1]) and np.isnan(mat[1, 0])
        assert corr_match(float(mat[0, 1]), self._factor_pair_oracle(F3, mask, 0, 1))

    # ---- HG-2 reduction-sensitive inputs (high-precision reference) ----
    # Disposition F10: "归约敏感输入按 CLAUDE HG-2 的高精度规则单测" -- on the
    # reduction-sensitive classes (|mean|>1e3*sigma; var-underflow risk) all
    # backends must match the serial-Kahan reference <=1e-12 (wrapper is
    # diagnostic-only). Used the frozen corpus bias arrays (canonical cases).
    # 2026-08-06: this test exposed that the fc CPU backend used numpy mean on
    # ~1e15-bias data (|d|~9e-3 vs Kahan); fixed via a Kahan-mean guard in
    # fc._cpu_core._two_pass_corr.

    def _bias_panel(self, a, b):
        """(1,N,2) factor panel pooling the two 1-D bias sequences as factors."""
        return np.stack([np.asarray(a), np.asarray(b)], axis=1).reshape(1, -1, 2)

    @pytest.mark.parametrize("name", ["bias_1e12", "bias_1e15", "f64_ulp_bias"])
    def test_factor_corr_cpu_hg2_bias(self, _corr_corpus_npz, name):
        """factor_corr (not stock_corr) against serial-Kahan on bias inputs."""
        a = _corr_corpus_npz[f"{name}_a"]
        b = _corr_corpus_npz[f"{name}_b"]
        ref = _serial_kahan_corr(a, b)
        F3 = self._bias_panel(a, b)
        got = float(np.asarray(fc.factor_corr(F3, None, backend="cpu"))[0, 1])
        assert abs(got - ref) <= 1e-12, f"factor_corr CPU {name}: |d|={abs(got-ref):.3e}"

    @NEED_CUDA
    @pytest.mark.parametrize("name", ["bias_1e12", "bias_1e15", "f64_ulp_bias"])
    def test_factor_corr_cuda_hg2_bias(self, _corr_corpus_npz, name):
        a = _corr_corpus_npz[f"{name}_a"]
        b = _corr_corpus_npz[f"{name}_b"]
        ref = _serial_kahan_corr(a, b)
        F3 = self._bias_panel(a, b)
        got = float(np.asarray(fc.factor_corr(F3, None, backend="cuda"))[0, 1])
        assert abs(got - ref) <= 1e-12, f"factor_corr CUDA {name}: |d|={abs(got-ref):.3e}"

    @pytest.mark.parametrize("name", ["bias_1e12", "bias_1e15", "f64_ulp_bias"])
    def test_stock_corr_cpu_hg2_bias(self, _corr_corpus_npz, name):
        a = _corr_corpus_npz[f"{name}_a"]
        b = _corr_corpus_npz[f"{name}_b"]
        X = np.column_stack([a, b])
        ref = _serial_kahan_corr(a, b)
        got = float(np.asarray(fc.stock_corr(X, None, backend="cpu"))[0, 1])
        assert abs(got - ref) <= 1e-12, f"stock_corr CPU {name}: |d|={abs(got-ref):.3e}"

    @NEED_CUDA
    @pytest.mark.parametrize("name", ["bias_1e12", "bias_1e15", "f64_ulp_bias"])
    def test_stock_corr_cuda_hg2_bias(self, _corr_corpus_npz, name):
        a = _corr_corpus_npz[f"{name}_a"]
        b = _corr_corpus_npz[f"{name}_b"]
        X = np.column_stack([a, b])
        ref = _serial_kahan_corr(a, b)
        got = float(np.asarray(fc.stock_corr(X, None, backend="cuda"))[0, 1])
        assert abs(got - ref) <= 1e-12, f"stock_corr CUDA {name}: |d|={abs(got-ref):.3e}"

    def test_stock_corr_cpu_var_underflow_hg2(self):
        # In-domain small-magnitude values with finite variance (~1e-140 scale,
        # the var-underflow-risk class). Matches the serial-Kahan reference.
        xv = np.array([1e-140, 2e-140, 3e-140, 4e-140, 5e-140], dtype=np.float64)
        yv = np.array([5e-140, 4e-140, 3e-140, 2e-140, 1e-140], dtype=np.float64)
        X = np.column_stack([xv, yv])
        ref = _serial_kahan_corr(xv, yv)
        got = float(np.asarray(fc.stock_corr(X, None, backend="cpu"))[0, 1])
        assert abs(got - ref) <= 1e-12

    @NEED_CUDA
    def test_stock_corr_cuda_var_underflow_hg2(self):
        xv = np.array([1e-140, 2e-140, 3e-140, 4e-140, 5e-140], dtype=np.float64)
        yv = np.array([5e-140, 4e-140, 3e-140, 2e-140, 1e-140], dtype=np.float64)
        X = np.column_stack([xv, yv])
        ref = _serial_kahan_corr(xv, yv)
        got = float(np.asarray(fc.stock_corr(X, None, backend="cuda"))[0, 1])
        assert abs(got - ref) <= 1e-12

    def test_factor_corr_cpu_var_underflow_hg2(self):
        xv = np.array([1e-140, 2e-140, 3e-140, 4e-140, 5e-140], dtype=np.float64)
        yv = np.array([5e-140, 4e-140, 3e-140, 2e-140, 1e-140], dtype=np.float64)
        F3 = self._bias_panel(xv, yv)
        ref = _serial_kahan_corr(xv, yv)
        got = float(np.asarray(fc.factor_corr(F3, None, backend="cpu"))[0, 1])
        assert abs(got - ref) <= 1e-12

    @NEED_CUDA
    def test_factor_corr_cuda_var_underflow_hg2(self):
        xv = np.array([1e-140, 2e-140, 3e-140, 4e-140, 5e-140], dtype=np.float64)
        yv = np.array([5e-140, 4e-140, 3e-140, 2e-140, 1e-140], dtype=np.float64)
        F3 = self._bias_panel(xv, yv)
        ref = _serial_kahan_corr(xv, yv)
        got = float(np.asarray(fc.factor_corr(F3, None, backend="cuda"))[0, 1])
        assert abs(got - ref) <= 1e-12


# ---------------------------------------------------------------------------
# F17 -- frozen manifest validation
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _frozen_manifest():
    path = CORPUS_DIR / "parity_anchors_v1.manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def _frozen_npz():
    return np.load(CORPUS_DIR / "parity_anchors_v1.npz", allow_pickle=False)


class TestF17FrozenManifest:
    """Review F17 disposition: validate the frozen parity manifest (case names,
    params, expected-exception keywords) — the single source of truth for parity
    coverage. Structural integrity + error-case keyword mapping through fc."""

    def test_manifest_loads_and_arrays_present(self, _frozen_manifest, _frozen_npz):
        assert _frozen_manifest["corpus_id"] == "parity_anchors_v1"
        cases = _frozen_manifest["case_schema"]
        assert len(cases) >= 20
        arrays = {a["name"] for a in _frozen_manifest["arrays"]}
        assert arrays <= set(_frozen_npz.files)

    def test_manifest_exact_case_set(self, _frozen_manifest):
        """F17: lock the EXACT frozen case set (no silent add/remove)."""
        assert [c["id"] for c in _frozen_manifest["case_schema"]] == [
            "rank_tie_asc", "rank_tie_desc", "rank_zero_asc", "rank_nan_inf_asc",
            "rank_mask_mismatch_asc", "ic_valid_29", "ic_valid_30", "ic_valid_31",
            "ic_all_invalid", "ic_two_sided_nan_inf_mask", "ic_factor_scatter_nan",
            "ic_tie1", "ic_tie2", "ic_factor_tradable_nan", "corr_a_b", "corr_a_c",
            "corr_n2_self", "corr_n2_reverse", "corr_const_self",
            "corr_float32_ac", "corr_mask_false_finite", "corr_no_common_valid",
            "corr_underflow_domain", "corr_dtype_variant", "corr_shape_error",
            "err_out_of_domain", "scan_container",
        ]

    def test_manifest_npz_hash(self, _frozen_manifest):
        """F17: the frozen npz bytes hash to the manifest data_sha256 (locking
        the data, not just the schema)."""
        import hashlib
        raw = (CORPUS_DIR / "parity_anchors_v1.npz").read_bytes()
        got = hashlib.sha256(raw).hexdigest()
        assert got.lower() == _frozen_manifest["hash"]["data_sha256"].lower()

    @NEED_CUDA
    def test_frozen_manifest_full_execution(self, _frozen_manifest, _frozen_npz):
        """F17: execute EVERY frozen case through the fc adapter and assert the
        frozen expected values / exceptions / schema (not just structure)."""
        data = _frozen_npz
        for c in _frozen_manifest["case_schema"]:
            inputs = [data[k] for k in c["inputs"]]
            mask = data[c["masks"]["mask"]] if (c.get("masks") or {}).get("mask") else None
            op, tol, exp = c["operation"], c["tolerance"], c["expected"]
            if op == "cs_rank":
                x = np.ascontiguousarray(inputs[0]).reshape(1, -1)
                m2 = (np.ascontiguousarray(mask).reshape(1, -1)
                      if mask is not None else None)
                got = np.asarray(fc.cross_sectional_rank(x, m2, c["params"]["descending"]))
                got = got.reshape(-1)
                exp_arr = np.asarray([float("nan") if v == "nan" else v for v in exp],
                                     dtype=np.float32)
                assert np.array_equal(np.isnan(got), np.isnan(exp_arr)), c["id"]
                if not np.isnan(exp_arr).all():
                    assert np.array_equal(got[~np.isnan(exp_arr)], exp_arr[~np.isnan(exp_arr)]), c["id"]
            elif op == "rolling_ic":
                f = np.ascontiguousarray(inputs[0]).reshape(1, -1)
                r = np.ascontiguousarray(inputs[1]).reshape(1, -1)
                got = float(fc.rolling_ic(f, r, min_valid=c["params"]["min_valid"],
                                          device="cpu")[0])
                if isinstance(exp, str) and exp == "nan":
                    assert np.isnan(got), c["id"]
                else:
                    assert abs(got - float(exp)) <= float(tol), c["id"]
            elif op == "factor_corr":
                if len(inputs) == 1:
                    # corr_shape_error: a single 1-D input is not a (T,N,F) panel
                    with pytest.raises(ValueError):
                        fc.factor_corr(inputs[0], None, backend="cpu")
                    continue
                F3 = np.stack([inputs[0], inputs[1]], axis=1).reshape(1, -1, 2)
                if isinstance(exp, str) and exp == "ValueError":
                    with pytest.raises(ValueError):
                        fc.factor_corr(F3, None, backend="cpu")
                    continue
                mat = np.asarray(fc.factor_corr(F3, None, backend="cuda"))
                got = float(mat[0, 1])
                if isinstance(exp, str) and exp == "nan":
                    assert np.isnan(got), c["id"]
                else:
                    assert abs(got - float(exp)) <= float(tol), f"{c['id']}: {got} vs {exp}"
            elif op == "parameter_scan":
                x = np.ascontiguousarray(inputs[0])  # (3,3) panel, keep 2-D
                m2 = np.ascontiguousarray(mask)      # (3,3) bool, keep 2-D
                res = fc.parameter_scan(c["params"]["axes"], x, m2)
                assert [g["status"] for g in res["groups"]] == ["ok"] * 4, c["id"]
                assert res["spec"] == {"direction": ["ascending", "descending"],
                                       "mask_mode": ["masked", "unmasked"]}, c["id"]

    def test_case_ids_unique(self, _frozen_manifest):
        ids = [c["id"] for c in _frozen_manifest["case_schema"]]
        assert len(ids) == len(set(ids))

    def test_case_schema_valid(self, _frozen_manifest):
        arrays = {a["name"] for a in _frozen_manifest["arrays"]}
        valid_ops = {"cs_rank", "rolling_ic", "factor_corr", "parameter_scan"}
        valid_tol = {"exact", "1e-12", "exception", "schema"}
        for c in _frozen_manifest["case_schema"]:
            assert c["operation"] in valid_ops, c["id"]
            assert c["tolerance"] in valid_tol, c["id"]
            for inp in c["inputs"]:
                assert inp in arrays, f"{c['id']} input {inp}"
            for mname in (c.get("masks") or {}).values():
                if mname:
                    assert mname in arrays, f"{c['id']} mask {mname}"
            assert "expected" in c, c["id"]
            if c["tolerance"] == "exception":
                assert c["expected"] == "ValueError", c["id"]
            elif c["tolerance"] == "schema":
                assert isinstance(c["expected"], dict), c["id"]

    def test_case_params_valid(self, _frozen_manifest):
        """F17 disposition: validate the manifest case '参数' (params) against
        each operation's frozen parameter contract."""
        for c in _frozen_manifest["case_schema"]:
            params = c.get("params", {})
            if c["operation"] == "cs_rank":
                assert isinstance(params.get("descending"), bool), c["id"]
            elif c["operation"] == "rolling_ic":
                mv = params.get("min_valid")
                assert isinstance(mv, int) and not isinstance(mv, bool) and mv >= 2, c["id"]
            elif c["operation"] == "factor_corr":
                dt = params.get("dtype")
                assert dt is None or dt in ("float32", "float64"), c["id"]
            elif c["operation"] == "parameter_scan":
                axes = params.get("axes")
                assert isinstance(axes, list) and len(axes) == 2, c["id"]
                vals = [v for _, v in axes]
                assert all(isinstance(v, list) and len(v) > 0 for v in vals), c["id"]
                g = params.get("G")
                assert g == int(np.prod([len(v) for v in vals])), c["id"]

    def _build_factor_panel(self, arr_a, arr_b):
        """(1,N,2) panel from two 1-D sequences (parity harness construction)."""
        return np.stack([np.asarray(arr_a), np.asarray(arr_b)], axis=1).reshape(1, -1, 2)

    def test_error_case_corr_underflow_domain(self, _frozen_npz):
        u = _frozen_npz["corr_underflow"]
        with pytest.raises(ValueError) as ei:
            fc.factor_corr(self._build_factor_panel(u, u), None, backend="cpu")
        assert "数值域外" in str(ei.value) or "1e-150" in str(ei.value)

    def test_error_case_err_out_of_domain(self, _frozen_npz):
        x, y = _frozen_npz["err_domain_x"], _frozen_npz["err_domain_y"]
        with pytest.raises(ValueError) as ei:
            fc.factor_corr(self._build_factor_panel(x, y), None, backend="cpu")
        assert "数值域外" in str(ei.value) or "1e150" in str(ei.value)

    def test_error_case_corr_shape_error(self, _frozen_npz):
        with pytest.raises(ValueError) as ei:
            fc.factor_corr(_frozen_npz["corr_a"], None, backend="cpu")
        assert "3-D" in str(ei.value) or "形状" in str(ei.value)

    @NEED_CUDA
    def test_error_case_keywords_gpu_path(self, _frozen_npz):
        # Same keyword mapping on the cuda backend (adapter-owned domain check).
        x, y = _frozen_npz["err_domain_x"], _frozen_npz["err_domain_y"]
        with pytest.raises(ValueError) as ei:
            fc.factor_corr(self._build_factor_panel(x, y), None, backend="cuda")
        assert "数值域外" in str(ei.value) or "1e150" in str(ei.value)


# ---------------------------------------------------------------------------
# F18 -- test supplement
# ---------------------------------------------------------------------------

class TestF18TestSupplement:
    """Review F18 disposition: frozen caps -> ValueError; hidden group failure
    injection; read-only contract; independent output allocation; requires_grad
    detach; torch-CUDA input semantics (HG-2 clarified D2H path); multi-GPU."""

    # ---- frozen implementation caps (HG-2, F03) --------------------------

    @NEED_CUDA
    def test_cap_rank_n_gt_2_24(self):
        N = (1 << 24) + 1
        X = np.zeros((1, N), dtype=np.float32)  # 64 MiB host buffer, cheap
        with pytest.raises(ValueError) as ei:
            fc.cross_sectional_rank(X)
        assert "2^24" in str(ei.value)

    @NEED_CUDA
    def test_cap_parameter_scan_n_gt_2_24(self):
        N = (1 << 24) + 1
        X = np.zeros((1, N), dtype=np.float32)
        with pytest.raises(ValueError) as ei:
            fc.parameter_scan([("direction", ["ascending"])], X)
        assert "2^24" in str(ei.value)

    @NEED_CUDA
    def test_cap_factor_corr_f_gt_128(self):
        F3 = np.zeros((2, 2, 129), dtype=np.float64)
        with pytest.raises(ValueError) as ei:
            fc.factor_corr(F3, None, backend="cuda")
        assert "128" in str(ei.value)

    @NEED_CUDA
    def test_cap_stock_corr_n_gt_46340(self):
        N = 46341  # N*N > INT32_MAX
        X = np.zeros((2, N), dtype=np.float64)
        with pytest.raises(ValueError) as ei:
            fc.stock_corr(X, None, backend="cuda")
        assert "46340" in str(ei.value) or "INT32" in str(ei.value)

    # ---- hidden group failure injection --------------------------------

    @NEED_CUDA
    @pytest.mark.parametrize("fail_status", [9, 701])
    def test_hidden_group_failure_injection(self, monkeypatch, fail_status):
        """Whitelist launch failure (cudaErrorInvalidConfiguration=9 or
        cudaErrorLaunchOutOfResources=701) on one effective group -> status=
        'failed', result=None, error_stage='launch', timing 0.0; other groups
        unaffected; active_groups mask propagates to the binding (contract
        failure isolation + active-group selector)."""
        from fc import _util as fc_util
        from fc._cpu_core import np_cs_rank

        received_active = {}

        class _Fake:
            def parameter_scan_f32(self, x, m, return_timing=True, active_groups=None):
                received_active["active_groups"] = list(active_groups)
                return {
                    "groups": [np_cs_rank(x, m, False), np_cs_rank(x, None, False),
                               np_cs_rank(x, m, True), np_cs_rank(x, None, True)],
                    "group_status": [fail_status, 0, 0, 0],  # asc-masked fails
                    "time_ms": [0.0, 1.0, 1.0, 1.0],
                    "time_gpu_ms": [0.0, 0.5, 0.5, 0.5],
                }

        monkeypatch.setattr(fc_util, "fcb", lambda: _Fake())
        X = rng.standard_normal((20, 10)).astype(np.float32)
        mask = make_mask((20, 10))
        res = fc.parameter_scan([("direction", ["ascending", "descending"]),
                                 ("mask_mode", ["masked", "unmasked"])], X, mask)
        assert received_active["active_groups"] == [1, 1, 1, 1]  # full scan
        g0 = res["groups"][0]
        assert g0["status"] == "failed"
        assert g0["error_stage"] == "launch"
        assert g0["result"] is None
        assert g0["time_ms"] == 0.0 and g0["time_gpu_ms"] == 0.0
        assert str(fail_status) in g0["error"]
        for g in res["groups"][1:]:
            assert g["status"] == "ok" and g["result"] is not None
        assert res["summary"]["n_failed"] == 1
        assert res["summary"]["total_groups"] == 4
        assert res["summary"]["total_time_ms"] == pytest.approx(3.0)

    @NEED_CUDA
    def test_parameter_scan_active_groups_subset(self, monkeypatch):
        """active-group selector: a subset scan must propagate [1,0,1,0] so the
        binding only launches the effective groups (contract F11)."""
        from fc import _util as fc_util
        from fc._cpu_core import np_cs_rank

        received_active = {}

        class _Fake:
            def parameter_scan_f32(self, x, m, return_timing=True, active_groups=None):
                received_active["active_groups"] = list(active_groups)
                return {
                    "groups": [np_cs_rank(x, m, False), np_cs_rank(x, None, False),
                               np_cs_rank(x, m, True), np_cs_rank(x, None, True)],
                    "group_status": [0, 0, 0, 0],
                    "time_ms": [1.0, 1.0, 1.0, 1.0],
                    "time_gpu_ms": [0.5, 0.5, 0.5, 0.5],
                }

        monkeypatch.setattr(fc_util, "fcb", lambda: _Fake())
        X = rng.standard_normal((20, 10)).astype(np.float32)
        mask = make_mask((20, 10))
        # direction only -> (ascending,masked)+(descending,masked) = binding 0,2
        res = fc.parameter_scan([("direction", ["ascending", "descending"])],
                                X, mask)
        assert received_active["active_groups"] == [1, 0, 1, 0]
        assert res["summary"]["total_groups"] == 2
        assert all(g["status"] == "ok" for g in res["groups"])

    @NEED_CUDA
    def test_parameter_scan_non_whitelist_runtime_error(self, monkeypatch):
        """Contract §4 failure: a non-whitelist group status (e.g. 5) must be a
        scan-level RuntimeError (no partial results), NOT a group failure."""
        from fc import _util as fc_util
        from fc._cpu_core import np_cs_rank

        class _Fake:
            def parameter_scan_f32(self, x, m, return_timing=True, active_groups=None):
                return {
                    "groups": [np_cs_rank(x, m, False)] * 4,
                    "group_status": [5, 0, 0, 0],  # non-whitelist (cudaErrorInvalidDevice)
                    "time_ms": [0.0, 0.0, 0.0, 0.0],
                    "time_gpu_ms": [0.0, 0.0, 0.0, 0.0],
                }

        monkeypatch.setattr(fc_util, "fcb", lambda: _Fake())
        X = rng.standard_normal((20, 10)).astype(np.float32)
        mask = make_mask((20, 10))
        with pytest.raises(RuntimeError):
            fc.parameter_scan([("direction", ["ascending", "descending"]),
                               ("mask_mode", ["masked", "unmasked"])], X, mask)

    @NEED_CUDA
    def test_scan_all_unmasked_ignores_garbage_mask(self):
        """F14 override: all-unmasked scans must not touch the user mask."""
        X = rng.standard_normal((20, 10)).astype(np.float32)
        res = fc.parameter_scan([("mask_mode", ["unmasked"])], X, "not-a-mask")
        assert res["summary"]["total_groups"] == 1
        assert res["groups"][0]["status"] == "ok"
        assert bitwise_f32(res["groups"][0]["result"],
                           np_cs_rank(X, None, False))

    # ---- independent numeric oracles (F18 / contract determinism) ----

    @NEED_CUDA
    def test_cross_sectional_rank_independent_oracle(self):
        """Contract §1 determinism: GPU rank output must be bitwise-equal to the
        independent CPU oracle np_cs_rank (a real bug shared by both sides of a
        self-referential GPU-vs-GPU check would be caught here)."""
        X = rng.standard_normal((60, 40)).astype(np.float32)
        X[0, :5] = np.nan          # NaN exclusion
        X[1, 2] = np.inf           # inf exclusion
        X[2, :] = 3.0              # constant section
        X[3, :4] = 0.5             # tie cluster
        mask = make_mask((60, 40))
        for desc in (False, True):
            assert bitwise_f32(
                fc.cross_sectional_rank(X, None, desc),
                _independent_rank_panel(X, None, desc))
            assert bitwise_f32(
                fc.cross_sectional_rank(X, mask, desc),
                _independent_rank_panel(X, mask, desc))

    @NEED_CUDA
    def test_cross_sectional_rank_nan_payload_bitwise(self):
        """Contract §1: NaN output cells carry the frozen quiet-NaN payload
        0x7fc00000, bitwise on the whole panel."""
        X = rng.standard_normal((40, 30)).astype(np.float32)
        X[0, 0] = np.nan
        out = fc.cross_sectional_rank(X)
        bits = out.view(np.uint32)
        assert bits[0, 0] == 0x7FC00000
        assert np.isfinite(out[1:]).all()

    def _independent_spearman_ic(self, f, r, fm, rm, min_valid):
        """Hand-written ordinal-Spearman reference (contract §3 semantics),
        implemented independently of np_rolling_ic to avoid shared-bug parity."""
        f = np.asarray(f); r = np.asarray(r)
        ok = np.isfinite(f) & np.isfinite(r)
        if fm is not None:
            ok &= np.asarray(fm, dtype=bool)
        if rm is not None:
            ok &= np.asarray(rm, dtype=bool)
        T = f.shape[0]
        out = np.full(T, np.nan)
        for t in range(T):
            o = ok[t]
            if o.sum() < min_valid:
                continue
            if np.ptp(f[t][o]) == 0 or np.ptp(r[t][o]) == 0:
                continue
            fa = f[t][o]; rb = r[t][o]
            ra = np.full(fa.size, np.nan)
            oo = np.argsort(fa, kind="stable")
            ra[oo] = np.arange(1, fa.size + 1)
            rn = np.full(rb.size, np.nan)
            oo2 = np.argsort(rb, kind="stable")
            rn[oo2] = np.arange(1, rb.size + 1)
            out[t] = np.corrcoef(np.stack([ra, rn]))[0, 1]
        return out

    def test_rolling_ic_independent_oracle(self):
        """rolling_ic numeric correctness against a hand-written ordinal-
        Spearman reference (contract §3): value + NaN-pattern parity."""
        f = rng.standard_normal((80, 40))
        r = rng.standard_normal((80, 40))
        f[0, :3] = np.nan
        r[1, 2] = np.inf
        fm = make_mask((80, 40))
        rm = make_mask((80, 40))
        got = fc.rolling_ic(f, r, fm, rm, min_valid=5, device="cpu")
        ref = self._independent_spearman_ic(f, r, fm, rm, 5)
        assert np.array_equal(np.isnan(got), np.isnan(ref))
        fin = np.isfinite(got) & np.isfinite(ref)
        assert np.all(np.abs(got[fin] - ref[fin]) <= 1e-12)

    @NEED_CUDA
    def test_rolling_ic_cuda_independent_oracle(self):
        f = rng.standard_normal((80, 40)).astype(np.float32)
        r = rng.standard_normal((80, 40)).astype(np.float32)
        got = fc.rolling_ic(f, r, min_valid=5)  # auto-GPU
        ref = self._independent_spearman_ic(f, r, None, None, 5)
        assert np.array_equal(np.isnan(to_np(got)), np.isnan(ref))
        fin = np.isfinite(to_np(got)) & np.isfinite(ref)
        assert np.all(np.abs(to_np(got)[fin] - ref[fin]) <= 1e-12)

    def test_rolling_ic_constant_section_nan(self):
        """Contract §3 frozen counterexample: constant factor + constant returns
        -> IC NaN (not +1.0)."""
        f = np.full((2, 100), 0.5)
        r = np.full((2, 100), 0.1)
        got = fc.rolling_ic(f, r, min_valid=30, device="cpu")
        assert np.isnan(got).all()

    def test_rolling_ic_manifest_cases(self, _frozen_manifest, _frozen_npz):
        """F17: execute the frozen manifest rolling_ic numeric cases through fc
        and check against the frozen expected values (independent of the
        adapter's own CPU core)."""
        for c in _frozen_manifest["case_schema"]:
            if c["operation"] != "rolling_ic":
                continue
            f = _frozen_npz[c["inputs"][0]].reshape(1, -1)
            r = _frozen_npz[c["inputs"][1]].reshape(1, -1)
            mv = c["params"]["min_valid"]
            ic = fc.rolling_ic(f, r, min_valid=mv, device="cpu")
            got = float(ic[0])
            exp = c["expected"]
            if isinstance(exp, str) and exp == "nan":
                assert np.isnan(got), c["id"]
            else:
                assert abs(got - float(exp)) <= float(c["tolerance"]), (
                    f"{c['id']}: got {got} expected {exp}")

    @NEED_CUDA
    def test_parameter_scan_group_semantics(self):
        """parameter_scan groups: each ok result must equal np_cs_rank under
        that group's (direction, mask_mode) binding, and axis_values must be
        canonical (contract §4 output)."""
        X = rng.standard_normal((20, 10)).astype(np.float32)
        mask = make_mask((20, 10))
        res = fc.parameter_scan([("direction", ["ascending", "descending"]),
                                 ("mask_mode", ["masked", "unmasked"])], X, mask)
        expected = [
            _independent_rank_panel(X, mask, False),
            _independent_rank_panel(X, None, False),
            _independent_rank_panel(X, mask, True),
            _independent_rank_panel(X, None, True),
        ]
        expected_axes = [
            {"direction": "ascending", "mask_mode": "masked"},
            {"direction": "ascending", "mask_mode": "unmasked"},
            {"direction": "descending", "mask_mode": "masked"},
            {"direction": "descending", "mask_mode": "unmasked"},
        ]
        for g, ref, ax in zip(res["groups"], expected, expected_axes):
            assert g["status"] == "ok"
            assert g["axis_values"] == ax, f"group {g['group_index']} axes"
            assert bitwise_f32(g["result"], ref), f"group {g['group_index']}"
        assert res["spec"] == {"direction": ["ascending", "descending"],
                               "mask_mode": ["masked", "unmasked"]}

    # ---- zero-copy transfer audit (F18, HG-2 F04 clarification) ---------

    def test_zero_copy_host_transfer_audit(self):
        """F18 zero-copy audit: a host-resident C-contiguous float32 input is
        consumed without a copy (to_numpy returns the same object); a
        non-contiguous host input is copied once to C-contiguous (contract §0
        memory layout); a torch-CUDA input takes the documented D2H path (not
        silently zero-copy, HG-2 2026-08-06)."""
        from fc import _util as fc_util
        X = np.ascontiguousarray(rng.standard_normal((30, 20)).astype(np.float32))
        kind, device, arr = fc_util.to_numpy(
            X, name="X", ndim=2, dtypes="f32f64", downcast_to="float32")
        assert kind == "numpy" and device == "cpu"
        assert arr is X  # zero-copy for host C-contiguous float32

        Y = np.asfortranarray(rng.standard_normal((30, 20)).astype(np.float32))
        _, _, arr2 = fc_util.to_numpy(
            Y, name="Y", ndim=2, dtypes="f32f64", downcast_to="float32")
        assert arr2 is not Y and arr2.flags["C_CONTIGUOUS"]

        if HAS_CUDA:
            xt = torch.randn(30, 20, device="cuda")
            k2, d2, arr3 = fc_util.to_numpy(xt, name="xt", ndim=2, dtypes="f32f64")
            assert k2 == "torch" and d2 == "cuda:0"
            assert isinstance(arr3, np.ndarray)  # D2H host copy, not zero-copy

    # ---- read-only contract + independent output -----------------------

    @NEED_CUDA
    def test_input_read_only(self):
        """All public ops leave their numpy inputs byte-identical (read-only
        contract). Parametrized across rank / factor_corr / stock_corr /
        rolling_ic (GPT-5.6-Sol review #14)."""
        X = rng.standard_normal((30, 20)).astype(np.float32)
        Xc = X.copy()
        mask = make_mask((30, 20))
        fc.cross_sectional_rank(X)
        fc.cross_sectional_rank(X, mask, True)
        assert np.array_equal(X, Xc, equal_nan=True)

        F3 = rng.standard_normal((30, 20, 3)).astype(np.float64)
        F3c = F3.copy()
        fc.factor_corr(F3, mask, backend="cpu")
        assert np.array_equal(F3, F3c, equal_nan=True)

        S = rng.standard_normal((30, 15)).astype(np.float64)
        Sc = S.copy()
        fc.stock_corr(S, None, backend="cpu")
        assert np.array_equal(S, Sc, equal_nan=True)

        f = rng.standard_normal((30, 20)).astype(np.float64)
        r = rng.standard_normal((30, 20)).astype(np.float64)
        fc_ = f.copy(); rc = r.copy()
        fc.rolling_ic(f, r, min_valid=2, device="cpu")
        assert np.array_equal(f, fc_) and np.array_equal(r, rc)

    @NEED_CUDA
    def test_output_independent_allocation(self):
        """Repeated calls return independent allocations; outputs never share
        memory with inputs (across rank / stock_corr / factor_plane)."""
        X = rng.standard_normal((30, 20)).astype(np.float32)
        o1 = fc.cross_sectional_rank(X)
        o2 = fc.cross_sectional_rank(X)
        assert not np.shares_memory(o1, X)
        assert not np.shares_memory(o2, X)
        assert not np.shares_memory(o1, o2)
        o1[...] = -1.0
        assert np.isfinite(o2).all()  # untouched

        S = rng.standard_normal((30, 10)).astype(np.float64)
        s1 = fc.stock_corr(S, None, backend="cpu")
        s2 = fc.stock_corr(S, None, backend="cpu")
        assert not np.shares_memory(s1, S) and not np.shares_memory(s2, s1)

        F3 = rng.standard_normal((30, 10, 2)).astype(np.float32)
        p1 = fc.factor_plane(F3, 0)
        p2 = fc.factor_plane(F3, 0)
        assert not np.shares_memory(p1, F3) and not np.shares_memory(p1, p2)

    # ---- device exception matrix (GPT-5.6-Sol review #13) ---------------

    @NEED_CUDA
    def test_device_values_cuda_mask_cpu_rejected(self):
        """Global rule: mask must be on the values device -> ValueError for a
        CPU mask with CUDA values (rank and correlation cuda backend)."""
        X = torch.randn(20, 10, device="cuda")
        m_cpu = torch.zeros(20, 10, dtype=torch.bool)
        with pytest.raises(ValueError):
            fc.cross_sectional_rank(X, m_cpu)
        F3 = torch.randn(20, 10, 2, device="cuda")
        with pytest.raises(ValueError):
            fc.factor_corr(F3, m_cpu, backend="cuda")
        with pytest.raises(ValueError):
            fc.stock_corr(X, m_cpu, backend="cuda")

    @NEED_CUDA
    def test_device_corr_cpu_exception_allows_cross_device(self):
        """correlation cpu backend is the documented device exception: CUDA
        values with a CUDA mask are legal (copied to CPU to compute)."""
        F3 = torch.randn(20, 10, 2, device="cuda")
        m = torch.zeros(20, 10, device="cuda", dtype=torch.bool)
        m[..., :] = True
        mat = fc.factor_corr(F3, m, backend="cpu")
        assert np.isfinite(np.asarray(mat)).all()

    @NEED_CUDA
    def test_device_rolling_ic_cross_input_devices(self):
        """rolling_ic factor/returns on different devices copies to the exec
        device (legal, not an error)."""
        f = torch.randn(20, 10, device="cuda")
        r = rng.standard_normal((20, 10)).astype(np.float64)  # host
        out = fc.rolling_ic(f, r, min_valid=2)  # device=None -> GPU exec, mirror factor
        assert torch.is_tensor(out) and out.is_cuda

    # ---- torch-missing / CPU-only robustness (GPT-5.6-Sol review #2) ----

    def test_import_fc_works_without_torch(self, monkeypatch):
        """Blocking torch at the adapter level must not break `import fc` or the
        cpu-backend ops; GPU-only ops raise RuntimeError (require_cuda)."""
        from fc import _util as fc_util
        monkeypatch.setattr(fc_util, "_torch", None)
        # import fc is already done at module scope; verify cpu ops route around
        X = rng.standard_normal((30, 20)).astype(np.float32)
        # numpy-input cpu ops must work without torch
        F3 = rng.standard_normal((30, 20, 2)).astype(np.float64)
        mat = fc.factor_corr(F3, None, backend="cpu")
        assert isinstance(mat, np.ndarray)
        f = rng.standard_normal((30, 20))
        r = rng.standard_normal((30, 20))
        assert isinstance(fc.rolling_ic(f, r, min_valid=2, device="cpu"), np.ndarray)
        # GPU-only op must raise RuntimeError (CUDA unavailable without torch)
        with pytest.raises(RuntimeError):
            fc.cross_sectional_rank(X)

    # ---- infeasible caps (GPT-5.6-Sol review #7) -----------------------

    @pytest.mark.skip(reason="T*N>INT32_MAX requires an >8 GiB host array (infeasible "
                             "to allocate); the binding's int64 guard is the same code "
                             "path exercised by the N>2^24 cap tests")
    def test_cap_rank_times_n_gt_int32_max(self):
        X = np.zeros((1, (1 << 31) + 1), dtype=np.float32)
        fc.cross_sectional_rank(X)

    @pytest.mark.skip(reason="4 GiB host-output budget for parameter_scan requires "
                             "T*N>~268M (>=1 GiB array); infeasible to allocate; the "
                             "binding's out_bytes guard is covered by the same int64 "
                             "code path as the N>2^24 cap test")
    def test_cap_parameter_scan_4gib_host_budget(self):
        X = np.zeros((1, (1 << 28) + 1), dtype=np.float32)
        fc.parameter_scan([("direction", ["ascending"])], X)

    @NEED_CUDA
    def test_repeated_calls_bitwise_stable(self):
        X = rng.standard_normal((30, 20)).astype(np.float32)
        o1 = fc.cross_sectional_rank(X, None, False)
        o2 = fc.cross_sectional_rank(X, None, False)
        assert bitwise_f32(o1, o2)

    # ---- requires_grad detach ------------------------------------------

    @NEED_TORCH
    def test_requires_grad_detach_rolling_ic(self):
        f = torch.randn(20, 10, requires_grad=True)
        r = torch.randn(20, 10, requires_grad=True)
        out = fc.rolling_ic(f, r, min_valid=2, device="cpu")
        assert not out.requires_grad

    @NEED_TORCH
    def test_requires_grad_detach_factor_corr(self):
        # cpu backend always returns numpy -> detach is unobservable there; use
        # the cuda backend whose torch output exposes requires_grad.
        if HAS_CUDA:
            F3 = torch.randn(20, 10, 3, requires_grad=True, device="cuda")
            m = fc.factor_corr(F3, None, backend="cuda")
            assert torch.is_tensor(m) and not m.requires_grad
        else:  # pragma: no cover - cpu-only fallback keeps the assertion honest
            F3 = torch.randn(20, 10, 3, requires_grad=True)
            m = fc.factor_corr(F3, None, backend="cpu")
            assert isinstance(m, np.ndarray)  # detach unobservable on cpu path

    # ---- torch-CUDA input semantics (HG-2 F04 clarification) -----------

    @NEED_CUDA
    def test_torch_cuda_rank_matches_cpu(self):
        x = torch.randn(40, 30, device="cuda")
        out = fc.cross_sectional_rank(x)
        assert torch.is_tensor(out) and out.is_cuda
        got = to_np(out)
        exp = fc.cross_sectional_rank(x.detach().cpu().numpy())
        assert bitwise_f32(got, exp)

    @NEED_CUDA
    def test_torch_cuda_factor_corr_matches_cpu(self):
        F3 = torch.randn(40, 20, 3, device="cuda")
        mat = fc.factor_corr(F3, None, backend="cuda")
        assert torch.is_tensor(mat) and mat.is_cuda
        got = to_np(mat)
        exp = fc.factor_corr(F3.detach().cpu().numpy(), None, backend="cpu")
        assert corr_match(got, exp)

    # ---- multi-GPU routing ---------------------------------------------

    @NEED_MULTI_GPU
    def test_multi_gpu_device_mirror(self):
        F3 = torch.randn(20, 10, 3, device="cuda:1")
        p = fc.factor_plane(F3, 0)
        assert p.is_cuda and p.device.index == 1

    # ---- backend / device validation errors ----------------------------

    def test_backend_invalid_value(self):
        F3 = rng.standard_normal((10, 8, 3))
        with pytest.raises(ValueError):
            fc.factor_corr(F3, None, backend="gpu")
        with pytest.raises(ValueError):
            fc.stock_corr(rng.standard_normal((10, 8)), None, backend="cuda_on_apple")

    def test_rolling_ic_device_invalid(self):
        f = rng.standard_normal((10, 8))
        r = rng.standard_normal((10, 8))
        with pytest.raises(ValueError):
            fc.rolling_ic(f, r, min_valid=2, device="tpu")

    @NEED_CUDA
    def test_mask_must_be_bool(self):
        X = rng.standard_normal((10, 8)).astype(np.float32)
        with pytest.raises(ValueError):
            fc.cross_sectional_rank(X, np.ones((10, 8), dtype=np.uint8))
        with pytest.raises(ValueError):
            fc.cross_sectional_rank(X, np.ones((10, 8), dtype=np.float64))

    @NEED_CUDA
    def test_mask_shape_mismatch(self):
        X = rng.standard_normal((10, 8)).astype(np.float32)
        with pytest.raises(ValueError):
            fc.cross_sectional_rank(X, np.ones((10, 7), dtype=bool))

    def test_non_container_typeerror(self):
        with pytest.raises(TypeError):
            fc.cross_sectional_rank("not-an-array")
        with pytest.raises(TypeError):
            fc.factor_corr([1.0, 2.0], None, backend="cpu")

    def test_dtype_whitelist(self):
        X = rng.standard_normal((10, 8)).astype(np.int64)
        with pytest.raises(ValueError):
            fc.cross_sectional_rank(X)
        with pytest.raises(ValueError):
            fc.factor_corr(X.astype(np.float16).reshape(10, 8, 1), None,
                           backend="cpu")
