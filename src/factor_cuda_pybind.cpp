// factor-cuda -- factor_cuda pybind11 binding (GIL release + NumPy paths).
//
// Thin Python wrapper over four GPU kernels (cs_rank / rolling_ic / stock_corr /
// parameter_scan), following the closed factor_corr_pybind pattern
// (src/factor_corr_pybind.cpp; reference_pybind11_lessons / pybind11-cuda-
// binding-pitfalls). The factor_corr binding stays separate and untouched
// (review-closed material).
//
// INTERFACE SCOPE: this module is a GPU-only LOW-LEVEL binding, not the frozen
// fc.* high-level operations (CLAUDE.md L0). Device policy, torch.Tensor /
// DLPack-capsule inputs, f64->f32 downcast, and contract parameter names
// (factor/forward_returns/factor_mask/...) are the future adapter's
// responsibility (design reviews/poc4_pybind_binding_design_2026-08-05.md).
//
// DTYPE GATES (design MAJOR-3, MINOR-17): cs_rank/parameter_scan take STRICT
// float32 (no forcecast -- a float64 input is a TypeError; downcast is adapter
// responsibility); rolling_ic/stock_corr accept float32/float64 (forcecast
// f32->f64 is exact); masks accept bool/uint8 only (a float64 mask is a
// ValueError, not silently truncated); non-ndarray objects are TypeErrors.
//
// GIL: the GPU call region runs with the GIL released; every host pointer
// (input data + output vectors) is materialized BEFORE the release block, and
// every cast result is kept in a NAMED local that outlives the release block
// (a temporary like mask.cast<ArrU8>().data() would dangle). Input arrays are
// zero-copy borrowed while the GPU call is in flight -- other threads must not
// mutate them concurrently.
//
// ASCII-only comments (nvcc/GBK pitfall). PoC 4.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cross_sectional_rank.cuh"
#include "parameter_scan.cuh"
#include "rolling_ic.cuh"
#include "stock_corr.cuh"

namespace py = pybind11;

using ArrD = py::array_t<double, py::array::c_style | py::array::forcecast>;
using ArrF32 = py::array_t<float, py::array::c_style>;
using ArrU8 = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

// ---- dtype gates (checked BEFORE any cast so forcecast never silently
// ---- changes semantics) ---------------------------------------------------

static void require_ndarray(py::handle obj, const char* name) {
  if (!py::isinstance<py::array>(obj)) {
    throw py::type_error(std::string(name) + " must be a numpy array");
  }
}

static py::dtype array_dtype(py::handle obj) {
  return py::reinterpret_borrow<py::array>(obj).dtype();
}

// Strict float32 input (no downcast here -- that is the adapter's job).
static ArrF32 require_f32(py::handle obj, const char* name) {
  require_ndarray(obj, name);
  if (array_dtype(obj) != py::dtype::of<float>()) {
    throw py::type_error(std::string(name) +
                         " must be float32 (float64 downcast is adapter responsibility)");
  }
  return py::cast<ArrF32>(obj);
}

// float32/float64 accepted; forcecast f32->f64 is exact (lossless upcast).
static ArrD upcast_f64(py::handle obj, const char* name) {
  require_ndarray(obj, name);
  py::dtype dt = array_dtype(obj);
  if (!(dt == py::dtype::of<float>() || dt == py::dtype::of<double>())) {
    throw py::type_error(std::string(name) + " must be float32 or float64");
  }
  return py::cast<ArrD>(obj);
}

// Optional mask (None = nullptr). bool/uint8 accepted; anything else (incl.
// float) is a ValueError -- never a silent truncation. The cast result is kept
// in a named local (holder) so its data() pointer survives the GIL-release
// GPU call.
static const uint8_t* mask_or_null(py::object obj, const char* name, py::ssize_t T,
                                   py::ssize_t N, ArrU8& holder) {
  if (obj.is_none()) return nullptr;
  require_ndarray(obj, name);
  py::dtype dt = array_dtype(obj);
  if (!(dt == py::dtype::of<bool>() || dt == py::dtype::of<uint8_t>())) {
    throw py::value_error(std::string(name) + " must be bool or uint8");
  }
  holder = py::cast<ArrU8>(obj);
  auto mb = holder.unchecked<2>();
  if (mb.shape(0) != T || mb.shape(1) != N) {
    throw std::invalid_argument(std::string(name) + " shape must be (T,N) matching input");
  }
  return holder.data();
}

// Integer argument that must be a Python int (a float like 30.5 is a
// ValueError, not a silent truncation).
static int require_int(py::handle obj, const char* name) {
  if (!py::isinstance<py::int_>(obj)) {
    throw std::invalid_argument(std::string(name) + " must be an integer");
  }
  return py::cast<int>(obj);
}

// Map a kernel error code: negative = contract error -> ValueError, positive
// cudaError_t = runtime/device failure -> RuntimeError.
static void check_rc(int rc, const char* fn) {
  if (rc == 0) return;
  if (rc < 0) {
    throw std::invalid_argument(std::string(fn) + " contract error code " + std::to_string(rc));
  }
  throw std::runtime_error(std::string(fn) + " failed with error code " + std::to_string(rc));
}

// ---- 1. cs_rank_f32(X, mask=None, descending=False, workspace=None) -------
// Strict float32 in/out. Returns the (T,N) float32 stable-ordinal rank matrix
// (valid cells = exact integer rank 1..K; invalid cells = quiet NaN payload
// 0x7fc00000, preserved bitwise). workspace: optional CsRankWorkspace handle
// (P3 adapter auto-cache, 2026-08-08). When provided, device buffers are reused
// across calls with the same shape/device (no per-call cudaMalloc/cudaFree); on
// shape/device mismatch the workspace clears + re-allocates. None = allocate and
// free every call (previous behavior). NOT thread-safe -- calls sharing a
// workspace must be serialized by the caller.
py::array_t<float> cs_rank_f32(py::object X, py::object mask, bool descending,
                               cs_rank_workspace* ws) {
  ArrF32 xa = require_f32(X, "X");
  auto xb = xa.unchecked<2>();
  const py::ssize_t T = xb.shape(0);
  const py::ssize_t N = xb.shape(1);
  if (T < 1 || N < 1) {
    throw std::invalid_argument("X must be (T,N) with T,N >= 1");
  }
  if (N > (1 << 24)) {
    throw std::invalid_argument("N > 2^24 (rank precision cap)");
  }
  if (static_cast<int64_t>(T) * N > INT32_MAX) {
    throw std::invalid_argument("T*N exceeds INT32_MAX (implementation cap)");
  }

  ArrU8 m_holder;
  const uint8_t* mp = mask_or_null(mask, "mask", T, N, m_holder);

  const size_t out_len = static_cast<size_t>(T) * static_cast<size_t>(N);
  std::vector<float> out(out_len);
  int rc = 0;
  {
    py::gil_scoped_release release;
    rc = cs_rank_gpu(xa.data(), mp, static_cast<int>(T), static_cast<int>(N),
                     descending, out.data(), nullptr, ws);
  }
  check_rc(rc, "cs_rank_gpu");

  py::array_t<float> result({T, N});
  std::memcpy(result.mutable_data(), out.data(), out_len * sizeof(float));
  return result;
}

// ---- 2. rolling_ic_f64(F, R, fmask=None, rmask=None, min_valid=30,
// ----        return_ranks=False, workspace=None) -----------------------------
// float64 in (f32 auto-upcast exactly). Default returns (T,) float64 IC;
// return_ranks=True returns (ic, rank_f, rank_r) where rank_* are (T,N)
// float64 with 0 for invalid cells (NON-contract diagnostics -- the kernel
// dumps these after the scatter stages; 0 means "not ranked", intentionally
// different from cs_rank's NaN convention). workspace: optional
// RollingIcWorkspace handle (P3 adapter auto-cache, 2026-08-08). Same reuse
// semantics as cs_rank_f32's workspace. NOT thread-safe.
py::object rolling_ic_f64(py::object F, py::object R, py::object fmask, py::object rmask,
                          py::object min_valid_obj, bool return_ranks,
                          rolling_ic_workspace* ws) {
  ArrD Fa = upcast_f64(F, "F");
  ArrD Ra = upcast_f64(R, "R");
  auto fb = Fa.unchecked<2>();
  auto rb = Ra.unchecked<2>();
  const py::ssize_t T = fb.shape(0);
  const py::ssize_t N = fb.shape(1);
  if (T < 1 || N < 1) {
    throw std::invalid_argument("F must be (T,N) with T,N >= 1");
  }
  if (rb.shape(0) != T || rb.shape(1) != N) {
    throw std::invalid_argument("R shape must match F (T,N)");
  }
  if (static_cast<int64_t>(T) * N > INT32_MAX) {
    throw std::invalid_argument("T*N exceeds INT32_MAX (implementation cap)");
  }
  const int min_valid = require_int(min_valid_obj, "min_valid");
  if (min_valid < 2) {
    throw std::invalid_argument("min_valid must be >= 2");
  }

  ArrU8 f_holder, r_holder;
  const uint8_t* fmp = mask_or_null(fmask, "fmask", T, N, f_holder);
  const uint8_t* rmp = mask_or_null(rmask, "rmask", T, N, r_holder);

  const size_t total = static_cast<size_t>(T) * static_cast<size_t>(N);
  std::vector<double> ic(static_cast<size_t>(T));
  std::vector<double> rank_f, rank_r;
  double* rf_p = nullptr;
  double* rr_p = nullptr;
  if (return_ranks) {
    rank_f.resize(total);
    rank_r.resize(total);
    rf_p = rank_f.data();
    rr_p = rank_r.data();
  }

  int rc = 0;
  {
    py::gil_scoped_release release;
    rc = rolling_ic_gpu(Fa.data(), Ra.data(), fmp, rmp, static_cast<int>(T),
                        static_cast<int>(N), min_valid, ic.data(), nullptr, rf_p, rr_p, ws);
  }
  check_rc(rc, "rolling_ic_gpu");

  py::array_t<double> ic_arr({T});
  std::memcpy(ic_arr.mutable_data(), ic.data(), static_cast<size_t>(T) * sizeof(double));
  if (!return_ranks) {
    return ic_arr;
  }
  py::array_t<double> rf_arr({T, N});
  py::array_t<double> rr_arr({T, N});
  std::memcpy(rf_arr.mutable_data(), rank_f.data(), total * sizeof(double));
  std::memcpy(rr_arr.mutable_data(), rank_r.data(), total * sizeof(double));
  return py::make_tuple(ic_arr, rf_arr, rr_arr);
}

// ---- 3. stock_corr_f64(X, mask=None, return_stats=False) -------------------
// float64 in (f32 auto-upcast exactly). Returns the (N,N) float64 correlation
// matrix; return_stats=True returns (corr, {"selected_path": int,
// "fallback_count": int}) -- NON-contract diagnostics.
py::object stock_corr_f64(py::object X, py::object mask, bool return_stats) {
  ArrD Xa = upcast_f64(X, "X");
  auto xb = Xa.unchecked<2>();
  const py::ssize_t T = xb.shape(0);
  const py::ssize_t N = xb.shape(1);
  if (T < 1 || N < 1) {
    throw std::invalid_argument("X must be (T,N) with T,N >= 1");
  }
  if (static_cast<int64_t>(T) * N > INT32_MAX) {
    throw std::invalid_argument("T*N exceeds INT32_MAX (implementation cap)");
  }
  if (static_cast<int64_t>(N) * N > INT32_MAX) {
    throw std::invalid_argument("N*N exceeds INT32_MAX (output grid cap, N <= 46340)");
  }

  ArrU8 m_holder;
  const uint8_t* mp = mask_or_null(mask, "mask", T, N, m_holder);

  const size_t out_len = static_cast<size_t>(N) * static_cast<size_t>(N);
  std::vector<double> out(out_len);
  StockCorrRunStats stats;
  int rc = 0;
  {
    py::gil_scoped_release release;
    rc = stock_corr_gpu(Xa.data(), mp, static_cast<int>(T), static_cast<int>(N),
                        out.data(), nullptr, return_stats ? &stats : nullptr);
  }
  check_rc(rc, "stock_corr_gpu");

  py::array_t<double> result({N, N});
  std::memcpy(result.mutable_data(), out.data(), out_len * sizeof(double));
  if (!return_stats) {
    return result;
  }
  py::dict sd;
  sd["selected_path"] = stats.selected_path;
  sd["fallback_count"] = stats.fallback_count;
  return py::make_tuple(result, sd);
}

// ---- 4. parameter_scan_f32(X, mask, return_timing=False, active_groups=None,
// ----        workspace=None) --------------------------------------------------
// Strict float32 X; mask REQUIRED (None -> ValueError -- otherwise the
// unmasked groups would silently run as all-finite, violating the frozen
// parameter_scan contract "masked mode requires a mask").
// Returns {"groups": [4x (T,N) float32 | None], "group_status": [4x int]}.
// This is a PoC-4-only LOW-LEVEL schema, not the frozen fc.parameter_scan
// {"spec","groups","summary"} output -- the adapter maps groups->records and
// synthesizes spec/summary/timing. Group downgrade is restricted to the
// contract whitelist (InvalidConfiguration / LaunchOutOfResources) -> that
// group is None and the rest still return; any other positive group code is a
// scan-level RuntimeError (no partial result).
py::dict parameter_scan_f32(py::object X, py::object mask, bool return_timing,
                            py::object active_groups, cs_rank_workspace* ws) {
  ArrF32 xa = require_f32(X, "X");
  auto xb = xa.unchecked<2>();
  const py::ssize_t T = xb.shape(0);
  const py::ssize_t N = xb.shape(1);
  if (T < 1 || N < 1) {
    throw std::invalid_argument("X must be (T,N) with T,N >= 1");
  }
  if (N > (1 << 24)) {
    throw std::invalid_argument("N > 2^24 (rank precision cap)");
  }
  const int64_t total64 = static_cast<int64_t>(T) * N;
  if (total64 > INT32_MAX) {
    throw std::invalid_argument("T*N exceeds INT32_MAX (implementation cap)");
  }

  // Active-group selector (Phase 1 adapter subset scans): None = all 4 groups;
  // otherwise a length-4 int sequence of 0/1 (1 = execute). Parsed BEFORE the
  // mask gate and the host-allocation loop so inactive groups do not
  // participate in mask validation or output allocation (review F3).
  int active_arr[4] = {1, 1, 1, 1};
  const int* h_active = nullptr;
  int active_count = 4;
  if (!active_groups.is_none()) {
    py::sequence seq = py::cast<py::sequence>(active_groups);
    if (py::len(seq) != 4) {
      throw std::invalid_argument("active_groups must have length 4");
    }
    active_count = 0;
    for (int g = 0; g < 4; ++g) {
      int v = py::cast<int>(seq[g]);
      if (v != 0 && v != 1) {
        throw std::invalid_argument("active_groups values must be 0 or 1");
      }
      active_arr[g] = v;
      active_count += v;
    }
    h_active = active_arr;
  }

  // Mask is required/validated ONLY when an active group is a masked group
  // (binding index 0 = ascending-masked, 2 = descending-masked). All-active-
  // groups-unmasked scans ignore the mask entirely (contract mask override;
  // review F3): None or any object is accepted without validation -- the
  // adapter passes an all-true mask for all-unmasked scans.
  const bool need_mask = (active_arr[0] || active_arr[2]);
  ArrU8 m_holder;
  const uint8_t* mp = nullptr;
  if (need_mask) {
    if (mask.is_none()) {
      throw std::invalid_argument(
          "parameter_scan requires a mask when any masked group is active");
    }
    mp = mask_or_null(mask, "mask", T, N, m_holder);
  }

  // Host output budget: only the ACTIVE groups allocate host buffers (review
  // F3); inactive groups pass nullptr h_out to the kernel.
  const size_t total = static_cast<size_t>(total64);
  const size_t out_bytes = static_cast<size_t>(active_count) * total * sizeof(float);
  if (out_bytes > 4ull * 1024 * 1024 * 1024) {  // 4 GiB host-output cap
    throw std::invalid_argument(
        "active*T*N*4 bytes exceeds the host output budget");
  }

  // Optional per-group timing (contract fc.parameter_scan time_ms/time_gpu_ms).
  // time_ms is double (wall-clock ms), time_gpu_ms is float (cudaEvent ms) --
  // the kernel writes float into h_time_gpu_ms, so the host vector MUST be
  // float (a double vector + reinterpret_cast<float*> misaligns).
  std::vector<double> time_ms;
  std::vector<float> time_gpu_ms;
  double* h_time_ms_p = nullptr;
  float* h_time_gpu_ms_p = nullptr;
  if (return_timing) {
    time_ms.resize(4);
    time_gpu_ms.resize(4);
    h_time_ms_p = time_ms.data();
    h_time_gpu_ms_p = time_gpu_ms.data();
  }

  std::vector<float> outs[4];
  float* hp[4] = {nullptr, nullptr, nullptr, nullptr};
  for (int g = 0; g < 4; ++g) {
    if (h_active == nullptr || h_active[g]) {
      outs[g].resize(total);
      hp[g] = outs[g].data();
    }
  }
  int group_status[4];
  int rc = 0;
  double elapsed_diag = 0.0;
  {
    py::gil_scoped_release release;
    // Whole-call diagnostic clock is gated by return_timing (review F4: the
    // default path must not read the clock at all).
    const auto t0 = return_timing ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point{};
    rc = parameter_scan_gpu(xa.data(), mp, static_cast<int>(T), static_cast<int>(N),
                            hp, group_status, nullptr, h_time_ms_p,
                            h_time_gpu_ms_p, h_active, ws);
    if (return_timing) {
      const auto t1 = std::chrono::steady_clock::now();
      elapsed_diag = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
  }
  check_rc(rc, "parameter_scan_gpu");

  py::dict result;
  py::list groups;
  py::list status_list;
  for (int g = 0; g < 4; ++g) {
    status_list.append(group_status[g]);
    if (group_status[g] == 0) {
      py::array_t<float> arr({T, N});
      std::memcpy(arr.mutable_data(), hp[g], total * sizeof(float));
      groups.append(arr);
    } else if (group_status[g] == cudaErrorInvalidConfiguration ||
               group_status[g] == cudaErrorLaunchOutOfResources ||
               group_status[g] == -100) {
      // Whitelist downgrade (result None) or a skipped group (active_groups=0
      // -> kGroupNotExecuted=-100): both surface as None; the caller reads
      // only the active groups' results.
      groups.append(py::none());
    } else {
      // Any non-whitelist positive group code is scan-level fatal.
      throw std::runtime_error("parameter_scan_gpu group-level error code " +
                               std::to_string(group_status[g]));
    }
  }
  result["groups"] = groups;
  result["group_status"] = status_list;
  if (return_timing) {
    py::list tms, tgms;
    for (int g = 0; g < 4; ++g) {
      tms.append(time_ms[g]);
      tgms.append(time_gpu_ms[g]);
    }
    result["time_ms"] = tms;
    result["time_gpu_ms"] = tgms;
    // Diagnostic only (review F12): the contract elapsed_ms authority is the
    // fc.parameter_scan entry wall-clock; this whole-call figure is not it.
    result["_elapsed_ms_diag"] = elapsed_diag;
  }
  return result;
}

PYBIND11_MODULE(factor_cuda_pybind, m) {
  m.doc() = "factor-cuda GPU kernel bindings (cs_rank / rolling_ic / stock_corr /\n"
            "parameter_scan), pybind11, GPU-only low-level layer.\n"
            "  cs_rank_f32(X, mask=None, descending=False, workspace=None) -> (T,N) f32\n"
            "  rolling_ic_f64(F, R, fmask=None, rmask=None, min_valid=30,\n"
            "                 return_ranks=False, workspace=None) -> (T,) f64 | (ic,rf,rr)\n"
            "  stock_corr_f64(X, mask=None, return_stats=False) -> (N,N) float64\n"
            "  parameter_scan_f32(X, mask, return_timing=False,\n"
            "                    active_groups=None, workspace=None) -> {groups, group_status}\n"
            "  CsRankWorkspace() / RollingIcWorkspace() -> cached device-buffer\n"
            "    handles; pass as workspace to reuse buffers across calls with the\n"
            "    same shape (None = allocate/free every call). .clear() releases\n"
            "    buffers (idempotent). NOT thread-safe -- serialize shared use.\n"
            "Inputs with matching dtype/layout may be zero-copy borrowed during\n"
            "the GPU call; conversions (f32->f64, bool->uint8, layout) allocate\n"
            "copies held until the call returns. Do not mutate inputs from\n"
            "another thread while a call is in flight.\n"
            "dtype gates: cs_rank/parameter_scan X strict float32; rolling_ic /\n"
            "stock_corr accept float32/float64 (lossless upcast); masks bool/uint8.";

  // Python default args are registered HERE (py::arg("mask") = py::none()
  // etc.) so callers may omit optional parameters -- the C++ functions carry
  // the same defaults but pybind11 only surfaces them via these bindings
  // (review 2026-08-06: missing defaults made every omit-optional call a
  // TypeError). parameter_scan's mask stays required (contract: masked mode
  // needs a mask).
  // Cached device-buffer workspace handles (P3 adapter auto-cache, 2026-08-08).
  // The C++ workspace types are non-copyable/non-movable owning aggregates;
  // these Python classes are thin handles so the adapter can hold a workspace
  // across calls (reusing device buffers per shape key) without exposing its
  // internals. clear() releases all cached buffers (idempotent; safe on empty)
  // via the owner MemTracker. Workspaces are NOT thread-safe -- calls sharing a
  // workspace must be serialized by the caller (the fc adapter holds one lock
  // per shape key).
  py::class_<cs_rank_workspace>(m, "CsRankWorkspace")
      .def(py::init<>())
      .def("clear", &cs_rank_workspace_clear,
           "Release all cached device buffers (idempotent; safe on empty). "
           "Next call re-allocates.");
  py::class_<rolling_ic_workspace>(m, "RollingIcWorkspace")
      .def(py::init<>())
      .def("clear", &rolling_ic_workspace_clear,
           "Release all cached device buffers (idempotent; safe on empty). "
           "Next call re-allocates.");

  m.def("cs_rank_f32", &cs_rank_f32, py::arg("X"), py::arg("mask") = py::none(),
        py::arg("descending") = false, py::arg("workspace") = py::none(),
        "Stable ordinal cross-sectional rank of a (T,N) float32 panel. "
        "Valid cells -> integer rank 1..K; invalid -> quiet NaN 0x7fc00000. "
        "workspace: optional CsRankWorkspace to reuse device buffers across "
        "calls with the same shape (None = allocate/free every call).");
  m.def("rolling_ic_f64", &rolling_ic_f64, py::arg("F"), py::arg("R"),
        py::arg("fmask") = py::none(), py::arg("rmask") = py::none(),
        py::arg("min_valid") = 30, py::arg("return_ranks") = false,
        py::arg("workspace") = py::none(),
        "Daily cross-sectional Spearman IC. Optional rank outputs are "
        "non-contract diagnostics (0 = not ranked). workspace: optional "
        "RollingIcWorkspace to reuse device buffers across calls with the "
        "same shape (None = allocate/free every call).");
  m.def("stock_corr_f64", &stock_corr_f64, py::arg("X"), py::arg("mask") = py::none(),
        py::arg("return_stats") = false,
        "(N,N) stock correlation matrix; optional dispatch stats are "
        "non-contract diagnostics.");
  m.def("parameter_scan_f32", &parameter_scan_f32, py::arg("X"), py::arg("mask"),
        py::arg("return_timing") = false, py::arg("active_groups") = py::none(),
        py::arg("workspace") = py::none(),
        "G=4 cross-sectional-rank parameter scan (direction x mask_mode). "
        "Low-level schema; whitelist-downgraded groups are None. "
        "return_timing=True adds per-group time_ms/time_gpu_ms (+ diagnostic "
        "_elapsed_ms_diag); active_groups=None runs all 4, else a length-4 "
        "0/1 list selects which groups to execute (skipped groups: -100). "
        "workspace: optional CsRankWorkspace (parameter_scan shares cs_rank's "
        "buffer set) to reuse device buffers across calls with the same shape "
        "(None = allocate/free every call).");
}
