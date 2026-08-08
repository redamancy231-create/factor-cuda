// factor-cuda -- factor_corr pybind11 binding (GIL release + NumPy paths).
//
// Thin Python wrapper around factor_corr_gpu (src/factor_corr.cu). F3 is a
// (T,N,F) float64 C-contiguous panel (forcecast upcasts f32 on the fly, so the
// caller does not need to pre-cast); mask is None or a (T,N) bool/uint8 array.
// T/N/F are taken from the F3 shape, so the Python signature is just
//   factor_corr_f64(F3, mask=None) -> (F,F) float64
// The GIL is released around the GPU call (the parallel region touches no
// Python object -- raw double*/uint8_t* pointers only), following the
// etf-pattern-match-pybind11 pattern (reference_pybind11_lessons: GIL release /
// py::arg count / c_style+forcecast / py::ssize_t indexing).
//
// Numerics: two-pass centered Pearson over the pooled valid rows of each factor
// pair + a Kahan (CompensatedSum) re-run when the corpus trigger fires
// (bias_metric > 1e8 / |r| > 1 / non-finite); diagonal 1.0 or NaN; strict lower
// triangle mirrored bitwise. See factor_corr.cuh for the full contract.
//
// ASCII-only comments (nvcc/GBK pitfall). PoC 3.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "factor_corr.cuh"

namespace py = pybind11;

using ArrD = py::array_t<double, py::array::c_style | py::array::forcecast>;
using ArrU8 = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

// Python wrapper for factor_corr_gpu.
//   F3   : (T,N,F) float64, C-contiguous (forcecast upcasts float32).
//   mask : None or (T,N) bool/uint8 (1 = participate); None = all finite.
// Returns the (F,F) float64 correlation matrix.
py::array_t<double> factor_corr_f64(ArrD F3, py::object mask) {
  auto fb = F3.unchecked<3>();
  const py::ssize_t T = fb.shape(0);
  const py::ssize_t N = fb.shape(1);
  const py::ssize_t F = fb.shape(2);
  if (T < 1 || N < 1 || F < 1) {
    throw std::invalid_argument("F3 shape must be (T,N,F) with T,N,F >= 1");
  }
  if (static_cast<int64_t>(T) * N > INT32_MAX) {
    throw std::invalid_argument("T*N exceeds INT32_MAX (implementation cap)");
  }
  if (F > 128) {
    throw std::invalid_argument("F > 128 (factor pair grid cap)");
  }
  // c_style | forcecast guarantees C-contiguous backing. The unchecked
  // reference's data() routes through operator()(indices) (index count == ndim),
  // so use py::array_t::data() (member) for the base pointer.
  const double* fp = F3.data();

  const uint8_t* mp = nullptr;
  // The cast holder must live in FUNCTION scope (not the if block): a
  // bool->uint8 forcecast allocates a NEW array owned by the holder, so a
  // block-local holder would destruct at the end of the if and leave mp
  // dangling for the GIL-released GPU call below (UB -- bool masks silently
  // produced wrong correlations). uint8 inputs are zero-copy (shared with the
  // caller) so they happened to work; this is the fix for both paths.
  ArrU8 m_holder;
  if (!mask.is_none()) {
    m_holder = mask.cast<ArrU8>();
    auto mb = m_holder.unchecked<2>();
    if (mb.shape(0) != T || mb.shape(1) != N) {
      throw std::invalid_argument("mask shape must be (T,N) matching F3");
    }
    mp = m_holder.data();
  }

  const size_t out_len = static_cast<size_t>(F) * static_cast<size_t>(F);
  std::vector<double> out(out_len);
  int rc = 0;
  {
    py::gil_scoped_release release;
    rc = factor_corr_gpu(fp, mp, static_cast<int>(T), static_cast<int>(N),
                         static_cast<int>(F), out.data());
  }
  if (rc != 0) {
    throw std::runtime_error("factor_corr_gpu failed with error code " + std::to_string(rc));
  }

  py::array_t<double> result({F, F});
  auto r = result.mutable_unchecked<2>();
  for (py::ssize_t i = 0; i < F; ++i) {
    for (py::ssize_t j = 0; j < F; ++j) {
      r(i, j) = out[static_cast<size_t>(i) * static_cast<size_t>(F) + static_cast<size_t>(j)];
    }
  }
  return result;
}

PYBIND11_MODULE(factor_corr_pybind, m) {
  m.doc() = "factor-cuda factor_corr (F,F) Pearson correlation matrix over pooled\n"
            "(T,N) valid rows, pybind11 binding.\n"
            "  factor_corr_f64(F3, mask=None) -> (F,F) float64\n"
            "    F3   : (T,N,F) float64, C-contiguous (float32 auto-upcast)\n"
            "    mask : None or (T,N) bool/uint8 (1 = participate)\n"
            "Numerics: two-pass centered Pearson + Kahan re-run on the corpus\n"
            "trigger (bias_metric>1e8 / |r|>1 / non-finite); diagonal 1.0 or NaN;\n"
            "strict lower triangle mirrored bitwise.";
  m.def("factor_corr_f64", &factor_corr_f64, py::arg("F3"), py::arg("mask") = py::none(),
        "Compute the (F,F) factor correlation matrix from a (T,N,F) float64 "
        "panel and an optional (T,N) bool/uint8 mask.");
}
