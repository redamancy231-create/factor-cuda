// factor-cuda -- PoC 3 cs_rank v0 selfcheck.
//
// Verifies src/cross_sectional_rank.cu against an in-process CPU reference
// (equivalent to benchmarks/backends.py np_cs_rank: stable ordinal per row,
// descending == ascending of negated, non-finite excluded -> NaN payload
// 0x7fc00000). Covers parity anchor cases (rank_tie / rank_zero / rank_nan_inf /
// rank_mask) plus randomized panels with NaN / +-inf / +-0 / ties / masks.
//
// ASCII-only comments (nvcc/GBK pitfall). Build: nvcc -arch=sm_89
//   -I src -I <cuda>/include/cccl poc/poc3_cs_rank_selfcheck.cu src/cross_sectional_rank.cu
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>
#include <cuda_runtime.h>
#include "cross_sectional_rank.cuh"

namespace {

float nan_payload() {
  uint32_t b = 0x7fc00000u;
  float f;
  std::memcpy(&f, &b, sizeof(float));
  return f;
}

uint32_t f32_bits(float f) {
  uint32_t b;
  std::memcpy(&b, &f, sizeof(float));
  return b;
}

// Deterministic LCG (fixed seed) for reproducible panels.
struct Lcg {
  uint64_t s;
  explicit Lcg(uint64_t seed) : s(seed) {}
  uint32_t next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(s >> 33);
  }
  float uniform() { return static_cast<float>(next() >> 8) / 16777216.0f; }  // [0,1)
};

// CPU reference: stable ordinal rank per row. Mirrors np_cs_rank exactly.
void cpu_cs_rank(const float* X, const uint8_t* mask, int T, int N, bool descending,
                 float* out) {
  std::vector<float> ov(static_cast<size_t>(T) * N, nan_payload());
  for (int t = 0; t < T; ++t) {
    const float* row = X + static_cast<size_t>(t) * N;
    std::vector<std::pair<float, int>> items;
    items.reserve(N);
    for (int j = 0; j < N; ++j) {
      float v = descending ? -row[j] : row[j];
      bool valid =
          std::isfinite(v) && (mask == nullptr || mask[static_cast<size_t>(t) * N + j]);
      if (valid) items.emplace_back(v, j);
    }
    std::stable_sort(items.begin(), items.end(),
                     [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                       return a.first < b.first;
                     });
    for (int k = 0; k < static_cast<int>(items.size()); ++k) {
      ov[static_cast<size_t>(t) * N + items[k].second] = static_cast<float>(k + 1);
    }
  }
  std::memcpy(out, ov.data(), static_cast<size_t>(T) * N * sizeof(float));
}

// Bitwise compare of two f32 arrays (NaN payloads must match exactly).
bool bitwise_eq(const float* a, const float* b, int n) {
  for (int i = 0; i < n; ++i) {
    if (f32_bits(a[i]) != f32_bits(b[i])) return false;
  }
  return true;
}

int g_fail = 0;

void report_case(const char* name, bool ok, const char* detail) {
  printf("  [%s] %s: %s\n", ok ? "PASS" : "FAIL", name, ok ? "" : detail);
  if (!ok) ++g_fail;
}

// Run GPU vs CPU reference on a concrete panel, expect bitwise equality.
void check_panel(const char* name, const float* X, const uint8_t* mask, int T, int N,
                 bool descending) {
  std::vector<float> gpu(static_cast<size_t>(T) * N);
  std::vector<float> cpu(static_cast<size_t>(T) * N);
  int rc = cs_rank_gpu(X, mask, T, N, descending, gpu.data());
  cpu_cs_rank(X, mask, T, N, descending, cpu.data());
  char detail[192];
  if (rc != 0) {
    std::snprintf(detail, sizeof(detail), "cs_rank_gpu rc=%d", rc);
    report_case(name, false, detail);
    return;
  }
  bool ok = bitwise_eq(gpu.data(), cpu.data(), T * N);
  if (!ok) {
    // find first mismatch for diagnostics
    std::snprintf(detail, sizeof(detail), "first mismatch at idx=%d (gpu=0x%08x cpu=0x%08x)",
                  [&]() {
                    for (int i = 0; i < T * N; ++i)
                      if (f32_bits(gpu[static_cast<size_t>(i)]) !=
                          f32_bits(cpu[static_cast<size_t>(i)]))
                        return i;
                    return -1;
                  }(),
                  [&]() {
                    for (int i = 0; i < T * N; ++i)
                      if (f32_bits(gpu[static_cast<size_t>(i)]) !=
                          f32_bits(cpu[static_cast<size_t>(i)]))
                        return f32_bits(gpu[static_cast<size_t>(i)]);
                    return 0u;
                  }(),
                  [&]() {
                    for (int i = 0; i < T * N; ++i)
                      if (f32_bits(gpu[static_cast<size_t>(i)]) !=
                          f32_bits(cpu[static_cast<size_t>(i)]))
                        return f32_bits(cpu[static_cast<size_t>(i)]);
                    return 0u;
                  }());
    report_case(name, false, detail);
  } else {
    report_case(name, true, "");
  }
}

// Workspace path (P0-2, 2026-08-05): run the panel through a persistent
// workspace and assert the output is bitwise identical to the non-workspace
// path -- proves the cached-buffer reuse is transparent (same numerics).
void check_workspace_panel(const char* name, const float* X, const uint8_t* mask,
                           int T, int N, bool descending, cs_rank_workspace* ws) {
  std::vector<float> plain(static_cast<size_t>(T) * N);
  std::vector<float> ws_out(static_cast<size_t>(T) * N);
  int rc1 = cs_rank_gpu(X, mask, T, N, descending, plain.data());
  int rc2 = cs_rank_gpu(X, mask, T, N, descending, ws_out.data(), nullptr, ws);
  if (rc1 != 0 || rc2 != 0) {
    report_case(name, false, "cs_rank_gpu rc failure");
    return;
  }
  bool ok = bitwise_eq(plain.data(), ws_out.data(), T * N);
  report_case(name, ok, ok ? "" : "workspace output differs from non-workspace");
}

// Explicit expected-value assertion (guards against a symmetric CPU-reference bug).
void check_expect(const char* name, const float* X, const uint8_t* mask, int T, int N,
                  bool descending, const std::vector<float>& expected) {
  std::vector<float> gpu(static_cast<size_t>(T) * N);
  int rc = cs_rank_gpu(X, mask, T, N, descending, gpu.data());
  if (rc != 0) {
    report_case(name, false, "cs_rank_gpu failed");
    return;
  }
  bool ok = expected.size() == static_cast<size_t>(T) * N;
  if (ok) {
    for (size_t i = 0; i < expected.size(); ++i) {
      if (f32_bits(gpu[i]) != f32_bits(expected[i])) {
        ok = false;
        break;
      }
    }
  }
  char detail[192];
  if (!ok) {
    std::snprintf(detail, sizeof(detail), "expected[0]=0x%08x got[0]=0x%08x",
                  expected.empty() ? 0u : f32_bits(expected[0]),
                  gpu.empty() ? 0u : f32_bits(gpu[0]));
  }
  report_case(name, ok, ok ? "" : detail);
}

// Randomize a panel with finite / NaN / +-inf / +-0 / ties, optional mask.
void random_panel(Lcg* rng, float* X, uint8_t* mask, int T, int N, bool with_mask) {
  float prev = 0.0f;
  for (int i = 0; i < T * N; ++i) {
    uint32_t r = rng->next() % 100;
    float v;
    if (r < 70) {  // finite random
      v = (rng->uniform() * 20.0f - 10.0f);
    } else if (r < 80) {  // exact zero (mix +0 / -0)
      v = (rng->next() & 1u) ? 0.0f : -0.0f;
    } else if (r < 86) {  // NaN (varied payloads)
      uint32_t nb = 0x7fc00000u | (rng->next() & 0x1FFFFFu);
      std::memcpy(&v, &nb, sizeof(float));
    } else if (r < 93) {  // +inf / -inf
      v = (rng->next() & 1u) ? INFINITY : -INFINITY;
    } else {  // duplicate previous value (dense ties)
      v = prev;
    }
    // small tie groups: ~15% chance to repeat previous finite value
    if (r >= 70 && (rng->next() % 100) < 15 && i > 0) v = prev;
    X[i] = v;
    prev = v;
    if (with_mask) mask[i] = ((rng->next() % 100) < 80) ? 1 : 0;
  }
  if (with_mask) {
    // force at least one all-False row and (if T>=2) one all-True row.
    // T==1 must not write row 1 (out of bounds on the (T*N) mask buffer).
    for (int j = 0; j < N; ++j) mask[j] = 0;  // row 0 all false
    if (T >= 2) {
      for (int j = 0; j < N; ++j) mask[static_cast<size_t>(N) + j] = 1;  // row 1 all true
    }
  }
}

}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);  // flush immediately for crash diagnostics
  int dev = 0;
  cudaError_t err = cudaGetDevice(&dev);
  if (err != cudaSuccess) {
    printf("cudaGetDevice FAIL: %s\n", cudaGetErrorString(err));
    return 1;
  }
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev);
  printf("GPU: %s (cc %d.%d), SM %d\n", prop.name, prop.major, prop.minor,
         prop.multiProcessorCount);
  printf("== cs_rank v0 selfcheck ==\n");

  // ---- parity anchor cases (from parity_anchors_v1 manifest / npz) ---------
  {
    // rank_tie: [3,3,1] -> asc [2,3,1]; desc (negate [-3,-3,-1]) -> [1,2,3]
    float x[] = {3.0f, 3.0f, 1.0f};
    check_expect("rank_tie_asc", x, nullptr, 1, 3, false, {2.0f, 3.0f, 1.0f});
    check_expect("rank_tie_desc", x, nullptr, 1, 3, true, {1.0f, 2.0f, 3.0f});
  }
  {
    // rank_zero: [0.0, -0.0, 1.0] -> asc [1,2,3] (+-0 fold to tie, col order)
    float x[] = {0.0f, -0.0f, 1.0f};
    check_expect("rank_zero_asc", x, nullptr, 1, 3, false, {1.0f, 2.0f, 3.0f});
  }
  {
    // rank_nan_inf: [1, nan, inf, -inf, 2] -> asc [1,nan,nan,nan,2]
    float x[] = {1.0f, nan_payload(), INFINITY, -INFINITY, 2.0f};
    std::vector<float> exp = {1.0f, nan_payload(), nan_payload(), nan_payload(), 2.0f};
    check_expect("rank_nan_inf_asc", x, nullptr, 1, 5, false, exp);
  }
  {
    // rank_mask: values [1,2,3,4], mask [T,F,T,T] -> [1,nan,2,3]
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    uint8_t m[] = {1, 0, 1, 1};
    std::vector<float> exp = {1.0f, nan_payload(), 2.0f, 3.0f};
    check_expect("rank_mask_asc", x, m, 1, 4, false, exp);
  }
  {
    // multi-row: constant section (all equal -> rank 1..K by column index),
    // empty section (all invalid -> all NaN), single-element row.
    float x[] = {2.0f, 2.0f, 2.0f,   // constant
                 nan_payload(), nan_payload(), -INFINITY,  // empty (all invalid)
                 5.0f, 7.0f, 6.0f};  // plain
    std::vector<float> exp = {1.0f, 2.0f, 3.0f, nan_payload(), nan_payload(),
                              nan_payload(), 1.0f, 3.0f, 2.0f};
    check_expect("multirow_const_empty_plain", x, nullptr, 3, 3, false, exp);
  }

  // ---- randomized panels: bitwise vs CPU reference -------------------------
  printf("== randomized bitwise checks ==\n");
  {
    const int kSizes[][2] = {
        {1, 1}, {1, 5}, {1, 32}, {3, 7}, {5, 100}, {1, 1000}, {2, 256},
        {20, 300}, {8, 128}, {6, 64}, {1, 2000}, {12, 1500}};
    for (int run = 0; run < 3; ++run) {
      for (const auto& sz : kSizes) {
        int T = sz[0], N = sz[1];
        Lcg rng(0xC0FFEEu + static_cast<uint32_t>(run) * 7919u + static_cast<uint32_t>(T) * 131u + static_cast<uint32_t>(N));
        std::vector<float> X(static_cast<size_t>(T) * N);
        std::vector<uint8_t> M(static_cast<size_t>(T) * N);
        for (int with_mask = 0; with_mask < 2; ++with_mask) {
          random_panel(&rng, X.data(), M.data(), T, N, with_mask != 0);
          char nm[128];
          std::snprintf(nm, sizeof(nm), "rand T=%d N=%d mask=%d desc=%d run=%d", T, N,
                        with_mask, 0, run);
          check_panel(nm, X.data(), with_mask ? M.data() : nullptr, T, N, false);
          std::snprintf(nm, sizeof(nm), "rand T=%d N=%d mask=%d desc=%d run=%d", T, N,
                        with_mask, 1, run);
          check_panel(nm, X.data(), with_mask ? M.data() : nullptr, T, N, true);
        }
      }
    }
  }

  // ---- workspace path: cached-buffer reuse must be bitwise transparent --------
  printf("== workspace path ==\n");
  {
    cs_rank_workspace ws;
    // 1. basic + masked + desc panels (reuse, both mask states)
    {
      float X1[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
      check_workspace_panel("ws basic asc", X1, nullptr, 1, 5, false, &ws);
      check_workspace_panel("ws basic desc", X1, nullptr, 1, 5, true, &ws);
      float X2[8] = {1, 2, 3, 4, 5, 6, 7, 8};
      uint8_t M2[8] = {1, 0, 1, 1, 0, 1, 1, 1};
      check_workspace_panel("ws masked", X2, M2, 1, 8, false, &ws);
      check_workspace_panel("ws masked desc", X2, M2, 1, 8, true, &ws);
    }
    // 2. shape switch (same ws, different T/N -> realloc + correct) and mask
    //    presence switch (has_mask is a shape key -> realloc)
    {
      Lcg rng(0x7A0E5E4Cu);
      const int T = 4, N = 16;
      std::vector<float> X3(static_cast<size_t>(T) * N);
      std::vector<uint8_t> M3(static_cast<size_t>(T) * N);
      for (size_t i = 0; i < X3.size(); ++i) {
        X3[i] = rng.uniform() * 20.0f - 10.0f;
        M3[i] = (rng.next() % 100) < 80 ? 1 : 0;
      }
      check_workspace_panel("ws shape-switch T4 N16", X3.data(), nullptr, T, N, false, &ws);
      check_workspace_panel("ws shape-switch desc", X3.data(), nullptr, T, N, true, &ws);
      check_workspace_panel("ws mask-on (realloc)", X3.data(), M3.data(), T, N, false, &ws);
      check_workspace_panel("ws mask-off (realloc)", X3.data(), nullptr, T, N, false, &ws);
    }
    // 3. clear then reuse (workspace is usable again after releasing buffers)
    {
      cs_rank_workspace_clear(&ws);
      // direct double-clear: second clear must be a safe no-op (review F04;
      // cudaFree / MemTracker::Free are nullptr no-ops, pointers are nulled).
      cs_rank_workspace_clear(&ws);
      float X4[3] = {9.0f, 3.0f, 7.0f};
      check_workspace_panel("ws after-clear reuse", X4, nullptr, 1, 3, false, &ws);
    }
    // 4. contract error with ws: invalid input rejected, workspace still usable
    {
      float out2[2];
      int rc = cs_rank_gpu(nullptr, nullptr, 1, 2, false, out2, nullptr, &ws);
      report_case("ws null input rejected", rc != 0, "expected nonzero rc");
      float X5[3] = {5.0f, 1.0f, 4.0f};
      check_workspace_panel("ws usable after error", X5, nullptr, 1, 3, false, &ws);
    }
    // 5. stale-buffer self-heal (review F05): corrupt the workspace's output
    //    buffer between calls, then a valid call must overwrite it and produce
    //    correct results (every buffer is written before read on the next call).
    {
      // poison d_out directly (raw scratch write) -- a mid-pipeline failure
      // leaves buffers half-written, and the next call must self-heal.
      if (ws.d_out != nullptr && ws.n_items > 0) {
        cudaMemset(ws.d_out, 0x7F, static_cast<size_t>(ws.n_items) * sizeof(float));
      }
      float X6[3] = {2.0f, 1.0f, 3.0f};
      check_workspace_panel("ws self-heal after poison", X6, nullptr, 1, 3, false, &ws);
    }
    cs_rank_workspace_clear(&ws);  // final cleanup
  }

  printf("== summary ==\n");
  if (g_fail == 0) {
    printf("ALL PASS (%s)\n", "cs_rank v0 bitwise parity with CPU reference");
    return 0;
  }
  printf("FAILURES: %d\n", g_fail);
  return 1;
}
