// factor-cuda -- PoC 3 parameter_scan v0 selfcheck.
//
// Verifies src/parameter_scan.cu against:
//   (a) an in-process CPU reference per group (same as np_cs_rank / np_parameter_scan
//       dictionary order: g0 (asc,masked), g1 (asc,unmasked), g2 (desc,masked),
//       g3 (desc,unmasked));
//   (b) per-group bitwise equality with a direct cs_rank_gpu call using the
//       matching (descending, use-mask) args (proves the reused pipeline is
//       exactly the single-call path);
//   (c) explicit expected-value anchors (dictionary order + mask semantics);
//   (d) determinism (two runs bitwise identical) and error-path smoke.
//
// ASCII-only comments (nvcc/GBK pitfall).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>
#include <cuda_runtime.h>
#include "parameter_scan.cuh"
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

struct Lcg {
  uint64_t s;
  explicit Lcg(uint64_t seed) : s(seed) {}
  uint32_t next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(s >> 33);
  }
  float uniform() { return static_cast<float>(next() >> 8) / 16777216.0f; }
};

// CPU reference per group (mirrors np_cs_rank / cs_rank selfcheck cpu_cs_rank).
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

bool bitwise_eq(const float* a, const float* b, int n) {
  for (int i = 0; i < n; ++i)
    if (f32_bits(a[i]) != f32_bits(b[i])) return false;
  return true;
}

int g_fail = 0;

void report_case(const char* name, bool ok, const char* detail) {
  printf("  [%s] %s: %s\n", ok ? "PASS" : "FAIL", name, ok ? "" : detail);
  if (!ok) ++g_fail;
}

// On a successful scan every group_status must be 0 (all groups attempted/ok).
bool group_status_all_ok(const int gs[4]) {
  for (int g = 0; g < 4; ++g)
    if (gs[g] != 0) return false;
  return true;
}

// Per-group check: scan group g == CPU reference (bitwise).
void check_scan_vs_cpu(const char* name, const float* X, const uint8_t* mask, int T, int N) {
  std::vector<float> gpu[4];
  std::vector<float> cpu[4];
  float* h_out[4];
  int gs[4];
  for (int g = 0; g < 4; ++g) {
    gpu[g].resize(static_cast<size_t>(T) * N);
    cpu[g].resize(static_cast<size_t>(T) * N);
    h_out[g] = gpu[g].data();
  }
  int rc = parameter_scan_gpu(X, mask, T, N, h_out, gs);
  if (rc != 0) {
    char d[96];
    std::snprintf(d, sizeof(d), "parameter_scan_gpu rc=%d", rc);
    report_case(name, false, d);
    return;
  }
  char detail[256];
  if (!group_status_all_ok(gs)) {
    std::snprintf(detail, sizeof(detail), "group_status not all ok (%d,%d,%d,%d)",
                  gs[0], gs[1], gs[2], gs[3]);
    report_case(name, false, detail);
    return;
  }
  for (int g = 0; g < 4; ++g) {
    bool descending = (g >= 2);
    const uint8_t* m = (g % 2 == 0) ? mask : nullptr;
    cpu_cs_rank(X, m, T, N, descending, cpu[g].data());
    if (!bitwise_eq(gpu[g].data(), cpu[g].data(), T * N)) {
      std::snprintf(detail, sizeof(detail), "group %d mismatch vs CPU", g);
      report_case(name, false, detail);
      return;
    }
  }
  report_case(name, true, "");
}

// Group g must equal a direct cs_rank_gpu call with the matching args.
void check_scan_vs_single(const char* name, const float* X, const uint8_t* mask, int T, int N) {
  std::vector<float> gpu[4];
  std::vector<float> single[4];
  float* h_out[4];
  int gs[4];
  for (int g = 0; g < 4; ++g) {
    gpu[g].resize(static_cast<size_t>(T) * N);
    single[g].resize(static_cast<size_t>(T) * N);
    h_out[g] = gpu[g].data();
  }
  int rc = parameter_scan_gpu(X, mask, T, N, h_out, gs);
  if (rc != 0) {
    char d[96];
    std::snprintf(d, sizeof(d), "parameter_scan_gpu rc=%d", rc);
    report_case(name, false, d);
    return;
  }
  char detail[256];
  if (!group_status_all_ok(gs)) {
    std::snprintf(detail, sizeof(detail), "group_status not all ok (%d,%d,%d,%d)",
                  gs[0], gs[1], gs[2], gs[3]);
    report_case(name, false, detail);
    return;
  }
  for (int g = 0; g < 4; ++g) {
    bool descending = (g >= 2);
    const uint8_t* m = (g % 2 == 0) ? mask : nullptr;
    rc = cs_rank_gpu(X, m, T, N, descending, single[g].data());
    if (rc != 0) {
      std::snprintf(detail, sizeof(detail), "cs_rank_gpu(g=%d) rc=%d", g, rc);
      report_case(name, false, detail);
      return;
    }
    if (!bitwise_eq(gpu[g].data(), single[g].data(), T * N)) {
      std::snprintf(detail, sizeof(detail), "group %d != single cs_rank", g);
      report_case(name, false, detail);
      return;
    }
  }
  report_case(name, true, "");
}

void random_panel(Lcg* rng, float* X, uint8_t* mask, int T, int N, bool with_mask) {
  float prev = 0.0f;
  for (int i = 0; i < T * N; ++i) {
    uint32_t r = rng->next() % 100;
    float v;
    if (r < 70) v = (rng->uniform() * 20.0f - 10.0f);
    else if (r < 80) v = (rng->next() & 1u) ? 0.0f : -0.0f;
    else if (r < 86) {
      uint32_t nb = 0x7fc00000u | (rng->next() & 0x1FFFFFu);
      std::memcpy(&v, &nb, sizeof(float));
    } else if (r < 93) v = (rng->next() & 1u) ? INFINITY : -INFINITY;
    else v = prev;
    if (r >= 70 && (rng->next() % 100) < 15 && i > 0) v = prev;
    X[i] = v;
    prev = v;
    if (with_mask) mask[i] = ((rng->next() % 100) < 80) ? 1 : 0;
  }
  if (with_mask) {
    for (int j = 0; j < N; ++j) mask[j] = 0;  // row 0 all false
    if (T >= 2) {
      for (int j = 0; j < N; ++j) mask[static_cast<size_t>(N) + j] = 1;  // row 1 all true
    }
  }
}

}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
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
  printf("== parameter_scan v0 selfcheck ==\n");

  // ---- error-path smoke ------------------------------------------------------
  {
    float dummy_in[4] = {1, 2, 3, 4};
    float dummy_out[4] = {0, 0, 0, 0};
    float* h_out4[4] = {dummy_out, dummy_out, dummy_out, dummy_out};
    int gs[4];
    int rc_null = parameter_scan_gpu(nullptr, nullptr, 1, 2, h_out4, gs);
    float* bad_out[4] = {dummy_out, dummy_out, nullptr, dummy_out};
    int rc_partial = parameter_scan_gpu(dummy_in, nullptr, 1, 2, bad_out, gs);
    int rc_t0 = parameter_scan_gpu(dummy_in, nullptr, 0, 2, h_out4, gs);
    int rc_n0 = parameter_scan_gpu(dummy_in, nullptr, 1, 0, h_out4, gs);
    // N > 2^24 returns -2 before touching X/out (only host preconditions).
    int rc_ncap = parameter_scan_gpu(dummy_in, nullptr, 1, (1 << 24) + 1, h_out4, gs);
    // N legal (2^24) but T*N > INT32_MAX returns -3 before any allocation.
    int rc_total = parameter_scan_gpu(dummy_in, nullptr, 129, (1 << 24), h_out4, gs);
    printf("error-path smoke: null=%d partial=%d T0=%d N0=%d Ncap=%d totalcap=%d\n",
           rc_null, rc_partial, rc_t0, rc_n0, rc_ncap, rc_total);
    bool ok = (rc_null == -1 && rc_partial == -1 && rc_t0 == -1 && rc_n0 == -1 &&
               rc_ncap == -2 && rc_total == -3);
    // Precondition failure must leave every group_status as "not executed".
    int not_exec = 0;
    for (int g = 0; g < 4; ++g) if (gs[g] == -100) ++not_exec;
    ok = ok && (not_exec == 4);
    report_case("error-path smoke (+ group_status all -100)", ok,
                ok ? "" : "unexpected rc codes or group_status");
  }

  // ---- explicit expected values (dictionary order + mask semantics) ----------
  {
    // X (1x5): [1, nan, 3, 4, 2]; mask [1,1,0,1,1].
    // g0 (asc,masked):  valid+masked cols {0,3,4} vals {1,4,2} -> {1,nan,nan,3,2}
    // g1 (asc,unmasked): finite cols {0,2,3,4} vals {1,3,4,2} -> {1,nan,3,4,2}
    // g2 (desc,masked): negate {-1,-4,-2} sorted 4,2,1 -> {3,nan,nan,1,2}
    // g3 (desc,unmasked): negate {-1,-3,-4,-2} sorted 4,3,2,1 -> {4,nan,2,1,3}
    float x[] = {1.0f, nan_payload(), 3.0f, 4.0f, 2.0f};
    uint8_t m[] = {1, 1, 0, 1, 1};
    std::vector<float> exp[4] = {
        {1.0f, nan_payload(), nan_payload(), 3.0f, 2.0f},
        {1.0f, nan_payload(), 3.0f, 4.0f, 2.0f},
        {3.0f, nan_payload(), nan_payload(), 1.0f, 2.0f},
        {4.0f, nan_payload(), 2.0f, 1.0f, 3.0f},
    };
    std::vector<float> gpu[4];
    float* h_out[4];
    int gs[4];
    for (int g = 0; g < 4; ++g) {
      gpu[g].resize(5);
      h_out[g] = gpu[g].data();
    }
    int rc = parameter_scan_gpu(x, m, 1, 5, h_out, gs);
    if (rc != 0) {
      report_case("dict_order_anchor", false, "parameter_scan_gpu failed");
    } else {
      bool ok = group_status_all_ok(gs);
      for (int g = 0; g < 4 && ok; ++g)
        if (!bitwise_eq(gpu[g].data(), exp[g].data(), 5)) ok = false;
      report_case("dict_order_anchor", ok, ok ? "" : "group value mismatch");
    }
  }

  // ---- randomized panels vs CPU reference (masked and unmasked) --------------
  printf("== randomized vs CPU reference ==\n");
  {
    const int kSizes[][2] = {{1, 1}, {1, 5}, {3, 7}, {5, 100}, {1, 1000},
                             {2, 256}, {20, 300}, {8, 128}, {6, 64}, {1, 2000}};
    for (int run = 0; run < 3; ++run) {
      for (const auto& sz : kSizes) {
        int T = sz[0], N = sz[1];
        Lcg rng(0x5CA5u + static_cast<uint32_t>(run) * 7919u + static_cast<uint32_t>(T) * 131u + static_cast<uint32_t>(N));
        std::vector<float> X(static_cast<size_t>(T) * N);
        std::vector<uint8_t> M(static_cast<size_t>(T) * N);
        for (int with_mask = 0; with_mask < 2; ++with_mask) {
          random_panel(&rng, X.data(), M.data(), T, N, with_mask != 0);
          char nm[128];
          std::snprintf(nm, sizeof(nm), "rand T=%d N=%d mask=%d run=%d", T, N, with_mask, run);
          check_scan_vs_cpu(nm, X.data(), with_mask ? M.data() : nullptr, T, N);
          std::snprintf(nm, sizeof(nm), "rand T=%d N=%d mask=%d run=%d (single)", T, N, with_mask, run);
          check_scan_vs_single(nm, X.data(), with_mask ? M.data() : nullptr, T, N);
        }
      }
    }
  }

  // ---- determinism ------------------------------------------------------------
  {
    Lcg rng(0xDEEF0u);
    const int T = 8, N = 6;
    std::vector<float> X(static_cast<size_t>(T) * N);
    std::vector<uint8_t> M(static_cast<size_t>(T) * N);
    random_panel(&rng, X.data(), M.data(), T, N, true);
    std::vector<float> o1[4], o2[4];
    float* h1[4], *h2[4];
    int gs1[4], gs2[4];
    for (int g = 0; g < 4; ++g) {
      o1[g].resize(static_cast<size_t>(T) * N);
      o2[g].resize(static_cast<size_t>(T) * N);
      h1[g] = o1[g].data();
      h2[g] = o2[g].data();
    }
    int rc1 = parameter_scan_gpu(X.data(), M.data(), T, N, h1, gs1);
    int rc2 = parameter_scan_gpu(X.data(), M.data(), T, N, h2, gs2);
    bool same = (rc1 == 0 && rc2 == 0) && group_status_all_ok(gs1) &&
                group_status_all_ok(gs2);
    for (int g = 0; g < 4 && same; ++g)
      if (!bitwise_eq(o1[g].data(), o2[g].data(), T * N)) same = false;
    report_case("determinism (two runs bitwise identical)", same, same ? "" : "rc1/rc2 or value diff");
  }

  // ---- workspace path (P2 PoC4 perf, 2026-08-08): cached-buffer reuse (the
  // buffer set is the cs_rank_workspace set) must be bitwise transparent --
  // ws vs non-ws identical, reuse, mask on/off, shape switch, clear-then-reuse,
  // and usable after a contract error.
  printf("== workspace path ==\n");
  {
    auto check_ws = [&](const char* name, const float* X, const uint8_t* mask,
                        int T, int N, cs_rank_workspace* ws) {
      std::vector<float> o1[4], o2[4];
      float* h1[4], *h2[4];
      int gs1[4], gs2[4];
      for (int g = 0; g < 4; ++g) {
        o1[g].resize(static_cast<size_t>(T) * N);
        o2[g].resize(static_cast<size_t>(T) * N);
        h1[g] = o1[g].data();
        h2[g] = o2[g].data();
      }
      int rc1 = parameter_scan_gpu(X, mask, T, N, h1, gs1);
      int rc2 = parameter_scan_gpu(X, mask, T, N, h2, gs2, nullptr, nullptr,
                                   nullptr, nullptr, ws);
      bool ok = (rc1 == 0 && rc2 == 0) && group_status_all_ok(gs1) &&
                group_status_all_ok(gs2);
      for (int g = 0; g < 4 && ok; ++g)
        if (!bitwise_eq(o1[g].data(), o2[g].data(), T * N)) ok = false;
      report_case(name, ok, ok ? "" : "workspace output differs from non-workspace");
    };
    cs_rank_workspace ws;
    Lcg rng(0x575331u);
    {
      const int T = 10, N = 12;
      std::vector<float> X(static_cast<size_t>(T) * N);
      std::vector<uint8_t> M(static_cast<size_t>(T) * N);
      random_panel(&rng, X.data(), M.data(), T, N, true);
      check_ws("ws basic masked", X.data(), M.data(), T, N, &ws);
      check_ws("ws reuse same shape", X.data(), M.data(), T, N, &ws);
      check_ws("ws mask-off (stale-mask guard)", X.data(), nullptr, T, N, &ws);
      const int T2 = 8, N2 = 16;
      std::vector<float> X2(static_cast<size_t>(T2) * N2);
      std::vector<uint8_t> M2(static_cast<size_t>(T2) * N2);
      random_panel(&rng, X2.data(), M2.data(), T2, N2, true);
      check_ws("ws shape-switch", X2.data(), M2.data(), T2, N2, &ws);
    }
    cs_rank_workspace_clear(&ws);
    {
      const int T = 5, N = 7;
      std::vector<float> X(static_cast<size_t>(T) * N);
      std::vector<uint8_t> M(static_cast<size_t>(T) * N);
      random_panel(&rng, X.data(), M.data(), T, N, true);
      check_ws("ws after-clear reuse", X.data(), M.data(), T, N, &ws);
    }
    // contract error with ws: invalid input rejected, workspace still usable.
    {
      float* h4[4] = {nullptr, nullptr, nullptr, nullptr};
      int gs[4];
      int rc = parameter_scan_gpu(nullptr, nullptr, 1, 2, h4, gs, nullptr, nullptr,
                                  nullptr, nullptr, &ws);
      const bool ok = (rc != 0);
      report_case("ws null input rejected", ok, ok ? "" : "expected nonzero rc");
      const int T = 5, N = 7;
      std::vector<float> X(static_cast<size_t>(T) * N);
      std::vector<uint8_t> M(static_cast<size_t>(T) * N);
      random_panel(&rng, X.data(), M.data(), T, N, true);
      check_ws("ws usable after error", X.data(), M.data(), T, N, &ws);
    }
    cs_rank_workspace_clear(&ws);
    // ws + MemTracker (F2/P2-PoC4-02): N reuse rounds must NOT grow
    // alloc_count/live_bytes (BLOCKER F1 leak regression -- a per-call temp
    // leak would grow alloc_count each round); clear returns live to 0.
    {
      factor_cuda::MemTracker mt;
      cs_rank_workspace ws2;
      const int T = 10, N = 12;
      std::vector<float> X(static_cast<size_t>(T) * N);
      std::vector<uint8_t> M(static_cast<size_t>(T) * N);
      random_panel(&rng, X.data(), M.data(), T, N, true);
      bool ok = true;
      size_t alloc0 = 0, live0 = 0;
      for (int rep = 0; rep < 5; ++rep) {
        std::vector<float> o[4]; float* h[4]; int gs[4];
        for (int g = 0; g < 4; ++g) {
          o[g].resize(static_cast<size_t>(T) * N);
          h[g] = o[g].data();
        }
        int rc = parameter_scan_gpu(X.data(), M.data(), T, N, h, gs, &mt, nullptr,
                                    nullptr, nullptr, &ws2);
        if (rc != 0 || !group_status_all_ok(gs)) { ok = false; break; }
        if (rep == 0) { alloc0 = mt.alloc_count(); live0 = mt.live_bytes(); }
        else if (mt.alloc_count() != alloc0 || mt.live_bytes() != live0) {
          ok = false;
          break;
        }
      }
      cs_rank_workspace_clear(&ws2);
      if (mt.live_bytes() != 0) ok = false;
      report_case("ws tracker reuse no-leak (5 rounds, clear live==0)", ok,
                  ok ? "" : "alloc/live grew across reuse or clear != 0");
    }
  }

  printf("== summary ==\n");
  if (g_fail == 0) {
    printf("ALL PASS (parameter_scan v0 bitwise parity)\n");
    return 0;
  }
  printf("FAILURES: %d\n", g_fail);
  return 1;
}
