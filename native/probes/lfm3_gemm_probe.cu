// G4 probe: native lfm::gemm against cuBLAS (called exactly as production calls it) on every production
// GEMM shape. For each case: error of both against an FP64 reference on sampled outputs (in BF16 ulps
// of the reference), agreement between the two, and median time (cold weights for M <= 8).
// Not production code; cuBLAS is linked here only as the comparison.
//   lfm3-gemm-probe [quick] > results.jsonl
#include <cublas_v2.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "lfm/gemm.cuh"

using bf16 = __nv_bfloat16;

static void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static void check(cublasStatus_t s) { if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublas " + std::to_string(s)); }

static std::mt19937_64 rng(1234);
static uint16_t to_bits(float f) {
  uint32_t u; std::memcpy(&u, &f, 4);
  return uint16_t((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
static float from_bits(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
static double ulp_of(double x) {
  // BF16 has 8 significant bits.
  if (x == 0) return std::ldexp(1.0, -133);
  int e; std::frexp(std::fabs(x), &e);
  return std::ldexp(1.0, e - 8);
}

struct Device {
  void* p = nullptr;
  explicit Device(size_t bytes) { check(cudaMalloc(&p, std::max<size_t>(bytes, 16))); }
  ~Device() { cudaFree(p); }
  template <class T> T* as() const { return static_cast<T*>(p); }
};

static void fill(std::vector<uint16_t>& v, float scale) {
  std::normal_distribution<float> d(0.f, scale);
  for (auto& x : v) x = to_bits(d(rng));
}

struct Stats { double max_ulp = 0, sum_ulp = 0; size_t exact = 0, n = 0; };
static void score(Stats& s, double ref, float got) {
  const double rounded = from_bits(to_bits(float(ref)));
  const double err = std::fabs(double(got) - ref) / ulp_of(ref);
  s.max_ulp = std::max(s.max_ulp, err); s.sum_ulp += err; s.exact += double(got) == rounded; ++s.n;
}

template <class F> static double median_ms(F&& f, int reps) {
  cudaEvent_t a, b; check(cudaEventCreate(&a)); check(cudaEventCreate(&b));
  std::vector<float> t;
  for (int i = 0; i < reps; ++i) {
    check(cudaEventRecord(a)); f(i); check(cudaEventRecord(b)); check(cudaEventSynchronize(b));
    float ms; check(cudaEventElapsedTime(&ms, a, b)); t.push_back(ms);
  }
  cudaEventDestroy(a); cudaEventDestroy(b);
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

static cublasHandle_t blas;
static Device* workspace;

// Projection Y[rows x out] = X · Wᵀ.
static void linear_case(const char* name, int rows, int in, int out, bool cold) {
  const size_t wn = size_t(out) * in, xn = size_t(rows) * in, yn = size_t(rows) * out;
  // Cold weights: rotate through copies totalling at least 1 GiB (Thor's L2 is far smaller).
  const int copies = cold ? int(std::max<size_t>(2, (size_t(1) << 30) / (wn * 2) + 1)) : 1;
  std::vector<uint16_t> hw(wn), hx(xn);
  fill(hw, 0.02f); fill(hx, 1.f);
  Device w(wn * 2 * copies), x(xn * 2), yn_(yn * 2), yc(yn * 2);
  for (int c = 0; c < copies; ++c) check(cudaMemcpy(w.as<bf16>() + c * wn, hw.data(), wn * 2, cudaMemcpyHostToDevice));
  check(cudaMemcpy(x.as<bf16>(), hx.data(), xn * 2, cudaMemcpyHostToDevice));
  auto native = [&](int i) { lfm::gemm::run(lfm::gemm::linear(x.as<bf16>(), w.as<bf16>() + (i % copies) * wn, yn_.as<bf16>(), rows, in, out)); };
  auto cublas = [&](int i) {
    float alpha = 1, beta = 0;
    check(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, out, rows, in, &alpha, w.as<bf16>() + (i % copies) * wn, CUDA_R_16BF, in,
                       x.as<bf16>(), CUDA_R_16BF, in, &beta, yc.as<bf16>(), CUDA_R_16BF, out, CUBLAS_COMPUTE_32F,
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  };
  native(0); cublas(0); check(cudaGetLastError()); check(cudaDeviceSynchronize());
  std::vector<uint16_t> hn(yn), hc(yn);
  check(cudaMemcpy(hn.data(), yn_.p, yn * 2, cudaMemcpyDeviceToHost));
  check(cudaMemcpy(hc.data(), yc.p, yn * 2, cudaMemcpyDeviceToHost));
  Stats sn, sc; size_t same = 0;
  for (size_t i = 0; i < yn; ++i) same += hn[i] == hc[i];
  std::uniform_int_distribution<size_t> pick(0, yn - 1);
  for (int s = 0; s < 512; ++s) {
    const size_t idx = s < 4 ? (s == 0 ? 0 : s == 1 ? yn - 1 : s == 2 ? size_t(out - 1) : size_t(rows - 1) * out) : pick(rng);
    const size_t r = idx / out, o = idx % out;
    double ref = 0;
    for (int kk = 0; kk < in; ++kk) ref += double(from_bits(hx[r * in + kk])) * double(from_bits(hw[o * in + kk]));
    score(sn, ref, from_bits(hn[idx])); score(sc, ref, from_bits(hc[idx]));
  }
  const int reps = rows >= 4096 ? 7 : 21;
  native(0); cublas(0);
  const double tn = median_ms(native, reps), tc = median_ms(cublas, reps);
  const double flops = 2.0 * rows * in * out, bytes = 2.0 * (wn + xn + yn);
  std::printf("{\"case\":\"%s\",\"kind\":\"linear\",\"m\":%d,\"n\":%d,\"k\":%d,\"cold\":%s,"
              "\"native\":{\"ms\":%.4f,\"tflops\":%.2f,\"gbps\":%.1f,\"max_ulp\":%.3f,\"mean_ulp\":%.4f,\"exact\":%.4f},"
              "\"cublas\":{\"ms\":%.4f,\"tflops\":%.2f,\"gbps\":%.1f,\"max_ulp\":%.3f,\"mean_ulp\":%.4f,\"exact\":%.4f},"
              "\"identical_outputs\":%.5f,\"speed_ratio\":%.3f}\n",
              name, rows, out, in, cold ? "true" : "false",
              tn, flops / tn / 1e9, bytes / tn / 1e6, sn.max_ulp, sn.sum_ulp / sn.n, double(sn.exact) / sn.n,
              tc, flops / tc / 1e9, bytes / tc / 1e6, sc.max_ulp, sc.sum_ulp / sc.n, double(sc.exact) / sc.n,
              double(same) / yn, tn / tc);
  std::fflush(stdout);
}

// Attention for one layer: 16 query heads over 4 KV heads, head_dim 128, `rows` queries, `end` keys.
// QK: scores[h] (rows x end, pitch) = Q_h · K_{h/4}ᵀ.  PV: out_h (rows x 128) = P_h (rows x pitch) · V_{h/4}.
static void attention_case(int rows, int end) {
  const int H = 16, G = 4, D = 128, hidden = H * D;
  const int pitch = (end + 7) / 8 * 8;
  const size_t qn = size_t(rows) * hidden, kn = size_t(H / G) * end * D, sn_ = size_t(H) * rows * pitch;
  std::vector<uint16_t> hq(qn), hk(kn), hv(kn), hp(sn_, 0);
  fill(hq, 1.f); fill(hk, 1.f); fill(hv, 1.f);
  std::uniform_real_distribution<float> u(0.f, 1.f / end * 2);
  for (int h = 0; h < H; ++h)
    for (int r = 0; r < rows; ++r)
      for (int e = 0; e < end; ++e) hp[(size_t(h) * rows + r) * pitch + e] = to_bits(u(rng));
  Device q(qn * 2), k(kn * 2), v(kn * 2), p(sn_ * 2), sN(sn_ * 2), sC(sn_ * 2), oN(qn * 2), oC(qn * 2);
  check(cudaMemcpy(q.p, hq.data(), qn * 2, cudaMemcpyHostToDevice));
  check(cudaMemcpy(k.p, hk.data(), kn * 2, cudaMemcpyHostToDevice));
  check(cudaMemcpy(v.p, hv.data(), kn * 2, cudaMemcpyHostToDevice));
  check(cudaMemcpy(p.p, hp.data(), sn_ * 2, cudaMemcpyHostToDevice));
  check(cudaMemset(sN.p, 0, sn_ * 2)); check(cudaMemset(sC.p, 0, sn_ * 2));
  lfm::gemm::Problem qk;
  qk.a = q.as<bf16>(); qk.b = k.as<bf16>(); qk.c = sN.as<bf16>(); qk.m = rows; qk.n = end; qk.k = D;
  qk.lda = hidden; qk.ldb = D; qk.ldc = pitch; qk.stride_a = D; qk.stride_b = size_t(end) * D; qk.stride_c = size_t(rows) * pitch;
  qk.batch = H; qk.group = G;
  lfm::gemm::Problem pv;
  pv.a = p.as<bf16>(); pv.b = v.as<bf16>(); pv.c = oN.as<bf16>(); pv.m = rows; pv.n = D; pv.k = pitch; pv.b_rows = end;
  pv.lda = pitch; pv.ldb = D; pv.ldc = hidden; pv.stride_a = size_t(rows) * pitch; pv.stride_b = size_t(end) * D; pv.stride_c = D;
  pv.batch = H; pv.group = G; pv.b_k_contiguous = false;
  Device scratch(std::max<size_t>(16, lfm::gemm::workspace_bytes(rows, D, pitch, H)));
  pv.workspace = scratch.as<float>(); pv.workspace_size = lfm::gemm::workspace_bytes(rows, D, pitch, H);
  std::vector<const void*> pa(H), pb(H); std::vector<void*> pc(H);
  Device da(H * 8), db(H * 8), dc(H * 8), da2(H * 8), db2(H * 8), dc2(H * 8);
  for (int h = 0; h < H; ++h) { pa[h] = k.as<bf16>() + size_t(h / G) * end * D; pb[h] = q.as<bf16>() + h * D; pc[h] = sC.as<bf16>() + size_t(h) * rows * pitch; }
  check(cudaMemcpy(da.p, pa.data(), H * 8, cudaMemcpyHostToDevice)); check(cudaMemcpy(db.p, pb.data(), H * 8, cudaMemcpyHostToDevice));
  check(cudaMemcpy(dc.p, pc.data(), H * 8, cudaMemcpyHostToDevice));
  for (int h = 0; h < H; ++h) { pa[h] = v.as<bf16>() + size_t(h / G) * end * D; pb[h] = p.as<bf16>() + size_t(h) * rows * pitch; pc[h] = oC.as<bf16>() + h * D; }
  check(cudaMemcpy(da2.p, pa.data(), H * 8, cudaMemcpyHostToDevice)); check(cudaMemcpy(db2.p, pb.data(), H * 8, cudaMemcpyHostToDevice));
  check(cudaMemcpy(dc2.p, pc.data(), H * 8, cudaMemcpyHostToDevice));
  float alpha = 1, beta = 0;
  auto nqk = [&](int) { lfm::gemm::run(qk); };
  auto cqk = [&](int) {
    check(cublasGemmBatchedEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, end, rows, D, &alpha, da.as<const void*>(), CUDA_R_16BF, D,
                              db.as<const void*>(), CUDA_R_16BF, hidden, &beta, dc.as<void*>(), CUDA_R_16BF, pitch, H,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  };
  auto npv = [&](int) { lfm::gemm::run(pv); };
  auto cpv = [&](int) {
    check(cublasGemmBatchedEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, D, rows, end, &alpha, da2.as<const void*>(), CUDA_R_16BF, D,
                              db2.as<const void*>(), CUDA_R_16BF, pitch, &beta, dc2.as<void*>(), CUDA_R_16BF, hidden, H,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  };
  nqk(0); cqk(0); npv(0); cpv(0); check(cudaGetLastError()); check(cudaDeviceSynchronize());
  std::vector<uint16_t> a1(sn_), a2(sn_), b1(qn), b2(qn);
  check(cudaMemcpy(a1.data(), sN.p, sn_ * 2, cudaMemcpyDeviceToHost)); check(cudaMemcpy(a2.data(), sC.p, sn_ * 2, cudaMemcpyDeviceToHost));
  check(cudaMemcpy(b1.data(), oN.p, qn * 2, cudaMemcpyDeviceToHost)); check(cudaMemcpy(b2.data(), oC.p, qn * 2, cudaMemcpyDeviceToHost));
  Stats s1n, s1c, s2n, s2c; size_t same1 = 0, same2 = 0, total1 = 0;
  for (int h = 0; h < H; ++h)
    for (int r = 0; r < rows; ++r)
      for (int e = 0; e < end; ++e, ++total1) { const size_t i = (size_t(h) * rows + r) * pitch + e; same1 += a1[i] == a2[i]; }
  for (size_t i = 0; i < qn; ++i) same2 += b1[i] == b2[i];
  std::uniform_int_distribution<int> ph(0, H - 1), pr(0, rows - 1), pe(0, end - 1), pd(0, D - 1);
  for (int s = 0; s < 256; ++s) {
    const int h = ph(rng), r = pr(rng), e = pe(rng), d = pd(rng);
    double ref = 0;
    for (int kk = 0; kk < D; ++kk) ref += double(from_bits(hq[size_t(r) * hidden + h * D + kk])) * from_bits(hk[(size_t(h / G) * end + e) * D + kk]);
    const size_t i = (size_t(h) * rows + r) * pitch + e;
    score(s1n, ref, from_bits(a1[i])); score(s1c, ref, from_bits(a2[i]));
    double ref2 = 0;
    for (int kk = 0; kk < end; ++kk) ref2 += double(from_bits(hp[(size_t(h) * rows + r) * pitch + kk])) * from_bits(hv[(size_t(h / G) * end + kk) * D + d]);
    const size_t j = size_t(r) * hidden + h * D + d;
    score(s2n, ref2, from_bits(b1[j])); score(s2c, ref2, from_bits(b2[j]));
  }
  const int reps = rows * double(end) > 1e7 ? 7 : 21;
  const double t1n = median_ms(nqk, reps), t1c = median_ms(cqk, reps), t2n = median_ms(npv, reps), t2c = median_ms(cpv, reps);
  for (int which = 0; which < 2; ++which) {
    const Stats& n = which ? s2n : s1n; const Stats& c = which ? s2c : s1c;
    const double tn = which ? t2n : t1n, tc = which ? t2c : t1c;
    std::printf("{\"case\":\"%s\",\"kind\":\"attention\",\"rows\":%d,\"keys\":%d,"
                "\"native\":{\"ms\":%.4f,\"max_ulp\":%.3f,\"mean_ulp\":%.4f,\"exact\":%.4f},"
                "\"cublas\":{\"ms\":%.4f,\"max_ulp\":%.3f,\"mean_ulp\":%.4f,\"exact\":%.4f},"
                "\"identical_outputs\":%.5f,\"speed_ratio\":%.3f}\n",
                which ? "pv" : "qk", rows, end, tn, n.max_ulp, n.sum_ulp / n.n, double(n.exact) / n.n,
                tc, c.max_ulp, c.sum_ulp / c.n, double(c.exact) / c.n,
                which ? double(same2) / qn : double(same1) / total1, tn / tc);
  }
  std::fflush(stdout);
}

int main(int argc, char** argv) try {
  const bool quick = argc > 1 && std::string(argv[1]) == "quick";
  check(cublasCreate(&blas));
  workspace = new Device(4096 * 1024 * 2 + 16 * 1024 * 8);
  check(cublasSetWorkspace(blas, workspace->p, 4096 * 1024 * 2 + 16 * 1024 * 8));
  check(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH));
  struct Shape { const char* name; int in, out; };
  const Shape shapes[] = {{"conv.in_proj", 2048, 6144}, {"o/out/q_proj", 2048, 2048}, {"k/v_proj", 2048, 512},
                          {"gate_proj", 2048, 16}, {"w1/w3", 2048, 7680}, {"w2", 7680, 2048}};
  const int rows_list[] = {1, 2, 8, 34, 256, 4096, 32768};
  for (const auto& s : shapes)
    for (int rows : rows_list) {
      if (quick && rows > 4096) continue;
      linear_case(s.name, rows, s.in, s.out, rows <= 8);
    }
  for (int rows : {1, 2}) linear_case("lm_head", rows, 2048, 128000, true);
  const int attn[][2] = {{1, 300}, {1, 4096}, {1, 32768}, {2, 4096}, {37, 37}, {256, 256}, {128, 4096}, {4096, 4096}, {128, 32768}};
  for (const auto& a : attn) {
    if (quick && a[1] > 4096) continue;
    attention_case(a[0], a[1]);
  }
  cublasDestroy(blas);
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "lfm3-gemm-probe: %s\n", e.what());
  return 1;
}
