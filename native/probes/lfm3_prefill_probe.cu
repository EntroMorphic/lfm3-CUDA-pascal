// Prefill (rows > 8) producer probe. Isolated experiment: no production integration, no acceptance.
// Compares v9::mma_prefill_partials<S,NT> with the frozen v4 runtime wmma_partials bit for bit
// (partials and finished outputs), on random inputs and the real layer-0 conv in_proj prefill
// fixture, and times both against production's cublasGemmEx with cold-cache weight rotation.
// Build: -DV4_HEADER=... -DPREFILL_S=8 -DPREFILL_NT=8 (see tools/thor/build_prefill_probe.sh).
#define THIN_PROBE_NO_MAIN 1
#include "lfm3_thin_v5_probe.cu"

#ifndef PREFILL_S
#define PREFILL_S 8
#endif
#ifndef PREFILL_NT
#define PREFILL_NT 8
#endif

namespace prefill {
using bf16 = __nv_bfloat16;

static void launch_v4(const bf16* w, const bf16* x, float* p, int m, int n, int k, int pn, int pm, int parts) {
  lfm3_wmma_candidate::wmma_partials<<<dim3(pm / 16, pn / 16, parts), 32>>>(w, x, p, m, n, k, pn, pm, parts);
}
static void launch_v9(const bf16* w, const bf16* x, float* p, int m, int n, int k, int pn, int pm, int parts) {
  int warps = std::min(4, pm / 16), tokens = 8 * PREFILL_NT;
  dim3 grid((pm / 16 + warps - 1) / warps, (n + tokens - 1) / tokens, parts);
  v9::mma_prefill_partials<PREFILL_S, PREFILL_NT><<<grid, 32 * warps>>>(w, x, p, m, n, k, pn, pm, parts);
}
static void finish(const float* p, bf16* out, int m, int n, int pm, int pn, int parts) {
  size_t count = size_t(m) * n;
  lfm3_wmma_candidate::wmma_finish<<<(count + 255) / 256, 256>>>(p, out, count, m, n, pm, pn, parts);
}

static int run(const Case& c, std::FILE* report) {
  if (c.n <= 8) throw std::runtime_error("prefill probe requires n > 8");
  int pm = (c.m + 15) / 16 * 16, pn = (c.n + 15) / 16 * 16;  // v4's prefill scratch geometry
  size_t wn = size_t(c.m) * c.k, xn = size_t(c.n) * c.k, pcount = size_t(pm) * pn * c.parts, out = size_t(c.m) * c.n;
  std::vector<uint16_t> hw(wn), hx(xn);
  for (auto& v : hw) v = random_bf16();
  for (auto& v : hx) v = random_bf16();
  bf16 *w, *x, *o4, *o9; float *p4, *p9;
  check(cudaMalloc(&w, wn * 2)); check(cudaMalloc(&x, xn * 2));
  check(cudaMalloc(&p4, pcount * 4)); check(cudaMalloc(&p9, pcount * 4));
  check(cudaMalloc(&o4, out * 2)); check(cudaMalloc(&o9, out * 2));
  check(cudaMemcpy(w, hw.data(), wn * 2, cudaMemcpyHostToDevice));
  check(cudaMemcpy(x, hx.data(), xn * 2, cudaMemcpyHostToDevice));
  check(cudaMemset(p4, 0xA5, pcount * 4)); check(cudaMemset(p9, 0x5A, pcount * 4));  // harness-only sentinels
  launch_v4(w, x, p4, c.m, c.n, c.k, pn, pm, c.parts);
  launch_v9(w, x, p9, c.m, c.n, c.k, pn, pm, c.parts);
  finish(p4, o4, c.m, c.n, pm, pn, c.parts);
  finish(p9, o9, c.m, c.n, pm, pn, c.parts);
  check(cudaGetLastError()); check(cudaDeviceSynchronize());
  std::vector<uint32_t> hp4(pcount), hp9(pcount); std::vector<uint16_t> ho4(out), ho9(out);
  check(cudaMemcpy(hp4.data(), p4, pcount * 4, cudaMemcpyDeviceToHost));
  check(cudaMemcpy(hp9.data(), p9, pcount * 4, cudaMemcpyDeviceToHost));
  check(cudaMemcpy(ho4.data(), o4, out * 2, cudaMemcpyDeviceToHost));
  check(cudaMemcpy(ho9.data(), o9, out * 2, cudaMemcpyDeviceToHost));
  // Partials the finish consumes: rows < m, columns < n, every partition.
  size_t pdiff = 0, pchecked = 0, odiff = 0;
  for (int part = 0; part < c.parts; ++part)
    for (int r = 0; r < c.m; ++r)
      for (int col = 0; col < c.n; ++col, ++pchecked) {
        size_t i = size_t(part) * pm * pn + size_t(r) * pn + col;
        pdiff += hp4[i] != hp9[i];
      }
  for (size_t i = 0; i < out; ++i) odiff += ho4[i] != ho9[i];
  double t4 = 0, t9 = 0, tb = 0;
  if (c.timed) {
    cudaEvent_t a, b; check(cudaEventCreate(&a)); check(cudaEventCreate(&b));
    size_t copies = std::max<size_t>(1, std::min<size_t>(256, (size_t(2) << 30) / (wn * 2) + 1));
    std::vector<bf16*> ws(copies, w);
    for (size_t i = 1; i < copies; ++i) { check(cudaMalloc(&ws[i], wn * 2)); check(cudaMemcpy(ws[i], w, wn * 2, cudaMemcpyDeviceToDevice)); }
    int reps = int(std::max<size_t>(copies, 10));
    auto time = [&](auto launch, float* p) {
      for (int i = 0; i < 3; ++i) launch(ws[i % copies], x, p, c.m, c.n, c.k, pn, pm, c.parts);
      check(cudaEventRecord(a));
      for (int i = 0; i < reps; ++i) launch(ws[(i + 3) % copies], x, p, c.m, c.n, c.k, pn, pm, c.parts);
      check(cudaEventRecord(b)); check(cudaEventSynchronize(b));
      float ms; check(cudaEventElapsedTime(&ms, a, b)); return double(ms) / reps * 1e-3;
    };
    t4 = time(launch_v4, p4);
    t9 = time(launch_v9, p9);
    cublasHandle_t h; if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate");
    float alpha = 1, beta = 0;
    auto gemm = [&](const bf16* wp, const bf16* xp, float*, int m, int n, int k, int, int, int) {
      if (cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, wp, CUDA_R_16BF, k, xp, CUDA_R_16BF, k, &beta, o4,
                       CUDA_R_16BF, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cublasGemmEx");
    };
    tb = time(gemm, p4);
    cublasDestroy(h);
    for (size_t i = 1; i < copies; ++i) check(cudaFree(ws[i]));
    check(cudaEventDestroy(a)); check(cudaEventDestroy(b));
  }
  bool exact = pdiff == 0 && odiff == 0, timing_valid = true;
  double bytes = double(wn) * 2, flops = 2.0 * c.m * c.n * c.k;
  if (c.timed)
    for (double t : {t4, t9, tb}) timing_valid = timing_valid && t > 0 && bytes / t / 1e9 <= dram_peak_gbs();
  std::fprintf(report,
               "{\"S\":%d,\"NT\":%d,\"m\":%d,\"n\":%d,\"k\":%d,\"parts\":%d,\"partials_checked\":%zu,\"partial_diff\":%zu,"
               "\"outputs_checked\":%zu,\"output_diff\":%zu,\"exact\":%s",
               PREFILL_S, PREFILL_NT, c.m, c.n, c.k, c.parts, pchecked, pdiff, out, odiff, exact ? "true" : "false");
  if (c.timed)
    std::fprintf(report,
                 ",\"v4_us\":%.2f,\"v9_us\":%.2f,\"cublas_us\":%.2f,\"v4_gbs\":%.1f,\"v9_gbs\":%.1f,\"cublas_gbs\":%.1f,"
                 "\"v9_tflops\":%.2f,\"cublas_tflops\":%.2f,\"timing_valid\":%s",
                 t4 * 1e6, t9 * 1e6, tb * 1e6, bytes / t4 / 1e9, bytes / t9 / 1e9, bytes / tb / 1e9, flops / t9 / 1e12,
                 flops / tb / 1e12, timing_valid ? "true" : "false");
  std::fprintf(report, "}\n"); std::fflush(report);
  check(cudaFree(w)); check(cudaFree(x)); check(cudaFree(p4)); check(cudaFree(p9)); check(cudaFree(o4)); check(cudaFree(o9));
  return exact && timing_valid ? 0 : 1;
}

// Real fixture: layer-0 conv in_proj prefill, 34 tokens, one partition (the runtime plan), vs reference.
static int real_fixture(const char* wpath, const char* xpath, const char* epath, std::FILE* report) {
  const int m = 6144, k = 2048, n = 34, parts = 1, pm = m, pn = 48;
  auto hw = read_file(wpath, size_t(m) * k * 2), hx = read_file(xpath, size_t(n) * k * 2), he = read_file(epath, size_t(m) * n * 2);
  bf16 *w, *x, *o; float* p;
  check(cudaMalloc(&w, hw.size())); check(cudaMalloc(&x, hx.size())); check(cudaMalloc(&o, he.size()));
  check(cudaMalloc(&p, size_t(pm) * pn * parts * 4));
  check(cudaMemcpy(w, hw.data(), hw.size(), cudaMemcpyHostToDevice));
  check(cudaMemcpy(x, hx.data(), hx.size(), cudaMemcpyHostToDevice));
  launch_v9(w, x, p, m, n, k, pn, pm, parts);
  finish(p, o, m, n, pm, pn, parts);
  check(cudaGetLastError()); check(cudaDeviceSynchronize());
  std::vector<char> ho(he.size()); check(cudaMemcpy(ho.data(), o, ho.size(), cudaMemcpyDeviceToHost));
  size_t diff = 0;
  for (size_t i = 0; i < size_t(m) * n; ++i) diff += std::memcmp(ho.data() + 2 * i, he.data() + 2 * i, 2) != 0;
  std::fprintf(report, "{\"real_fixture\":\"layer0 conv.in_proj prefill n=34\",\"checked\":%d,\"mismatches\":%zu}\n", m * n, diff);
  check(cudaFree(w)); check(cudaFree(x)); check(cudaFree(o)); check(cudaFree(p));
  return diff ? 1 : 0;
}
}  // namespace prefill

int main(int argc, char** argv) try {
  const bool tails_only = argc == 2 && std::string(argv[1]) == "--tails";
  std::vector<Case> cases;
  if (!tails_only) {
    // Prefill plans from the runtime log (n=34 prompt): partitions as the heuristic selected them.
    const int shapes[][3] = {{7680, 2048, 1}, {2048, 7680, 5}, {6144, 2048, 1}, {2048, 2048, 1}, {512, 2048, 1}, {16, 2048, 2}};
    for (auto& s : shapes) cases.push_back({s[0], 34, s[1], s[2], s[0] >= 512});
    for (int n : {9, 16, 17, 64, 257})
      for (auto& s : shapes) cases.push_back({s[0], n, s[1], s[2], false});
    // Long prompts (32K-gate geometry, one partition), exactness plus throughput.
    cases.push_back({2048, 4096, 2048, 1, true});
    cases.push_back({7680, 4096, 2048, 1, true});
    cases.push_back({2048, 32768, 2048, 1, false});
  }
  for (int m : {1, 17, 63, 65})
    for (int n : {9, 15, 16, 17, 33, 65})
      for (int k : {16, 32, 48, 64, 272})
        for (int parts : {1, 2, 3, 4})
          if (k % parts == 0 && (k / parts) % 16 == 0) cases.push_back({m, n, k, parts, false});
  int failures = 0;
  for (const auto& c : cases) failures += prefill::run(c, stdout);
  if (!tails_only && argc == 4) failures += prefill::real_fixture(argv[1], argv[2], argv[3], stdout);
  std::fprintf(stdout, "{\"S\":%d,\"NT\":%d,\"cases\":%zu,\"failures\":%d,\"full_acceptance\":false}\n", PREFILL_S,
               PREFILL_NT, cases.size(), failures);
  return failures ? 2 : 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "FAIL: %s\n", e.what());
  return 1;
}
