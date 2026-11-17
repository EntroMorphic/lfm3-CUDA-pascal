// Lossless weights (gemm_lossless.cuh): every packed LFM3 weight matrix, run through the compressed and the plain
// wide kernel in several tile shapes, M = 1, 7, 32, 64, and split-K: the outputs must be bitwise identical.
//   lfm3-lossless-check MODEL_DIR
#define LFM3_RUNTIME_ONLY 1
#include "../src/lfm3_main.cu"
#include "../src/lfm3_batch.cuh"

#include <random>

int main(int argc, char** argv) try {
  if (argc != 2) throw std::runtime_error("usage: lfm3-lossless-check MODEL_DIR");
  setenv("LFM3_LOSSLESS", "1", 1);
  auto cfg = lfm::ModelConfig::load(std::string(argv[1]) + "/config.json", 256);
  Runner runner(argv[1], cfg);
  const auto [packed_bytes, raw_bytes] = lfm3_batch::pack_model(runner);
  using namespace lfm::gemm;
  using namespace lfm3_batch::tiles;
  const int max_m = 64, max_k = 7680, max_n = 128000;
  Buffer<bf16> x(size_t(max_m) * max_k), y0(size_t(max_m) * max_n), y1(size_t(max_m) * max_n);
  Buffer<float> f0(size_t(4) * max_m * 7680), f1(size_t(4) * max_m * 7680);
  {
    std::mt19937 g(5); std::normal_distribution<float> d(0, 1);
    std::vector<bf16> h(size_t(max_m) * max_k); for (auto& v : h) v = __float2bfloat16(d(g));
    cuda_check(cudaMemcpy(x.p, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
  }
  long checks = 0, mismatches = 0, tensors = 0;
  auto same = [&](const void* a, const void* b, size_t bytes) {
    std::vector<unsigned char> u(bytes), v(bytes);
    cuda_check(cudaMemcpy(u.data(), a, bytes, cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(v.data(), b, bytes, cudaMemcpyDeviceToHost));
    ++checks;
    if (u != v) ++mismatches;
  };
  for (auto& [w, pw] : packed_weights()) {
    ++tensors;
    for (int m : {1, 7, 32, 64}) {
      Problem p = linear(x.p, w, y0.p, m, pw.k, pw.n);
      Problem q = p; q.c = y1.p;
      const size_t bytes = size_t(m) * pw.n * 2;
      auto both = [&](auto plain, auto packed) { plain(p); packed(q); cuda_check(cudaGetLastError()); same(y0.p, y1.p, bytes); };
      both([](const Problem& a) { detail::launch_wide<W32, true>(a, 0); }, [&](const Problem& a) { detail::launch_wide_lossless<W32>(a, pw, 0); });
      both([](const Problem& a) { detail::launch_wide<C32, true>(a, 0); }, [&](const Problem& a) { detail::launch_wide_lossless<C32>(a, pw, 0); });
      both([](const Problem& a) { detail::launch_wide<S16, true>(a, 0); }, [&](const Problem& a) { detail::launch_wide_lossless<S16>(a, pw, 0); });
      both([](const Problem& a) { detail::launch_wide<H64, true>(a, 0); }, [&](const Problem& a) { detail::launch_wide_lossless<H64>(a, pw, 0); });
      // the tiles production uses (lfm3_batch::launch)
      both([](const Problem& a) { detail::launch_wide<W32, true>(a, 0); }, [&](const Problem& a) { detail::launch_wide_lossless<lfm3_batch::LosslessS>(a, pw, 0); });
      both([](const Problem& a) { detail::launch_wide<W32, true>(a, 0); }, [&](const Problem& a) { detail::launch_wide_lossless<lfm3_batch::LosslessM>(a, pw, 0); });
      // split-K over 4 slices (FP32 partials), when K allows it
      if (pw.k % 256 == 0 && pw.n <= 7680) {
        Problem s = linear(x.p, w, nullptr, m, pw.k / 4, pw.n);
        s.lda = pw.k, s.ldb = pw.k, s.c_f32 = f0.p, s.batch = 4, s.stride_a = pw.k / 4, s.stride_b = pw.k / 4,
        s.stride_c = (long long)m * pw.n;
        Problem t = s; t.c_f32 = f1.p;
        detail::launch_wide<C32, true>(s, 0);
        detail::launch_wide_lossless<C32>(t, pw, 0);
        cuda_check(cudaGetLastError());
        same(f0.p, f1.p, size_t(4) * m * pw.n * 4);
      }
    }
  }
  // timing: a decode step's projections (every packed tensor once) at M = 1 and 32: plain (W32/C32) vs packed tiles
  using detail::Tile;
  using LW8 = Tile<32, 128, 64, 2, 4, 4>;   // 8 warps
  using LW8s3 = Tile<32, 128, 64, 2, 4, 3>;
  using LN8 = Tile<32, 64, 64, 2, 4, 4>;    // 8 warps, 64 wide
  using LS4 = Tile<16, 64, 64, 1, 4, 4>;    // 4 warps, small: more blocks per SM
  using LS3 = Tile<16, 64, 64, 1, 4, 3>;
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  auto time_step = [&](int m, auto launch_one) {
    auto step = [&] { for (auto& [w, pw] : packed_weights()) launch_one(linear(x.p, w, y0.p, m, pw.k, pw.n), pw); };
    step(); cuda_check(cudaDeviceSynchronize()); cuda_check(cudaGetLastError());
    std::vector<float> t;
    for (int i = 0; i < 5; ++i) { cudaEventRecord(e0); step(); cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); t.push_back(ms); }
    std::sort(t.begin(), t.end()); return t[2];
  };
  using A2 = Tile<16, 128, 64, 1, 8, 3>;    // 8 warps
  using A3 = Tile<16, 64, 64, 1, 2, 3>;     // 2 warps, many blocks
  using B1 = Tile<32, 128, 64, 2, 8, 3>;    // 16 warps
  using B2 = Tile<32, 64, 64, 2, 4, 3>;     // 8 warps, 64 wide, shallow
  auto both = [&](auto wide, auto narrow) { return [=](const Problem& p, const Packed& w) { w.n >= 6144 ? wide(p, w) : narrow(p, w); }; };
#define L(T) [](const Problem& p, const Packed& w) { detail::launch_wide_lossless<T>(p, w, 0); }
  std::cerr << "M=1  plain " << time_step(1, [](const Problem& p, const Packed& w) { w.n >= 6144 ? detail::launch_wide<W32, true>(p, 0) : detail::launch_wide<C32, true>(p, 0); })
            << " | LS4 " << time_step(1, L(LS4)) << " | A2 " << time_step(1, L(A2))
            << " | A3 " << time_step(1, L(A3)) << " | A2/LS4 " << time_step(1, both(L(A2), L(LS4))) << "\n";
  std::cerr << "M=32 plain " << time_step(32, [](const Problem& p, const Packed& w) { w.n >= 6144 ? detail::launch_wide<W32, true>(p, 0) : detail::launch_wide<C32, true>(p, 0); })
            << " | LW8/LN8 " << time_step(32, both(L(LW8), L(LN8))) << " | B1 " << time_step(32, L(B1)) << " | B2 " << time_step(32, L(B2))
             << " | B1/B2 " << time_step(32, both(L(B1), L(B2))) << " | LW8/B2 " << time_step(32, both(L(LW8), L(B2))) << "\n";
#undef L
  std::cout << "{\"packed_tensors\":" << tensors << ",\"raw_bytes\":" << raw_bytes << ",\"packed_bytes\":" << packed_bytes
            << ",\"ratio\":" << double(packed_bytes) / raw_bytes << ",\"checks\":" << checks << ",\"mismatches\":" << mismatches
            << "}\n";
  return mismatches ? 1 : 0;
} catch (const std::exception& e) {
  std::cerr << "lfm3-lossless-check: " << e.what() << "\n";
  return 1;
}
