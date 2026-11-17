// Decode-path fidelity (P0 speed gate): teacher-forced NLL of the batched decoder (src/lfm3_batch.cuh) and of the
// accepted serial runner's decode, on the same windows and positions.
//
//   lfm3-batch-teacher MODEL_DIR WINDOWS.i32 LENGTH PREFIX ROWS OUT_STEM
// The first ROWS windows: each is prefilled with its first PREFIX tokens by the Runner (the validated prefill), then
// tokens PREFIX..LENGTH-1 are scored one decode step at a time, the true token fed back.
// - OUT_STEM.batch.f32: all ROWS windows decoded together in one batch;
// - OUT_STEM.serial.f32: each window through Runner::forward one token at a time (the accepted engine's decode).
// Both are ROWS x (LENGTH - PREFIX) float32: the NLL of token t at index t - PREFIX (ppl_reference's index t - 1),
// computed by the same device kernel: log(sum exp(x - max)) + max - x[target] in FP64 over all logits.
#define LFM3_RUNTIME_ONLY 1
#include "../src/lfm3_main.cu"
#include "../src/lfm3_batch.cuh"

#include <chrono>

namespace {
__global__ void __launch_bounds__(1024) nll_rows(const float* logits, int vocab, const int* target, float* out) {
  const int b = blockIdx.x, t = threadIdx.x;
  const float* x = logits + size_t(b) * vocab;
  const int chunk = (vocab + 1023) / 1024, lo = min(vocab, t * chunk), hi = min(vocab, lo + chunk);
  __shared__ float sm[32];
  __shared__ double sums[1024];
  float m = -INFINITY;
  for (int i = lo; i < hi; ++i) m = fmaxf(m, x[i]);
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if (t % 32 == 0) sm[t / 32] = m;
  __syncthreads();
  if (t < 32) {
    m = sm[t];
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (t == 0) sm[0] = m;
  }
  __syncthreads();
  m = sm[0];
  double s = 0;
  for (int i = lo; i < hi; ++i) s += exp(double(x[i]) - double(m));
  sums[t] = s;
  __syncthreads();
  if (t == 0) {
    double total = 0;
    for (int i = 0; i < 1024; ++i) total += sums[i];
    out[b] = float(log(total) + double(m) - double(x[target[b]]));
  }
}
std::vector<int> read_ids(const char* path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot read windows");
  std::vector<int> v(size_t(in.tellg()) / 4);
  in.seekg(0);
  in.read(reinterpret_cast<char*>(v.data()), std::streamsize(v.size() * 4));
  return v;
}
void write(const std::string& path, const std::vector<float>& v) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * 4));
  if (!f) throw std::runtime_error("cannot write " + path);
}
}  // namespace

int main(int argc, char** argv) try {
  if (argc != 7) throw std::runtime_error("usage: lfm3-batch-teacher MODEL_DIR WINDOWS.i32 LENGTH PREFIX ROWS OUT_STEM");
  const int length = std::atoi(argv[3]), prefix = std::atoi(argv[4]), rows = std::atoi(argv[5]);
  const auto ids = read_ids(argv[2]);
  if (length < 3 || prefix < 1 || prefix >= length || rows < 1 || size_t(rows) * length > ids.size())
    throw std::runtime_error("invalid LENGTH/PREFIX/ROWS for the windows file");
  auto cfg = lfm::ModelConfig::load(std::string(argv[1]) + "/config.json", length);
  Runner runner(argv[1], cfg);
  const int steps = length - prefix, vocab = cfg.shape.vocabulary;
  Buffer<int> target(rows);
  Buffer<float> nll(rows);
  auto window = [&](int r) { return ids.begin() + size_t(r) * length; };

  // Batched decode: every window in one batch.
  std::vector<float> batch_nll(size_t(rows) * steps);
  {
    lfm3_batch::BatchDecoder dec(runner, rows);
    for (int r = 0; r < rows; ++r) {
      runner.reset();
      runner.forward(std::vector<int>(window(r), window(r) + prefix));
      dec.load(r);
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int t = prefix; t < length; ++t) {
      std::vector<int> next(rows);
      for (int r = 0; r < rows; ++r) next[r] = window(r)[t];
      cuda_check(cudaMemcpy(target.p, next.data(), rows * 4, cudaMemcpyHostToDevice));
      nll_rows<<<rows, 1024>>>(dec.logits.p, vocab, target.p, nll.p);
      std::vector<float> got(rows);
      cuda_check(cudaMemcpy(got.data(), nll.p, rows * 4, cudaMemcpyDeviceToHost));
      for (int r = 0; r < rows; ++r) batch_nll[size_t(r) * steps + (t - prefix)] = got[r];
      if (t + 1 < length) dec.step(next);
    }
    std::cout << "{\"batch_decode_seconds\":" << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }
  write(std::string(argv[6]) + ".batch.f32", batch_nll);

  // The accepted serial decode, one window at a time, scored by the same kernel.
  std::vector<float> serial_nll(size_t(rows) * steps);
  const auto t1 = std::chrono::steady_clock::now();
  for (int r = 0; r < rows; ++r) {
    runner.reset();
    runner.forward(std::vector<int>(window(r), window(r) + prefix));
    for (int t = prefix; t < length; ++t) {
      const int next = window(r)[t];
      cuda_check(cudaMemcpy(target.p, &next, 4, cudaMemcpyHostToDevice));
      nll_rows<<<1, 1024>>>(runner.logits.p, vocab, target.p, nll.p);
      cuda_check(cudaMemcpy(&serial_nll[size_t(r) * steps + (t - prefix)], nll.p, 4, cudaMemcpyDeviceToHost));
      if (t + 1 < length) runner.forward({next});
    }
  }
  std::cout << ",\"serial_decode_seconds\":" << std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count()
            << ",\"rows\":" << rows << ",\"steps\":" << steps << "}\n";
  write(std::string(argv[6]) + ".serial.f32", serial_nll);
  return 0;
} catch (const std::exception& e) {
  std::cerr << "lfm3-batch-teacher: " << e.what() << "\n";
  return 1;
}
