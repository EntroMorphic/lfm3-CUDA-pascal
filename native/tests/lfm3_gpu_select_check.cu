// Device token selection (src/lfm3_batch.cuh: greedy_rows, sample_rows) against the host sampler (lfm::TextSampler,
// long double, the engine's validated reference). No model is loaded.
//
//   lfm3-gpu-select-check OUT_DIR
// 1. Greedy: 4,096 random rows (with planted ties at the maximum and suppressed maxima) must give exactly the host
//    sampler's greedy token (lowest id on ties, suppression honoured).
// 2. Sampling: for each case (temperature, logit scale, suppression), DRAWS draws of one logit row (rows of a launch
//    differ only in seed) are counted; OUT_DIR/case-K.counts.i64 and OUT_DIR/case-K.probs.f64 (the host sampler's exact
//    distribution, in id order) are written for the chi-squared test (tools/check_gpu_select.py).
#define LFM3_RUNTIME_ONLY 1
#include "../src/lfm3_main.cu"
#include "../src/lfm3_batch.cuh"

#include <random>

int main(int argc, char** argv) try {
  if (argc != 2) throw std::runtime_error("usage: lfm3-gpu-select-check OUT_DIR");
  const std::string out = argv[1];
  const int vocab = 128000;
  std::mt19937_64 g(7);
  std::vector<int> suppressed = {125029, 125030, 125031, 125032, 125012, 125015, 17, 4093};
  std::vector<unsigned char> mask(vocab, 0);
  for (int id : suppressed) mask[id] = 1;
  Buffer<unsigned char> mask_dev(vocab);
  cuda_check(cudaMemcpy(mask_dev.p, mask.data(), vocab, cudaMemcpyHostToDevice));

  // 1. Greedy
  {
    const int rows = 4096;
    std::vector<float> x(size_t(rows) * vocab);
    std::normal_distribution<float> n(0, 3);
    for (auto& v : x) v = n(g);
    for (int r = 0; r < rows; ++r) {
      float* row = x.data() + size_t(r) * vocab;
      const float top = 40.0f + float(r % 7);
      if (r % 3 == 0) row[r % 1000 + 5000] = top, row[r % 1000 + 9000] = top;  // a tie: the lower id must win
      if (r % 5 == 0) row[suppressed[r % suppressed.size()]] = 99.0f;          // a suppressed maximum
      if (r % 11 == 0) for (int i = 0; i < 300; ++i) row[(i * 397) % vocab] = top + 1;  // many ties
    }
    Buffer<float> logits(size_t(rows) * vocab);
    Buffer<unsigned char> active(rows);
    Buffer<int> picked(rows);
    cuda_check(cudaMemcpy(logits.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    cuda_check(cudaMemset(active.p, 1, rows));
    int mismatches = 0;
    for (int with_mask = 0; with_mask < 2; ++with_mask) {
      lfm3_batch::greedy_rows<<<rows, 1024>>>(logits.p, vocab, with_mask ? mask_dev.p : nullptr, active.p, picked.p);
      std::vector<int> got(rows);
      cuda_check(cudaMemcpy(got.data(), picked.p, rows * 4, cudaMemcpyDeviceToHost));
      lfm::TextSamplingOptions o;  // do_sample=false: greedy
      lfm::TextSampler host(o);
      for (int r = 0; r < rows; ++r) {
        const int want = host.sample(x.data() + size_t(r) * vocab, vocab, {}, with_mask ? suppressed : std::vector<int>{});
        mismatches += want != got[r];
      }
    }
    std::cout << "{\"greedy_rows\":" << 2 * rows << ",\"greedy_mismatches\":" << mismatches;
    if (mismatches) throw std::runtime_error("greedy mismatch");
  }

  // 2. Sampling distribution
  struct Case { double temperature; float scale; bool mask; };
  const Case cases[] = {{0.528, 1.0f, true}, {1.0, 1.0f, false}, {0.528, 3.0f, true}, {0.1, 1.0f, false}};
  const int rows = 512, launches = 400;  // 204,800 draws per case
  Buffer<float> logits(size_t(rows) * vocab);
  Buffer<unsigned char> active(rows);
  Buffer<int> picked(rows);
  Buffer<uint64_t> seeds(rows), steps(rows);
  cuda_check(cudaMemset(active.p, 1, rows));
  for (size_t k = 0; k < std::size(cases); ++k) {
    const auto& c = cases[k];
    // a peaked, long-tailed row like real logits: a few strong candidates over a gaussian bulk
    std::vector<float> row(vocab);
    std::normal_distribution<float> n(0, 1.5f);
    for (auto& v : row) v = c.scale * n(g);
    for (int i = 0; i < 40; ++i) row[(i * 7919 + 13) % vocab] = c.scale * (6.0f + 0.15f * i);
    if (c.mask) row[suppressed[0]] = 50.0f;  // suppressed: must never be drawn
    std::vector<float> all(size_t(rows) * vocab);
    for (int r = 0; r < rows; ++r) std::copy(row.begin(), row.end(), all.begin() + size_t(r) * vocab);
    cuda_check(cudaMemcpy(logits.p, all.data(), all.size() * 4, cudaMemcpyHostToDevice));
    std::vector<int64_t> counts(vocab, 0);
    std::vector<uint64_t> seed(rows), step(rows);
    for (int r = 0; r < rows; ++r) seed[r] = 1000003ull * (k + 1) + r;
    for (int l = 0; l < launches; ++l) {
      for (int r = 0; r < rows; ++r) step[r] = l;
      cuda_check(cudaMemcpy(seeds.p, seed.data(), rows * 8, cudaMemcpyHostToDevice));
      cuda_check(cudaMemcpy(steps.p, step.data(), rows * 8, cudaMemcpyHostToDevice));
      lfm3_batch::sample_rows<<<rows, 1024>>>(logits.p, vocab, c.mask ? mask_dev.p : nullptr, active.p, c.temperature,
                                              seeds.p, steps.p, picked.p);
      std::vector<int> got(rows);
      cuda_check(cudaMemcpy(got.data(), picked.p, rows * 4, cudaMemcpyDeviceToHost));
      for (int t : got) {
        if (t < 0 || t >= vocab) throw std::runtime_error("sampled id out of range");
        ++counts[t];
      }
    }
    lfm::TextSamplingOptions o;
    o.do_sample = true, o.temperature = c.temperature;
    lfm::TextSampler host(o);
    std::vector<double> probs(vocab, 0.0);
    for (const auto& cand : host.distribution(row.data(), vocab, {}, c.mask ? suppressed : std::vector<int>{}))
      probs[cand.token_id] = double(cand.probability);
    std::ofstream fc(out + "/case-" + std::to_string(k) + ".counts.i64", std::ios::binary);
    fc.write(reinterpret_cast<const char*>(counts.data()), vocab * 8);
    std::ofstream fp(out + "/case-" + std::to_string(k) + ".probs.f64", std::ios::binary);
    fp.write(reinterpret_cast<const char*>(probs.data()), vocab * 8);
    int64_t drawn_suppressed = 0;
    if (c.mask) for (int id : suppressed) drawn_suppressed += counts[id];
    std::cout << ",\"case" << k << "\":{\"temperature\":" << c.temperature << ",\"scale\":" << c.scale
              << ",\"draws\":" << int64_t(rows) * launches << ",\"suppressed_drawn\":" << drawn_suppressed << "}";
    if (drawn_suppressed) throw std::runtime_error("a suppressed id was drawn");
  }
  std::cout << "}\n";
  return 0;
} catch (const std::exception& e) {
  std::cerr << "lfm3-gpu-select-check: " << e.what() << "\n";
  return 1;
}
