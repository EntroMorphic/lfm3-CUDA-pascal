// A row that runs to its capacity and finishes must not touch the rows beside it (validation/speech-batch-v1,
// red-team finding 1: BatchDecoder::step wrote every row's K/V at its cursor, so a finished row at the capacity wrote
// past its cache into the next row's).
//
//   lfm3-batch-capacity-check MODEL_DIR
// Capacity 64, two rows, greedy. Row 0 (A, 8 prompt tokens) decodes until its cursor reaches 64 and finishes; row 1
// (B, 2 prompt tokens) decodes 6 steps more. The run is repeated with row 0 never loaded. Prints whether B's logits
// were the same, bit for bit, at every step, and exits 1 if not.
#define LFM3_RUNTIME_ONLY 1
#include "../src/lfm3_main.cu"
#include "../src/lfm3_batch.cuh"

#include <cstring>

namespace {
// B's logits after every step it takes; `with_a`: row 0 holds A (B then steps until 6 steps after A finished);
// otherwise B takes `b_steps` steps alone.
std::vector<std::vector<float>> run(Runner& runner, bool with_a, int& a_steps, size_t b_steps = 0) {
  const std::vector<int> a = {124894, 124899, 23630, 207, 89056, 415, 117727, 2803}, b = {124894, 124899};
  lfm3_batch::BatchDecoder dec(runner, 2);
  if (with_a) { runner.reset(); runner.forward(a); dec.load(0); }
  runner.reset(); runner.forward(b); dec.load(1);
  std::vector<std::vector<float>> out;
  a_steps = 0;
  int b_left = -1;  // steps B takes after A finishes
  for (;;) {
    const auto pick = dec.select(0.0, {0, 0}, {0, 0});
    std::vector<int> feed(2, 0);
    if (with_a && dec.active[0]) {
      if (dec.cursor[0] >= runner.s.context) dec.finish(0);  // A ran to its capacity
      else feed[0] = pick[0], ++a_steps;
    }
    if (with_a) {
      if (!dec.active[0] && b_left < 0) b_left = 6;
      if (b_left == 0) break;
      if (b_left > 0) --b_left;
    } else if (out.size() == b_steps) {
      break;
    }
    if (dec.cursor[1] >= runner.s.context) break;
    feed[1] = pick[1];
    dec.step(feed);
    std::vector<float> row(runner.s.vocabulary);
    cuda_check(cudaMemcpy(row.data(), dec.logits.p + size_t(runner.s.vocabulary), row.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    out.push_back(std::move(row));
  }
  return out;
}
}  // namespace

int main(int argc, char** argv) try {
  if (argc != 2) throw std::runtime_error("usage: lfm3-batch-capacity-check MODEL_DIR");
  auto cfg = lfm::ModelConfig::load(std::string(argv[1]) + "/config.json", 64);
  Runner runner(argv[1], cfg);
  int a_steps = 0, unused = 0;
  const auto with_a = run(runner, true, a_steps);
  const auto alone = run(runner, false, unused, with_a.size());
  if (alone.size() != with_a.size() || a_steps < 50) throw std::runtime_error("the runs did not take the planned steps");
  const size_t n = with_a.size();
  size_t first_diff = n;
  for (size_t i = 0; i < n && first_diff == n; ++i)
    if (std::memcmp(with_a[i].data(), alone[i].data(), with_a[i].size() * sizeof(float))) first_diff = i;
  std::cout << "{\"a_steps\":" << a_steps << ",\"b_steps_compared\":" << n << ",\"identical\":"
            << (first_diff == n ? "true" : "false") << ",\"first_difference\":" << long(first_diff == n ? -1 : first_diff)
            << "}\n";
  return first_diff == n ? 0 : 1;
} catch (const std::exception& e) {
  std::cerr << "lfm3-batch-capacity-check: " << e.what() << "\n";
  return 1;
}
