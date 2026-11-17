// lfm3-batch: batched decode driver and gate tool for the LFM3 text engine (P0 speed; src/lfm3_batch.cuh).
//
//   lfm3-batch MODEL_DIR REQUEST.json OUT.json
// REQUEST: {"capacity": C, "batch": B, "prompts": [[ids...], ...] (row r takes prompt r % count),
//           "max_new_tokens": N, "temperature": T (0: greedy), "seed": S, "row_seeds": [...] (optional),
//           "stop": [ids] (default <|im_end|>), "serial": false, "warmup": false}
// warmup: the whole decode runs once untimed first (Thor's GPU clock ramps up over about a second).
// "all_prompts": true decodes every prompt once, in successive batches of `batch` rows (data generation).
// serial: every row decodes alone in a batch of one (the same code path), for the row-independence gate.
// OUT: per row its generated ids and how it ended; prefill and decode seconds; aggregate decode tokens/s (generated
// tokens after the first, over decode-loop time, which includes selection and host bookkeeping).
#define LFM3_RUNTIME_ONLY 1
#include "../src/lfm3_main.cu"
#include "../src/lfm3_batch.cuh"

#include <chrono>

namespace {
uint64_t mix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}
struct Row { std::vector<int> prompt, generated; std::string end; uint64_t seed = 0; };

// Decodes rows[first, first + count) together; returns decode-loop seconds and the steps run.
std::pair<double, int> decode(Runner& runner, std::vector<Row>& rows, int first, int count, int max_new,
                              double temperature, const std::vector<int>& stop, double& prefill_seconds) {
  lfm3_batch::BatchDecoder dec(runner, count);
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < count; ++r) {
    runner.reset();
    runner.forward(rows[first + r].prompt);
    dec.load(r);
  }
  cuda_check(cudaDeviceSynchronize());
  auto t1 = std::chrono::steady_clock::now();
  prefill_seconds += std::chrono::duration<double>(t1 - t0).count();
  std::vector<uint64_t> seeds(count), steps(count, 0);
  for (int r = 0; r < count; ++r) seeds[r] = rows[first + r].seed;
  int iterations = 0;
  for (;;) {
    auto pick = dec.select(temperature, seeds, steps);
    std::vector<int> feed(count, -1);
    bool any = false;
    for (int r = 0; r < count; ++r) {
      if (!dec.active[r]) continue;
      Row& row = rows[first + r];
      row.generated.push_back(pick[r]);
      ++steps[r];
      if (std::find(stop.begin(), stop.end(), pick[r]) != stop.end()) row.end = "stop_token", dec.finish(r);
      else if (int(row.generated.size()) >= max_new) row.end = "max_new_tokens", dec.finish(r);
      else if (dec.cursor[r] >= runner.s.context) row.end = "capacity", dec.finish(r);
      else feed[r] = pick[r], any = true;
    }
    if (!any) break;
    dec.step(feed);
    ++iterations;
  }
  cuda_check(cudaDeviceSynchronize());
  return {std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count(), iterations};
}
}  // namespace

int main(int argc, char** argv) try {
  if (argc != 4) throw std::runtime_error("usage: lfm3-batch MODEL_DIR REQUEST.json OUT.json");
  const auto req = read_json(argv[2]);
  const int capacity = req.at("capacity").integer(), batch = req.at("batch").integer();
  const int max_new = req.at("max_new_tokens").integer();
  const double temperature = req.has("temperature") ? req.at("temperature").number() : 0.0;
  const bool serial = req.has("serial") && req.at("serial").boolean();
  const bool warmup = req.has("warmup") && req.at("warmup").boolean();
  if (batch < 1 || batch > 128 || max_new < 1 || !(temperature >= 0)) throw std::runtime_error("invalid request");
  auto cfg = lfm::ModelConfig::load(std::string(argv[1]) + "/config.json", capacity);
  Runner runner(argv[1], cfg);
  std::vector<int> stop = {124900};
  if (req.has("stop")) { stop.clear(); for (auto& x : req.at("stop").array()) stop.push_back(x.integer()); }
  const auto& prompts = req.at("prompts").array();
  if (prompts.empty()) throw std::runtime_error("no prompts");
  const bool all_prompts = req.has("all_prompts") && req.at("all_prompts").boolean();
  const int count = all_prompts ? int(prompts.size()) : batch;
  std::vector<Row> rows(count);
  const uint64_t seed = req.has("seed") ? uint64_t(req.at("seed").integer()) : 0;
  for (int r = 0; r < count; ++r) {
    for (auto& x : prompts[r % prompts.size()].array()) rows[r].prompt.push_back(x.integer());
    if (rows[r].prompt.empty() || int(rows[r].prompt.size()) >= capacity) throw std::runtime_error("bad prompt length");
    rows[r].seed = req.has("row_seeds") ? uint64_t(req.at("row_seeds").array().at(r).integer()) : mix(seed + r);
  }
  if (warmup) {
    std::vector<Row> spare = rows;
    double ignored = 0;
    if (serial) for (int r = 0; r < batch; ++r) decode(runner, spare, r, 1, max_new, temperature, stop, ignored);
    else decode(runner, spare, 0, batch, max_new, temperature, stop, ignored);
  }
  double prefill = 0, seconds = 0;
  int steps = 0;
  size_t bytes = lfm3_batch::BatchDecoder(runner, serial ? 1 : batch).device_bytes();
  if (all_prompts) {
    for (int first = 0; first < count; first += batch) {
      auto [t, n] = decode(runner, rows, first, std::min(batch, count - first), max_new, temperature, stop, prefill);
      seconds += t, steps += n;
      std::cerr << "decoded " << std::min(count, first + batch) << " / " << count << "\n";
    }
  } else if (serial) {
    for (int r = 0; r < batch; ++r) { auto [t, n] = decode(runner, rows, r, 1, max_new, temperature, stop, prefill); seconds += t; steps += n; }
  } else {
    auto [t, n] = decode(runner, rows, 0, batch, max_new, temperature, stop, prefill);
    seconds = t, steps = n;
  }
  size_t generated = 0;
  for (auto& r : rows) generated += r.generated.size();
  std::ofstream f(argv[3]);
  f << "{\"batch\":" << batch << ",\"serial\":" << (serial ? "true" : "false") << ",\"capacity\":" << capacity
    << ",\"temperature\":" << temperature << ",\"prefill_seconds\":" << prefill << ",\"decode_seconds\":" << seconds
    << ",\"decode_steps\":" << steps << ",\"generated_tokens\":" << generated
    << ",\"decode_tokens_per_second\":" << double(generated - count) / seconds
    << ",\"batch_device_bytes\":" << bytes << ",\"rows\":[";
  for (int r = 0; r < count; ++r) {
    f << (r ? "," : "") << "{\"seed\":" << rows[r].seed << ",\"end\":\"" << rows[r].end << "\",\"generated_ids\":";
    print_ids(f, rows[r].generated);
    f << "}";
  }
  f << "]}\n";
  if (!f) throw std::runtime_error("cannot write output");
  std::cout << "{\"batch\":" << batch << ",\"serial\":" << serial << ",\"generated\":" << generated
            << ",\"decode_seconds\":" << seconds << ",\"decode_tok_s\":" << double(generated - count) / seconds
            << ",\"prefill_seconds\":" << prefill << "}\n";
  return 0;
} catch (const std::exception& e) {
  std::cerr << "lfm3-batch: " << e.what() << "\n";
  return 1;
}
