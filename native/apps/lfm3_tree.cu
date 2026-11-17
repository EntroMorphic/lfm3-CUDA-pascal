// lfm3-tree: tree-verify decode driver and exactness gate (src/lfm3_tree.cuh).
//
//   lfm3-tree MODEL_DIR REQUEST.json OUT.json
// REQUEST: {"capacity": C, "prompts": [[ids...]...], "max_new_tokens": N, "temperature": T, "seed": S,
//           "stop": [ids], "rows": R (the pass's row budget), "drafter": "none" | "ngram", "mode": "tree" | "plain",
//           "warmup": false}
// One slot per prompt. "plain" decodes with lfm3_batch::BatchDecoder (the reference); "tree" with the tree decoder:
// every pass gives each active slot its pending token plus up to floor(R / active) - 1 drafted nodes. The output's
// tokens must equal plain's, greedy or sampled. OUT: per slot its tokens and end; passes; decode seconds; tok/s; and
// tokens per pass.
#define LFM3_RUNTIME_ONLY 1
#include "../src/lfm3_main.cu"
#include "../src/lfm3_batch.cuh"
#include "../src/lfm3_tree.cuh"

#include <chrono>

namespace {
uint64_t mix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}
struct Slot { std::vector<int> prompt, gen; std::string end; uint64_t seed = 0; bool active = true; };
struct Result { double seconds = 0; long passes = 0; };

Result plain(Runner& runner, std::vector<Slot>& slots, int max_new, double T, const std::vector<int>& stop) {
  const int n = int(slots.size());
  lfm3_batch::BatchDecoder dec(runner, n);
  for (int r = 0; r < n; ++r) { runner.reset(); runner.forward(slots[r].prompt); dec.load(r); }
  cuda_check(cudaDeviceSynchronize());
  auto t0 = std::chrono::steady_clock::now();
  std::vector<uint64_t> seeds(n), steps(n, 0);
  for (int r = 0; r < n; ++r) seeds[r] = slots[r].seed;
  Result res;
  for (;;) {
    auto pick = dec.select(T, seeds, steps);
    std::vector<int> feed(n, -1);
    bool any = false;
    for (int r = 0; r < n; ++r) {
      if (!dec.active[r]) continue;
      Slot& sl = slots[r];
      sl.gen.push_back(pick[r]); ++steps[r];
      if (std::find(stop.begin(), stop.end(), pick[r]) != stop.end()) sl.end = "stop_token", dec.finish(r);
      else if (int(sl.gen.size()) >= max_new) sl.end = "max_new_tokens", dec.finish(r);
      else if (dec.cursor[r] >= runner.s.context) sl.end = "capacity", dec.finish(r);
      else feed[r] = pick[r], any = true;
    }
    if (!any) break;
    dec.step(feed);
    ++res.passes;
  }
  cuda_check(cudaDeviceSynchronize());
  res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return res;
}

Result tree(Runner& runner, std::vector<Slot>& slots, int max_new, double T, const std::vector<int>& stop, int rows,
            bool ngram) {
  using namespace lfm3_tree;
  const int n = int(slots.size()), C = runner.s.context;
  TreeDecoder dec(runner, n, rows);
  // prefill, then the first token from the prefill logits (row 0 of the decoder's logits), as plain decode picks it
  for (int r = 0; r < n; ++r) {
    runner.reset(); runner.forward(slots[r].prompt); dec.load(r);
    cuda_check(cudaMemcpy(dec.logits.p, runner.logits.p, size_t(runner.s.vocabulary) * 4, cudaMemcpyDeviceToDevice));
    const int t = dec.pick(1, T, {slots[r].seed}, {0})[0];
    Slot& sl = slots[r];
    sl.gen.push_back(t);
    if (std::find(stop.begin(), stop.end(), t) != stop.end()) sl.end = "stop_token", sl.active = false;
    else if (int(sl.gen.size()) >= max_new) sl.end = "max_new_tokens", sl.active = false;
    else if (dec.committed[r] >= C) sl.end = "capacity", sl.active = false;
  }
  cuda_check(cudaDeviceSynchronize());
  auto t0 = std::chrono::steady_clock::now();
  Result res;
  for (;;) {
    std::vector<int> live;
    for (int r = 0; r < n; ++r) if (slots[r].active) live.push_back(r);
    if (live.empty()) break;
    const int per = std::max(1, rows / int(live.size()));
    std::vector<Node> nodes;
    std::vector<uint64_t> seeds, steps;
    std::vector<int> root_of(n, -1);
    for (int r : live) {
      Slot& sl = slots[r];
      root_of[r] = int(nodes.size());
      nodes.push_back({r, sl.gen.back(), -1, 0});
      seeds.push_back(sl.seed), steps.push_back(sl.gen.size());
      if (!ngram || per <= 1) continue;
      std::vector<int> ctx = sl.prompt;
      ctx.insert(ctx.end(), sl.gen.begin(), sl.gen.end());
      const int depth_cap = std::min({kMaxDepth, C - dec.committed[r] - 1, max_new - int(sl.gen.size())});
      const int base = root_of[r];
      for (const auto& d : ngram_tree(ctx, std::min(per - 1, rows - int(nodes.size())), depth_cap)) {
        nodes.push_back({r, d.token, d.parent < 0 ? base : base + 1 + d.parent, d.depth});
        seeds.push_back(sl.seed), steps.push_back(sl.gen.size() + d.depth);
      }
    }
    dec.pass(nodes);
    const auto pick = dec.pick(int(nodes.size()), T, seeds, steps);
    ++res.passes;
    // children lists
    std::vector<std::vector<int>> kids(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) if (nodes[i].parent >= 0) kids[nodes[i].parent].push_back(int(i));
    std::vector<std::vector<int>> accepted(n);
    for (int r : live) {
      Slot& sl = slots[r];
      int row = root_of[r];
      const int c0 = dec.committed[r];
      for (;;) {
        accepted[r].push_back(row);
        const int t = pick[row];
        sl.gen.push_back(t);
        if (std::find(stop.begin(), stop.end(), t) != stop.end()) { sl.end = "stop_token", sl.active = false; break; }
        if (int(sl.gen.size()) >= max_new) { sl.end = "max_new_tokens", sl.active = false; break; }
        if (c0 + nodes[row].depth + 1 >= C) { sl.end = "capacity", sl.active = false; break; }
        int next = -1;
        for (int kid : kids[row]) if (nodes[kid].token == t) { next = kid; break; }
        if (next < 0) break;
        row = next;
      }
    }
    dec.commit(accepted);
  }
  cuda_check(cudaDeviceSynchronize());
  res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return res;
}
}  // namespace

int main(int argc, char** argv) try {
  if (argc != 4) throw std::runtime_error("usage: lfm3-tree MODEL_DIR REQUEST.json OUT.json");
  const auto req = read_json(argv[2]);
  const int capacity = req.at("capacity").integer(), max_new = req.at("max_new_tokens").integer();
  const double T = req.has("temperature") ? req.at("temperature").number() : 0.0;
  const int rows = req.has("rows") ? req.at("rows").integer() : 32;
  const std::string mode = req.has("mode") ? req.at("mode").string() : "tree";
  const bool ngram = req.has("drafter") && req.at("drafter").string() == "ngram";
  const bool warmup = req.has("warmup") && req.at("warmup").boolean();
  auto cfg = lfm::ModelConfig::load(std::string(argv[1]) + "/config.json", capacity);
  Runner runner(argv[1], cfg);
  std::vector<int> stop = {124900};
  if (req.has("stop")) { stop.clear(); for (auto& x : req.at("stop").array()) stop.push_back(x.integer()); }
  std::vector<Slot> slots;
  const uint64_t seed = req.has("seed") ? uint64_t(req.at("seed").integer()) : 0;
  for (auto& p : req.at("prompts").array()) {
    Slot sl;
    for (auto& x : p.array()) sl.prompt.push_back(x.integer());
    sl.seed = mix(seed + slots.size());
    slots.push_back(sl);
  }
  auto run = [&](std::vector<Slot>& s) {
    return mode == "plain" ? plain(runner, s, max_new, T, stop) : tree(runner, s, max_new, T, stop, rows, ngram);
  };
  if (warmup) { auto spare = slots; run(spare); }
  const Result res = run(slots);
  size_t generated = 0;
  for (auto& s : slots) generated += s.gen.size();
  std::ofstream f(argv[3]);
  f << "{\"mode\":\"" << mode << "\",\"drafter\":\"" << (ngram ? "ngram" : "none") << "\",\"rows\":" << rows
    << ",\"temperature\":" << T << ",\"passes\":" << res.passes << ",\"decode_seconds\":" << res.seconds
    << ",\"generated_tokens\":" << generated
    << ",\"decode_tokens_per_second\":" << double(generated - slots.size()) / res.seconds
    << ",\"tokens_per_pass\":" << double(generated - slots.size()) / std::max(1L, res.passes) << ",\"slots\":[";
  for (size_t r = 0; r < slots.size(); ++r) {
    f << (r ? "," : "") << "{\"end\":\"" << slots[r].end << "\",\"generated_ids\":";
    print_ids(f, slots[r].gen);
    f << "}";
  }
  f << "]}\n";
  std::cout << "{\"mode\":\"" << mode << "\",\"drafter\":\"" << (ngram ? "ngram" : "none") << "\",\"rows\":" << rows
            << ",\"generated\":" << generated << ",\"passes\":" << res.passes << ",\"tok_s\":"
            << double(generated - slots.size()) / res.seconds << ",\"tok_per_pass\":"
            << double(generated - slots.size()) / std::max(1L, res.passes) << "}\n";
  return 0;
} catch (const std::exception& e) {
  std::cerr << "lfm3-tree: " << e.what() << "\n";
  return 1;
}
