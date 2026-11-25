// continuation-v1: Runner::extend, a cached prefix continued in one call, each new row computed as a prefill of the
// whole prompt computes it.
//   lfm3-continuation-check MODEL_DIR TOKENS.i32 exact|exact-small|decoded|speed
// exact (rule 1): a prompt prefilled in parts (a prefill, then extensions) against the same prompt prefilled at once:
//   the logits after the last part, the whole cache (every convolution state; every attention layer's keys and values
//   at the positions it holds, and a compact windowed layer's slot bookkeeping) and the logits of 8 greedy decode
//   steps after it, each byte for byte. The parts of a case stay in one regime of the prefill's launch shapes (ends of
//   32-2,048 tokens, 2,049-4,096, or above 4,096 at a capacity above 4,096; see validation/continuation-v1).
// decoded (rule 2, recorded): a prompt, 32 greedy decode steps, then new tokens, continued by extend() and by one
//   forward a token at a time (the parity build's path), each against the whole sequence prefilled at once.
// speed (rule 3): seconds for extend() of N tokens after a prefill of L, the whole prompt's prefill (L + N), and, for
//   N <= 1,024, the token-at-a-time continuation.
// One JSON line per case on stdout; exit 0 when every exact case is identical (the other modes always exit 0 unless
// something fails to run).
#define LFM3_RUNTIME_ONLY 1
#include "../src/lfm3_main.cu"

#if !LFM3_PREFILL_CONTINUATION
#error "lfm3-continuation-check needs the shipping build (Runner::extend)"
#endif

#include <cmath>
#include <cstring>
#include <iomanip>

namespace {
using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

std::vector<int> read_tokens(const std::string& path, size_t n) {
  std::ifstream f(path, std::ios::binary);
  std::vector<int> t(n);
  f.read(reinterpret_cast<char*>(t.data()), std::streamsize(n * sizeof(int)));
  if (!f) throw std::runtime_error("cannot read " + std::to_string(n) + " tokens from " + path);
  return t;
}
int argmax(const std::vector<float>& x) { return int(std::max_element(x.begin(), x.end()) - x.begin()); }
bool same(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}
uint64_t fnv(uint64_t h, const void* p, size_t n) {
  auto c = static_cast<const unsigned char*>(p);
  for (size_t i = 0; i < n; ++i) h = (h ^ c[i]) * 1099511628211ull;
  return h;
}
// Every byte of the cache the next step reads: convolution states; per attention layer, the positions it holds.
uint64_t cache_hash(Runner& r) {
  cuda_check(cudaDeviceSynchronize());
  uint64_t h = 1469598103934665603ull;
  std::vector<bf16> host;
  auto add = [&](const bf16* p, size_t elements) {
    host.resize(elements);
    cuda_check(cudaMemcpy(host.data(), p, elements * sizeof(bf16), cudaMemcpyDeviceToHost));
    h = fnv(h, host.data(), elements * sizeof(bf16));
  };
  for (int i = 0; i < r.s.layers; ++i) {
    if (r.states[i]) add(r.states[i]->p, size_t(r.s.hidden) * 7);
    if (!r.keys[i]) continue;
    const int stride = r.kv_stride[i], n = r.compact(i) ? r.slots[i].count : r.cursor;  // compact: slots [0, count)
    if (r.compact(i)) { const int meta[2] = {r.slots[i].base, r.slots[i].count}; h = fnv(h, meta, sizeof meta); }
    for (int head = 0; head < r.s.kv_heads; ++head) {
      const size_t at = size_t(head) * stride * r.s.head_dim;
      add(r.keys[i]->p + at, size_t(n) * r.s.head_dim);
      add(r.values[i]->p + at, size_t(n) * r.s.head_dim);
    }
  }
  return h;
}
// reset(), then the compact layers' shared full-length pair filled with NaN (all ones: a BF16 NaN). reset() leaves that
// pair as the last call left it, so without this a run after another would find its keys there: a span read before
// the held window that extend() failed to zero would then pass. With it, such a read gives non-finite logits, which
// forward() refuses (continuation-v1 rule 2, after the independent review).
void fresh(Runner& r) {
  r.reset();
  if (r.window_keys) {
    cuda_check(cudaMemset(r.window_keys->p, 0xFF, r.s.kv_layer_stride() * sizeof(bf16)));
    cuda_check(cudaMemset(r.window_values->p, 0xFF, r.s.kv_layer_stride() * sizeof(bf16)));
  }
}
struct Model {
  std::string directory;
  std::shared_ptr<lfm::Safetensors> weights;
  std::unique_ptr<Runner> runner;
  int capacity = 0;
  Runner& at(int c) {
    if (!runner || capacity != c) {
      runner.reset();
      if (!weights) weights = std::make_shared<lfm::Safetensors>(directory + "/model.safetensors");
      auto cfg = lfm::ModelConfig::load(directory + "/config.json", c);
      runner = std::make_unique<Runner>(directory, cfg, weights);
      capacity = c;
    }
    fresh(*runner);
    return *runner;
  }
};
std::string ints(const std::vector<int>& v) {
  std::string s = "[";
  for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
  return s + "]";
}
// The greedy decode steps after a prompt: each step's logits.
std::vector<std::vector<float>> decode(Runner& r, std::vector<float> last, int steps, std::vector<int>& ids) {
  std::vector<std::vector<float>> out;
  for (int i = 0; i < steps; ++i) {
    ids.push_back(argmax(last));
    last = r.forward({ids.back()});
    out.push_back(last);
  }
  return out;
}

bool any_compact(const Runner& r) {
  for (int i = 0; i < r.s.layers; ++i) if (r.keys[i] && r.compact(i)) return true;
  return false;
}
struct Case { int capacity; std::vector<int> ends; };
bool exact(Model& m, const std::vector<int>& tokens, bool small) {
  // ends: the prefix's end, then each extension's; the last is the prompt's length.
  // small (for memcheck): a few cases, among them one through the compact layers' zeroed span (8,192: 4,097 on).
  const std::vector<Case> small_cases = {{4096, {40, 100}}, {4096, {100, 300, 301, 305}}, {8192, {4097, 4224, 4300, 4400}}};
  const std::vector<Case> all_cases = {
      {4096, {40, 100}}, {4096, {500, 501}}, {4096, {500, 505}}, {4096, {1000, 2048}},
      {4096, {100, 300, 301, 1200, 2000}}, {4096, {33, 34, 41, 2047}},
      {4096, {2049, 2050}}, {4096, {2100, 3000, 4000}}, {4096, {3500, 4088}},
      {8192, {3000, 4000}}, {8192, {4500, 8000}}, {8192, {4097, 4224, 4225, 4300, 8184}},
      {32768, {200, 1500}}, {32768, {2500, 4000}}, {32768, {4097, 4098}}, {32768, {5000, 5130}},
      {32768, {6000, 6127, 6128, 6129, 20000}}, {32768, {16384, 32760}}, {32768, {4200, 32760}},
  };
  const auto& cases = small ? small_cases : all_cases;
  const int steps = 8;
  bool all = true;
  for (const auto& c : cases) {
    const int total = c.ends.back();
    std::vector<int> prompt(tokens.begin(), tokens.begin() + total);
    auto& r = m.at(c.capacity);
    auto t0 = Clock::now();
    auto whole = r.forward(prompt);
    const double whole_s = seconds_since(t0);
    const uint64_t whole_cache = cache_hash(r);
    std::vector<int> whole_ids;
    auto whole_steps = decode(r, whole, steps, whole_ids);
    fresh(r);
    t0 = Clock::now();
    auto parts = r.forward(std::vector<int>(prompt.begin(), prompt.begin() + c.ends[0]));
    const double prefix_s = seconds_since(t0);
    t0 = Clock::now();
    for (size_t i = 1; i < c.ends.size(); ++i)
      parts = r.extend(std::vector<int>(prompt.begin() + c.ends[i - 1], prompt.begin() + c.ends[i]));
    const double extend_s = seconds_since(t0);
    const uint64_t parts_cache = cache_hash(r);
    std::vector<int> parts_ids;
    auto parts_steps = decode(r, parts, steps, parts_ids);
    int same_steps = 0;
    for (int i = 0; i < steps; ++i) same_steps += same(whole_steps[i], parts_steps[i]);
    const bool ok = same(whole, parts) && whole_cache == parts_cache && same_steps == steps && whole_ids == parts_ids;
    all &= ok;
    std::cout << "{\"mode\":\"exact\",\"capacity\":" << c.capacity << ",\"compact\":" << (any_compact(r) ? "true" : "false")
              << ",\"ends\":" << ints(c.ends) << ",\"logits_identical\":" << (same(whole, parts) ? "true" : "false")
              << ",\"cache_identical\":" << (whole_cache == parts_cache ? "true" : "false")
              << ",\"decode_steps_identical\":" << same_steps << ",\"decode_ids\":" << ints(parts_ids)
              << ",\"whole_prefill_s\":" << whole_s << ",\"prefix_prefill_s\":" << prefix_s << ",\"extensions_s\":" << extend_s
              << ",\"ok\":" << (ok ? "true" : "false") << "}" << std::endl;
  }
  return all;
}

void decoded(Model& m, const std::vector<int>& tokens) {
  // generated: decode steps before the continuation; 400 pass the 256-token window, so the compact layers' slots
  // have moved (WindowSlots::make_room) before extend() reads them.
  struct Case { int capacity, prefix, generated, added; };
  const std::vector<Case> cases = {{4096, 1000, 32, 500}, {4096, 300, 32, 2000}, {4096, 1000, 400, 600},
                                   {32768, 8000, 32, 1024}, {32768, 5000, 32, 20000}, {32768, 30000, 32, 1000},
                                   {32768, 8000, 400, 1024}, {32768, 4500, 400, 300}};
  const int steps = 16;
  for (const auto& c : cases) {
    const int generated = c.generated;
    auto& r = m.at(c.capacity);
    std::vector<int> seq(tokens.begin(), tokens.begin() + c.prefix);
    auto last = r.forward(seq);
    std::vector<int> ids;
    for (int i = 0; i < generated; ++i) { ids.push_back(argmax(last)); last = r.forward({ids.back()}); }
    seq.insert(seq.end(), ids.begin(), ids.end());
    const std::vector<int> added(tokens.begin() + c.prefix, tokens.begin() + c.prefix + c.added);
    // the state after the decoded positions, copied once for the two continuations
    const auto measure = [&](auto&& cont, std::vector<int>& follow) {
      std::vector<float> got = cont();
      std::vector<int> f;
      decode(r, got, steps, f);
      follow = f;
      return got;
    };
    std::vector<int> f_extend, f_tokens, f_whole;
    // extend()
    auto e = measure([&] { return r.extend(added); }, f_extend);
    // token at a time, on the same decoded state: rebuild it (prefill and the same greedy steps)
    fresh(r);
    last = r.forward(std::vector<int>(seq.begin(), seq.begin() + c.prefix));
    for (int id : ids) last = r.forward({id});
    auto t = measure([&] { std::vector<float> x; for (int id : added) x = r.forward({id}); return x; }, f_tokens);
    // the whole sequence prefilled at once
    fresh(r);
    std::vector<int> all = seq; all.insert(all.end(), added.begin(), added.end());
    auto w = measure([&] { return r.forward(all); }, f_whole);
    auto diff = [](const std::vector<float>& a, const std::vector<float>& b) {
      double mx = 0; for (size_t i = 0; i < a.size(); ++i) mx = std::max(mx, double(std::fabs(a[i] - b[i]))); return mx;
    };
    auto agree = [](const std::vector<int>& a, const std::vector<int>& b) {
      int n = 0; while (n < int(a.size()) && a[n] == b[n]) ++n; return n;
    };
    std::cout << "{\"mode\":\"decoded\",\"capacity\":" << c.capacity << ",\"prefix\":" << c.prefix << ",\"generated\":" << generated
              << ",\"added\":" << c.added << ",\"max_abs_extend_vs_whole\":" << diff(e, w) << ",\"max_abs_tokens_vs_whole\":" << diff(t, w)
              << ",\"max_abs_extend_vs_tokens\":" << diff(e, t) << ",\"argmax\":[" << argmax(e) << "," << argmax(t) << "," << argmax(w)
              << "],\"follow_agree_extend_whole\":" << agree(f_extend, f_whole) << ",\"follow_agree_tokens_whole\":" << agree(f_tokens, f_whole)
              << ",\"follow_agree_extend_tokens\":" << agree(f_extend, f_tokens) << ",\"of\":" << steps << "}" << std::endl;
  }
}

void speed(Model& m, const std::vector<int>& tokens) {
  struct Case { int capacity, prefix, added; };
  const std::vector<Case> cases = {
      {4096, 3072, 1}, {4096, 3072, 8}, {4096, 3072, 64}, {4096, 3072, 1024}, {4096, 1024, 1024}, {4096, 1, 4095},
      {32768, 31744, 1}, {32768, 31744, 8}, {32768, 31744, 64}, {32768, 31744, 1024}, {32768, 8192, 1024},
      {32768, 16384, 16384}, {32768, 1024, 31744}, {32768, 1, 32767},
  };
  {  // warm the card and its clocks: one prefill of each capacity's largest case is not timed
    auto& r = m.at(4096); r.forward(std::vector<int>(tokens.begin(), tokens.begin() + 4096));
    auto& q = m.at(32768); q.forward(std::vector<int>(tokens.begin(), tokens.begin() + 8192));
  }
  for (const auto& c : cases) {
    const int total = c.prefix + c.added;
    std::vector<int> prompt(tokens.begin(), tokens.begin() + total);
    auto& r = m.at(c.capacity);
    auto t0 = Clock::now();
    r.forward(prompt);
    const double whole = seconds_since(t0);
    fresh(r);
    r.forward(std::vector<int>(prompt.begin(), prompt.begin() + c.prefix));
    t0 = Clock::now();
    r.extend(std::vector<int>(prompt.begin() + c.prefix, prompt.end()));
    const double extend = seconds_since(t0);
    double tokens_s = -1;
    if (c.added <= 1024) {
      fresh(r);
      r.forward(std::vector<int>(prompt.begin(), prompt.begin() + c.prefix));
      t0 = Clock::now();
      for (int i = c.prefix; i < total; ++i) r.forward({prompt[i]});
      tokens_s = seconds_since(t0);
    }
    std::cout << "{\"mode\":\"speed\",\"capacity\":" << c.capacity << ",\"prefix\":" << c.prefix << ",\"added\":" << c.added
              << ",\"extend_s\":" << extend << ",\"whole_prefill_s\":" << whole << ",\"token_at_a_time_s\":" << tokens_s << "}" << std::endl;
  }
}
}  // namespace

int main(int argc, char** argv) try {
  if (argc != 4) throw std::runtime_error("usage: lfm3-continuation-check MODEL_DIR TOKENS.i32 exact|exact-small|decoded|speed");
  const std::string mode = argv[3];
  Model m{argv[1]};
  const auto tokens = read_tokens(argv[2], 32768 + 1024);
  std::cout << std::setprecision(9);
  if (mode == "exact" || mode == "exact-small") {
    const bool ok = exact(m, tokens, mode == "exact-small");
    std::cout << "{\"exact\":" << (ok ? "\"PASS\"" : "\"FAIL\"") << "}" << std::endl;
    return ok ? 0 : 1;
  }
  if (mode == "decoded") { decoded(m, tokens); return 0; }
  if (mode == "speed") { speed(m, tokens); return 0; }
  throw std::runtime_error("unknown mode");
} catch (const std::exception& e) {
  std::cerr << "lfm3-continuation-check: " << e.what() << std::endl;
  return 2;
}
