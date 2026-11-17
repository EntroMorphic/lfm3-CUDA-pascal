// Host check of lfm3_speech::run_chunks (lfm/speech_run.hpp) against a scripted fake session: each chunk "wants" a
// number of steps; the fake takes min(wanted, max_new_tokens), the last a text pick and the rest frames, and ends by EOS
// (EOS frame, then <|im_end|>) only when it was not cut by its cap. Checks the calls run_chunks makes (seed, codec
// continue/finalize, cap), the step bound and the truncated flag. Prints "ok N" or the first failure (exit 1).
#include <lfm/speech_run.hpp>

#include <iostream>
#include <random>
#include <sstream>

namespace {
struct Call { uint64_t seed; int cap; bool cont, fin; };
struct Fake {
  std::vector<size_t> want;
  std::vector<Call> calls;
  lfm3_speech::Result generate(const std::vector<int>& prompt, const lfm3_speech::Options& o) {
    const size_t k = calls.size();
    calls.push_back({o.seed, o.max_new_tokens, o.codec_continue, o.codec_finalize});
    (void)prompt;
    lfm3_speech::Result r;
    const size_t steps = std::min(want[k], size_t(o.max_new_tokens));
    const bool natural = want[k] <= size_t(o.max_new_tokens);
    for (size_t i = 0; i + 1 < steps; ++i) {
      lfm::audio::Frame f{};
      if (natural && i + 2 == steps) f[0] = lfm::audio::kEosCode;
      r.raw_frames.push_back(f);
      if (f[0] != lfm::audio::kEosCode) r.frames.push_back(f), r.slots.push_back(125032);
    }
    if (steps) r.text_tokens.push_back(natural ? lfm::audio::ModeTokens{}.eos[0] : 1), r.slots.push_back(r.text_tokens.back());
    r.wav.assign(steps, 0.0f);
    return r;
  }
};
int failures = 0, checks = 0;
void expect(bool c, const std::string& what) { ++checks; if (!c && !failures++) std::cout << "FAIL: " << what << "\n"; }

void run(const std::string& name, std::vector<size_t> want, int cap, size_t budget, bool want_truncated, size_t want_run) {
  Fake f{want, {}};
  std::vector<std::vector<int>> prompts(want.size(), std::vector<int>{1});
  lfm3_speech::Options o;
  o.max_new_tokens = cap; o.seed = 7;
  std::vector<lfm3_speech::ChunkInfo> info;
  bool truncated = true;
  const auto r = lfm3_speech::run_chunks(f, prompts, o, true, info, budget, &truncated);
  size_t used = r.raw_frames.size() + r.text_tokens.size();
  expect(truncated == want_truncated, name + ": truncated");
  expect(f.calls.size() == want_run && info.size() == want_run, name + ": chunks run");
  if (budget) expect(used <= budget, name + ": steps within the budget");
  for (size_t k = 0; k < f.calls.size(); ++k) {
    expect(f.calls[k].seed == 7000 + k, name + ": seed");
    expect(f.calls[k].cont == (k > 0), name + ": codec continue");
    expect(f.calls[k].fin == (k + 1 == f.calls.size()), name + ": codec finalized exactly on the last chunk run");
    expect(k + 1 == f.calls.size() || f.calls[k].cap == cap, name + ": cap kept before the last chunk");
  }
}
}  // namespace

int main() {
  run("no budget", {50, 60, 70}, 100, 0, false, 3);
  run("budget never near", {50, 60, 70}, 100, 6144, false, 3);
  run("capped chunk ends naturally, last", {100, 100, 30}, 100, 230, false, 3);   // 3rd: remaining 30 <= 100, wants 30
  run("capped chunk ends naturally, chunks left", {100, 20, 30, 40}, 100, 200, true, 2);  // 2nd capped at 100, ends at 20
  run("capped chunk uses its whole cap", {100, 100, 100}, 100, 250, true, 3);     // 3rd capped at 50, wants 100
  run("remaining exactly the cap", {100, 100}, 100, 200, false, 2);               // 2nd: remaining 100 <= 100, wants 100
  run("remaining equals the cap, last chunk cut by its own cap", {100, 150}, 100, 200, false, 2);  // not the budget's doing
  run("remaining equals the cap, chunks left", {100, 100, 50}, 100, 200, true, 2);  // 3rd skipped: the budget's doing
  run("remaining one above the cap", {100, 100, 5}, 100, 201, true, 3);           // 2nd keeps cap; 3rd gets 1, wants 5: cut
  run("remaining two, short end", {100, 100, 2}, 100, 202, false, 3);  // 3rd gets 2 and wants just 2 (EOS frame, im_end)
  // property: random chunks and budgets
  std::mt19937 rng(1);
  for (int t = 0; t < 20000; ++t) {
    const size_t n = 1 + rng() % 12, cap = 2 + rng() % 59, budget = 1 + rng() % 400;
    std::vector<size_t> want(n);
    for (auto& w : want) w = 2 + rng() % 79;  // a real ending takes at least 2 steps (EOS frame, <|im_end|>)
    if (cap >= budget) continue;  // the server's case: max_new_tokens < budget
    Fake f{want, {}};
    std::vector<std::vector<int>> prompts(n, std::vector<int>{1});
    lfm3_speech::Options o; o.max_new_tokens = int(cap);
    std::vector<lfm3_speech::ChunkInfo> info;
    bool truncated = false;
    const auto r = lfm3_speech::run_chunks(f, prompts, o, true, info, budget, &truncated);
    const size_t used = r.raw_frames.size() + r.text_tokens.size();
    // `truncated` reports the budget only (a chunk cut by its own max_new_tokens shows as ended_by_eos false): chunks
    // skipped, or the chunk the budget capped (remaining <= cap before it) cut short of its text
    size_t before_last = 0;
    for (size_t k = 0; k + 1 < f.calls.size(); ++k) before_last += std::min(want[k], size_t(f.calls[k].cap));
    const bool budget_capped_last = budget - before_last < cap;  // the budget lowered the last chunk's cap
    // an independent statement of which chunks run with which caps: chunk k runs; if what remains is at most the cap,
    // it gets what remains and is the last
    {
      size_t used_ref = 0, k = 0;
      std::vector<int> caps;
      for (; k < n; ++k) {
        const size_t rem = budget - used_ref;
        if (rem <= cap) { caps.push_back(int(rem)); break; }
        caps.push_back(int(cap));
        used_ref += std::min(want[k], cap);
      }
      bool same = caps.size() == f.calls.size();
      for (size_t j = 0; same && j < caps.size(); ++j) same = caps[j] == f.calls[j].cap;
      expect(same, "reference schedule (chunks run and their caps)");
    }
    // far from the budget: exactly the calls and output of no budget at all
    size_t natural = 0;
    for (const auto w : want) natural += std::min(w, cap);
    if (natural + cap < budget) {
      Fake g{want, {}};
      std::vector<lfm3_speech::ChunkInfo> info0;
      const auto r0 = lfm3_speech::run_chunks(g, prompts, o, true, info0);
      bool same = g.calls.size() == f.calls.size() && r0.raw_frames.size() == r.raw_frames.size() && r0.wav == r.wav;
      for (size_t j = 0; same && j < g.calls.size(); ++j)
        same = g.calls[j].cap == f.calls[j].cap && g.calls[j].fin == f.calls[j].fin && g.calls[j].seed == f.calls[j].seed;
      expect(same, "far from the budget: identical to no budget");
    }
    const bool expected = f.calls.size() < n || (budget_capped_last && want[f.calls.size() - 1] > size_t(f.calls.back().cap));
    std::ostringstream id; id << "property case " << t << " (n " << n << ", cap " << cap << ", budget " << budget << ")";
    expect(used <= budget, id.str() + ": steps within the budget");
    expect(truncated == expected, id.str() + ": truncated exactly when the budget stopped the text");
    expect(f.calls.back().fin && (f.calls.size() == 1 || !f.calls[f.calls.size() - 2].fin), id.str() + ": finalize once, last");
  }
  if (failures) return 1;
  std::cout << "ok " << checks << "\n";
}
