// Oracle for lfm/audio_modes.hpp (A4): replays forced tokens and frames through the mode machine and prints the
// per-step trace (what was sampled, what was fed back, whether a decode runs, the suppression set), the emitted
// slots and frames, and the unconsumed inputs. Each speech step also prints the repetition history a FrameSampler
// holds before it (length and hash), fed as the integration must: accept every frame whose Outcome::eos is false.
// stdin:
//   MODE(none|text|audio|interleaved) SUPPRESS_MIXED SUPPRESS_AUDIO_OUT RETURN_AUDIO_EOS MAX_NEW LAST_PROMPT_TOKEN
//   CALLER_CACHE REPLAYED [12 codes of the prompt's last replayed frame, when REPLAYED is 1]
//   N_TOKENS tokens...   N_FRAMES then 12 codes per frame
#include <lfm/audio_modes.hpp>
#include <lfm/audio_sampling.hpp>

#include <deque>
#include <iostream>
#include <string>

using namespace lfm::audio;

template <class C>
static void list(const C& c) {
  bool first = true;
  for (auto v : c) std::cout << (first ? "" : ",") << v, first = false;
}

// tools/lfm3_mode_reference.py's history_hash: frames in order, codes in order, mod 2^61 - 1
static unsigned long long history_hash(const FrameSampler& s, size_t& length) {
  std::vector<int> columns[kCodebooks];
  for (int c = 0; c < kCodebooks; ++c) columns[c] = s.history_codes(c);
  length = columns[0].size();
  const unsigned long long m = (1ull << 61) - 1;
  unsigned long long h = 0;
  for (size_t f = 0; f < length; ++f)
    for (int c = 0; c < kCodebooks; ++c)
      h = (unsigned long long)(((unsigned __int128)h * 1000003u + unsigned(columns[c][f]) + 1) % m);
  return h;
}

int main() try {
  std::string mode;
  int sm, sa, re, max_new, last, cc, replayed;
  std::cin >> mode >> sm >> sa >> re >> max_new >> last >> cc >> replayed;
  Frame last_frame{};
  if (replayed) for (int& c : last_frame) std::cin >> c;
  ModeOptions o;
  o.mode = mode == "text" ? Requested::Text : mode == "audio" ? Requested::Audio
         : mode == "interleaved" ? Requested::Interleaved : Requested::None;
  o.suppress_mixed = sm, o.suppress_audio_out = sa, o.return_audio_eos = re, o.max_new_tokens = max_new;
  o.caller_cache = cc;
  int n;
  std::cin >> n;
  std::deque<int> tokens{}; tokens.resize(size_t(n));
  for (int& t : tokens) std::cin >> t;
  std::cin >> n;
  std::deque<Frame> frames{}; frames.resize(size_t(n));
  for (Frame& f : frames) for (int& c : f) std::cin >> c;
  if (!std::cin) throw std::runtime_error("bad input");

  ModeMachine m(ModeTokens{}, o);
  FrameSampler sampler;  // only its history is used
  std::cout << "marker " << m.begin(last, replayed ? &last_frame : nullptr) << "\n";
  std::vector<int> slots;
  std::vector<Frame> emitted;
  for (Step s; (s = m.next()) != Step::Done;) {
    Outcome r;
    if (s == Step::Text) {
      if (tokens.empty()) throw std::runtime_error("forced tokens exhausted");
      const int token = tokens.front();
      tokens.pop_front();
      const auto suppressed = m.suppressed();
      r = m.on_text(token);
      std::cout << "T " << token << " " << r.decode << " " << r.token << " ";
      list(suppressed);
      std::cout << "\n";
    } else {
      if (frames.empty()) throw std::runtime_error("forced frames exhausted");
      const Frame f = frames.front();
      frames.pop_front();
      size_t length;
      const unsigned long long h = history_hash(sampler, length);
      r = m.on_frame(f);
      if (!r.eos) sampler.accept(r.fed);
      std::cout << "A ";
      list(f);
      std::cout << " " << r.decode << " ";
      list(r.fed);
      std::cout << " H " << length << " " << h << "\n";
      if (r.emitted) emitted.push_back(r.fed);
    }
    if (r.emitted) slots.push_back(r.slot);
  }
  std::cout << "slots ";
  list(slots);
  std::cout << "\nframes " << emitted.size() << "\n";
  for (const Frame& f : emitted) list(f), std::cout << "\n";
  std::cout << "left " << tokens.size() << " " << frames.size() << "\n";
  return 0;
} catch (const std::exception& e) {
  std::cerr << "audio_modes_oracle: " << e.what() << "\n";
  return 1;
}
