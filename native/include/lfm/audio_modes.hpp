#pragma once
// The text/speech mode machine of LFM3-Omni generation (HF `Lfm3OmniForConditionalGeneration._generate`, batch 1),
// #2 step 3, bar A4 (journal/lfm3-tts-quality-bar.md). Host logic only: the caller samples what `next()` asks for,
// hands the result to `on_text` / `on_frame`, feeds the returned input to the language model when `decode` is set,
// and stops at Step::Done.
//
// Rules, as HF:
// - The requested mode appends its marker to the prompt unless the prompt already ends with one: <|audio_start|>
//   (or <|audio_out|>) for audio, <|mixed_start|> for interleaved. With no mode, the prompt's last token chooses,
//   except that a prompt ending on a replayed <|audio_out|> frame holding an EOS code resumes text, as a frame
//   sampled inside the call would.
// - Audio mode: every step is a frame. A frame with any EOS code (2016) is rewritten to all-EOS, fed back but not
//   emitted (unless return_audio_eos), and switches to text.
// - Interleaved: 6 text slots, then 12 frames, alternating; <|text_end|> ends the text and starts the audio tail,
//   which runs until an EOS frame. Inside an interleaved block the mode markers are ordinary text.
// - Text: <|text_end|> ends the turn outside a block; an eos id ends it anywhere; outside a block, <|audio_start|>
//   switches to audio and <|mixed_start|> opens a new interleaved block.
// - Text steps never sample the placeholder tokens (<|audio|>, <|audio_out|>, image, video); text-only requests also
//   never sample the two markers; suppress_mixed / suppress_audio_out remove either marker always.
// - The last step (turn finished, or max_new_tokens reached) is not decoded, unless the caller supplied the cache
//   (HF's `past_key_values`): then it is, so the cache covers every returned slot. max_new_tokens must be positive.
// - The speech repetition history is the caller's (lfm::audio::FrameSampler): accept every frame whose
//   Outcome::eos is false, for the whole request.
#include <lfm/audio_sampling.hpp>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace lfm::audio {

struct ModeTokens {
  int audio_start = 125012, text_end = 125014, mixed_start = 125015, audio_out = 125032;
  std::vector<int> placeholders = {125031, 125032, 125030, 125029};  // audio, audio_out, image, video
  std::vector<int> eos = {124900, 125014};                            // <|im_end|>, <|text_end|>
  int n_text = 6, n_audio = 12;
};

enum class Mode { Text, Audio, Interleaved };
enum class Requested { None, Text, Audio, Interleaved };
enum class Step { Text, Audio, Done };

struct ModeOptions {
  Requested mode = Requested::None;
  bool suppress_mixed = false, suppress_audio_out = false, return_audio_eos = false;
  bool caller_cache = false;  // the caller supplied the KV cache: decode the last step too
  int max_new_tokens = 0;
};

struct Outcome {
  bool emitted = false;  // a slot was emitted: `slot` (a text token, or audio_out for a frame)
  int slot = -1;
  bool is_frame = false;  // what to feed back: the frame `fed` (codes) or the token `slot`'s input `token`
  int token = -1;
  Frame fed{};
  bool decode = false;  // run the language model on the fed input
  bool eos = false;     // an EOS frame: not for the penalty history (FrameSampler::accept only when false)
};

class ModeMachine {
 public:
  ModeMachine(ModeTokens tokens, ModeOptions options) : t_(std::move(tokens)), o_(options) {
    // HF refuses max_new_tokens = 0 (GenerationConfig.validate)
    if (o_.max_new_tokens < 1) throw std::invalid_argument("max_new_tokens must be positive");
  }

  // Before the prefill: the marker to append to the prompt (-1 for none). `last_replayed_frame`: the codes of the
  // prompt's last <|audio_out|> placeholder, when the prompt replays speech (HF `audio_out_codes[-1]`).
  // A prompt ending on <|audio_out|> must replay its frame: HF refuses placeholders without `audio_out_codes`.
  int begin(int last_prompt_token, const Frame* last_replayed_frame = nullptr) {
    if (last_prompt_token == t_.audio_out && !last_replayed_frame)
      throw std::invalid_argument("ModeMachine: a prompt ending on <|audio_out|> needs its replayed frame");
    // HF: only a prompt that ends on <|audio_out|>, and only an EOS code anywhere in that frame
    const bool last_frame_eos = last_replayed_frame && last_prompt_token == t_.audio_out &&
                                std::find(last_replayed_frame->begin(), last_replayed_frame->end(), kEosCode) !=
                                    last_replayed_frame->end();
    int marker = -1;
    switch (o_.mode) {
      case Requested::None:
        mode_ = last_frame_eos                                                          ? Mode::Text
                : last_prompt_token == t_.audio_start || last_prompt_token == t_.audio_out ? Mode::Audio
                : last_prompt_token == t_.mixed_start                                   ? Mode::Interleaved
                                                                                        : Mode::Text;
        break;
      case Requested::Text: mode_ = Mode::Text; text_only_ = true; break;
      case Requested::Audio:
        mode_ = Mode::Audio;
        if (last_prompt_token != t_.audio_start && last_prompt_token != t_.audio_out) marker = t_.audio_start;
        break;
      case Requested::Interleaved:
        mode_ = Mode::Interleaved;
        if (last_prompt_token != t_.mixed_start) marker = t_.mixed_start;
        break;
    }
    slots_left_ = t_.n_text;
    speaking_ = text_done_ = finished_ = false;
    step_ = 0;
    begun_ = true;
    return marker;
  }

  Step next() const {
    if (!begun_) throw std::logic_error("ModeMachine: begin() first");
    if (finished_ || step_ >= o_.max_new_tokens) return Step::Done;
    return mode_ == Mode::Audio || (mode_ == Mode::Interleaved && speaking_) ? Step::Audio : Step::Text;
  }

  // The tokens a text step must not sample, ascending.
  std::vector<int> suppressed() const {
    std::vector<int> s = t_.placeholders;
    if (o_.suppress_mixed || text_only_) s.push_back(t_.mixed_start);
    if (o_.suppress_audio_out || text_only_) s.push_back(t_.audio_start);
    std::sort(s.begin(), s.end());
    s.erase(std::unique(s.begin(), s.end()), s.end());
    return s;
  }

  Outcome on_text(int token) {
    if (next() != Step::Text) throw std::logic_error("ModeMachine: not a text step");
    const bool interleaving = mode_ == Mode::Interleaved;
    Outcome r;
    r.emitted = true;
    r.slot = r.token = token;
    if (token == t_.text_end) {
      if (interleaving) text_done_ = speaking_ = true;
      else finished_ = true;
    } else if (std::find(t_.eos.begin(), t_.eos.end(), token) != t_.eos.end()) {
      finished_ = true;
    } else if (interleaving) {
      // markers are ordinary text inside a block
    } else if (token == t_.audio_start) {
      mode_ = Mode::Audio;
    } else if (token == t_.mixed_start) {
      mode_ = Mode::Interleaved;
      slots_left_ = t_.n_text;
      speaking_ = text_done_ = false;
    }
    return finish_step(r, interleaving);
  }

  Outcome on_frame(const Frame& raw) {
    if (next() != Step::Audio) throw std::logic_error("ModeMachine: not an audio step");
    const bool interleaving = mode_ == Mode::Interleaved;
    Outcome r;
    r.is_frame = true;
    r.fed = raw;
    const bool eos = std::find(raw.begin(), raw.end(), kEosCode) != raw.end();
    if (eos) {
      r.fed.fill(kEosCode);
      mode_ = Mode::Text;
    }
    if (!eos || o_.return_audio_eos) r.emitted = true, r.slot = t_.audio_out;
    r.eos = eos;
    return finish_step(r, interleaving);
  }

  Mode mode() const { return mode_; }
  int steps() const { return step_; }

 private:
  Outcome finish_step(Outcome r, bool interleaving) {
    if (interleaving && !text_done_ && --slots_left_ == 0) {
      speaking_ = !speaking_;
      slots_left_ = speaking_ ? t_.n_audio : t_.n_text;
    }
    ++step_;
    r.decode = o_.caller_cache || !(finished_ || step_ >= o_.max_new_tokens);
    return r;
  }

  ModeTokens t_;
  ModeOptions o_;
  Mode mode_ = Mode::Text;
  bool text_only_ = false, speaking_ = false, text_done_ = false, finished_ = false, begun_ = false;
  int slots_left_ = 0, step_ = 0;
};

}  // namespace lfm::audio
