// Host check of lfm3_speech::check_audio_spans and check_image_spans (lfm/speech_run.hpp; validation/images-v1 rule 5):
// the engine's last word on a prompt's media spans before any state changes. Each case is a prompt, the media's token
// counts, and the refusal expected (or none). Prints each failure; exit 1 on any. Run by tests/test_media_spans.py.
#include <lfm/speech_run.hpp>

#include <iostream>
#include <string>
#include <vector>

namespace {
constexpr int S = 125009, E = 125010, A = 125031, I = 125030, T = 1000;  // vision_start/end, <|audio|>, <|image|>, text

std::vector<int> span(int token, size_t n) {
  std::vector<int> v{S};
  v.insert(v.end(), n, token);
  v.push_back(E);
  return v;
}
std::vector<int> cat(std::initializer_list<std::vector<int>> parts) {
  std::vector<int> v;
  for (const auto& p : parts) v.insert(v.end(), p.begin(), p.end());
  return v;
}

int failures = 0;
void expect(const std::string& name, const std::vector<int>& prompt, const std::vector<size_t>& clips,
            const std::vector<size_t>& images, const std::string& refusal) {
  std::string got;
  try {
    lfm3_speech::check_audio_spans(prompt, clips);
    lfm3_speech::check_image_spans(prompt, images);
  } catch (const std::exception& e) {
    got = e.what();
  }
  if (got != refusal) {
    ++failures;
    std::cout << "FAIL " << name << ": expected \"" << refusal << "\", got \"" << got << "\"\n";
  }
}
}  // namespace

int main() {
  const std::vector<int> head{T, T}, tail{T};
  // accepted
  expect("text only", cat({head, tail}), {}, {}, "");
  expect("one image", cat({head, span(I, 64), tail}), {}, {64}, "");
  expect("two images", cat({head, span(I, 64), span(I, 256), tail}), {}, {64, 256}, "");
  expect("audio then image", cat({head, span(A, 13), span(I, 64), tail}), {13}, {64}, "");
  expect("image then audio", cat({head, span(I, 64), {T}, span(A, 13), tail}), {13}, {64}, "");
  expect("interleaved", cat({span(A, 2), span(I, 4), span(A, 3), span(I, 5)}), {2, 3}, {4, 5}, "");
  // image refusals
  expect("image count", cat({head, span(I, 63), tail}), {}, {64},
         "speech: image 0 needs 64 placeholders, the prompt has 63");
  expect("second image count", cat({head, span(I, 64), span(I, 255), tail}), {}, {64, 256},
         "speech: image 1 needs 256 placeholders, the prompt has 255");
  expect("image without images", cat({head, span(I, 64), tail}), {}, {}, "speech: more <|image|> spans than images");
  expect("images without spans", cat({head, tail}), {}, {64}, "speech: fewer <|image|> spans than images");
  expect("image span missing its start", cat({head, {I, I, E}, tail}), {}, {2},
         "speech: an <|image|> span must be wrapped in <|vision_start|> and <|vision_end|>");
  expect("image span missing its end", cat({head, {S, I, I}, tail}), {}, {2},
         "speech: an <|image|> span must be wrapped in <|vision_start|> and <|vision_end|>");
  expect("image span at the very start", {I, I, E}, {}, {2},
         "speech: an <|image|> span must be wrapped in <|vision_start|> and <|vision_end|>");
  expect("image span at the very end", {S, I, I}, {}, {2},
         "speech: an <|image|> span must be wrapped in <|vision_start|> and <|vision_end|>");
  // a span of one medium is not the other's: the counts belong to each medium in order
  expect("media swapped", cat({span(I, 64), span(A, 13)}), {64}, {13},
         "speech: audio clip 0 needs 64 placeholders, the prompt has 13");
  // audio refusals keep their exact wording (validation/audio-in-v1)
  expect("audio count", cat({head, span(A, 12), tail}), {13}, {},
         "speech: audio clip 0 needs 13 placeholders, the prompt has 12");
  expect("audio without clips", cat({head, span(A, 13), tail}), {}, {}, "speech: more <|audio|> spans than audio clips");
  expect("clips without spans", cat({head, tail}), {13}, {}, "speech: fewer <|audio|> spans than audio clips");
  expect("audio span unwrapped", cat({head, {A, A}, tail}), {2}, {},
         "speech: an <|audio|> span must be wrapped in <|vision_start|> and <|vision_end|>");
  std::cout << (failures ? "FAILED" : "all passed") << "\n";
  return failures ? 1 : 0;
}
