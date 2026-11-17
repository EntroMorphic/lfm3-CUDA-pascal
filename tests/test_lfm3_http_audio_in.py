"""Host tests for audio input on POST /speech (#3; native/include/lfm/lfm3_http.hpp) behind the harness's fake engine.

The fake answers op "speech" as WorkerCore does: it decodes each clip and checks the prompt's <|audio|> spans against
them (lfm3_speech::check_audio_spans), and reports each chunk's prompt length as its "frames", so these tests see the
prompt the Service built: the template's single <|audio|> per clip expanded to the clip's encoder token count
(lfm::audio_features::tokens). Every build variant of the harness runs (plain, ASan+UBSan, TSan). The engine itself,
the renderer against HF's processor and the recognition quality are checked on Thor (validation/audio-in-v1).
"""
import base64
import io
import math
import struct
import wave

import pytest

from test_lfm3_http_native import harness_binary, running  # noqa: F401  (fixture)


def wav(seconds, rate=16_000, channels=1, width=2, samples=None):
    n = int(round(seconds * rate)) if samples is None else samples
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels), w.setsampwidth(width), w.setframerate(rate)
        frames = b"".join(struct.pack("<h", int(8000 * math.sin(i / 7))) * channels for i in range(n)) if width == 2 \
            else bytes(128 for _ in range(n * channels))
        w.writeframes(frames)
    return base64.b64encode(buf.getvalue()).decode()


def tokens(samples):
    """lfm::audio_features::tokens (Lfm3OmniFeatureExtractor.num_frames, then ceil(frames / 8))."""
    padded = 400 if samples < 400 else samples + (-(samples - 400) % 160)
    return -(-(1 + (padded - 400) // 160) // 8)


def ask(clips, mode="text", text=None, system="Perform ASR in English.", max_new_tokens=256, **extra):
    content = [{"type": "audio", "audio": c} for c in clips] + ([{"type": "text", "text": text}] if text else [])
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
    return dict(messages=messages, mode=mode, max_new_tokens=max_new_tokens, **extra)


def prompt_length(h, body):
    status, reply = h.call("POST", "/speech", body)
    assert status == 200 and reply["cancelled"] is False, (status, reply)
    assert len(reply["chunks"]) == 1
    return reply, reply["chunks"][0]["frames"]


def test_each_clip_expands_to_its_token_count(harness_binary):
    with running(harness_binary) as h:
        # the same conversation with clips of different lengths: the prompts differ by exactly the token counts
        _, short = prompt_length(h, ask([wav(1.0)]))
        _, longer = prompt_length(h, ask([wav(7.3)]))
        assert longer - short == tokens(116_800) - tokens(16_000)
        _, both = prompt_length(h, ask([wav(1.0), wav(7.3)]))
        # a second clip adds its tokens and its own <|vision_start|>, <|vision_end|>
        assert both - short == tokens(116_800) + 2
        # the token count at the edges of the feature frames (400-sample window, 160-sample hop, 8 frames a token)
        for samples in (1, 399, 400, 401, 559, 560, 1520, 1521, 1680, 1681):
            _, n = prompt_length(h, ask([wav(0, samples=samples)]))
            assert n - short == tokens(samples) - tokens(16_000), samples


def test_text_mode_and_spoken_replies(harness_binary):
    with running(harness_binary) as h:
        text, _ = prompt_length(h, ask([wav(2.0)]))
        assert text["mode"] == "text" and text["split"] is False and text["text"].startswith("Café")
        # speech to speech: audio input in audio mode is not split by default
        reply, _ = prompt_length(h, ask([wav(2.0)], mode="audio", system="Respond with audio."))
        assert reply["mode"] == "audio" and reply["split"] is False and reply["text"] == ""
        reply, _ = prompt_length(h, ask([wav(2.0)], mode="interleaved", text="What did I say?", system=None))
        assert reply["mode"] == "interleaved" and reply["split"] is False
        # the fake's text-mode slots are its phrase alone; interleaved, the phrase with <|audio_out|> between, which the
        # Service drops: the same text either way
        assert text["text"] == reply["text"]
        # text mode with text input only works too
        status, reply = h.call("POST", "/speech", dict(messages=[{"role": "user", "content": "Hello."}], mode="text",
                                                        max_new_tokens=16))
        assert status == 200 and reply["split"] is False


def test_ninety_seconds_fits_the_request_limits(harness_binary):
    # 90 s of 16-bit audio is about 3.8 MB of base64, inside the 4 MiB body and engine-frame limits (a longer clip is
    # a 413 body_limit, tested generically in test_lfm3_http_native.py)
    with running(harness_binary) as h:
        reply, n = prompt_length(h, ask([wav(90.0)]))
        assert n > tokens(90 * 16_000)


@pytest.mark.parametrize("make", [
    lambda: ask(["UklGRiwAAABXQVZF!!!!"]),                 # not base64
    lambda: ask([wav(1.0)[:-4]]),                          # valid base64 of a truncated WAV (data chunk cut short)
    lambda: ask(["QUJD"]),                                 # base64 of "ABC": not a WAV file
    lambda: ask([wav(1.0, rate=24_000)]),                  # wrong sample rate
    lambda: ask([wav(1.0, channels=2)]),                   # stereo
    lambda: ask([wav(0.0)]),                               # no samples
    lambda: ask([wav(1.0, width=1)]),                      # 8-bit PCM
    lambda: ask([""]),                                     # empty audio string
    lambda: ask([wav(1.0)], split=True, mode="audio"),     # audio input is not split
    lambda: dict(messages=[{"role": "user", "content": "Hi."},
                           {"role": "assistant", "content": [{"type": "audio", "audio": wav(1.0)}]},
                           {"role": "user", "content": "Again."}], mode="text", max_new_tokens=16),  # assistant audio
    lambda: dict(messages=[{"role": "system", "content": [{"type": "audio", "audio": wav(1.0)}]},
                           {"role": "user", "content": "Hi."}], mode="text", max_new_tokens=16),  # system audio
    lambda: dict(messages=[{"role": "user", "content": [{"type": "audio", "audio": wav(1.0), "extra": 1}]}],
                 mode="text", max_new_tokens=16),          # extra key in the part
    lambda: dict(messages=[{"role": "user", "content": [{"type": "audio", "audio": 7}]}], mode="text",
                 max_new_tokens=16),                       # audio not a string
    lambda: dict(messages=[{"role": "user", "content": "transcribe <|audio|> please"}], mode="text",
                 max_new_tokens=16),                       # a placeholder typed as text
    lambda: ask([wav(60.0), wav(30.0)], max_new_tokens=3000),  # prompt + output over the 4096 capacity
])
def test_audio_refusals_leave_the_server_serving(harness_binary, make):
    bad = make()
    with running(harness_binary) as h:
        status, body = h.call("POST", "/speech", bad)
        assert status == 400 and body == {"error": "invalid_request"}, (status, body)
        status, body = h.call("POST", "/speech", ask([wav(1.0)]))
        assert status == 200 and body["cancelled"] is False


def float_wav(seconds):
    x = struct.pack(f"<{int(seconds * 16_000)}f", *(0.25 * math.sin(i / 9) for i in range(int(seconds * 16_000))))
    fmt = struct.pack("<HHIIHH", 3, 1, 16_000, 64_000, 4, 32)
    body = b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", len(x)) + x
    return base64.b64encode(b"RIFF" + struct.pack("<I", len(body)) + body).decode()


def test_float_wav_and_text_around_clips(harness_binary):
    with running(harness_binary) as h:
        _, a = prompt_length(h, ask([wav(1.5)]))
        _, b = prompt_length(h, ask([float_wav(1.5)]))
        assert a == b  # the same token count from a 32-bit float file
        # text parts on either side of a clip are checked as separate runs: "<|" + clip + "x|>" never forms a token
        body = dict(messages=[{"role": "user", "content": [{"type": "text", "text": "say <|"},
                                                            {"type": "audio", "audio": wav(1.0)},
                                                            {"type": "text", "text": "x|> now"}]}],
                    mode="text", max_new_tokens=16)
        status, reply = h.call("POST", "/speech", body)
        assert status == 200, reply
        # within one run the joined text is still checked
        body["messages"][0]["content"] = [{"type": "text", "text": "say <|"}, {"type": "text", "text": "x|>"},
                                          {"type": "audio", "audio": wav(1.0)}]
        status, reply = h.call("POST", "/speech", body)
        assert status == 400 and reply == {"error": "invalid_request"}


def test_clip_count_cap(harness_binary):
    with running(harness_binary) as h:
        tiny = wav(0, samples=400)
        _, n = prompt_length(h, ask([tiny] * 64))
        status, reply = h.call("POST", "/speech", ask([tiny] * 65))
        assert status == 400 and reply == {"error": "invalid_request"}
        status, reply = h.call("POST", "/speech", ask([wav(1.0)]))
        assert status == 200


def test_no_audio_encoder_is_501_and_speech_still_serves(harness_binary):
    with running(harness_binary, "--audio-input", "off") as h:
        status, reply = h.call("POST", "/speech", ask([wav(1.0)]))
        assert status == 501 and reply == {"error": "audio_input_unsupported"}, (status, reply)
        status, reply = h.call("POST", "/speech", dict(messages=[{"role": "user", "content": "Read this aloud."}],
                                                        mode="audio", max_new_tokens=64))
        assert status == 200 and reply["cancelled"] is False
