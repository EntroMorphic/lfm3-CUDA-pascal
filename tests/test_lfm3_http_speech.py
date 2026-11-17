"""Host tests for POST /speech (native/include/lfm/lfm3_http.hpp) behind the harness's fake engine.

The fake answers op "speech" as WorkerCore does, with each chunk's frame count set to its prompt length, so these tests
see exactly the prompts the Service sent: the splitter's sentences (lfm/speech_text.hpp), each rendered as a single
request. Every build variant of the harness runs (plain, ASan+UBSan, TSan); responses are checked byte for byte
against Python's json.dumps of their content (parse_response). The engine itself is checked on Thor
(validation/speech-server-v1: the server's WAV against lfm3-speech's).
"""
import importlib.util
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

from test_lfm3_http_native import CHECKPOINT, ROOT, harness_binary, running  # noqa: F401  (fixture)

SPEC = importlib.util.spec_from_file_location("chunked", ROOT / "tools/lfm3_speech_chunked_hf.py")
chunked = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chunked)

SYSTEM = "Perform TTS in American English using a male voice."
LONG = ("Lighthouses were once the most important safety systems on any coast. Before radio and satellite navigation, a "
        "ship's captain relied on a steady beam of light to judge distance and direction in the dark. Each lighthouse had "
        "its own pattern of flashes, so sailors could tell one from another.")
FAKE_WAV = "UklGRiwAAABXQVZFZm10IBAAAAABAAEAwF0AAIC7AAACABAAZGF0YQgAAAAAAAEA//8AgA=="
KEYS = ["mode", "text", "audio_wav_base64", "sample_rate", "audio_seconds", "frames", "chunks", "seed", "split", "normalize",
        "cancelled", "truncated", "prefill_seconds", "generate_seconds"]


@pytest.fixture(scope="module")
def text_tool(tmp_path_factory):
    if shutil.which("g++") is None:
        pytest.skip("g++ required")
    binary = tmp_path_factory.mktemp("speech-text") / "lfm3-text-tool"
    subprocess.run(["g++", "-std=c++20", "-O2", "-I", str(ROOT / "native/include"),
                    str(ROOT / "native/tests/lfm3_text_tool.cpp"), "-o", str(binary)], check=True)
    return binary


def tool_rows(tool, tmp_path, mode, rows):
    src, dst = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    subprocess.run([str(tool), str(CHECKPOINT / "tokenizer.json"), mode, str(src), str(dst)], check=True, capture_output=True)
    return [json.loads(line) for line in dst.read_text(encoding="utf-8").split("\n") if line]


def tts(text, **extra):
    return dict(messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}], mode="audio",
                max_new_tokens=512, **extra)


def test_audio_is_split_and_each_sentence_rendered_alone(harness_binary, text_tool, tmp_path):
    want = tool_rows(text_tool, tmp_path, "speech", [dict(messages=tts(LONG)["messages"], capacity=4096, max_new_tokens=512)])[0]
    assert want["texts"] == chunked.sentences(LONG) and len(want["texts"]) == 3
    with running(harness_binary) as h:
        status, body = h.call("POST", "/speech", tts(LONG, seed=7))
        assert status == 200 and list(body) == KEYS, body
        assert body["mode"] == "audio" and body["text"] == "" and body["audio_wav_base64"] == FAKE_WAV
        assert body["sample_rate"] == 24000 and body["seed"] == 7 and body["split"] is True and body["cancelled"] is False
        assert body["truncated"] is False
        assert [c["text"] for c in body["chunks"]] == want["texts"]
        assert [c["frames"] for c in body["chunks"]] == [len(ids) for ids in want["ids"]]  # the prompts sent
        assert [c["ended_by_eos"] for c in body["chunks"]] == [True, False, True]
        assert body["frames"] == sum(len(ids) for ids in want["ids"])
        assert body["prefill_seconds"] == 0.25 and body["generate_seconds"] == 0.5


def test_audio_unsplit_and_default_seed(harness_binary, text_tool, tmp_path):
    want = tool_rows(text_tool, tmp_path, "chat", [dict(messages=tts(LONG)["messages"], capacity=4096, max_new_tokens=512)])[0]
    with running(harness_binary) as h:
        status, body = h.call("POST", "/speech", tts(LONG, split=False))
        assert status == 200 and body["split"] is False and body["seed"] == 0
        assert body["chunks"] == [{"frames": len(want["ids"]), "ended_by_eos": True}]


def test_interleaved_returns_the_decoded_text(harness_binary):
    messages = [{"role": "system", "content": "Respond with interleaved text and audio."},
                {"role": "user", "content": "What are three tips for staying focused while studying?"}]
    with running(harness_binary) as h:
        status, body = h.call("POST", "/speech", dict(messages=messages, mode="interleaved", max_new_tokens=256))
        assert status == 200 and body["mode"] == "interleaved" and body["split"] is False
        # the fake's slots are its phrase with <|audio_out|> between; the Service drops those and decodes the rest
        assert body["text"] == "Café \"quoted\" back\\slash\ttab\nline   \x7f end \U0001F642 done."
        status, body = h.call("POST", "/speech", dict(messages=messages, mode="interleaved", max_new_tokens=256, split=True))
        assert status == 400 and body == {"error": "invalid_request"}


@pytest.mark.parametrize("bad", [
    {"messages": [{"role": "user", "content": "Read this aloud for me please now."}], "mode": "audio"},  # no max_new_tokens
    dict(tts(LONG), extra=1), dict(tts(LONG), mode="transcribe"), dict(tts(LONG), mode=None),
    dict(tts(LONG), max_new_tokens=0), dict(tts(LONG), max_new_tokens=4096), dict(tts(LONG), max_new_tokens=1.5),
    dict(tts(LONG), seed=-1), dict(tts(LONG), seed=2147483648), dict(tts(LONG), seed=True), dict(tts(LONG), split="yes"),
    dict(tts(LONG), normalize="yes"), dict(tts(LONG), normalize=1), dict(tts(LONG), mode="interleaved", normalize=True),
    dict(tts(LONG), mode="text", normalize=True),
    dict(tts(LONG), messages="hi"), dict(tts(LONG), messages=[]),
    dict(tts(LONG), messages=[{"role": "user", "content": "Hi."}, {"role": "assistant", "content": "Hello."},
                              {"role": "user", "content": "Read this aloud for me please now."}]),
    dict(tts(LONG), messages=[{"role": "user", "content": "say <|audio_out|> now please, friend"}]),
    dict(tts(" ".join(["Seven words make up this short sentence."] * 513)), max_new_tokens=8),  # 513 chunks > 512
    # prompt tokens over 131072: a ~2,700-token system message repeated in each of 60 chunks
    dict(tts(" ".join(["Seven words make up this short sentence."] * 60)), max_new_tokens=8,
         messages=[{"role": "system", "content": " ".join(f"zq{i}x" for i in range(900))},
                   {"role": "user", "content": " ".join(["Seven words make up this short sentence."] * 60)}]),
])
def test_refusals_leave_the_server_serving(harness_binary, bad):
    with running(harness_binary) as h:
        status, body = h.call("POST", "/speech", bad)
        assert status == 400 and body == {"error": "invalid_request"}, (bad, status, body)
        status, body = h.call("POST", "/speech", tts(LONG))
        assert status == 200 and body["cancelled"] is False
        status, _ = h.call("POST", "/speech", tts(LONG), auth=False)
        assert status == 401


def test_text_sessions_are_unaffected(harness_binary):
    with running(harness_binary) as h:
        status, created = h.call("POST", "/sessions", {"capacity": 512})
        assert status == 200
        req = dict(session_id=created["session_id"], max_new_tokens=8,
                   messages=[{"role": "user", "content": "hello there"}])
        before = h.call("POST", "/generate", req)
        h.call(f"POST", f"/sessions/{created['session_id']}/reset", {})
        speech = h.call("POST", "/speech", tts(LONG))
        after = h.call("POST", "/generate", req)
        assert before[0] == after[0] == 200 and speech[0] == 200
        assert before[1]["generated_ids"] == after[1]["generated_ids"] and before[1]["text"] == after[1]["text"]


def test_shutdown_cancels_a_running_speech(harness_binary):
    with running(harness_binary, "--token-delay-ms", "400") as h:
        result = {}
        t = threading.Thread(target=lambda: result.update(r=h.call("POST", "/speech", tts(LONG * 3))))
        t.start()
        time.sleep(0.6)
        h.stop()
        t.join(30)
        status, body = result["r"]
        assert status == 200 and body == {"mode": "audio", "cancelled": True, "seed": 0, "split": True, "normalize": True}


def test_floats_are_written_as_python_does(tmp_path):
    # py_float (the only float writer in responses) against float.__repr__, which json.dumps uses
    import math
    import random
    import struct
    if shutil.which("g++") is None:
        pytest.skip("g++ required")
    src = tmp_path / "pyfloat.cpp"
    src.write_text('#include <lfm/lfm3_http.hpp>\n#include <cstdlib>\n#include <iostream>\nint main() { std::string t; '
                   'while (std::cin >> t) std::cout << lfm::http::py_float(std::strtod(t.c_str(), nullptr)) << "\\n"; }\n')
    subprocess.run(["g++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "native/include"), str(src),
                    "-pthread", "-o", str(tmp_path / "pyfloat")], check=True)
    rng = random.Random(1)
    vals = [0.25, 1.0, 1e-05, 1e-4, 0.0001234, 1e16, 1e15, 123456789012345.6, 1.5e16, 38.24, -2.5, 5e-324, -0.0,
            1.7976931348623157e308, 2.2250738585072014e-308, 0.1, 1e22, 9.999999999999999e15]
    vals += [rng.uniform(0, 1000) for _ in range(20000)] + [10 ** rng.uniform(-30, 30) for _ in range(20000)]
    vals += [v for v in (struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0] for _ in range(20000)) if math.isfinite(v)]
    out = subprocess.run([str(tmp_path / "pyfloat")], input="\n".join(v.hex() for v in vals), capture_output=True,
                         text=True, check=True).stdout.split()
    assert out == [repr(v) for v in vals]


def test_unsupported_speech_is_refused_without_failing_the_server(harness_binary):
    # a text-only checkpoint or a cuBLAS build: the engine would refuse op "speech", which fails the server closed, so
    # the Service refuses /speech itself and keeps serving text
    with running(harness_binary, "--speech", "off") as h:
        for _ in range(2):
            status, body = h.call("POST", "/speech", tts(LONG))
            assert status == 501 and body == {"error": "speech_unsupported"}
        status, created = h.call("POST", "/sessions", {"capacity": 512})
        assert status == 200
        status, _ = h.call("POST", "/generate", dict(session_id=created["session_id"], max_new_tokens=4,
                                                    messages=[{"role": "user", "content": "hello there"}]))
        assert status == 200


def test_engine_frame_too_large_is_a_400_not_a_failed_service(harness_binary):
    # the server red-team's reproduction: a ~900-word system message repeated in each of 1000 chunks made the engine
    # request exceed 4 MiB, which the Service noticed only inside the job and failed closed. Now the chunk and
    # prompt-token caps refuse it before any engine call (which also keeps every engine frame under ~0.9 MiB; the
    # frame-size check in encode_speech is a backstop), and the server keeps serving.
    system = " ".join(f"zq{i}x" for i in range(900))
    for user in (" ".join(["Seven words make up this short sentence."] * 1000),   # 1000 chunks
                 " ".join(["Seven words make up this short sentence."] * 100)):   # 100 chunks, ~270k prompt tokens
        with running(harness_binary) as h:
            status, body = h.call("POST", "/speech", dict(messages=[{"role": "system", "content": system},
                                                                   {"role": "user", "content": user}], mode="audio", max_new_tokens=1))
            assert status == 400 and body == {"error": "invalid_request"}, (status, body)
            status, body = h.call("POST", "/speech", tts(LONG))
            assert status == 200 and body["cancelled"] is False


def test_requests_the_old_static_budget_refused_now_run(harness_binary):
    # 20 sentences at 1000 frames each was over the first version's chunks x max_new_tokens cap; the engine now bounds
    # the work itself (step budget, then truncation)
    with running(harness_binary) as h:
        status, body = h.call("POST", "/speech", dict(tts(" ".join(["Short sentence here number one now."] * 20)), max_new_tokens=1000))
        assert status == 200 and len(body["chunks"]) == 20 and body["truncated"] is False


def test_a_truncated_reply_carries_only_the_chunks_run(harness_binary):
    with running(harness_binary, "--speech-truncate", "2") as h:
        status, body = h.call("POST", "/speech", tts(LONG))
        assert status == 200 and body["truncated"] is True
        assert [c["text"] for c in body["chunks"]] == chunked.sentences(LONG)[:2]


def test_speech_needs_the_engine_timeout_its_budget_was_sized_for(harness_binary):
    with running(harness_binary, "--engine-timeout", "120") as h:
        status, body = h.call("POST", "/speech", tts(LONG))
        assert status == 501 and body == {"error": "speech_unsupported"}
    with running(harness_binary, "--engine-timeout", "300") as h:
        assert h.call("POST", "/speech", tts(LONG))[0] == 200


NUMBERS = ("The meeting starts at 7:45 and ends by 9:00. Order number 389 cost $3.50, about 2.75% more than in 1984. "
           "Dr. Patel met Mr. Lee at St. Mary's on Nov. 9, 2026.")


def test_numbers_are_read_as_words_by_default(harness_binary, text_tool, tmp_path):
    """Audio mode normalizes the user's text before the split (lfm/speech_normalize.hpp); the prompts sent are the
    splitter's sentences of the normalized text, each rendered alone; normalize false sends the text as written."""
    spec = importlib.util.spec_from_file_location("speech_normalize", ROOT / "tools/speech_normalize.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    for normalize, text in ((True, reference.normalize(NUMBERS)), (False, NUMBERS)):
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
        want = tool_rows(text_tool, tmp_path, "speech", [dict(messages=messages, capacity=4096, max_new_tokens=512)])[0]
        with running(harness_binary) as h:
            body = tts(NUMBERS) if normalize else tts(NUMBERS, normalize=False)
            status, reply = h.call("POST", "/speech", body)
            assert status == 200 and reply["normalize"] is normalize, reply
            assert [c["text"] for c in reply["chunks"]] == want["texts"]
            assert [c["frames"] for c in reply["chunks"]] == [len(ids) for ids in want["ids"]]
    assert "seven forty-five" in reference.normalize(NUMBERS) and "Doctor Patel" in reference.normalize(NUMBERS)


# audio-assurance-v1 rule 10: the normalizer is American English only; by default it runs only when the system prompt
# names American or British English, or there is no system prompt. An explicit normalize true with another language
# is refused.
@pytest.mark.parametrize("system, default", [
    ("Perform TTS in Spanish using a female voice.", False),
    ("Perform TTS in German using a male voice.", False),
    ("Perform TTS in French using a female voice.", False),
    ("Perform TTS in Japanese using a male voice.", False),
    ("Perform TTS in American English using a female voice.", True),
    ("Perform TTS in British English using a male voice.", True),
    (None, True),
])
def test_normalization_follows_the_language(harness_binary, system, default):
    text = {"Spanish": "Tengo 3 gatos y 12 perros.", "German": "Ich habe 3 Katzen und 12 Hunde.",
            "French": "J'ai 3 chats et 12 chiens.", "Japanese": "猫が3匹います。"}
    lang = next((k for k in text if system and k in system), None)
    user = text[lang] if lang else NUMBERS
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    with running(harness_binary) as h:
        status, reply = h.call("POST", "/speech", dict(messages=messages, mode="audio", max_new_tokens=512))
        assert status == 200 and reply["normalize"] is default, (system, status, reply)
        if not default:
            assert "".join(c["text"] for c in reply["chunks"]).replace(" ", "") == user.replace(" ", ""), reply
        status, reply = h.call("POST", "/speech", dict(messages=messages, mode="audio", max_new_tokens=512, normalize=True))
        if default:
            assert status == 200 and reply["normalize"] is True
        else:
            assert status == 400 and reply == {"error": "invalid_request"}, (system, status, reply)


def test_speech_text_has_a_byte_limit(harness_binary):
    """audio-assurance-v1 red-team: a 2 MB text took 60 GB in the split; /speech text is held to 64 KiB, checked first."""
    with running(harness_binary) as h:
        for body in (tts("a" * (64 * 1024 + 1)), tts("1," * 500_000), tts("Hi there now friend.", normalize=False) | dict(
                messages=[{"role": "system", "content": "x" * (64 * 1024)}, {"role": "user", "content": "Hi there."}])):
            t0 = time.monotonic()
            status, reply = h.call("POST", "/speech", body, timeout=30)
            assert status == 400 and reply == {"error": "invalid_request"} and time.monotonic() - t0 < 5
        status, _ = h.call("POST", "/speech", tts(LONG))
        assert status == 200


@pytest.mark.parametrize("text", ["1," * 30000, "1" * 60000 + "st", "$" + "9" * 60000, "1,000 " * 10000,
                                  "aa bb cc dd ee ff. " * 3000],
                         ids=["commas", "ordinal", "money", "grouped", "sentences"])
def test_hostile_text_is_answered_quickly(harness_binary, text):
    """Texts that were quadratic in the normalizer or the split, or overflowed its stack (audio-assurance-v1 red-team),
    just under the byte limit: an answer (200 or 400) in seconds, and the server serves on."""
    with running(harness_binary) as h:
        t0 = time.monotonic()
        status, _ = h.call("POST", "/speech", tts(text) | dict(max_new_tokens=8), timeout=120)
        assert status in (200, 400) and time.monotonic() - t0 < 30, (status, time.monotonic() - t0)
        status, _ = h.call("POST", "/speech", tts(LONG))
        assert status == 200
