"""A4 (journal/lfm3-tts-quality-bar.md): the native mode machine (native/include/lfm/audio_modes.hpp) against HF's
own generation control flow, bit for bit. tests/data_mode_reference.json is tools/lfm3_mode_reference.py's output:
HF's `_generate` on a meta-device model with only the neural calls stubbed, run on forced tokens and frames. It holds
22 synthetic scenarios covering every branch (replayed prompt frames and a caller cache
included), and the six traced cases, whose emitted slots and frames the harness
reproduced exactly. For every scenario: the marker, each step (what was sampled, what was fed back, whether a decode
ran, the text step's suppression set, the speech step's repetition history), the emitted slots and frames, and the unconsumed inputs must all be equal.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# tools/lfm3_audio_control_report.py points this at a copy of the headers with a planted bug
INCLUDE = os.environ.get("LFM_AUDIO_INCLUDE", str(ROOT / "native/include"))
SCENARIOS = json.loads(Path(__file__).with_name("data_mode_reference.json").read_text())["scenarios"]


@pytest.fixture(scope="module")
def oracle(tmp_path_factory):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ required")
    exe = tmp_path_factory.mktemp("audio_modes") / "oracle"
    subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-I", INCLUDE,
                    str(ROOT / "native/tests/audio_modes_oracle.cpp"), "-o", str(exe)], check=True)
    return exe


def replayed(sc):
    # the prompt's last replayed frame, as HF's audio_out_codes[-1]; native decides what it implies
    frames = sc.get("prompt_frames") or []
    return f"1 {' '.join(map(str, frames[-1]))}" if frames else "0"


def native(oracle, sc):
    lines = [f"{sc['mode'] or 'none'} {int(sc['suppress_mixed'])} {int(sc['suppress_audio_out'])} "
             f"{int(sc['return_audio_eos'])} {sc['max_new_tokens']} {sc['last_prompt_token']} "
             f"{int(sc['caller_cache'])} " + replayed(sc),
             f"{len(sc['tokens'])} " + " ".join(map(str, sc["tokens"])),
             f"{len(sc['frames'])} " + " ".join(str(c) for f in sc["frames"] for c in f)]
    out = subprocess.run([str(oracle)], input="\n".join(lines) + "\n", capture_output=True, text=True, check=True)
    return out.stdout.splitlines()


def expected(ref):
    csv = lambda xs: ",".join(map(str, xs))  # noqa: E731
    lines = [f"marker {ref['marker']}"]
    for s in ref["steps"]:
        if s["kind"] == "T":
            lines.append(f"T {s['sampled']} {int(s['decode'])} {s['fed']} {csv(s['suppress'])}")
        else:
            lines.append(f"A {csv(s['sampled'])} {int(s['decode'])} {csv(s['fed'])} H {s['history'][0]} {s['history'][1]}")
    lines += [f"slots {csv(ref['slots'])}", f"frames {len(ref['frames'])}", *map(csv, ref["frames"]),
              f"left {ref['tokens_left']} {ref['frames_left']}"]
    return lines


@pytest.mark.parametrize("sc", SCENARIOS, ids=[s["name"] for s in SCENARIOS])
def test_native_equals_hf(oracle, sc):
    assert native(oracle, sc) == expected(sc["reference"])


def test_reference_covers_the_branches():
    names = {s["name"] for s in SCENARIOS}
    assert len(names) == 28 and sum(n.startswith("trace-") for n in names) == 6
    assert all(s.get("reproduces_trace") for s in SCENARIOS if s["name"].startswith("trace-"))
    kinds = [st["kind"] for s in SCENARIOS for st in s["reference"]["steps"]]
    assert "T" in kinds and "A" in kinds
    by = {s["name"]: s for s in SCENARIOS}
    # the branches the red-team added: replayed EOS frame, caller cache, a full history window
    assert by["replay-eos-frame-resumes-text"]["reference"]["steps"][0]["kind"] == "T"
    assert by["replay-speech-frame-continues-audio"]["reference"]["steps"][0]["kind"] == "A"
    assert by["caller-cache-text"]["reference"]["steps"][-1]["decode"]
    assert max(st["history"][0] for s in SCENARIOS for st in s["reference"]["steps"] if st["kind"] == "A") == 64


def test_max_new_tokens_zero_is_refused(oracle):
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run([str(oracle)], input="audio 0 0 0 0 7 0 0\n0\n0\n", capture_output=True, text=True, check=True)


def test_eos_frame_is_ignored_unless_the_prompt_ends_on_audio_out(oracle):
    # HF only looks at the frame when the prompt's last token is <|audio_out|>: ending on <|audio_start|>, it speaks
    eos = " ".join(["2016"] * 12)
    out = subprocess.run([str(oracle)], input=f"none 0 0 0 1 125012 0 1 {eos}\n0\n1 " + " ".join(["5"] * 12) + "\n",
                         capture_output=True, text=True, check=True).stdout.split("\n")
    assert out[1].startswith("A ")
    out = subprocess.run([str(oracle)], input=f"none 0 0 0 1 125032 0 1 {eos}\n1 1000\n0\n", capture_output=True,
                         text=True, check=True).stdout.split("\n")
    assert out[1].startswith("T ")   # resumes text


def test_audio_out_without_its_frame_is_refused(oracle):
    # HF's check_speech_placeholders: <|audio_out|> placeholders require audio_out_codes
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run([str(oracle)], input="none 0 0 0 1 125032 0 0\n0\n1 " + " ".join(["5"] * 12) + "\n",
                       capture_output=True, text=True, check=True)
