"""Host tests for the audio-input parsers and front end (#3): lfm/wav.hpp, lfm/base64.hpp, lfm/audio_features.hpp,
through native/tests/audio_input_tool.cpp, built plain and with ASan+UBSan.

- WAV: valid 16-bit PCM, 32-bit float and WAVE_FORMAT_EXTENSIBLE files read exactly as soundfile reads them
  (s / 32768; floats as stored); a streamed data size (0xFFFFFFFF) runs to the end of the file; odd-sized chunks are
  skipped with their padding; every malformed or unsupported file is refused with a message; 3,000 random mutations
  of a valid file never crash.
- base64: equal to Python's on every length 0..64; refuses bad characters, lengths, misplaced padding, whitespace and
  nonzero bits under the padding (one accepted encoding per byte string).
- features: the frame and token counts at the window/hop edges, and the log-mel values against an FP64 reference built
  from transformers' own mel_filter_bank (the GPU round compares against HF's FP64 features too: validation/audio-in-v1).
"""
import base64
import json
import random
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {"plain": ("g++", ["-O2"]), "asan": ("g++", ["-O1", "-g", "-fsanitize=address,undefined", "-fno-sanitize-recover=all"])}


@pytest.fixture(scope="module", params=sorted(VARIANTS))
def tool(tmp_path_factory, request):
    compiler, flags = VARIANTS[request.param]
    if not shutil.which(compiler):
        pytest.skip(f"{compiler} required")
    binary = tmp_path_factory.mktemp("audio-input") / f"audio-input-tool-{request.param}"
    subprocess.run([compiler, "-std=c++20", "-Wall", "-Wextra", "-Werror", *flags, "-I", str(ROOT / "native/include"),
                    str(ROOT / "native/tests/audio_input_tool.cpp"), "-o", str(binary)], check=True)
    return binary


def run(tool, mode, paths):
    r = subprocess.run([str(tool), mode, *map(str, paths)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    return [json.loads(line) for line in r.stdout.splitlines()]


def riff(chunks):
    body = b"WAVE" + b"".join(cid + struct.pack("<I", len(data) if size is None else size) + data + (b"\0" if len(data) % 2 else b"")
                              for cid, data, size in chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def fmt(fmt_tag=1, channels=1, rate=16_000, bits=16, align=None, byte_rate=None, extensible_sub=None):
    align = channels * bits // 8 if align is None else align
    byte_rate = rate * align if byte_rate is None else byte_rate
    data = struct.pack("<HHIIHH", fmt_tag, channels, rate, byte_rate, align, bits)
    if extensible_sub is not None:
        data += struct.pack("<HHI", 22, bits, 4) + struct.pack("<H", extensible_sub) + b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
    return (b"fmt ", data, None)


def pcm16(n, seed=0):
    return np.random.default_rng(seed).integers(-32768, 32768, n).astype("<i2")


def write(tmp, name, data):
    p = tmp / name
    p.write_bytes(data)
    return p


def test_wav_valid_files_read_exactly(tool, tmp_path):
    s = pcm16(1001)
    f = np.random.default_rng(1).uniform(-1, 1, 777).astype("<f4")
    files = {
        "pcm16": riff([fmt(), (b"data", s.tobytes(), None)]),
        "float32": riff([fmt(3, bits=32), (b"data", f.tobytes(), None)]),
        "extensible": riff([fmt(0xFFFE, extensible_sub=1), (b"data", s.tobytes(), None)]),
        "odd-chunk-first": riff([fmt(), (b"LIST", b"abc", None), (b"data", s.tobytes(), None)]),
        "streamed": riff([fmt(), (b"data", s.tobytes() + b"\x01", 0xFFFFFFFF)])[:-1],  # trailing half sample dropped
        "rate-24k": riff([fmt(rate=24_000), (b"data", s.tobytes(), None)]),
    }
    paths = [write(tmp_path, k + ".wav", v) for k, v in files.items()]
    got = dict(zip(files, run(tool, "wav", paths)))
    want16 = float(np.sum(s.astype(np.float32) / np.float32(32768), dtype=np.float64))
    for k in ("pcm16", "extensible", "odd-chunk-first", "streamed"):
        assert got[k]["ok"] and got[k]["samples"] == 1001 and got[k]["rate"] == 16_000, (k, got[k])
        assert abs(got[k]["sum"] - want16) < 1e-6 * max(1, abs(want16)), (k, got[k], want16)
    assert got["float32"]["ok"] and got["float32"]["samples"] == 777
    assert abs(got["float32"]["sum"] - float(np.sum(f, dtype=np.float64))) < 1e-5
    assert got["rate-24k"]["ok"] and got["rate-24k"]["rate"] == 24_000  # the rate is the caller's to check


def test_wav_malformed_files_are_refused(tool, tmp_path):
    s = pcm16(100).tobytes()
    nan = np.array([0.0, np.nan], "<f4").tobytes()
    files = {
        "empty-file": (b"", "not a RIFF/WAVE"),
        "riff-only": (b"RIFF\x04\x00\x00\x00WAVE", "no data chunk"),
        "not-riff": (b"RIFX" + b"\0" * 40, "not a RIFF/WAVE"),
        "no-fmt": (riff([(b"data", s, None)]), "data before fmt"),
        "truncated-data": (riff([fmt(), (b"data", s, None)])[:-10], "truncated data chunk"),
        "huge-chunk": (riff([fmt(), (b"LIST", b"", 0x7FFFFFF0), (b"data", s, None)]), "truncated LIST chunk"),
        "short-fmt": (riff([(b"fmt ", b"\x01\x00\x01\x00", None), (b"data", s, None)]), "short fmt chunk"),
        "short-extensible": (riff([(b"fmt ", fmt(0xFFFE)[1], None), (b"data", s, None)]), "short extensible"),
        "stereo": (riff([fmt(channels=2), (b"data", s, None)]), "expected mono"),
        "pcm8": (riff([fmt(bits=8), (b"data", s, None)]), "16-bit PCM or 32-bit float"),
        "pcm24": (riff([fmt(bits=24), (b"data", s[:99], None)]), "16-bit PCM or 32-bit float"),
        "float64": (riff([fmt(3, bits=64), (b"data", s[:96], None)]), "16-bit PCM or 32-bit float"),
        "bad-align": (riff([fmt(align=4), (b"data", s, None)]), "block align or byte rate"),
        "bad-byte-rate": (riff([fmt(byte_rate=1), (b"data", s, None)]), "block align or byte rate"),
        "zero-rate": (riff([fmt(rate=0), (b"data", s, None)]), "invalid sample rate"),
        "odd-16bit": (riff([fmt(), (b"data", s[:99], None)]), "odd 16-bit data size"),
        "float-nan": (riff([fmt(3, bits=32), (b"data", nan, None)]), "non-finite sample"),
        "float-ragged": (riff([fmt(3, bits=32), (b"data", s[:98], None)]), "not a multiple of 4"),
    }
    paths = [write(tmp_path, k + ".wav", v[0]) for k, v in files.items()]
    for (k, (_, words)), r in zip(files.items(), run(tool, "wav", paths)):
        assert not r["ok"] and words in r["error"], (k, r)


def test_wav_random_mutations_never_crash(tool, tmp_path):
    rng = random.Random(7)
    base = bytearray(riff([fmt(), (b"LIST", b"xyz", None), (b"data", pcm16(64).tobytes(), None)]))
    paths = []
    for i in range(3000):
        b = bytearray(base)
        for _ in range(rng.randint(1, 6)):
            op = rng.random()
            if op < 0.5:
                b[rng.randrange(len(b))] = rng.randrange(256)
            elif op < 0.75:
                del b[rng.randrange(len(b)):][: rng.randint(1, 20)]
            else:
                at = rng.randrange(len(b))
                b[at:at] = bytes(rng.randrange(256) for _ in range(rng.randint(1, 8)))
        paths.append(write(tmp_path, f"m{i}.wav", bytes(b)))
    results = run(tool, "wav", paths)  # an ASan or UBSan report would fail the run
    assert len(results) == 3000 and all("ok" in r for r in results)
    assert any(r["ok"] for r in results) and any(not r["ok"] for r in results)


def test_base64_matches_python_and_is_strict(tool, tmp_path):
    rng = np.random.default_rng(3)
    blobs = [rng.integers(0, 256, n).astype(np.uint8).tobytes() for n in range(65)]
    paths = [write(tmp_path, f"ok{i}.txt", base64.b64encode(b)) for i, b in enumerate(blobs)]
    for b, r in zip(blobs, run(tool, "b64", paths)):
        assert r["ok"] and r["hex"] == b.hex(), (b.hex(), r)
    bad = ["QUJD!", "QUJ", "QU=D", "Q===", "====", "QUJD\n", " QUJD", "QR==", "QUF=", "QUJDQQ==QUJD", "QUJD=", "-_8=", "Q\0==",
           "QUI=QUI="]
    paths = [write(tmp_path, f"bad{i}.txt", s.encode()) for i, s in enumerate(bad)]
    for s, r in zip(bad, run(tool, "b64", paths)):
        assert not r["ok"] and "invalid base64" in r["error"], (s, r)


def reference_log_mel(x):
    """Lfm3OmniFeatureExtractor's features in FP64 throughout (window, FFT, the mel filters as float64)."""
    from transformers.audio_utils import mel_filter_bank
    filters = mel_filter_bank(num_frequency_bins=201, num_mel_filters=128, min_frequency=0.0, max_frequency=8000,
                              sampling_rate=16_000, norm="slaney", mel_scale="slaney")
    x = np.asarray(x, np.float64)
    padded = 400 if len(x) < 400 else len(x) + (-(len(x) - 400) % 160)
    x = np.pad(x, (0, padded - len(x)))
    frames = 1 + (padded - 400) // 160
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 400)
    spec = np.fft.rfft(np.stack([x[t * 160: t * 160 + 400] * window for t in range(frames)]), axis=-1)
    mel = (np.abs(spec) ** 2) @ filters
    return ((np.maximum(np.log10(np.maximum(mel, 1e-300)), -12.0) + 4.0) / 4.0).T  # [128][frames]


@pytest.mark.parametrize("samples", [1, 399, 400, 401, 559, 560, 561, 1680, 1681, 16_000, 52_321])
def test_features_match_fp64_reference(tool, tmp_path, samples):
    pytest.importorskip("transformers.audio_utils")
    rng = np.random.default_rng(samples)
    t = np.arange(samples) / 16_000
    s = np.clip(0.3 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(samples), -1, 1)
    q = (s * 32767).astype("<i2")
    wav = write(tmp_path, "clip.wav", riff([fmt(), (b"data", q.tobytes(), None)]))
    out = tmp_path / "mel.f32"
    r = subprocess.run([str(tool), "mel", str(wav), str(out)], capture_output=True, text=True, check=True)
    info = json.loads(r.stdout)
    ref = reference_log_mel(q.astype(np.float32) / 32768)
    assert info["frames"] == ref.shape[1] and info["tokens"] == -(-ref.shape[1] // 8)
    got = np.fromfile(out, np.float32).reshape(128, -1)
    # float32 window and filter values in the engine; the rest in double (validation/audio-in-v1: within 1.2e-4 of
    # FP64 on real speech, closer than HF's float32 arithmetic)
    assert np.abs(got - ref).max() < 2e-4, np.abs(got - ref).max()


def test_base64_encode_matches_python(tmp_path):
    """lfm::base64::encode (the worker's WAV and streamed-audio encoder) against Python's, and through the strict
    decoder back to the bytes."""
    import random
    if shutil.which("g++") is None:
        pytest.skip("g++ required")
    src = tmp_path / "enc.cpp"
    src.write_text('#include <lfm/base64.hpp>\n#include <fstream>\n#include <iostream>\n#include <iterator>\n'
                   'int main(int, char** argv) { std::ifstream f(argv[1], std::ios::binary); '
                   'std::string b((std::istreambuf_iterator<char>(f)), {}); const auto e = lfm::base64::encode(b); '
                   'if (lfm::base64::decode(e) != b) return 2; std::cout << e; }\n')
    subprocess.run(["g++", "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined",
                    "-fno-sanitize-recover=all", "-I", str(ROOT / "native/include"), str(src), "-o", str(tmp_path / "enc")],
                   check=True)
    rng = random.Random(5)
    blobs = [bytes(rng.getrandbits(8) for _ in range(n)) for n in range(301)]
    blobs += [b"\0" * 7, b"\xff" * 8, bytes(rng.getrandbits(8) for _ in range(1 << 20))]
    for i, b in enumerate(blobs):
        p = tmp_path / "in.bin"
        p.write_bytes(b)
        out = subprocess.run([str(tmp_path / "enc"), str(p)], capture_output=True, check=True).stdout
        assert out == base64.b64encode(b), (i, len(b))
