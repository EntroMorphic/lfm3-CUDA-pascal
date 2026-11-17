#!/usr/bin/env python3
"""A4 reference: HF's own generation control flow (`Lfm3OmniForConditionalGeneration._generate`) on forced inputs.

The model is instantiated on the meta device (no weights). Only the neural calls are stubbed: the prefill and
decode forwards, lm_head, the text sampler `_sample` and `sample_audio_frame`, which return the scenario's forced
tokens and frames in order. Everything else, i.e. modes, markers, cadence, EOS rewrite, suppression, stopping and
the speech repetition history, is HF's code. Records per step what was sampled, what was fed back (token embedding
or frame codes), whether a decode ran, each text step's suppression set and each speech step's repetition history
(length and history_hash); then the emitted slots and frames. Scenarios may replay frames in the prompt
(`audio_out_codes`) and supply the cache (`past_key_values`).
Scenarios: synthetic ones covering every branch, and the six traced cases (their sampled tokens and raw frames).
Usage (Thor, lfm3-omni-dev): lfm3_mode_reference.py MODEL_DIR TRACE_DIR OUT_JSON
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

IM_END, TEXT_END, AUDIO_START, MIXED, AUDIO_OUT = 124900, 125014, 125012, 125015, 125032


def history_hash(frames):
    """Polynomial hash of a list of frames (12 codes each), in order; native/tests/audio_modes_oracle.cpp's twin."""
    h, m = 0, (1 << 61) - 1
    for f in frames:
        for c in f:
            h = (h * 1000003 + int(c) + 1) % m
    return h


def frames_of(rng, n, eos_at=None, eos_codebooks=(3,)):
    f = rng.integers(0, 2016, (n, 12)).tolist()
    if eos_at is not None:
        for c in eos_codebooks:
            f[eos_at][c] = 2016
    return f


def synthetic():
    rng = np.random.default_rng(0)
    t = lambda n, base=1000: list(range(base, base + n))  # noqa: E731 ordinary tokens
    S = []
    add = lambda name, mode, last, tokens, frames, max_new=200, **flags: S.append(dict(  # noqa: E731
        name=name, mode=mode, last_prompt_token=last, tokens=tokens, frames=frames, max_new_tokens=max_new,
        suppress_mixed=flags.get("suppress_mixed", False), suppress_audio_out=flags.get("suppress_audio_out", False),
        return_audio_eos=flags.get("return_audio_eos", False), caller_cache=flags.get("caller_cache", False),
        prompt_frames=flags.get("prompt_frames")))
    add("text-only", "text", 7, t(3) + [IM_END], [])
    add("text-to-audio", None, 7, [1000, AUDIO_START] + [1001, IM_END], frames_of(rng, 3, eos_at=2))
    add("text-to-interleaved", None, 7,
        [MIXED] + t(6) + t(6, 2000) + t(3, 3000) + [TEXT_END, IM_END], frames_of(rng, 12 + 12 + 6, eos_at=29))
    add("interleaved-eos-mid-block", "interleaved", 7, t(6) + [1100, AUDIO_START, IM_END],
        frames_of(rng, 5, eos_at=4) + frames_of(rng, 3, eos_at=2, eos_codebooks=(0, 11)))
    add("interleaved-im-end-in-text", "interleaved", 7, t(3) + [IM_END], [])
    add("text-end-outside-block", None, 7, t(2) + [TEXT_END], [])
    add("audio-max-new-cut", "audio", 7, [], frames_of(rng, 10), max_new=5)
    add("audio-prompt-has-marker-return-eos", "audio", AUDIO_START, [IM_END], frames_of(rng, 4, eos_at=3),
        return_audio_eos=True)
    add("markers-inside-block-are-text", "interleaved", 7, [1, AUDIO_START, MIXED, 4, 5, 6] + [TEXT_END, IM_END],
        frames_of(rng, 12 + 2, eos_at=13))
    add("suppress-flags", None, 7, [1000, IM_END], [], suppress_mixed=True, suppress_audio_out=True)
    add("audio-eos-first-frame", "audio", 7, [1000, IM_END], frames_of(rng, 1, eos_at=0))
    add("interleaved-prompt-has-marker", "interleaved", MIXED, t(6) + [TEXT_END, IM_END], frames_of(rng, 13, eos_at=12))
    add("two-blocks", None, 7, [MIXED] + t(6) + [MIXED] + t(6, 2000) + t(1, 3000) + [TEXT_END, IM_END],
        frames_of(rng, 3, eos_at=2) + frames_of(rng, 12 + 4, eos_at=15))
    add("max-new-cut-in-interleaved", "interleaved", 7, t(6) + t(6, 2000), frames_of(rng, 12), max_new=15)
    add("text-max-new-cut", "text", 7, t(10), [], max_new=4)
    # replayed speech in the prompt: a terminal EOS frame resumes text unless a mode is requested
    add("replay-eos-frame-resumes-text", None, AUDIO_OUT, [1000, IM_END], [],
        prompt_frames=frames_of(rng, 2, eos_at=1))
    add("replay-speech-frame-continues-audio", None, AUDIO_OUT, [IM_END], frames_of(rng, 3, eos_at=2),
        prompt_frames=frames_of(rng, 2))
    add("replay-eos-frame-requested-audio", "audio", AUDIO_OUT, [IM_END], frames_of(rng, 2, eos_at=1),
        prompt_frames=frames_of(rng, 1, eos_at=0, eos_codebooks=(0, 5)))
    # the caller's cache: the last step is decoded too
    add("caller-cache-text", "text", 7, t(2) + [IM_END], [], caller_cache=True)
    add("caller-cache-audio-max-cut", "audio", 7, [], frames_of(rng, 6), max_new=4, caller_cache=True)
    add("caller-cache-interleaved", None, 7, [MIXED] + t(6) + [TEXT_END, IM_END], frames_of(rng, 13, eos_at=12),
        caller_cache=True)
    # the history spans audio -> text -> audio and outlives the 64-frame window
    add("history-across-switch-and-window", None, 7, [AUDIO_START, 1000, AUDIO_START, IM_END],
        frames_of(rng, 40, eos_at=39) + frames_of(rng, 70, eos_at=69))
    return S


def traced(trace_dir):
    S = []
    for case in sorted(p.name for p in Path(trace_dir).iterdir() if (p / "summary.json").exists()):
        d = Path(trace_dir) / case
        s = json.loads((d / "summary.json").read_text())
        seq = torch.load(d / "sequences.pt")
        f = torch.load(d / "frames.pt")
        slots = seq["slots"].tolist()
        last = int(seq["prompt_ids"][-1])
        # the trace counted from before the marker generate appends: its first slot is that marker
        marker = (MIXED if s["mode"] == "interleaved" and last != MIXED else
                  AUDIO_START if s["mode"] == "audio" and last not in (AUDIO_START, AUDIO_OUT) else None)
        if marker is not None:
            assert slots[0] == marker, case
            slots = slots[1:]
        S.append(dict(name=f"trace-{case}", mode=s["mode"], last_prompt_token=last,
                      tokens=[x for x in slots if x != AUDIO_OUT],
                      frames=f["raw_codes"].reshape(-1, 12).tolist(), max_new_tokens=s["max_new_tokens"],
                      suppress_mixed=False, suppress_audio_out=False, return_audio_eos=False, caller_cache=False,
                      prompt_frames=None,
                      expected_slots=slots, expected_frames=seq["audio_codes"][0].T.tolist()))
    return S


def run(model, sc):
    tokens, frames = list(sc["tokens"]), [torch.tensor(x) for x in sc["frames"]]
    steps, prefill = [], {}
    hidden = torch.zeros(1, 1, 2048)

    def forward(input_ids=None, **kw):
        prefill["ids"] = input_ids[0].tolist()
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=None)

    def decode(**kw):
        steps[-1]["decode"] = True
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=None)

    def embed(token):
        steps[-1]["fed"] = int(token.item())
        return torch.zeros(1, 2048)

    def audio_embeds(codes):
        steps[-1]["fed"] = codes[0].tolist()
        return torch.zeros(1, 2048)

    def sample(logits, temperature, top_k, penalty=None, history=None, suppress=(), **kw):
        if not tokens:
            raise RuntimeError("forced tokens exhausted")
        steps.append(dict(kind="T", sampled=tokens[0], suppress=sorted(suppress), decode=False))
        return torch.tensor(tokens.pop(0))

    def sample_frame(hidden_state, temperature, top_k, penalty, history, *a, **kw):
        if not frames:
            raise RuntimeError("forced frames exhausted")
        h = [] if history is None else history.tolist()
        steps.append(dict(kind="A", sampled=frames[0].tolist(), decode=False, history=[len(h), history_hash(h)]))
        return frames.pop(0)

    stub = SimpleNamespace(language_model=decode, get_audio_out_embeds=audio_embeds,
                           get_input_embeddings=lambda: embed)
    object.__setattr__(model, "model", stub)
    object.__setattr__(model, "lm_head", lambda h: torch.zeros(1))
    object.__setattr__(model, "get_input_embeddings", lambda: embed)
    object.__setattr__(model, "_sample", sample)
    object.__setattr__(model, "sample_audio_frame", sample_frame)
    stub.__call__ = forward
    object.__setattr__(model, "model", type("Stub", (), {"__call__": staticmethod(forward), "language_model": staticmethod(decode),
                                                         "get_audio_out_embeds": staticmethod(audio_embeds)})())
    replay = sc.get("prompt_frames")
    # the stubbed prefill skips HF's check_speech_placeholders, which refuses <|audio_out|> without its codes
    assert sc["last_prompt_token"] != AUDIO_OUT or replay, sc["name"]
    prompt = [1, 2, sc["last_prompt_token"]]
    if replay:  # one <|audio_out|> per replayed frame, the last at the end
        assert len(replay) <= 2 and sc["last_prompt_token"] == AUDIO_OUT
        prompt = [1] * (3 - len(replay)) + [AUDIO_OUT] * len(replay)
    extra = {}
    if replay:
        extra["audio_out_codes"] = torch.tensor(replay)
    if sc.get("caller_cache"):
        extra["past_key_values"] = SimpleNamespace(get_seq_length=lambda: 0)
    out = model._generate(torch.tensor([prompt]), generation_mode=sc["mode"], max_new_tokens=sc["max_new_tokens"],
                          do_sample=False, suppress_mixed=sc["suppress_mixed"],
                          suppress_audio_out=sc["suppress_audio_out"], return_audio_eos=sc["return_audio_eos"],
                          return_dict_in_generate=True, **extra)
    n = len(prefill["ids"])
    slots = out.sequences[0, n:int(out.sequence_lengths[0])].tolist()
    frames_out = out.audio_codes[0, :, :int(out.audio_lengths[0])].T.tolist()
    return dict(marker=prefill["ids"][-1] if n > len(prompt) else -1, steps=steps, slots=slots, frames=frames_out,
                tokens_left=len(tokens), frames_left=len(frames))


def main(model_dir, trace_dir, out_json):
    from transformers import AutoConfig, GenerationConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    cls = get_class_from_dynamic_module("modeling_lfm3_omni.Lfm3OmniForConditionalGeneration", model_dir)
    with torch.device("meta"):
        model = cls(config)
    model.generation_config = GenerationConfig.from_pretrained(model_dir)
    scenarios = synthetic() + traced(trace_dir)
    for sc in scenarios:
        sc["reference"] = run(model, sc)
        r = sc["reference"]
        if "expected_slots" in sc:  # the harness must reproduce what the real run emitted
            assert r["slots"] == sc.pop("expected_slots") and r["frames"] == sc.pop("expected_frames"), sc["name"]
            sc["reproduces_trace"] = True
        print(sc["name"], "steps", len(r["steps"]), "slots", len(r["slots"]), "frames", len(r["frames"]),
              "marker", r["marker"], "left", r["tokens_left"], r["frames_left"])
    Path(out_json).write_text(json.dumps(dict(scenarios=scenarios)) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
