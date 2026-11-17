#!/usr/bin/env python3
"""#2 step 4, long-form TTS by sentences (the fix the sliding-window finding points to), prototyped in HF.

Why: the backbone's five sliding-attention layers (window 256) lose sight of the prompt text the model is reading
about 180 frames into an utterance (validation/speech-long-form-v1); from there only the six full-attention layers
keep the place, and long utterances skip and loop. Utterances under ~50 words never reach that point.

The text is split by sentences() (validation/speech-split-v1: sentence ends with a title and abbreviation guard, short
parts joined, chunks capped at CAP_WORDS words at clause marks; the native splitter, lfm/speech_text.hpp, equals it).
  fresh:    each chunk is its own request (system + user chunk), as the single-shot TTS request is built.
  continue: one conversation; each chunk is a new user turn after the previous turn exactly as generated
            (<|audio_start|>, its frames with the end-of-audio frame, <|im_end|>), so the earlier speech is in
            context (a whole prefill per chunk: simple, not fast).
Every chunk: torch.manual_seed(seed * 1000 + chunk), generate(generation_mode="audio", do_sample=False,
audio_temperature=0.4, audio_repetition_penalty=1.2). The accepted frames of all chunks are decoded by the codec in
one pass. Writes OUT_DIR/ENGINE-MODE/CASE-sSEED/{summary.json, response.wav, response.txt} (tools/lfm3_asr.py's
layout; the summary's conversation holds the whole text as one user turn, which ASR scores against).
Usage (lfm3-omni-dev): lfm3_speech_chunked_hf.py MODEL_DIR PROMPTS.json OUT_DIR fresh|continue|requests [bfloat16|float32]
  requests: write OUT_DIR/requests/CASE-sSEED.req.json for lfm3-speech ("chunks": each sentence's fresh prompt ids;
  the native engine seeds chunk k with SEED * 1000 + k and runs the codec stream on across chunks); no model loaded.
"""
import json
import re
import sys
import time
from pathlib import Path


MIN_WORDS = 6
CAP_WORDS = 49  # validation/speech-split-v1: phase A's longest safe single chunk; phase C (Tripp): comma cuts, few seams

# Candidate rules for the splitter's open gap (no length cap), prototyped here for the split experiment (validation/
# speech-split-v1). LFM3_SPLIT_RULE selects one: "sentences" (the default, the rule the experiment chose), or
# "sentences-v1" (the first rule) and the candidates below, built on sentences_v1. LFM3_SPLIT_CAP is their cap in words.
#   cap-comma: sentences_v1(), then a chunk over CAP words is cut at its last clause mark (, ; : or a dash) at or before
#              word CAP that leaves both sides MIN_WORDS long; with none, at the even cut below; repeated.
#   cap-hard:  sentences_v1(), then a chunk over CAP words is cut into the fewest equal pieces of at most CAP words.
#   full:      paragraphs (blank lines) first; sentence ends also after a closing quote or bracket; titles (Mr. Dr. ...)
#              never end a sentence, other abbreviations (a.m. U.S. Jan. ...) only before a capital; then cap-comma.
# the experiment's lists and cap, frozen as they ran (the default's own are below, refined in the red-team)
X_TITLES = {"mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "jr.", "sr.", "mt.", "no."}
X_ABBREVIATIONS = {"a.m.", "p.m.", "u.s.", "u.k.", "e.g.", "i.e.", "etc.", "vs.", "jan.", "feb.", "mar.", "apr.",
                   "jun.", "jul.", "aug.", "sep.", "sept.", "oct.", "nov.", "dec."}

# The default rule's lists. Titles never end a sentence; an abbreviation (or an ellipsis "...") ends one only before a
# word that starts with an ASCII capital, after any opening quote or bracket. Red-team 2026-11-09: no. jr. sr. moved
# from titles (they often end sentences: "told him no."), vs. moved to titles (it essentially never does).
TITLES = {"mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "mt.", "vs."}
ABBREVIATIONS = {"a.m.", "p.m.", "u.s.", "u.k.", "e.g.", "i.e.", "etc.", "no.", "jr.", "sr.", "jan.", "feb.", "mar.",
                 "apr.", "jun.", "jul.", "aug.", "sep.", "sept.", "oct.", "nov.", "dec."}
OPENERS = "\"'\u201c\u2018(["
DASHES = ("-", "--", "\u2014", "\u2013")
WORD = re.compile(r"\S+")


def even_cut(n, cap):
    pieces = -(-n // cap)
    return -(-n // pieces)


def x_cap_comma(chunk, cap):
    """The experiment's cap-comma (recursive; whitespace kept in chunks under the cap)."""
    words = chunk.split()
    if len(words) <= cap:
        return [chunk]
    marks = [i + 1 for i in range(len(words) - 1)
             if (words[i][-1] in ",;:\u2014\u2013" or words[i + 1] in ("-", "\u2014", "\u2013"))
             and MIN_WORDS <= i + 1 <= cap and len(words) - (i + 1) >= MIN_WORDS]
    cut = max(marks) if marks else even_cut(len(words), cap)
    return [" ".join(words[:cut])] + x_cap_comma(" ".join(words[cut:]), cap)


def cap_hard(chunk, cap):
    words = chunk.split()
    if len(words) <= cap:
        return [chunk]
    cut = even_cut(len(words), cap)
    return [" ".join(words[:cut])] + cap_hard(" ".join(words[cut:]), cap)


def merge_short(parts):
    out = []
    for p in parts:
        if out and len(out[-1].split()) < MIN_WORDS:
            out[-1] += " " + p
        else:
            out.append(p)
    if len(out) > 1 and len(out[-1].split()) < MIN_WORDS:
        last = out.pop()
        out[-1] += " " + last
    return out


def full_sentences(paragraph):
    words, parts, cur = paragraph.split(), [], []
    for k, w in enumerate(words):
        cur.append(w)
        bare = w.rstrip("\"'\u201d\u2019)]")
        if k + 1 < len(words) and bare[-1:] in (".", "!", "?"):
            low = bare.lower()
            nxt = words[k + 1].lstrip("\"'\u201c\u2018([")
            if low in X_TITLES or (low in X_ABBREVIATIONS and not nxt[:1].isupper()):
                continue
            parts.append(" ".join(cur)); cur = []
    if cur:
        parts.append(" ".join(cur))
    return parts


def split_text(text, rule=None, cap=None):
    import os
    rule = rule or os.environ.get("LFM3_SPLIT_RULE", "sentences")
    cap = int(cap or os.environ.get("LFM3_SPLIT_CAP") or "0")
    if rule == "sentences":
        return sentences(text)
    if rule == "sentences-v1":
        return sentences_v1(text)
    assert cap >= 2 * MIN_WORDS, "LFM3_SPLIT_CAP must be at least 12"
    if rule in ("cap-comma", "cap-hard"):
        f = x_cap_comma if rule == "cap-comma" else cap_hard
        return [q for c in sentences_v1(text) for q in f(c, cap)]
    if rule == "full":
        paragraphs = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
        chunks = merge_short([q for p in paragraphs for q in full_sentences(p)])
        return [q for c in chunks for q in x_cap_comma(c, cap)]
    raise SystemExit(f"unknown LFM3_SPLIT_RULE {rule}")


def sentences_v1(text):
    """The first rule (validated 2026-11-08, speech-long-form-v1): no abbreviation guard, no length cap."""
    parts = [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p]
    return merge_short(parts)


def ascii_lower(s):
    return "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in s)


def sentence_parts(text):
    """Cut at each whitespace run that follows . ! or ?, except after a title (never a sentence end) or after an
    abbreviation when the next word does not start with an ASCII capital. Both lists compare ASCII-lowercased."""
    t = text.strip()
    parts, start, prev_end = [], 0, 0
    for m in re.finditer(r"\s+", t):
        i, j = m.start(), m.end()
        before, prev_end = t[prev_end:i], j
        if before[-1] not in ".!?":
            continue
        low = ascii_lower(before)
        nxt = WORD.match(t, j).group().lstrip(OPENERS)  # (not t[j:]: a copy per sentence would be quadratic)
        if low in TITLES or ((low in ABBREVIATIONS or before.endswith("...")) and not "A" <= nxt[:1] <= "Z"):
            continue
        parts.append(t[start:i])
        start = j
    if t:
        parts.append(t[start:])
    return parts


def is_mark(word, nxt):
    """A clause mark ends `word`: it ends in , ; : -- or an em or en dash, or the next word is a lone dash."""
    return word[-1] in ",;:\u2014\u2013" or word.endswith("--") or nxt in DASHES


def cap_words(chunk, cap):
    """The chunk's words joined by single spaces; over `cap` words, cut at the last clause mark at or before word `cap`
    that leaves both sides MIN_WORDS long, or with none into the fewest equal pieces; repeated on the rest (a loop:
    linear in the text)."""
    words, out, start = chunk.split(), [], 0
    while len(words) - start > cap:
        n, cut = len(words) - start, 0
        for i in range(start, min(start + cap, len(words) - 1)):
            k = i + 1 - start
            if k >= MIN_WORDS and n - k >= MIN_WORDS and is_mark(words[i], words[i + 1]):
                cut = k
        cut = cut or even_cut(n, cap)
        out.append(" ".join(words[start:start + cut]))
        start += cut
    out.append(" ".join(words[start:]))
    return out


def sentences(text):
    """The default rule (validation/speech-split-v1): sentence_parts, parts under MIN_WORDS joined, then every chunk
    normalized to single spaces and any over CAP_WORDS cut at clause marks (cap_words)."""
    return [q for c in merge_short(sentence_parts(text)) for q in cap_words(c, CAP_WORDS)]


def main(model_dir, prompts, out_dir, mode, dtype="bfloat16"):
    import numpy as np
    import soundfile as sf
    import torch
    torch.set_grad_enabled(False)
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    assert mode in ("fresh", "continue", "requests")
    if mode == "requests":
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        spec = json.load(open(prompts))
        (Path(out_dir) / "requests").mkdir(parents=True, exist_ok=True)
        for c in spec["cases"]:
            system = spec["systems"][c["system"]]
            chunks = [processor.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": t}],
                                                    add_generation_prompt=True, tokenize=True, return_dict=True,
                                                    return_tensors="pt")["input_ids"][0].tolist() for t in split_text(c["user"])]
            for seed in spec["seeds"]:
                json.dump(dict(prompt_ids=chunks[0], chunks=chunks, chunk_texts=split_text(c["user"]), mode="audio", max_new_tokens=c["max_new_tokens"],
                               seed=seed, conversation=[{"role": "system", "content": system}, {"role": "user", "content": c["user"]}]),
                          open(Path(out_dir) / "requests" / f"{c['case']}-s{seed}.req.json", "w"))
        return
    if dtype == "float32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForMultimodalLM.from_pretrained(model_dir, dtype=getattr(torch, dtype), device_map="cuda",
                                                     trust_remote_code=True, attn_implementation="sdpa").eval()
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    tok = processor.tokenizer
    codec = processor.audio_tokenizer.to(device="cuda").eval()
    ao, eos_code = model.config.audio_out_token_id, 2016
    engine = ("hf" if dtype == "bfloat16" else "hf-fp32") + "-" + mode
    spec = json.load(open(prompts))
    for c in spec["cases"]:
        system = spec["systems"][c["system"]]
        chunks = split_text(c["user"])
        for seed in spec["seeds"]:
            name = f"{c['case']}-s{seed}"
            d = Path(out_dir) / engine / name
            d.mkdir(parents=True, exist_ok=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            history_ids, history_codes, frames, per_chunk = None, [], [], []
            for k, chunk in enumerate(chunks):
                if mode == "fresh" or history_ids is None:
                    conv = [{"role": "system", "content": system}, {"role": "user", "content": chunk}]
                    ids = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=True,
                                                        return_dict=True, return_tensors="pt")["input_ids"][0].tolist()
                    codes_in = []
                else:
                    suffix = tok.encode(f"\n<|im_start|>user\n{chunk}<|im_end|>\n<|im_start|>assistant\n<think></think>",
                                        add_special_tokens=False)
                    ids, codes_in = history_ids + suffix, history_codes
                inp = dict(input_ids=torch.tensor([ids], device="cuda"),
                           attention_mask=torch.ones(1, len(ids), dtype=torch.long, device="cuda"))
                if codes_in:
                    inp["audio_out_codes"] = torch.tensor(codes_in, device="cuda")
                torch.manual_seed(seed * 1000 + k)
                o = model.generate(**inp, max_new_tokens=c["max_new_tokens"], generation_mode="audio", do_sample=False,
                                   audio_temperature=0.4, audio_repetition_penalty=1.2, return_audio_eos=True)
                slots = o.sequences[0, len(ids):o.sequence_lengths[0]].tolist()
                n = int(o.audio_lengths[0])
                got = o.audio_codes[0, :, :n].T.tolist()  # generation order, end-of-audio frames kept
                speech = [f for f in got if eos_code not in f]
                frames += speech
                ended = bool(got) and eos_code in got[-1] and slots[-1] == tok.convert_tokens_to_ids("<|im_end|>")
                per_chunk.append(dict(text=chunk, frames=len(speech), slots=len(slots), ended_by_eos=ended))
                history_ids = ids + slots
                history_codes = codes_in + got
                assert history_ids.count(ao) == len(history_codes), (history_ids.count(ao), len(history_codes))
            wav = codec.decode(torch.tensor(frames, device="cuda").T[None]).audio_values[0].squeeze(0).float().cpu().numpy() \
                if frames else np.zeros(0, np.float32)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - t0
            sf.write(d / "response.wav", wav, 24_000)
            (d / "response.txt").write_text("\n")
            (d / "summary.json").write_text(json.dumps(dict(
                case=name, mode="audio", seed=seed, engine=engine,
                conversation=[{"role": "system", "content": system}, {"role": "user", "content": c["user"]}], text="",
                audio_seconds=len(wav) / 24_000, frames=len(frames), slots=sum(p["slots"] for p in per_chunk),
                chunks=per_chunk, generate_seconds=seconds,
                real_time_factor=(len(wav) / 24_000) / seconds if seconds else 0), indent=1) + "\n")
            print(name, engine, f"{len(chunks)} chunks, {len(frames)} frames, {len(wav) / 24_000:.1f}s audio in {seconds:.1f}s,",
                  "all ended by EOS" if all(p["ended_by_eos"] for p in per_chunk) else "NOT all ended by EOS", flush=True)


if __name__ == "__main__":
    if len(sys.argv) not in (5, 6):
        sys.exit(__doc__)
    main(*sys.argv[1:])
