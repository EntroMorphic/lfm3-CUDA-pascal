#!/usr/bin/env python3
"""HF's answers to MMStar and POPE (validation/images-v1 rule 3's reference): LFM3-Omni-2B in BF16 on the GPU
(attn_implementation "sdpa"), greedy, one request at a time, through the checkpoint's own chat template.

Each item: the user turn is the image (its file path, so HF's `load_image` decodes it: EXIF transpose, then RGB) and
the question with the lmms-eval instruction:
  MMStar: "{question}\\nAnswer with the option's letter from the given choices directly."
  POPE:   "{question} Please answer yes or no."
No system turn. max_new_tokens 512 for MMStar (the model often reasons before answering, and 346 of its questions
ask for the letter "at the end": at 16 tokens 28% of replies, at 64 tokens 27%, were cut off before an answer), 16 for
POPE; do_sample False, generation_mode "text".
Scoring (VLMEvalKit's approach, without its LLM judge): MMStar, in order: a reply that begins with a letter A-D
("B", "(B)", "B.", "B:"; not "A man ..."); else a last line that is only a
letter ("B", "**B**", "(B)", "B."); else "answer is B" or "Answer: B" (the last such); else exactly one standalone option letter in the reply ("A" not counted in a reply of more
than 3 words, where it is the article); else exactly one option's text (2 characters or more, at word boundaries) found in the reply; else no
answer. POPE, the
first "yes" or "no" word. No answer is wrong. The rule that matched is recorded. Per item, written as one JSONL row: the prompt's token count, the generated ids, the reply, the parsed answer,
correct, and seconds. Rows already in OUT_JSONL are skipped (resume). The first row records the environment.
Usage: run_lfm3_vision_bench_hf.py MODEL_DIR BENCH_DIR OUT_JSONL [LIMIT_PER_SPLIT] [--dtype float32] [--benchmark NAME]
  --dtype float32: the model and pixel values in FP32 instead of BF16 (rule 3b's reference); default bfloat16.
  --benchmark NAME: only that benchmark's items (mmstar or pope); default both.
"""
import json
import platform
import re
import sys
import time
from pathlib import Path

MMSTAR = "{q}\nAnswer with the option's letter from the given choices directly."
POPE = "{q} Please answer yes or no."
MAX_NEW = dict(mmstar=512, pope=16)


def prompt_text(row):
    return (MMSTAR if row["benchmark"] == "mmstar" else POPE).format(q=row["question"])


def options(question):
    """MMStar's options as {letter: text}, from either layout: "Options: A: ..., B: ..." or lines "(A) ..." (most after
    "Choices:")."""
    lines = re.findall(r"^\(([A-D])\)\s*(.*)$", question.rsplit("Choices:", 1)[-1], flags=re.M)
    if lines:
        found = lines
    else:
        tail = question.rsplit("Options:", 1)[-1]
        found = re.findall(r"(?:^|,)\s*([A-D]):\s*(.*?)(?=,\s*[A-D]:|$)", tail, flags=re.S)
    return {k: v.strip().rstrip(".").strip() for k, v in found}


def parse(row, reply):
    """(answer or None, the rule that matched)."""
    text = reply.strip()
    if row["benchmark"] == "pope":
        m = re.search(r"\b(yes|no)\b", text.lower())
        return (m.group(1), "word") if m else (None, "none")
    m = re.match(r"^\(?([A-D])(?:[).:]|$)", text)
    if m:
        return m.group(1), "leading letter"
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        m = re.fullmatch(r"[*_\s]*\(?([A-D])\)?[.:]?[*_\s]*", lines[-1])
        if m:
            return m.group(1), "last line"
    found = re.findall(r"(?i:answer(?:\s+is)?\s*[:：]?\s*)[*_(\s]*([A-D])\b", text)
    if found:
        return found[-1], "answer is"
    words = re.findall(r"[A-Za-z]+", text)
    letters = {w for w in words if w in ("A", "B", "C", "D")}
    if len(words) > 3:
        letters.discard("A")
    if len(letters) == 1:
        return letters.pop(), "one letter"
    low = text.lower()
    hits = [k for k, v in options(row["question"]).items()
            if len(v) >= 2 and re.search(r"(?<!\w)" + re.escape(v.lower()) + r"(?!\w)", low)]
    if len(hits) == 1:
        return hits[0], "option text"
    return None, "none"


def key(row):
    return f"{row['benchmark']}/{row['split']}/{row['id']}"


def main(model_dir, bench_dir, out_jsonl, limit=None, dtype_name="bfloat16", benchmark=None):
    import PIL
    import torch
    import torchvision
    import transformers
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    bench, out = Path(bench_dir), Path(out_jsonl)
    rows = [json.loads(l) for l in (bench / "manifest.jsonl").read_text().splitlines()]
    if benchmark:
        rows = [r for r in rows if r["benchmark"] == benchmark]
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_name]
    if limit:
        per, kept = {}, []
        for r in rows:
            k = (r["benchmark"], r["split"])
            if per.get(k, 0) < int(limit):
                per[k] = per.get(k, 0) + 1
                kept.append(r)
        rows = kept
    done = set()
    if out.exists():
        for line in out.read_text().splitlines()[1:]:
            done.add(json.loads(line)["key"])
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    t0 = time.monotonic()
    model = AutoModelForMultimodalLM.from_pretrained(model_dir, dtype=dtype, device_map="cuda",
                                                     attn_implementation="sdpa", trust_remote_code=True).eval()
    env = dict(machine=platform.machine(), transformers=transformers.__version__, torch=torch.__version__,
               torchvision=torchvision.__version__, pillow=PIL.__version__, cuda=torch.version.cuda,
               device=torch.cuda.get_device_name(), dtype=dtype_name, attn="sdpa", max_new_tokens=MAX_NEW,
               load_seconds=round(time.monotonic() - t0, 1), prompts=dict(mmstar=MMSTAR, pope=POPE))
    with out.open("a") as f:
        if not done:
            f.write(json.dumps(dict(env=env)) + "\n")
        for i, row in enumerate(rows):
            if key(row) in done:
                continue
            conv = [{"role": "user", "content": [{"type": "image", "image": str(bench / row["image"])},
                                                 {"type": "text", "text": prompt_text(row)}]}]
            t = time.monotonic()
            inputs = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=True, return_dict=True,
                                                   return_tensors="pt").to("cuda")
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=MAX_NEW[row["benchmark"]], generation_mode="text",
                                     do_sample=False)
            n = inputs["input_ids"].shape[1]
            ids = gen.sequences[0, n:gen.sequence_lengths[0]].tolist()
            reply = processor.decode(ids, skip_special_tokens=True)
            answer, rule = parse(row, reply)
            rec = dict(key=key(row), benchmark=row["benchmark"], split=row["split"], id=row["id"],
                       prompt_tokens=n, grid_thw=inputs["image_grid_thw"].tolist(), generated_ids=ids, reply=reply,
                       answer=answer, parse_rule=rule, truth=row["answer"], correct=answer == row["answer"],
                       seconds=round(time.monotonic() - t, 3))
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 250 == 0:
                print(f"{i}/{len(rows)} {rec['key']} {reply!r} -> {answer} ({row['answer']})", flush=True)
    print("HF BENCH DONE")


if __name__ == "__main__":
    args, opts = [], {}
    argv = sys.argv[1:]
    while argv:
        a = argv.pop(0)
        if a in ("--dtype", "--benchmark") and argv:
            opts[a[2:]] = argv.pop(0)
        else:
            args.append(a)
    if len(args) not in (3, 4) or opts.get("dtype", "bfloat16") not in ("bfloat16", "float32") \
            or opts.get("benchmark") not in (None, "mmstar", "pope"):
        sys.exit(__doc__)
    main(*args, dtype_name=opts.get("dtype", "bfloat16"), benchmark=opts.get("benchmark"))
