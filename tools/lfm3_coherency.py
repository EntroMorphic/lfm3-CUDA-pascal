#!/usr/bin/env python3
"""Coherency check for the LFM3 text engines: real chat generations, read by a person, plus mechanical flags.

  cases  CHECKPOINT WINDOWS_I32 OUT_DIR    write one "messages" request per case (greedy, native chat template)
  report CHECKPOINT OUT_DIR ENGINE...      print a Markdown report of every case for each engine

Cases cover every prefill heads-per-call band of the budget rule (capacity 512: 16 heads, 2,560: 8, 4,096: 4,
32,768: 16 in 128-row tiles): five short prompts at each capacity, and one long document per band above 512
(WikiText-2 test text, the perplexity windows' tokens) with a fact planted in its middle and a question that needs
both the fact and the document. Flags per output: the planted fact recalled, the engine's text equals the HF
tokenizer's decode of its generated ids, whether generation stopped by itself, repeated 4-grams (degeneration), and
the common greedy prefix with the first engine named.
"""
import json
import sys
from pathlib import Path

SHORT = {
    "sky": [{"role": "user", "content": "Explain why the sky is blue in three sentences."}],
    "code": [{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number "
                                         "iteratively, then state its time complexity."}],
    "arith": [{"role": "user", "content": "A train leaves at 3:40 pm and the trip takes 2 hours 35 minutes. "
                                          "When does it arrive? Explain step by step."}],
    "poem": [{"role": "user", "content": "Write a four-line poem about a lighthouse in winter."}],
    "memory": [{"role": "user", "content": "My name is Priya and I keep bees."},
               {"role": "assistant", "content": "That's wonderful! How many hives do you have?"},
               {"role": "user", "content": "Three. What is my name, and what should I check in my hives before "
                                           "winter?"}],
}
SHORT_CAPACITIES = (512, 2560, 4096, 32768)
NEEDLE = "The archivist's secret passphrase is violet harbor 42."
NEEDLE_KEY = "violet harbor 42"
QUESTION = ("What is the archivist's secret passphrase mentioned in the text above? Answer with the passphrase, "
            "then summarize what the text is about in two sentences.")
LONG = {2560: 2000, 4096: 3500, 32768: 30000}  # capacity -> document tokens
SHORT_NEW, LONG_NEW = 256, 200


def cases(checkpoint, windows, out):
    import numpy as np
    tok = tokenizer(checkpoint)
    ids = np.fromfile(windows, np.int32).tolist()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, messages in SHORT.items():
        for capacity in SHORT_CAPACITIES:
            manifest[f"{name}-{capacity}"] = dict(messages=messages, max_new_tokens=SHORT_NEW, capacity=capacity)
    for capacity, length in LONG.items():
        text = tok.decode(ids[:length], skip_special_tokens=True)  # the windows begin with BOS
        cut = text.find(". ", len(text) // 2) + 2  # sentence boundary just past the middle
        doc = text[:cut] + NEEDLE + " " + text[cut:]
        manifest[f"doc-{capacity}"] = dict(messages=[{"role": "user", "content": doc + "\n\n" + QUESTION}],
                                           max_new_tokens=LONG_NEW, capacity=capacity, needle=NEEDLE_KEY)
    for name, case in manifest.items():
        request = {k: v for k, v in case.items() if k != "needle"}
        request["do_sample"] = False
        if request["capacity"] > 4096:
            request["experimental_long_context"] = True
        prompt = tok.apply_chat_template(case["messages"], add_generation_prompt=True, tokenize=True,
                                         return_dict=False)
        case["hf_prompt_tokens"] = len(prompt)
        assert len(prompt) + request["max_new_tokens"] <= request["capacity"], name
        (out / f"req-{name}.json").write_text(json.dumps(request))
    (out / "manifest.json").write_text(json.dumps(
        {k: {x: v[x] for x in v if x != "messages"} for k, v in manifest.items()}, indent=1))


def tokenizer(checkpoint):
    # The fast tokenizer class directly: AutoTokenizer would offer to run the checkpoint's remote code.
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast.from_pretrained(checkpoint)


def repeated_4grams(ids):
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)


def report(checkpoint, out, engines):
    tok = tokenizer(checkpoint)
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text())
    print("| Case | Engine | Prompt tokens | New tokens | Stopped | Text = HF decode | Repeated 4-grams | "
          f"Common prefix with {engines[0]} | Fact recalled |")
    print("|---|---|---|---|---|---|---|---|---|")
    texts, failures = {}, []
    for name, case in manifest.items():
        first = None
        for engine in engines:
            path = out / f"{engine}-{name}.json"
            if not path.exists():
                failures.append(f"{engine} {name}: no output (see {engine}-{name}.log)")
                print(f"| {name} | {engine} | **failed: no output** | | | | | | |")
                texts[(name, engine)] = "(no output)"
                continue
            r = json.loads(path.read_text())
            gen = r["generated_ids"]
            # Generation ends at a stop token (dropped from "text") or at the limit (a stop exactly at the
            # limit is told apart by which decode the text equals).
            stopped = len(gen) < case["max_new_tokens"]
            same = r["text"] == tok.decode(gen[:-1] if stopped else gen, skip_special_tokens=True)
            if not same and not stopped and r["text"] == tok.decode(gen[:-1], skip_special_tokens=True):
                same = stopped = True
            prefix = next((i for i, (a, b) in enumerate(zip(first, gen)) if a != b), min(len(first), len(gen))) \
                if first is not None else len(gen)
            first = gen if first is None else first
            recalled = "" if "needle" not in case else ("yes" if case["needle"] in r["text"].lower() else "**no**")
            if not same:
                failures.append(f"{engine} {name}: text differs from the HF decode")
            print(f"| {name} | {engine} | {len(r['input_ids'])} (HF {case['hf_prompt_tokens']}) | {len(gen)} | "
                  f"{'yes' if stopped else '**limit**'} | {'yes' if same else '**no**'} | "
                  f"{repeated_4grams(gen):.2f} | {prefix} | {recalled} |")
            texts[(name, engine)] = r["text"]
    for name in manifest:
        print(f"\n### {name}\n")
        for engine in engines:
            print(f"**{engine}:**\n\n```text\n{texts[(name, engine)]}\n```\n")
    print("\n" + ("\n".join(failures) if failures else "Every case produced output, and every text equals the HF "
                                                         "tokenizer's decode of its ids."))


if __name__ == "__main__":
    if sys.argv[1] == "cases":
        cases(*sys.argv[2:5])
    elif sys.argv[1] == "report":
        report(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        sys.exit(__doc__)
