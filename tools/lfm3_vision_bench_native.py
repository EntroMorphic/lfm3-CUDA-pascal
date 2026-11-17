#!/usr/bin/env python3
"""The native engine's answers to MMStar and POPE (validation/images-v1 rule 3), the same requests as HF's
(tools/run_lfm3_vision_bench_hf.py): no system turn; the user turn is the image (its file path) then the question
with the lmms-eval instruction; greedy, text mode, max_new_tokens 512 (MMStar) or 16 (POPE).

  requests BENCH_DIR OUT_DIR [LIMIT_PER_SPLIT]
      OUT_DIR/req/<benchmark>__<split>__<id>.json (lfm3-speech requests: "messages", mode "text", record nothing) and
      OUT_DIR/list.txt (REQUEST OUT_STEM per line, the stems in OUT_DIR/out/), for lfm3-speech --batch or
      --concurrent. BENCH_DIR as the container sees it (the paths are written into the requests).
  collect BENCH_DIR OUT_DIR OUT_JSONL [ENV_JSON]
      Reads each output OUT_DIR/out/<stem>.json and writes HF's row format (key, benchmark, split, id, prompt_tokens,
      generated_ids = the text tokens, reply = the decoded text, answer and parse_rule by the HF tool's own parse(),
      truth, correct, seconds = prefill + generate), for tools/score_lfm3_vision_bench.py. A request without an output
      is an error. The first row records ENV_JSON's object (build, binary sha256, mode).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_lfm3_vision_bench_hf as hfb  # noqa: E402


def rows_of(bench, limit=None):
    rows = [json.loads(l) for l in (bench / "manifest.jsonl").read_text().splitlines()]
    if limit:
        per, kept = {}, []
        for r in rows:
            k = (r["benchmark"], r["split"])
            if per.get(k, 0) < int(limit):
                per[k] = per.get(k, 0) + 1
                kept.append(r)
        rows = kept
    return rows


def stem(row):
    return f"{row['benchmark']}__{row['split']}__{row['id']}"


def requests(bench_dir, out_dir, limit=None):
    bench, out = Path(bench_dir), Path(out_dir)
    (out / "req").mkdir(parents=True, exist_ok=True)
    (out / "out").mkdir(parents=True, exist_ok=True)
    rows = rows_of(bench, limit)
    with open(out / "list.txt", "w") as lst:
        for row in rows:
            conv = [{"role": "user", "content": [{"type": "image", "image": str(bench / row["image"])},
                                                 {"type": "text", "text": hfb.prompt_text(row)}]}]
            req = dict(messages=conv, mode="text", max_new_tokens=hfb.MAX_NEW[row["benchmark"]])
            (out / "req" / f"{stem(row)}.json").write_text(json.dumps(req))
            lst.write(f"{out / 'req' / stem(row)}.json {out / 'out' / stem(row)}\n")
    print(len(rows), "requests")


def collect(bench_dir, out_dir, out_jsonl, env_json=None):
    bench, out = Path(bench_dir), Path(out_dir)
    stems = [l.split()[1] for l in (out / "list.txt").read_text().splitlines() if l.strip()]
    by_stem = {stem(r): r for r in rows_of(bench)}
    env = json.loads(Path(env_json).read_text()) if env_json else {}
    missing = 0
    with open(out_jsonl, "w") as f:
        f.write(json.dumps(dict(env=env)) + "\n")
        for s in stems:
            p = Path(s + ".json")
            if not p.exists():
                missing += 1
                continue
            j = json.loads(p.read_text())
            row = by_stem[Path(s).name]
            reply = j["text"]
            answer, rule = hfb.parse(row, reply)
            rec = dict(key=hfb.key(row), benchmark=row["benchmark"], split=row["split"], id=row["id"],
                       prompt_tokens=len(j["prompt"]),  # text mode appends no marker
                       generated_ids=j["text_tokens"], reply=reply, answer=answer,
                       parse_rule=rule, truth=row["answer"], correct=answer == row["answer"],
                       seconds=round(j["prefill_seconds"] + j["generate_seconds"], 3))
            f.write(json.dumps(rec) + "\n")
    if missing:
        sys.exit(f"{missing} requests have no output")
    print(len(stems), "rows")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "requests":
        requests(*sys.argv[2:])
    elif len(sys.argv) >= 5 and sys.argv[1] == "collect":
        collect(*sys.argv[2:])
    else:
        sys.exit(__doc__)
