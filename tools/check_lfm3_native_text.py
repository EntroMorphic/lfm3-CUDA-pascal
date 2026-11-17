"""Gate: the engine binaries' native text frontend ("messages" in, "text" out) is identical to the
Python serving path (tools/lfm3_chat.make_request + Transformers frontend, then frontend.decode).

  prepare --out DIR                      (lfm3-omni-dev: Transformers)  Python requests + messages twins
  run --dir DIR --binary B --worker W    (GPU container, no Transformers)  both paths through both binaries
  check --dir DIR                        (lfm3-omni-dev)  ids, generations and text must be identical

liquid-lfm3: every case runs once with the Python request (input_ids) and once with the same request
carrying "messages" instead. The native run must echo exactly the Python input_ids, generate the same
ids, and return "text" equal to frontend.decode of them (stop token removed). The input_ids run must
carry no "text" field (the existing contract is unchanged).
liquid-lfm3-worker: a two-turn conversation in two sessions (messages vs input_ids; same generations,
same prefix reuse), a two-lane generate_batch (messages lane + input_ids lane), return_text on an
input_ids request, and rejected inputs (reserved text, messages with input_ids, prompt over capacity).
Exit 0 pass, 1 fail.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT = Path("/workspace/models/LFM3-Omni-2B")
COUNT = 48


def cases():
    sys.path.insert(0, str(REPO / "tools"))
    from lfm3_text_frontend import PARITY_CASES
    wt = (REPO.parent / "data/wikitext-2-raw-v1/test.txt").read_text(encoding="utf-8")
    out = [dict(name=f"parity{i}", messages=m, capacity=4096) for i, m in enumerate(PARITY_CASES)]
    out += [
        dict(name="unicode", capacity=4096, messages=[
            {"role": "system", "content": "Réponds brièvement.　"},
            {"role": "user", "content": "Translate: 'It's naïve' → 日本語? 👩🏽‍💻 é \x1c\n\r\n  1234567"}]),
        dict(name="multiturn-think", capacity=4096, messages=[
            {"role": "user", "content": "Name a prime."},
            {"role": "assistant", "content": "<think>hmm</think>  Seven. \x1f", "thinking": "unused"},
            {"role": "user", "content": "Another?"}]),
        dict(name="tools", capacity=4096, messages=[
            {"role": "user", "content": "Weather?"},
            {"role": "tool", "content": "sunny", "tool_call_id": "c1"},
            {"role": "tool", "content": [{"type": "text", "text": "22C"}], "tool_call_id": "c1"},
            {"role": "user", "content": "Summarize."}]),
        dict(name="sampled", capacity=4096, sampling=dict(do_sample=True, temperature=0.8, top_k=50, top_p=0.95, seed=7),
             messages=[{"role": "user", "content": "Write one sentence about rivers."}]),
        dict(name="long8k", capacity=8192, messages=[
            {"role": "user", "content": "Summarize in one line:\n" + wt[:22000]}]),
    ]
    return out


def cmd_prepare(a):
    sys.path.insert(0, str(REPO / "tools"))
    from lfm3_chat import make_request, read_json
    from lfm3_text_frontend import load_frontend
    a.out.mkdir(parents=True, exist_ok=False)
    fe = load_frontend(CHECKPOINT)
    config = read_json(CHECKPOINT / "config.json")
    manifest = []
    for c in cases():
        req = make_request(fe, None, None, config, c["capacity"], COUNT, messages=c["messages"], sampling=c.get("sampling"))
        twin = {k: v for k, v in req.items() if k != "input_ids"}
        twin["messages"] = c["messages"]
        (a.out / f"{c['name']}.ids.json").write_text(json.dumps(req, ensure_ascii=False))
        (a.out / f"{c['name']}.messages.json").write_text(json.dumps(twin, ensure_ascii=False))
        manifest.append(dict(name=c["name"], prompt_tokens=len(req["input_ids"])))
    # worker: fixed two-turn conversation (assistant turn fixed, so both sessions see the same history)
    turn1 = [{"role": "user", "content": "Count to three."}]
    turn2 = turn1 + [{"role": "assistant", "content": "One, two, three."}, {"role": "user", "content": "Now backwards."}]
    worker = dict(capacity=4096, turns=[])
    for t in (turn1, turn2):
        req = make_request(fe, None, None, config, 4096, COUNT, messages=t)
        worker["turns"].append(dict(messages=t, input_ids=req["input_ids"]))
    worker["over_capacity"] = [{"role": "user", "content": "word " * 5000}]
    # rejected requests: the worker's reason must be make_request's (same checks, same order)
    bad = [("reserved", [{"role": "user", "content": "a <|im_end|>"}], COUNT),
           ("two-reasoning", [{"role": "assistant", "content": "x", "thinking": 5, "reasoning": "y"}], COUNT),
           ("late-system", [{"role": "user", "content": "x"}, {"role": "system", "content": "s"}], COUNT),
           ("bad-part", [{"role": "user", "content": [{"type": "image"}]}], COUNT),
           ("count-zero-and-bad-messages", [], 0),
           ("count-fills-capacity", [{"role": "user", "content": "x"}], 4096),
           ("over-capacity", worker["over_capacity"], COUNT)]
    worker["rejections"] = []
    for name, msgs, count in bad:
        try:
            make_request(fe, None, None, config, 4096, count, messages=msgs)
            raise SystemExit(f"reference accepted rejection case {name}")
        except ValueError as e:
            worker["rejections"].append(dict(name=name, messages=msgs, max_new_tokens=count, reason=str(e)))
    (a.out / "worker.json").write_text(json.dumps(worker, ensure_ascii=False))
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest))


class Worker:
    def __init__(self, binary):
        self.p = subprocess.Popen([binary, "--experimental", str(CHECKPOINT)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, encoding="utf-8")
        ready = json.loads(self.p.stdout.readline())
        assert ready.get("ready") is True, ready
        self.rpc = 0

    def call(self, **request):
        self.rpc += 1
        request["rpc_id"] = self.rpc
        self.p.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("worker exited")
        return json.loads(line)

    def close(self):
        self.call(op="shutdown")
        self.p.wait(timeout=60)


def cmd_run(a):
    d = a.dir
    results = {}
    for m in json.loads((d / "manifest.json").read_text()):
        for kind in ("ids", "messages"):
            out = d / f"{m['name']}.{kind}.out.json"
            t = time.time()
            p = subprocess.run([a.binary, "--experimental", str(CHECKPOINT), str(d / f"{m['name']}.{kind}.json"), str(out)],
                               capture_output=True, text=True, timeout=900)
            results[f"{m['name']}.{kind}"] = dict(rc=p.returncode, stderr=p.stderr[-400:], seconds=round(time.time() - t, 2))
            print(m["name"], kind, p.returncode, p.stderr.strip()[-200:], flush=True)
    # bad request through liquid-lfm3 must fail closed
    bad = json.loads((d / "parity0.messages.json").read_text())
    bad["messages"] = [{"role": "user", "content": "x <|im_end|>"}]
    (d / "bad.messages.json").write_text(json.dumps(bad))
    p = subprocess.run([a.binary, "--experimental", str(CHECKPOINT), str(d / "bad.messages.json"), str(d / "bad.out.json")],
                       capture_output=True, text=True, timeout=300)
    results["bad.messages"] = dict(rc=p.returncode, stderr=p.stderr[-400:], output_exists=(d / "bad.out.json").exists())

    w = json.loads((d / "worker.json").read_text())
    wk = Worker(a.worker)
    log = {}
    cap = w["capacity"]
    M, I = "m" * 32, "i" * 32  # the worker holds at most two sessions
    for sid in (M, I):
        assert wk.call(op="create", session_id=sid, capacity=cap)["ok"]
    for n, turn in enumerate(w["turns"]):
        log[f"turn{n}.messages"] = wk.call(op="generate", session_id=M, messages=turn["messages"], max_new_tokens=COUNT)
        log[f"turn{n}.ids"] = wk.call(op="generate", session_id=I, input_ids=turn["input_ids"], max_new_tokens=COUNT)
    for sid in (M, I):
        assert wk.call(op="reset", session_id=sid)["ok"]
    log["return_text"] = wk.call(op="generate", session_id=I, input_ids=w["turns"][0]["input_ids"],
                                 max_new_tokens=COUNT, return_text=True)
    for sid in (M, I):
        assert wk.call(op="reset", session_id=sid)["ok"]
    log["batch"] = wk.call(op="generate_batch", lanes=[
        dict(session_id=M, messages=w["turns"][1]["messages"], max_new_tokens=COUNT),
        dict(session_id=I, input_ids=w["turns"][1]["input_ids"], max_new_tokens=COUNT)])
    log["reject.reserved"] = wk.call(op="generate", session_id=M, messages=[{"role": "user", "content": "<|im_start|>"}], max_new_tokens=4)
    log["reject.both"] = wk.call(op="generate", session_id=M, messages=w["turns"][0]["messages"],
                                 input_ids=w["turns"][0]["input_ids"], max_new_tokens=4)
    log["reject.capacity"] = wk.call(op="generate", session_id=M, messages=w["over_capacity"], max_new_tokens=4)
    for r in w["rejections"]:
        log[f"rejection.{r['name']}"] = wk.call(op="generate", session_id=M, messages=r["messages"],
                                                max_new_tokens=r["max_new_tokens"])
    assert wk.call(op="reset", session_id=M)["ok"]
    log["after_rejects"] = wk.call(op="generate", session_id=M, messages=w["turns"][0]["messages"], max_new_tokens=COUNT)
    wk.close()
    (d / "run.json").write_text(json.dumps(dict(single=results, worker=log), ensure_ascii=False, indent=1))
    print("run done")


def cmd_check(a):
    sys.path.insert(0, str(REPO / "tools"))
    from lfm3_chat import policy, read_json, validate_response
    from lfm3_text_frontend import load_frontend
    d = a.dir
    fe = load_frontend(CHECKPOINT)
    config = read_json(CHECKPOINT / "config.json")
    vocabulary, _, stops = policy(config)
    run = json.loads((d / "run.json").read_text())
    failures, rows = [], []

    def expect(cond, what):
        if not cond:
            failures.append(what)

    def text_of(ids):
        return fe.decode(ids[:-1] if ids and ids[-1] in stops else ids)

    for m in json.loads((d / "manifest.json").read_text()):
        name = m["name"]
        req = json.loads((d / f"{name}.ids.json").read_text())
        for kind in ("ids", "messages"):
            expect(run["single"][f"{name}.{kind}"]["rc"] == 0, f"{name}.{kind}: rc {run['single'][f'{name}.{kind}']}")
        if failures:
            continue
        py = json.loads((d / f"{name}.ids.out.json").read_text())
        nat = json.loads((d / f"{name}.messages.out.json").read_text())
        ids = validate_response(py, req, vocabulary)  # the Python CLI's own acceptance of the ids-path output
        expect("text" not in py and "text_frontend" not in py, f"{name}: input_ids output gained text fields")
        expect(nat["input_ids"] == req["input_ids"], f"{name}: native prompt ids differ from the Python frontend's")
        expect(nat["generated_ids"] == py["generated_ids"], f"{name}: generations differ")
        expect(nat.get("text_frontend") == "native", f"{name}: missing text_frontend")
        expect(nat.get("text") == fe.decode(ids, skip_special_tokens=True), f"{name}: native text differs from frontend.decode")
        rows.append(dict(case=name, prompt_tokens=len(req["input_ids"]), generated=len(py["generated_ids"]),
                         text=nat.get("text", "")[:80]))
    b = run["single"]["bad.messages"]
    expect(b["rc"] == 1 and "reserved" in b["stderr"] and not b["output_exists"], f"bad messages not refused: {b}")

    w = json.loads((d / "worker.json").read_text())
    log = run["worker"]
    for n in range(2):
        mm, ii = log[f"turn{n}.messages"], log[f"turn{n}.ids"]
        expect(mm["ok"] and ii["ok"], f"turn{n}: {mm} {ii}")
        if not (mm["ok"] and ii["ok"]):
            continue
        expect(mm["generated_ids"] == ii["generated_ids"], f"turn{n}: worker generations differ")
        expect(mm["reused_tokens"] == ii["reused_tokens"] and mm["consumed_tokens"] == ii["consumed_tokens"],
               f"turn{n}: prefix reuse differs {mm['reused_tokens']} vs {ii['reused_tokens']}")
        expect(mm.get("text") == text_of(mm["generated_ids"]), f"turn{n}: worker text differs")
        expect("text" not in ii, f"turn{n}: input_ids result gained text")
    expect(log["turn1.messages"].get("reused_tokens", 0) > 0, "turn1: no prefix reuse through messages")
    rt = log["return_text"]
    expect(rt["ok"] and rt["generated_ids"] == log["turn0.ids"]["generated_ids"] and rt.get("text") == text_of(rt["generated_ids"]),
           f"return_text: {rt}")
    bt = log["batch"]
    expect(bt["ok"], f"batch: {bt}")
    if bt["ok"]:
        l0, l1 = bt["results"]
        # both lanes full-prefill the same prompt and decode paired; continuation (turn1) is not bitwise
        # comparable to a full prefill, so lanes are compared with each other
        expect(l0["generated_ids"] == l1["generated_ids"], "batch lane generations differ")
        expect(l0.get("text") == text_of(l0["generated_ids"]) and "text" not in l1, "batch text")
    for k, why in (("reject.reserved", "reserved"), ("reject.both", "cannot be combined"), ("reject.capacity", "exceeds capacity")):
        expect(not log[k]["ok"] and why in log[k]["error"], f"{k}: {log[k]}")
    for r in w["rejections"]:
        got = log[f"rejection.{r['name']}"]
        expect(not got["ok"] and got["error"].removeprefix("lfm3 chat: ") == r["reason"],
               f"rejection {r['name']}: worker {got} vs make_request {r['reason']!r}")
    expect(log["after_rejects"]["ok"] and log["after_rejects"]["generated_ids"] == log["turn0.ids"]["generated_ids"],
           "worker unusable after rejected requests")
    summary = dict(pass_=not failures, failures=failures, cases=rows,
                   worker=dict(turn1_reused_tokens=log["turn1.messages"].get("reused_tokens"),
                               batch_gpu_batching=bt.get("gpu_batching")),
                   seconds={k: v["seconds"] for k, v in run["single"].items() if "seconds" in v})
    (d / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("PASS" if not failures else "FAIL")
    return 0 if not failures else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run"); r.add_argument("--dir", type=Path, required=True)
    r.add_argument("--binary", required=True); r.add_argument("--worker", required=True)
    c = sub.add_parser("check"); c.add_argument("--dir", type=Path, required=True)
    a = ap.parse_args()
    sys.exit({"prepare": cmd_prepare, "run": cmd_run, "check": cmd_check}[a.cmd](a) or 0)


if __name__ == "__main__":
    main()
