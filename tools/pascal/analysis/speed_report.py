"""speed-v1's rules 6 and 7 to 14 (validation/speed-v1/README.md) from two runs of tools/pascal/speed.sh, the baseline
build's and the candidate's, made in the same job.
    python3 tools/pascal/analysis/speed_report.py BASELINE_DIR CANDIDATE_DIR [--output FILE]
Each directory holds CASE.N.json records and summary.json. Prints one line per rule and exits 0 only if all hold.
"""
import argparse, json, os, sys

BARS = {  # rule: (case, the bar as a function of the baseline median, description)
    7: ("prefill-4096", lambda b: 4.6, "<= 4.6 s"),
    8: ("prefill-32768", lambda b: 44.7, "<= 44.7 s"),
    9: ("prefill-257", lambda b: 0.60, "<= 0.60 s"),
    10: ("prefill-32", lambda b: 0.30, "<= 0.30 s"),
    # Rules 11 and 12 as amended by the owner (2026-11-18): the bytes a step must read at 245 GB/s, plus 1.5 ms.
    11: ("decode-1k", lambda b: 15.61, "<= 15.61 ms per step (amended)"),
    # Rule 12's figure corrected 2026-11-19: 4 key/value heads, 380.9 MB of keys and values (was 18.72, from 8 heads).
    12: ("decode-32k", lambda b: 17.16, "<= 17.16 ms per step (amended, corrected)"),
}
NO_SLOWER = ("decode-8k", "prefill-1024")


def load(d):
    summary = json.load(open(os.path.join(d, "summary.json")))
    runs = {}
    for f in os.listdir(d):
        if f.endswith(".json") and f != "summary.json" and not f.startswith("warmup"):
            r = json.load(open(os.path.join(d, f)))
            runs.setdefault(r["case"], []).append(r)
    return summary, runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline"); ap.add_argument("candidate"); ap.add_argument("--output")
    a = ap.parse_args()
    (bs, br), (cs, cr) = load(a.baseline), load(a.candidate)
    lines, ok_all = [], True
    cases = sorted(set(bs) | set(cs))
    if set(bs) != set(cs): lines.append(f"cases differ: baseline {sorted(bs)}, candidate {sorted(cs)}"); ok_all = False
    # Rule 6: every run of every case gives the baseline's ids.
    same = True
    for case in cases:
        ref = br.get(case, [None])[0]
        for r in cr.get(case, []) + br.get(case, []):
            if ref is None or r["teacher_argmax"] != ref["teacher_argmax"] or r["generated_ids"] != ref["generated_ids"]:
                same = False; lines.append(f"rule 6: {case} ids differ")
    lines.append(f"rule 6 (same ids as the baseline in every run): {'PASS' if same else 'FAIL'}"); ok_all &= same
    for rule, (case, bar, text) in BARS.items():
        if case not in bs or case not in cs: lines.append(f"rule {rule}: {case} missing: FAIL"); ok_all = False; continue
        b, c = bs[case]["median"], cs[case]["median"]
        ok = c <= bar(b)
        lines.append(f"rule {rule} {case}: baseline {b:.4g}, now {c:.4g} ({c / b:.2f}x), bar {text}: {'PASS' if ok else 'FAIL'}")
        ok_all &= ok
    for case in NO_SLOWER:
        b, c = bs[case]["median"], cs[case]["median"]
        ok = c <= b * 1.01
        lines.append(f"rule 13 {case}: baseline {b:.4g}, now {c:.4g} ({c / b:.2f}x), bar no slower (+1%): {'PASS' if ok else 'FAIL'}")
        ok_all &= ok
    mem_ok = True
    for case in cases:
        b, c = bs[case]["peak_device_mib"], cs[case]["peak_device_mib"]
        ok = c <= b + 16
        mem_ok &= ok
        lines.append(f"  memory {case}: baseline {b} MiB, now {c} MiB{'' if ok else ' (OVER)'}")
    lines.append(f"rule 14 (peak memory no higher, +16 MiB): {'PASS' if mem_ok else 'FAIL'}"); ok_all &= mem_ok
    lines.append(f"SPEED RULES {'PASS' if ok_all else 'FAIL'}")
    text = "\n".join(lines)
    print(text)
    if a.output: open(a.output, "w").write(text + "\n")
    sys.exit(0 if ok_all else 1)


main()
