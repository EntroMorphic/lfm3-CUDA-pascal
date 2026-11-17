#!/usr/bin/env python3
"""Mutation test of native/include/lfm/speech_normalize.hpp: 18 planted bugs, one per rule or helper; each must change
the output on the equality corpus of tests/test_speech_normalize.py (experiment sentences, real text, fuzz).
Usage: mutate_speech_normalize.py REPO_ROOT SCRATCH_DIR
"""
import json, subprocess, sys, shutil, os
from pathlib import Path
ROOT = Path(sys.argv[1]); S = Path(sys.argv[2])
sys.path.insert(0, str(ROOT / "tests")); sys.path.insert(0, str(ROOT / "tools"))
import test_speech_normalize as t
from speech_normalize import normalize
texts = [c["text"] for s in ("dev","heldout") for c in json.load(open(ROOT/f"validation/speech-normalize-v1/cases-{s}.json"))["cases"]]
texts += t.corpus_lines()[:40000] + t.fuzz(11, 30000)
want = [normalize(x) for x in texts]
(S/"in.jsonl").write_text("".join(json.dumps(x)+"\n" for x in texts), encoding="utf-8")
H = ROOT/"native/include/lfm/speech_normalize.hpp"; src = H.read_text()
M = [
 ("phone oh", 'spell_digits(x.s.substr(a, 3), "oh")', 'spell_digits(x.s.substr(a, 3), "zero")'),
 ("money cents", "cents == 1 ? units[cur][2]", "cents == 2 ? units[cur][2]"),
 ("date month index", "kMonths[small(x.s.substr(i, m)) - 1]", "kMonths[small(x.s.substr(i, m)) % 12]"),
 ("clock minutes", "(mm >= 10 ? cardinal(long(mm))", "(mm > 10 ? cardinal(long(mm))"),
 ("meridiem capitals", "a == 'a' || a == 'A' || a == 'p' || a == 'P'", "a == 'a' || a == 'p' || a == 'P'"),
 ("degree singular", '(value == "1" ? "" : "s")', '(value == "2" ? "" : "s")'),
 ("percent word", '" percent"', '" per cent"'),
 ("address digits", "de - i > 5", "de - i > 6"),
 ("unit period", "(dot && sentence_end(x, end, true) ? \".\" : \"\")", "(dot && sentence_end(x, end, false) ? \".\" : \"\")"),
 ("fraction quarter", "else if (equals(d, 4))", "else if (equals(d, 3))"),
 ("range years", "in_range(p, 1000, 2099) && in_range(q, 1000, 2099)", "in_range(p, 1100, 2099) && in_range(q, 1000, 2099)"),
 ("digit strings", "s.size() >= 7", "s.size() >= 8"),
 ("year cue", '"in", "since", "by"', '"in", "by"'),
 ("saint/street", 'name_follows() ? "Saint"', '!name_follows() ? "Saint"'),
 ("collapse spaces", "if (j - i < 2) return 0;", "if (j - i < 3) return 0;"),
 ("pairs oh", "(lo >= 10 ? cardinal(lo)", "(lo > 10 ? cardinal(lo)"),
 ("ordinal twelfth", '{"twelve", "twelfth"}', '{"twelve", "twelveth"}'),
 ("integer lookbehind", 'if (!end && not_after(x, i, ".,", true) && clear(de))', 'if (!end && not_after(x, i, ".", true) && clear(de))'),
]
caught = 0
for name, a, b in M:
    assert src.count(a) == 1, name
    inc = S / "inc" / "lfm"; shutil.rmtree(S/"inc", ignore_errors=True); shutil.copytree(ROOT/"native/include/lfm", inc)
    (inc/"speech_normalize.hpp").write_text(src.replace(a, b))
    subprocess.run(["g++","-std=c++20","-O2","-I",str(S/"inc"),str(ROOT/"native/tests/speech_normalize_tool.cpp"),"-o",str(S/"tool")], check=True)
    subprocess.run([str(S/"tool"), str(S/"in.jsonl"), str(S/"out.jsonl")], check=True, capture_output=True)
    got = [json.loads(l) for l in (S/"out.jsonl").read_text(encoding="utf-8").splitlines()]
    diff = sum(g != w for g, w in zip(got, want))
    caught += diff > 0
    print(f"{'CAUGHT' if diff else 'MISSED'} {name}: {diff} differ")
print(f"{caught} of {len(M)} caught")
