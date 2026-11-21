"""speech-v1 rule 9: the listening set's shipping clips are the final build's, case by case, without revealing A/B.

KEY is the listening set's key (tools/pascal/listening_set.py's OUT.key.json); FINAL is `sha256sum` of the final
shipping build's seed-1 clips as the rule run writes it (validation/speech-v1/job.sh: text/CASE-s1.wav, and
text-split/tts-long-male-s1.wav, the split run, which the set uses for that case). For each case of the key, its
shipping_source_sha256 must equal the final build's clip of that case (tts-long-male: the split one). Prints only the
count and the cases that differ; nothing of which side is which. FINAL is kept out of the repository until the owner's
verdicts are in (red-team-2 D4: with the clips' hashes public, hashing A.wav and B.wav names the shipping side).
    python3 tools/pascal/rule9_key_check.py KEY FINAL
"""
import json
import sys
from pathlib import Path

if len(sys.argv) != 3: sys.exit(__doc__)
key = json.loads(Path(sys.argv[1]).read_text())["cases"]
final = {}
for line in Path(sys.argv[2]).read_text().splitlines():
    h, path = line.split(maxsplit=1)
    p = Path(path)
    case = p.name.removesuffix("-s1.wav")
    split = p.parent.name == "text-split"
    if (case == "tts-long-male") == split: final[case] = h  # tts-long-male: only the split run; others: text/
missing = sorted(set(key) - set(final))
differ = sorted(c for c in key if c in final and key[c]["shipping_source_sha256"] != final[c])
equal = len(key) - len(missing) - len(differ)
print(f"rule 9: {equal} of {len(key)} cases' shipping clips equal the final build's"
      + (f"; missing from FINAL: {missing}" if missing else "") + (f"; differing: {differ}" if differ else ""))
sys.exit(0 if equal == len(key) == 10 else 1)
