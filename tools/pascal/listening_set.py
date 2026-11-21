"""speech-v1 rule 9 (C3): the owner's blind listening set, the shipping build against HF, seed 1 of the ten cases.

HF's clip of each case is the HF side of the reference's listening-c3-v1 (by its key; tts-long-male: HF BF16 unsplit).
The shipping build's clip is its seed-1 run (tts-long-male split by sentence, as the reference ships long text and as
listening-c3-v1's engine side was made). For each case, which of A and B is the shipping build is drawn from the OS's
entropy (secrets). Writes OUT/CASE/{A,B}.wav, OUT/index.md (each case's mode, system and prompt only: no durations and
no texts, since HF's clips are byte-identical to the HF side of listening-c3-v1, whose index printed both and whose key
the owner has opened; red-team review) and the key beside the set, OUT.key.json (who is A, every file's SHA-256, the
sources' SHA-256), to be opened only after every verdict. Refuses to overwrite.
    python3 tools/pascal/listening_set.py REFERENCE_OUTPUTS SHIPPING_DIR PROMPTS_JSON OUT
SHIPPING_DIR holds the shipping build's CASE-s1.wav and sources.sha256, `sha256sum` of the build's runs with their
paths (speech_refs.sh's layout): every clip must equal its source, tts-long-male's the split run's
(speech-chunk-native-v1/native/), every other case's speech-e2e-v1/native/'s.
"""
import hashlib
import json
import secrets
import shutil
import sys
from pathlib import Path

ref, ship, prompts, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
key_path = out.with_name(out.name + ".key.json")
if out.exists() or key_path.exists(): sys.exit(f"{out} or its key exists")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


sources = {}
for line in (ship / "sources.sha256").read_text().splitlines():
    h, path = line.split(maxsplit=1); sources[Path(path).name, Path(path).parent.parent.name] = h
ref_key = json.loads((ref / "listening-c3-v1.key.json").read_text())
assert ref_key["seed"] == 1
P = json.loads(prompts.read_text())
cases = {c["case"]: c for c in P["cases"]}
order = [c["case"] for c in P["cases"]]
assert sorted(order) == sorted(ref_key["key"]), (order, list(ref_key["key"]))
out.mkdir(parents=True)
key, lines = dict(seed=1, set="listening-speech-v1", made_by="tools/pascal/listening_set.py", cases={}), [
    "# Phase 4 listening set (blind)", "",
    "Seed 1 of every case. Each case has A.wav and B.wav: one is this engine's shipping build on the GTX 1080, one is "
    "HF (the reference's own HF clip, the same file as in listening-c3-v1: please do not consult that set, its index "
    "or its key while listening). The key is kept separately; please listen to every case and give a verdict before "
    "opening it.", ""]
for case in order:
    c = cases[case]
    hf_side = next(s for s, who in ref_key["key"][case].items() if who == "hf")
    hf_wav, ship_wav = ref / "listening-c3-v1" / case / f"{hf_side}.wav", ship / f"{case}-s1.wav"
    want_dir = "speech-chunk-native-v1" if case == "tts-long-male" else "speech-e2e-v1"
    if sources.get((f"{case}-s1.wav", want_dir)) != sha(ship_wav): sys.exit(f"{case}: not the {want_dir} run")
    shipping_is_a = secrets.randbits(1) == 1
    sides = dict(A=ship_wav, B=hf_wav) if shipping_is_a else dict(A=hf_wav, B=ship_wav)
    (out / case).mkdir()
    for s, src in sides.items(): shutil.copyfile(src, out / case / f"{s}.wav")
    system = P["systems"][c["system"]]
    lines += [f"## {case} ({c['mode']}, {system})", "", f"Prompt: {c['user']}", "", f"- {case}/A.wav", f"- {case}/B.wav", ""]
    key["cases"][case] = dict(A="shipping" if shipping_is_a else "hf", B="hf" if shipping_is_a else "shipping",
                              A_sha256=sha(out / case / "A.wav"), B_sha256=sha(out / case / "B.wav"),
                              shipping_source_sha256=sha(ship_wav), hf_source=f"listening-c3-v1/{case}/{hf_side}.wav",
                              hf_source_sha256=sha(hf_wav))
(out / "index.md").write_text("\n".join(lines))
key_path.write_text(json.dumps(key, indent=1) + "\n")
print(f"{len(order)} cases in {out}; key in {key_path} (not to be opened before the verdicts)")
