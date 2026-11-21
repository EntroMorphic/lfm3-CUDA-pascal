#!/usr/bin/env python3
"""speech-v1 rules 7 (C1, intelligibility; C4 for the interleaved cases) and 8 (C2, endings and length), the shipping
build against HF's committed records, as tools/lfm3_speech_report.py judges them. That tool reads both engines' case
directories; HF's are on the reference machine, so here HF's side is its committed transcripts and audio lengths, and
tts-long-male is the split run (speech-v1 rule 7) on both sides, which the tool's single prompt set cannot express.

  c1c2 PROMPTS.json ASR_E2E.json ASR_SPLIT.json E2E_CASES SPLIT_NATIVE REFERENCE OUT.json
    ASR_E2E.json, ASR_SPLIT.json: tools/lfm3_asr.py over the build's prompts.json cases and its split lighthouse runs
    (speech_refs.sh's speech-e2e-v1/native-cases and speech-chunk-native-v1/native-cases); E2E_CASES their case
    directories (summary.json: slots); SPLIT_NATIVE the split runs' lfm3-speech json (chunks[].ended_by_eos);
    REFERENCE validation/speech-v1/inputs/reference.
    C1, per case over the seeds of PROMPTS.json: flagged iff the shipping build's mean WER exceeds HF's worst seed.
    C2: every utterance ended by the model's EOS (a single request: slots under max_new_tokens, the report tool's
    test; a split one: every chunk ended_by_eos), and the ratio of mean audio length to HF's within 0.85 to 1.15. The
    length is audio seconds on both sides (frames x 1920 samples at 24 kHz: the same ratio as frames; HF's records
    carry seconds, not frames).
    Exit 0 if rule 8 passes and no case is flagged; 3 if rule 8 passes and a case is flagged (rule 7 escalates);
    1 if rule 8 fails.
  escalate CASE ASR[,ASR...] REFERENCE OUT.json [PARITY_ASR[,PARITY_ASR...]]
    A flagged case at seeds 1 to 15 against HF's committed 15 seeds where they exist (tts-numbers-female,
    tts-long-male, il-fridge), otherwise against the parity build's 15 (rule 7; PARITY_ASR required then).
"""
import json
import statistics as st
import sys
from pathlib import Path

SPLIT_CASE = "tts-long-male"


def results(path):
    return {r["case"]: r for r in json.loads(Path(path).read_text())["results"]}


def c1c2(prompts, asr_e2e, asr_split, e2e_cases, split_native, reference, out_json):
    spec = json.loads(Path(prompts).read_text())
    ref = Path(reference)
    ship = {"e2e": results(asr_e2e), "split": results(asr_split)}
    hf = {"e2e": results(ref / "speech-e2e-v1/rounds/e2e/asr-hf.json"),
          "split": results(ref / "speech-long-form-v1/asr/hf-bf16-split.json")}
    rows, flagged, rule8 = {}, [], True
    print("| case | mode | shipping WER mean [per seed] | HF WER mean [worst] | C1 | shipping/HF audio length | ended by EOS |")
    print("|---|---|---|---|---|---|---|")
    for c in spec["cases"]:
        kind = "split" if c["case"] == SPLIT_CASE else "e2e"
        names = [f"{c['case']}-s{s}" for s in spec["seeds"]]
        s_runs, h_runs = [ship[kind][n] for n in names], [hf[kind][n] for n in names]
        if kind == "split":
            eos = [all(k["ended_by_eos"] for k in json.loads((Path(split_native) / f"{n}.json").read_text())["chunks"])
                   for n in names]
        else:
            eos = [json.loads((Path(e2e_cases) / n / "summary.json").read_text())["slots"] < c["max_new_tokens"]
                   for n in names]
        s_wer, h_wer = [r["wer"] for r in s_runs], [r["wer"] for r in h_runs]
        ratio = st.mean(r["audio_seconds"] for r in s_runs) / st.mean(r["audio_seconds"] for r in h_runs)
        row = dict(mode=c["mode"], kind=kind, shipping_wer=s_wer, shipping_mean=st.mean(s_wer), hf_wer=h_wer,
                   hf_worst=max(h_wer), c1_flag=st.mean(s_wer) > max(h_wer),
                   shipping_seconds=[r["audio_seconds"] for r in s_runs], hf_seconds=[r["audio_seconds"] for r in h_runs],
                   length_ratio=ratio, length_ok=0.85 <= ratio <= 1.15, ended_by_eos=eos)
        rows[c["case"]] = row
        flagged += [c["case"]] if row["c1_flag"] else []
        rule8 &= row["length_ok"] and all(eos)
        print(f"| {c['case']} | {c['mode']}{' (split)' if kind == 'split' else ''} | {row['shipping_mean']:.3f} "
              f"[{', '.join(f'{w:.3f}' for w in s_wer)}] | {st.mean(h_wer):.3f} [{max(h_wer):.3f}] | "
              f"{'FLAG' if row['c1_flag'] else 'ok'} | {ratio:.3f}{'' if row['length_ok'] else ' OUT'} | "
              f"{sum(eos)}/{len(eos)} |")
    Path(out_json).write_text(json.dumps(dict(seeds=spec["seeds"], cases=rows, c1_flagged=flagged, rule8_pass=rule8),
                                         indent=1) + "\n")
    print(f"rule 7 (C1): flagged {flagged or 'none'}{': escalate to seeds 1-15' if flagged else ': PASS'}")
    print(f"rule 8 (C2): {'PASS' if rule8 else 'FAIL'}")
    sys.exit(1 if not rule8 else 3 if flagged else 0)


HF_15 = ("tts-numbers-female", SPLIT_CASE, "il-fridge")  # HF's committed seeds 1-15 (rounds/long, hf-bf16-split, rounds/fridge)


def escalate(case, ship_asrs, reference, out_json, parity_asrs=None):
    """A flagged case at seeds 1 to 15: the shipping build's 15 WERs (the comma-separated ASR records, seeds 1-3 and
    4-15) against HF's committed 15 (tts-long-male split: hf-bf16-split.json; otherwise rounds/e2e seeds 1-3 and
    rounds/long or rounds/fridge seeds 4-15). Fails iff the shipping mean is higher and the two-sided Mann-Whitney
    p < 0.05 (scipy.stats.mannwhitneyu, as tools/lfm3_speech_report.py). Exit 0 pass, 1 fail."""
    from scipy.stats import mannwhitneyu
    ref = Path(reference)
    names = [f"{case}-s{s}" for s in range(1, 16)]
    ship = {}
    for p in ship_asrs.split(","):
        ship.update(results(p))
    against = "hf"
    if case not in HF_15:  # rule 7: against the parity build's 15 seeds (by rule 3, the Thor engine's)
        if not parity_asrs: sys.exit(f"{case}: HF's 15 seeds are not committed; the parity build's ASR records are required")
        hf, against = {}, "parity"
        for p in parity_asrs.split(","): hf.update(results(p))
    elif case == SPLIT_CASE:
        hf = results(ref / "speech-long-form-v1/asr/hf-bf16-split.json")
    else:
        hf = results(ref / "speech-e2e-v1/rounds/e2e/asr-hf.json")
        for r in ("long", "fridge"):
            hf.update(results(ref / f"speech-e2e-v1/rounds/{r}/asr-hf.json"))
    s, h = [ship[n]["wer"] for n in names], [hf[n]["wer"] for n in names]
    p = 1.0 if s == h or len(set(s + h)) == 1 else float(mannwhitneyu(s, h).pvalue)
    fail = st.mean(s) > st.mean(h) and p < 0.05
    Path(out_json).write_text(json.dumps(dict(case=case, against=against, seeds=list(range(1, 16)), shipping_wer=s, hf_wer=h,
                                              shipping_mean=st.mean(s), hf_mean=st.mean(h), mann_whitney_p=p,
                                              pass_=not fail), indent=1) + "\n")
    print(f"{case}: shipping mean {st.mean(s):.4f} [{', '.join(f'{w:.3f}' for w in s)}]")
    print(f"{case}: {'HF' if against == 'hf' else 'parity build'} mean {st.mean(h):.4f} [{', '.join(f'{w:.3f}' for w in h)}]")
    print(f"rule 7 escalation {case}: Mann-Whitney p {p:.3f}: {'FAIL' if fail else 'PASS'}")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    if len(sys.argv) == 9 and sys.argv[1] == "c1c2":
        c1c2(*sys.argv[2:])
    elif len(sys.argv) in (6, 7) and sys.argv[1] == "escalate":
        escalate(*sys.argv[2:])
    else:
        sys.exit(__doc__)
