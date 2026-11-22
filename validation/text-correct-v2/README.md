# text-correct-v2

Phase 1 of [PLAN.md](../../PLAN.md) again, after the owner's two decisions of 2026-11-18 (PLAN decisions 5 and 6):

- **Decision 5, the prefill kernel.** Prompt position 0 is computed with the tensor-core arithmetic, every other
  row with the plain kernel (`native/include/lfm/gemm.cuh`). Before this round that existed only as an experiment
  switch in the validation build (`LFM_TCM_ROWS=1`, [prefill-arithmetic-v1](../prefill-arithmetic-v1/)); here it is
  the engine.
- **Decision 6, the bar.** `tools/pascal/bar_amended.py`: a length fails only if the engine is "worse" than HF SDPA
  and "worse" than the FP32 model, by `ppl_compare`'s own statistic. `ppl_compare` is still run, unchanged, and
  reported first.

Build under test: `text-v6` (the tag after `text-v5`, the last plain-kernel build). [text-correct-v1](../text-correct-v1/) is the record of the plain kernel and is not
changed.

## What is already known (so that the rules are read with it)

- The experiment hybrid passed the original bar at 256 and 257. It was not run at 4,096 or 32,768.
- The amended rule was written after the data that motivated it. Applied to files already on hand it gives: plain
  kernel PASS (257: worse than SDPA, not worse than FP32); experiment hybrid PASS; round-once [BOS] row FAIL at 256
  and 257. Those are not results of this round.

## Rules (committed before `text-v6` has run on the card)

1. **Build.** As text-correct-v1 rule 1.
2. **Tokens, determinism, reset.** The four forward fixtures: the Thor pins' ids, two runs identical, reset passing;
   with `LFM3_COMPACT_WINDOWS` unset and set to 1.
3. **Identity with the experiment.** `text-v6`'s NLL files at 256 and 257 are byte-identical to the experiment
   hybrid's: `c47912b0…` and `75430bb6…`. If not, the engine is not computing what was decided, and nothing below
   counts.
4. **Position 0 is the Thor engine's.** At 256, 257, 4,096 and 32,768 the NLL at position 1 (computed from row 0
   alone) equals the Thor engine's (`nll-native-gemm-L.f32`) value for value in every window.
5. **Perplexity, original bar.** `tools/ppl_compare.py`, unchanged, against HF SDPA at all four lengths, with
   `text-v6`'s own files at every length. No length "worse".
6. **Perplexity, amended bar.** `tools/pascal/bar_amended.py` at all four lengths: PASS.
7. **Fidelity.** `tools/ppl_fidelity.py`, unchanged: PASS at all four lengths.
8. **Parity gate.** `tools/pascal/gate_pins.sh` on this tree: 7,826 of 7,826.
9. **Sanitizer.** `compute-sanitizer --tool memcheck` on the four fixtures: 0 errors, ids equal.
10. **Memory and time.** Peak device memory and wall time per length, next to the plain kernel's. Recorded, not
    judged; a length that does not fit is a FAIL of that length.
11. **Nothing else changes.** The host suite's result is the one recorded (2,397 passed, 5 skipped, 1 not runnable
    here), plus any test added.

Phase 1 passes if rules 1 to 9 and 11 pass.

## Results (2026-11-18)

Tree `48502422…`, commit `c791102` (the job: `job.sh`, its log: `records/job.log`). Build `text-v6`:
`liquid-lfm3` `7464f580…`, `lfm3-score` `a27d18ec…`. NLL files, their metadata and the three reports in `records/`.

| Rule | Result |
|---|---|
| 1. Build | **PASS**: 0 warnings; C and C++ runtime only |
| 2. Tokens, determinism, reset | **PASS**: the Thor pins' ids on all four fixtures, two runs identical, reset ok; with compact windows unset. (Red-team-2 C1: this job never ran the `LFM3_COMPACT_WINDOWS=1` leg, which the rule names; red-team-2's verification ran it on `text-v6`'s own binary, hash checked: FIXTURES PASS, [red-team-2](../red-team-2/).) |
| 3. Identity with the experiment | **PASS**: 256 `c47912b0…`, 257 `75430bb6…`, byte-identical to the experiment hybrid's files |
| 4. Position 0 is the Thor engine's | **PASS**: the NLL at position 1 equals the Thor engine's value for value in 1,160 of 1,160, 1,155 of 1,155, 72 of 72 and 9 of 9 windows |
| 5. Perplexity, original bar | **PASS**: "equal" at all four lengths |
| 6. Perplexity, amended bar | **PASS** |
| 7. Fidelity | **PASS**: closer to the FP32 model than SDPA at all four lengths |
| 8. Parity gate | **PASS**: 7,826 of 7,826 (validation build `63dfd52f…` of this tree) |
| 9. Sanitizer | **PASS**: 0 errors on all four fixtures, ids equal |
| 10. Memory and time | recorded below |
| 11. Nothing else changes | **PASS**: 2,397 passed, 5 skipped; the one test that needs Transformers 5.x not run, as before |

**Phase 1 passes.**

### Rule 5, `tools/ppl_compare.py` unchanged, against HF SDPA

| Length | Windows | PPL SDPA | PPL here | dNLL | 95% CI | Verdict | Plain kernel (text-correct-v1) |
|---|---:|---:|---:|---:|---|---|---|
| 256 | 1,160 | 26.5821 | 26.5798 | −8.68e-05 | [−2.14e-04, +4.17e-05] | equal | +6.89e-05, equal |
| 257 | 1,155 | 26.5045 | 26.5055 | +3.87e-05 | [−9.09e-05, +1.67e-04] | equal | +2.27e-04, **worse** |
| 4,096 | 72 | 13.0596 | 13.0583 | −1.05e-04 | [−2.49e-04, +3.77e-05] | equal | −5.29e-05, equal |
| 32,768 | 9 | 12.5771 | 12.5765 | −5.05e-05 | [−2.51e-04, +1.61e-04] | equal | −1.96e-04, better |

### Rule 6, `tools/pascal/bar_amended.py`

| Length | Against SDPA | Against FP32 | FP32 against SDPA (information) | Reading |
|---|---|---|---|---|
| 256 | −8.68e-05, equal | −1.06e-04 [−2.08e-04, −7.8e-07], equal | +1.92e-05, equal | pass |
| 257 | +3.87e-05, equal | −1.14e-04 [−2.20e-04, −7.3e-06], equal | +1.53e-04, worse | pass |
| 4,096 | −1.05e-04, equal | +1.98e-04 [+8.85e-05, +3.01e-04], worse | −3.02e-04, better | pass |
| 32,768 | −5.05e-05, equal | +3.72e-04 [+1.99e-04, +5.78e-04], worse | −4.23e-04, better | pass |

No length needed the amendment: the original bar passes by itself. At 4,096 and 32,768 the engine's mean NLL is
above the FP32 model's, as HF SDPA's is by more.

### Rule 7, `tools/ppl_fidelity.py`, mean |NLL − NLL_FP32| per token

| Length | This engine | HF SDPA | Difference, 95% CI |
|---|---:|---:|---|
| 256 | 1.683e-02 | 1.940e-02 | [−2.66e-03, −2.49e-03] |
| 257 | 1.675e-02 | 1.948e-02 | [−2.82e-03, −2.64e-03] |
| 4,096 | 1.498e-02 | 1.753e-02 | [−2.65e-03, −2.45e-03] |
| 32,768 | 1.581e-02 | 1.818e-02 | [−2.43e-03, −2.28e-03] |

### Against the Thor engine (information)

Mean NLL minus the Thor engine's: +6.7e-05, −4.5e-05, −3.2e-05, +2.2e-05 at the four lengths. Tokens with the same
NLL value: 0.7% to 1.0% (position 1 in every window, and chance elsewhere).

### Rule 10

| Length | Mean NLL | Perplexity | Wall | Peak device memory | NLL sha256 |
|---|---:|---:|---:|---:|---|
| 256 | 3.280150656 | 26.579777 | 932 s | 4,409 MiB | `c47912b0…` |
| 257 | 3.277352868 | 26.505516 | 1,191 s | 4,415 MiB | `75430bb6…` |
| 4,096 | 2.569422345 | 13.058279 | 791 s | 5,029 MiB | `3a00cb32…` |
| 32,768 | 2.531826883 | 12.576461 | 907 s | 6,577 MiB | `86582837…` |

**The cost of position 0.** The same tree built with `-DLFM_GEMM_FIRST_ROW=0` (`text-v6-plain`), fixtures, warm:

| Fixture | Plain kernel alone | Engine |
|---|---:|---:|
| 34 tokens: prefill | 0.236 s | 0.394 s |
| 31 tokens | 0.263 s | 0.423 s |
| 257 tokens | 0.763 s | 0.915 s |

About 0.155 s per prefill whatever its length, for one row. The model kernel works in tiles of 8 rows and here
computes 7 rows of padding with it. Not changed in this round, so that every result above is of one binary. For
phase 2: skip the padded rows (no output bit changes; the check is these NLL files byte for byte), or compute the
row once per first token and keep it.

### Not covered

- The other modalities' GEMMs also go through this kernel and would give their row 0 the tensor-core arithmetic.
  Whether that is wanted there is for phases 4 and 5.
- A prefill of 8 rows or fewer uses the decode kernels for every row, position 0 included, as the Thor engine does.
