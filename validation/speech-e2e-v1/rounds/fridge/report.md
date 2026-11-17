seeds: [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

| case | mode | native WER (mean over seeds) | HF WER (mean [worst]) | C1 | native - HF (95% CI), Mann-Whitney p | native frames mean [min-max] | HF frames | ratio | ended by EOS (native, HF) |
|---|---|---|---|---|---|---|---|---|---|
| il-fridge | interleaved | 0.002 | 0.001 [0.014] | ok | +0.001 (-0.002, +0.005), 0.581 | 304 [244-391] | 296 [176-390] | 1.03 | 12/12, 12/12 |

C1 pooled WER per seed: native {4: 0.0, 5: 0.013888888888888888, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.014705882352941176, 13: 0.0, 14: 0.0, 15: 0.0}, HF {4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.014285714285714285, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0}
C1 per case (native mean WER > HF's worst seed): flagged none
C2 mean frames ratio native/HF over cases: 1.028
real-time factor (audio seconds per second, mean over runs): native 2.85 decode loop, 2.84 end to end; HF 2.08 end to end
