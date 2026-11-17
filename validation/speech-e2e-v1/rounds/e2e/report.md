seeds: [1, 2, 3]

| case | mode | native WER (mean over seeds) | HF WER (mean [worst]) | C1 | native - HF (95% CI), Mann-Whitney p | native frames mean [min-max] | HF frames | ratio | ended by EOS (native, HF) |
|---|---|---|---|---|---|---|---|---|---|
| tts-fox-female | audio | 0.000 | 0.000 [0.000] | ok | +0.000 (+0.000, +0.000), 1.000 | 43 [40-44] | 45 [42-48] | 0.96 | 3/3, 3/3 |
| tts-fox-male | audio | 0.000 | 0.000 [0.000] | ok | +0.000 (+0.000, +0.000), 1.000 | 44 [40-48] | 46 [44-50] | 0.95 | 3/3, 3/3 |
| tts-museum-female | audio | 0.028 | 0.028 [0.042] | ok | +0.000 (-0.028, +0.028), 1.000 | 152 [150-154] | 151 [146-158] | 1.01 | 3/3, 3/3 |
| tts-museum-male | audio | 0.028 | 0.028 [0.042] | ok | +0.000 (-0.028, +0.028), 1.000 | 155 [150-162] | 153 [150-154] | 1.02 | 3/3, 3/3 |
| tts-numbers-female | audio | 0.263 | 0.158 [0.158] | FLAG | +0.105 (+0.000, +0.316), 0.505 | 117 [112-124] | 117 [108-124] | 1.00 | 3/3, 3/3 |
| tts-numbers-male | audio | 0.158 | 0.175 [0.211] | ok | -0.018 (-0.053, +0.000), 0.505 | 121 [116-126] | 116 [112-118] | 1.05 | 3/3, 3/3 |
| tts-long-male | audio | 0.887 | 0.158 [0.268] | FLAG | +0.729 (-0.003, +1.801), 0.268 | 767 [493-1181] | 442 [399-492] | 1.74 | 3/3, 3/3 |
| il-reflective | interleaved | 0.000 | 0.000 [0.000] | ok | +0.000 (+0.000, +0.000), 1.000 | 322 [252-376] | 325 [259-397] | 0.99 | 3/3, 3/3 |
| il-study | interleaved | 0.000 | 0.000 [0.000] | ok | +0.000 (+0.000, +0.000), 1.000 | 266 [228-299] | 276 [267-293] | 0.96 | 3/3, 3/3 |
| il-fridge | interleaved | 0.000 | 0.000 [0.000] | ok | +0.000 (+0.000, +0.000), 1.000 | 324 [317-331] | 285 [218-325] | 1.14 | 3/3, 3/3 |

C1 pooled WER per seed: native {1: 0.5, 2: 0.06801007556675064, 3: 0.14617169373549885}, HF {1: 0.04895104895104895, 2: 0.0812807881773399, 3: 0.0410958904109589}
C1 per case (native mean WER > HF's worst seed): flagged ['tts-numbers-female', 'tts-long-male']
C2 mean frames ratio native/HF over cases: 1.080
real-time factor (audio seconds per second, mean over runs): native 3.24 decode loop, 3.20 end to end; HF 1.91 end to end
