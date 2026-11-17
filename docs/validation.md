# How the engine is validated

The goal is an engine that is *at least as good as the reference* — Hugging Face Transformers running the same
checkpoint — on every capability it claims. "As good" is decided by evidence gathered under rules fixed in advance.

## A round

Each capability or change is a round: a directory `validation/<name>-vN/` whose `README.md` holds, in order,

1. **What it will be** — the scope, and what is out of scope.
2. **Decisions** — choices made with the project owner, dated and quoted.
3. **Rules** — numbered pass/fail rules, committed before the first native run. An amendment is dated and committed
   before the data it would affect.
4. **Results** — per rule, with the records that support them (compressed JSON, logs, scripts as run).
5. **Failures and what they taught** — kept, not erased.

The index of rounds is [validation/README.md](../validation/README.md).

## Kinds of rule

| Kind | Example | Standard |
|---|---|---|
| Exactness | tokenizer ids; decoded pixels; resize output | bit-identical to the reference |
| Accuracy | encoder features; logits | relative error to an FP32 reference no worse than HF BF16's, with per-token minimum cosine no worse |
| Task quality | perplexity; speech recognition WER; MMStar and POPE accuracy | paired comparison with HF on the same items; bootstrap lower bound above a pre-set margin |
| Coherency | generated replies; synthesized speech | read and listened to end to end by a person; listening is the final gate for speech |
| Robustness | malformed and hostile inputs; refusals | refused before any state changes, with a specific message; fuzzed |
| Sanitizers | memcheck, racecheck, synccheck, initcheck | 0 errors, outputs identical to the plain run |
| Regression | every earlier round's outputs | byte-identical |
| Speed | decode loop or end to end, labeled | reported against a pre-set bar on an idle machine; never traded for exactness |

## References

References are pinned: library versions, container digests, dataset revisions and file checksums are recorded with
each round. When the reference's own numerics differ between machines (for example torchvision's CPU kernels on x86 and
aarch64), the disagreement is measured and one machine is declared the reference before native runs.

## Where the engine departs from the reference

Departures are deliberate, decided with the project owner, and recorded with their reason — for example, the image
front end shows 16-bit PNGs at full precision and refuses PNGs whose checksums fail, where the reference shows damaged
pixels ([images-v1](../validation/images-v1/)).
