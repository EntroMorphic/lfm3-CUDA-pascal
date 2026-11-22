# Validation rounds

Each round's README states what it is for, its rules (committed before its data) and its results; its job scripts
and records sit beside it. [docs/validation.md](../docs/validation.md) describes the method. In order:

| Round | Phase | What |
|---|---|---|
| [phase0-foundations-v1](phase0-foundations-v1/) | 0 | The card, container, toolchain, checkpoint and host tests the port stands on |
| [text-correct-v1](text-correct-v1/) | 1 | The text backbone with the simplest correct kernels, against the perplexity bar |
| [tensor-core-model-v1](tensor-core-model-v1/) | 1 | An exact CPU model of Blackwell's tensor-core BF16 arithmetic (the parity build's basis) |
| [prefill-arithmetic-v1](prefill-arithmetic-v1/) | 1 | Where the plain prefill kernel's difference from the Thor engine comes from |
| [round-once-v1](round-once-v1/) | 1 | Exact sums rounded once: a third arithmetic, measured |
| [decode-atoms-v1](decode-atoms-v1/) | 1 | What a decode step costs, kernel by kernel |
| [prefill-speed-v1](prefill-speed-v1/) | 1 | The prefill GEMM re-tiled |
| [text-correct-v2](text-correct-v2/) | 1 | Phase 1's bar after decisions 5 and 6 (passed) |
| [long-decode-v1](long-decode-v1/) | 1 | Decode above 4,096 tokens of capacity |
| [memory-v1](memory-v1/) | 1 | Compact window caches and MLP blocks: less memory per token, the same outputs |
| [red-team-1](red-team-1/) | 1 | Review of phase 1's code, end to end |
| [speed-baseline-v1](speed-baseline-v1/) | 2 | Phase 2's starting point and the card's ceilings |
| [speed-v1](speed-v1/) | 2 | Phase 2's bar (closed with rule 12 failed) |
| [serve-v1](serve-v1/) | 3 | Text serving: the reference's server checks on both builds (passed) |
| [speech-v1](speech-v1/) | 4 | Speech out and audio in (13 of 14 rules; the owner's listening pending) |
| [red-team-2](red-team-2/) | 4 | Review of the code of phases 1 to 4, end to end (passed) |

Imported from the Thor engine (inputs and records its rules and tools use, unchanged):
[speech-e2e-v1](speech-e2e-v1/) (the speech prompts), [speech-normalize-v1](speech-normalize-v1/) (text normalization),
[g4-gates-v2](g4-gates-v2/) (the parity gate's pins), [cublas-modes-13.1-vs-13.10-v2](cublas-modes-13.1-vs-13.10-v2/)
and [initcheck-version-shapes-v2](initcheck-version-shapes-v2/) (host-test inputs).
