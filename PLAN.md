# Plan

## Goal

The smallest single-binary engine that runs LFM3-Omni-2B on an NVIDIA GTX 1080 (Pascal, `sm_61`, 8 GB) at the same
or better quality than the Hugging Face reference. Text first, then the other modalities.

This is the goal of the Thor engine ([liquid-CUDA](https://github.com/EntroMorphic), imported here at commit
`e2f8b7f`) with a new target. Its discipline carries over: rules before data, measure everything, earlier results
must not change.

## Decisions (project owner, 2026-11-17)

1. **Quality reference.** The Thor project's pinned Hugging Face references and FP32 baselines, with the Thor
   engine's own outputs as a second comparator. No Hugging Face stack is stood up on the GTX 1080.
2. **Code.** Copied from the reference at a recorded commit, not shared. The engines diverge in kernels and are
   validated separately.
3. **Context length.** The 32,768-token bar is kept unless it causes problems; dealt with when it does.

## The constraint

| | MiB |
|---|---:|
| All weights (BF16), with the audio codec | 4,273 |
| Language model alone | 3,297 |
| Free on the card (measured in the container, idle) | 7,996 |

Memory is a budget tracked from phase 0 on.

## Phases

A phase has one goal and one exit gate. A phase does not start until the gate before it has passed. Every phase ends
with red-team, remediate, verify, commit.

| # | Phase | What is built | Exit gate |
|---|---|---|---|
| 0 | Foundations | Repository; one command that syncs, builds in the container, runs, and returns exit code and hashes; checkpoint on the target; host tests running locally | Sync verified by SHA-256 on both ends; host tests pass; a measured memory budget |
| 1 | Text, correct | The backbone with the simplest correct kernels (FP32 over BF16 weights, no tensor cores); prefill and single-token decode | Greedy tokens identical on the reference's fixtures; perplexity bar passed at 256, 257 and 4,096 tokens (32,768: decision 3) |
| 2 | Text, fast | A GEMM tuned for Pascal; decode-loop throughput | A speed bar set before measuring; phase 1 outputs unchanged |
| 3 | Text serving | The HTTP server, sessions, batching, admission limits sized to 8 GB | The reference's server checks pass; requests that would exceed memory are refused, never crash |
| 4 | Speech | Depth decoder and codec (out), audio encoder (in) | The reference's speech rounds: word error rate against HF, and listening |
| 5 | Images | Vision encoder and projector | The reference's image round: encoder accuracy, POPE, MMStar |
| 6 | Whole engine | All modalities resident in one binary within 8 GB; soak and red-team | Every earlier gate re-run and unchanged, under sustained mixed load |

## Guardrails

- **No speed work before phase 1 passes.** A slow correct engine is the baseline everything else is measured against.
- **Port, don't redesign.** Host-side code (tokenizer, template, JSON, HTTP, image and audio front ends) stays as
  imported; GPU kernels are rewritten.
- **Nothing beyond the reference.** No video, no new features. Ideas go to the parking list below and are not acted
  on.

## Status

| Phase | State |
|---|---|
| 0 | passed 2026-11-17 ([phase0-foundations-v1](validation/phase0-foundations-v1/)) |
| 1 | not started |

## Parking list

- **LFM3-1.7B at its 262,144-token context** (owner's question, 2026-11-17). Same backbone as Omni-2B's language
  model (3,297 MiB). By calculation it fits in 8 GB only with all three of: keys and values kept for the 6 full
  layers alone (12 KiB per token: 3,072 MiB), row-tiled prefill scratch, and prefill in chunks through the cache
  (the engine prefills only into an empty cache today). Nothing measured. Written up in
  [docs/long-context-memory.md](docs/long-context-memory.md).
- **A software model of Blackwell's tensor-core arithmetic** (owner's question, 2026-11-17). The Thor prefill
  kernel's `mma.sync` (BF16 in, FP32 accumulate) can be computed on Pascal in plain FP32; whether it can be
  reproduced bit for bit depends on the tensor core's undocumented summation order and rounding. If it can, the
  Thor engine's pinned trace hashes (`g4-gates-v2/pins/`, already in the Thor repository) become exact gates for this
  engine, and candidate models can be tested against those hashes on the GTX 1080 without Thor. Speed is not
  recoverable this way. Nothing tried.
