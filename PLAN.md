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
4. **Parity gate** (2026-11-17, after [tensor-core-model-v1](validation/tensor-core-model-v1/); the owner: "We are
   going for parity so it makes sense."). Exact equality with the Thor engine is a standing gate: in the validation
   build (the prefill GEMM with Blackwell's tensor-core arithmetic), every trace file of the four forward fixtures
   must equal the Thor pins, 7,826 of 7,826 (`tools/pascal/gate_pins.sh`). It is run after every change to the
   engine, in every phase from here on. It does not replace the perplexity and fidelity rules, which judge the
   shipping build, whose prefill GEMM sums in a different order.

5. **The prefill kernel** (2026-11-18, after [prefill-arithmetic-v1](validation/prefill-arithmetic-v1/) and
   [round-once-v1](validation/round-once-v1/)). Prompt position 0 is computed with the tensor-core arithmetic, as
   the Thor engine computes it; every other prefill row with the plain kernel. That row is shared by every prompt,
   every later position attends to it, and it carries the whole measured difference between the two arithmetics.
   Not chosen: the tensor-core arithmetic everywhere (bit-identical to Thor, several times the prefill cost), the
   plain kernel alone, and exact sums rounded once (nearest the FP32 model at position 1, fails the bar).
6. **The perplexity bar, amended** (2026-11-18). The bar's windows all start with the same token, so one row is
   shared by every window and the bootstrap counts a single draw once per window; and its reference, HF SDPA in
   BF16, is itself "worse" than the FP32 model's run at 257 tokens by the bar's own rule. From here a length fails
   only if the engine is "worse" than HF SDPA and "worse" than the FP32 model (`tools/pascal/bar_amended.py`).
   `tools/ppl_compare.py` stays as imported and is still reported. A proposal for the Thor project, whose bar has
   the same property: [docs/proposal-perplexity-bar.md](docs/proposal-perplexity-bar.md).

7. **The decode bar and the memory clock** (2026-11-18). The card's memory clock stays where the driver puts it
   under compute load (4,513 MHz; the owner: "85% cap extends the life of the hardware"). The decode bar is 245 GB/s,
   the read rate measured at that clock: a decode step may take what it must read at 245 GB/s plus 1.5 ms
   ([speed-v1](validation/speed-v1/), rules 11 and 12, amended).

8. **Text serving** (2026-11-19, before phase 3's rules; [serve-v1](validation/serve-v1/)).
   - *Control:* the reference's server checks run on both builds. The parity build is held to Thor's pinned 86-token
     control unchanged; on the card it reproduces that control exactly, while the shipping build first differs at
     token 58 (decision 5). The shipping build is held to its own CLI's output on the same request: the checks'
     "accepted native control".
   - *Build:* a text-only switch in `lfm3_worker.cu` leaves out speech, the batch decoder and vision, whose kernels
     use Thor-only instructions. The HTTP layer and the host front ends stay as imported; phases 4 and 5 bring the
     rest back.
   - *Batching:* paired decode (`/generate_batch`) learns the compact window caches, so it works at the reference's
     limits (2 sessions, 32,768 tokens in total), held to every logit bit equal to serial decode.

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
  *Lifted by the owner on 2026-11-17* ("lift the guardrail on speed. I think we're close enough to make this call"),
  first for a faster exact prefill kernel. Phase 1's open rules stay open and are not waived by this.
- **Port, don't redesign.** Host-side code (tokenizer, template, JSON, HTTP, image and audio front ends) stays as
  imported; GPU kernels are rewritten.
- **Parity is checked, not assumed.** No engine change is committed as done until the parity gate (decision 4) has
  passed on the card with that tree.
- **Nothing beyond the reference.** No video, no new features. Ideas go to the parking list below and are not acted
  on.

## Status

| Phase | State |
|---|---|
| 0 | passed 2026-11-17 ([phase0-foundations-v1](validation/phase0-foundations-v1/)) |
| 1 | passed 2026-11-18 ([text-correct-v2](validation/text-correct-v2/)): the original bar and the amended one at all four lengths, with position 0 on the tensor-core arithmetic (decision 5). The plain kernel alone did not pass the original bar at 257 ([text-correct-v1](validation/text-correct-v1/)). Decode above 4,096 tokens of capacity, where the compact window buffers are the default: identical to the full-length layout and consistent with prefill ([long-decode-v1](validation/long-decode-v1/)) |
| Red team | passed 2026-11-18 ([red-team-1](validation/red-team-1/)): two rounds of independent review of everything written since the import; every finding fixed; recorded results unchanged byte for byte |
| 2 | closed 2026-11-19 by the owner with rule 12 failed ([speed-v1](validation/speed-v1/)): 13 of 14 rules pass. Prefill 3.4 times faster at 4,096 tokens, 3.1 at 32,768, 2.7 at 257, 2 at 32; decode 1% faster at 1,000 tokens (at the 245 GB/s bar, decision 7) and 16% at 31,000, where it misses its bar by 0.18 ms; outputs unchanged; no more memory. Carried: position 0 still costs about 150 ms per prefill |

## Parking list

- **LFM3-1.7B at its 262,144-token context** (owner's question, 2026-11-17). Same backbone as Omni-2B's language
  model (3,297 MiB). By calculation it fits in 8 GB only with all three of: keys and values kept for the 6 full
  layers alone (12 KiB per token: 3,072 MiB), row-tiled prefill scratch, and prefill in chunks through the cache
  (the engine prefills only into an empty cache today). Nothing measured. Written up in
  [docs/long-context-memory.md](docs/long-context-memory.md).
- **A software model of Blackwell's tensor-core arithmetic** (owner's question, 2026-11-17; started the same day at
  the owner's request). Done for the four forward fixtures: a validation build of this engine with the published
  tensor-core model in its prefill GEMM reproduces all 7,826 of the Thor engine's pinned trace files on the GTX 1080
  ([tensor-core-model-v1](validation/tensor-core-model-v1/)). Adopted as a standing gate (decision 4). Open: pins
  for the paths the fixtures do not reach (long context, the scorer, the other modalities), which would have to
  come from Thor; and, for phase 2, whether the shipping prefill kernel should itself be the exact one.
- **For the Thor engine, after the exact Pascal kernel is fast** (owner, 2026-11-17: "#3 will come after #1 … make
  sure we don't lose sight of that one"). The Thor prefill kernel reaches 36-42 TFLOPS of a 57 TFLOPS `mma.sync`
  ceiling by its own record, and a `tcgen05` kernel is its planned stage B. What this project can hand it: the
  tensor core's arithmetic is known and reproduced here, so any Thor kernel that keeps 16 products per step in
  ascending k gives identical bits, and its tiling, data movement and pipeline can be restructured freely and checked
  against the pins already in its repository; and a `tcgen05` kernel's arithmetic can be tested the same way (the
  published model has a separate row for it). That work belongs to the Thor repository, which is read-only from here:
  it is a proposal to bring to that team, not something to do in this one.
- **Long-context decode attention** (speed-v1 rule 12, failed by 0.18 ms). At 31,000 keys the key and value products
  read at 205 to 220 GB/s of the 245, and the softmax over the cache costs 0.18 ms. Next lever: one fused kernel for
  scores, softmax and values with each output's arithmetic unchanged.
- **The card's clocks** ([decode-atoms-v1](validation/decode-atoms-v1/)). Decode is 90% weight reads at the memory's
  rate. The cold-start slowdown was the driver re-initializing the card when nothing held it open; the host's
  persistence daemon now runs in persistence mode, permanently, by a systemd drop-in (2026-11-18, at the owner's
  instruction; confirmed across a reboot). Still open: the memory clock under compute load (4,513 MHz in state
  P2 against 5,005 MHz in P0, about 10% of decode), for which no Linux switch was found. Closed by decision 7: the clock stays.
