# red-team-2: the code this project wrote, end to end, after phase 4

Status: **rules set; reviews to follow** (2026-11-21). The owner: "let's red-team the code end-to-end. Remediate
findings, verify, and iterate until pristine." Rules and predictions committed before the first review.

## Scope and method

Everything written or changed since the import (`git diff fc130a3`), on every path the four phases built: the GEMM and
the tensor-core model, the text runner and scorer, the batch decoder, worker and server, the speech engine's changes
(the shared codec checkpoint, the batch decoder on `lfm::gemm`), the build scripts, and the validation tooling of
every phase (`tools/pascal/`, the card copies of the reference's checks, the `validation/*/` job scripts). The imported
code is read where the new code calls it or depends on what it does. red-team-1 covered phase 1's code; this covers it
again with phases 2 to 4.

Five independent reviews by reading, one per area, each told to find defects and given none of the project's
conclusions:

- **A. Engine numerics and the text path:** `gemm.cuh`, `tensor_core_model.hpp`, `fp32_ops.cuh`, `greedy_pick.cuh`,
  `window_slots.hpp`, `lfm3_main.cu`, `lfm3_score.cu`, `lfm3_batch.cuh`, `lfm3_tree.cuh`, `native/tests/`,
  `native/bench/`, `native/probes/`.
- **B. Serving and speech host code:** `lfm3_server.cu`, `lfm3_worker.cu`, `lfm3_decode_batch.cuh`, `lfm3_speech.cuh`,
  `codec.cuh`, `vision_absent.hpp`, and the imported HTTP layer, sessions, limits and speech batch where they meet them;
  the build scripts.
- **C. Phase 1 and 2 tooling:** the text, score, bar, sweep, speed, tensor-core and analysis tools, `remote.sh`.
- **D. Phase 3 and 4 tooling:** the card copies of the reference's checks against the originals, the server and speech
  measurement and judging tools, the job scripts of serve-v1 and speech-v1.
- **E. End to end:** what a user of the public repository gets: building from a clean checkout, the binaries'
  command lines and defaults, the server's API and refusals, and what the documents say against what the code does.

Each finding is checked before it is fixed (reproduced where it can be); one that does not hold is recorded with why.
After the fixes, a further independent review of the fixes and of anything they touch; the rounds repeat until a round
finds nothing to fix.

## Rules

1. Every finding is fixed, or recorded as not holding with the evidence, or left unchanged by the owner's decision.
2. A finding in what a binary computes or accepts is reproduced on the old binary where the card can show it, and does
   not reproduce on the new one.
3. Nothing recorded changes. Device code of every final binary is compared function for function with `text-v35` and
   `parity-v35`; every rule of phases 1 to 4 whose path a fix reaches runs again on the final tree, and the rest are
   carried as decision 11 sets (identical device code; byte-identical reruns of their inputs).
4. The parity gate (7,826 of 7,826), the fixtures, rule 3 of speech-v1 (the Thor engine's 16 clips byte for byte) and
   the host suite (as recorded) on the final tree.
5. The last review round finds nothing to fix.

## Predictions

- Most findings are in the tooling, as in red-team-1: ways a check could pass when it should not.
- Some are in the server and speech host code that phases 3 and 4 changed (defaults, limits, refusals).
- No finding changes a recorded result: the outputs every phase judged stay byte-identical.
