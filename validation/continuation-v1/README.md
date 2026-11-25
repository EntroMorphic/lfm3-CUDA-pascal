# continuation-v1

A session's cached prefix continued in one batched pass. Before this change, a later turn was forwarded a token at a time,
at decode speed (about 16 ms a token, 22 ms near 32K). red-team-2 (R5-1) capped that at 1,024 tokens, so a 28,000-token
continuation could no longer outlast the 300 s engine timeout. A longer continuation was a full prefill of the whole
prompt instead. The owner's item 3 in red-team-2 is this cap, a departure from the Thor engine's server.

## What it will be

- **`Runner::extend(tokens)`** (`native/src/lfm3_main.cu`, shipping build only). It takes the tokens that follow the cached
  positions `[0, cursor)` and computes them in one call, each row as a prefill of the whole prompt (cursor + N
  positions) computes it, on the same cache:
  - **Launch shapes.** Every launch whose shape depends on the row count takes the whole prompt's: the norms' thread
    counts (`shape_rows`), and the GEMMs' prefill kernel whenever the prompt has more than 8 rows (`Problem::wide`).
    The GEMMs' reduction order is fixed and does not depend on tiling (`lfm/gemm.cuh`), as the MLP's blocks already
    rely on.
  - **Attention tiles.** Tiles start at multiples of the prefill's tile from position 0. Each tile reads the key span
    of the prefill tile that holds it (`lfm::text::prefill_key_range`), so every row's scores, softmax and products
    have the prefill's spans and summation orders.
  - **Convolutions.** `convolution7_continue` is `convolution7_prefill`'s arithmetic, with the prefix's last inputs
    read from the stored state where the prefill reads its own earlier rows. `convolution7_continue_state` then updates
    the state.
  - **Compact windowed layers** (capacity above 4,096). The prefill's path: the held window and the new rows go into
    the shared full-length buffer at their absolute positions, the layer attends as a prefill does, and the last window
    is kept (`WindowSlots::extend`). Keys a span reaches before the held window are masked for every new row. They are
    zeroed so their products add exact zeros.
- **Server.** The shipping build's worker continues every cached prefix with one `extend` (`/generate`, and each lane of
  `/generate_batch`), with no cap. A continuation costs what the whole prompt's prefill costs for its last N rows, so
  never more than that prefill (28.9 s at 32,768 on this card), well inside the engine timeout.
- **Parity build.** Unchanged: as the Thor engine, a token at a time, at most 1,024 tokens (red-team-2's cap).

**What the arithmetic gives.** When the cache holds what the whole prompt's prefill puts there, the logits are that
prefill's byte for byte. That holds when the prefix was itself prefilled, or prefilled and extended, with launch shapes
equal to the whole prompt's. Two of those shapes depend on the prompt's length:
- the softmax kernel, by span (2,048 keys or fewer, or more);
- the attention tiles' spans (prompts of more than 4,096 tokens at a capacity above 4,096 are tiled and windowed).

So the identity holds when every part of the prompt ends in the same range: 32 to 2,048 tokens, 2,049 to 4,096, or
above 4,096 at a capacity above 4,096. Prompts of 32 tokens or more give the norms the same thread count. Beyond 8 the
GEMMs use one kernel. Masked keys beyond a shorter part's end add exact zeros.

After a decoded turn the cache holds decode's arithmetic, so a continuation equals no single prefill. Its new rows are
the prefill's arithmetic on that cache, as a fresh prompt's are on an empty one. The token-at-a-time path it replaces
computed them with decode's kernels.

Baseline: `text-v41` (shipping, [images-v1](../images-v1/) remedy 2), `parity-v40`, `text-v39s` (text-only server,
[memory-v2](../memory-v2/); unchanged since). Under test: `text-v42`, `parity-v42`, `text-v42s`.

## Rules (committed before the v42 builds have run on the card)

Development 1 ([dev/dev1.sh](dev/dev1.sh), job `cont-dev1`, uncommitted tree on `06f242b`) found two warnings.
`Runner::extend` was unreferenced in the CLI, and the cap's constant unused in the shipping worker. Now `extend` is a
member template, and the constant exists in the parity build only. The development run also gave the speed figures
the gates in rule 3 are set from.

An independent review of the change (no findings in the engine or worker) found two gaps in the check, both closed
before these rules were committed:
- **The zeroing was not tested.** `Runner::reset()` leaves the compact layers' shared full-length pair as the last
  call left it, so the parts of an `exact` case found the whole prompt's keys there. A span read before the held
  window that was not zeroed would have passed. The check now fills that pair with NaN after every reset, so such a
  read gives non-finite logits, which `forward()` refuses.
- **No judged case continued after a long decode.** Each chat turn continues after decoded tokens, and the compact
  layers' slots move only after 256 decoded steps. `decoded` now has three cases with 400 steps before the
  continuation, and rule 6 judges that every case runs.

1. **Build.** `text-v42`, `parity-v42` and `text-v42s` build. The speech builds and `lfm3-continuation-check` have 0
   warnings. The text-only worker has its 6 #177-D warnings, the same list as `text-v39s` (memory-v2 rule 1).
2. **The identity** (`lfm3-continuation-check exact`, `text-v42`). 19 prompts at capacities 4,096, 8,192 and 32,768,
   each prefilled in parts: a prefix, then one to four extensions of 1 to 28,560 tokens, at tile-aligned and unaligned
   positions, in each of the three ranges above. Each is compared against the same prompt prefilled at once. In every
   case these are identical byte for byte:
   - the logits;
   - the whole cache: every convolution state, every attention layer's keys and values at the positions it holds,
     and the compact layers' slot bookkeeping;
   - the logits of 8 greedy decode steps after it.

   This holds with `LFM3_COMPACT_WINDOWS` unset, `=1` (every capacity compact) and `=0` (none compact). The compact
   layers' shared pair holds NaN before every run, so a key read that should have been zeroed fails the case. Under
   compute-sanitizer memcheck, `exact-small` (three cases, one through the compact layers' zeroed span) passes with 0
   errors, unset and `=1`.
3. **Speed** (`speed`, 14 cases at capacities 4,096 and 32,768). In every case `extend` of N tokens after L cached
   takes at most 1.03 times the whole prompt's prefill (L + N). For N of 8 or more it is faster than the token-at-a-time
   continuation wherever that is measured (N up to 1,024). 1,024 tokens after 31,744 cached take at most 2.0 s
   (red-team-2 R5-1's situation; development 1: 1.31 s, against 17.9 s a token at a time and 28.6 s for the full
   prefill). 1,024 after 3,072 take at most 1.0 s (development: 0.66 s). N of 1 is recorded, not judged: one row
   through the prefill kernels took 61-78 ms in development, a decode step 16-18 ms. A served chat turn adds the
   template's tokens, more than 8.
4. **The served paths** (`tools/pascal/continuation_serve_check.py`).
   - **Shipping worker** (`text-v42`, `text-v42s`). A session continued by 1,024, 1,025 and 7,000 tokens reuses its
     cached 1,000 each time.
   - **Shipping servers.** R5-1's case: a session of 29,696 seeded with about 1,000 tokens and continued by about
     28,000. It answers 200 within 60 s, reusing the whole seed, and `/health` is ready after. A further turn reuses
     everything before it. A `/generate_batch` whose two lanes each continue a seeded session by about 5,000 tokens
     reuses both seeds.
   - **Parity worker** (`parity-v42`). Keeps the cap: 1,024 reused, 1,025 and 7,000 not.
5. **Nothing else changes.**
   - **Device code.** Every binary's device code is identical to its baseline's (`text-v41`, `parity-v40`,
     `text-v39s`), function for function, but for `convolution7_continue` and `convolution7_continue_state`, in v42
     alone. (`sass_compare.py --only-in-a`: its "removed as expected" counts them.)
   - **Text.** The parity gate: 7,826 of 7,826 pins. The forward fixtures with `LFM3_COMPACT_WINDOWS` unset and `=1`,
     and under memcheck with 0 errors. The scorer's NLL files at 256, 257 and 4,096, both builds, byte for byte
     against images-v1's v40. Above 4,096: the A1 prompts' logits against `text-v41` and `parity-v40`.
   - **Speech.** speech-v1 rule 3 on `parity-v42`: 16 of 16 Thor clips, the rendering 15 of 15. The shipping seed-1
     clips, the split lighthouse and the 19 LibriSpeech utterances on both builds, byte for byte against speech-v1's.
   - **Paired decode** (serve-v1 rule 10) passes.
6. **After decoded positions** (`decoded`, 8 cases, `LFM3_COMPACT_WINDOWS` unset and `=1`). A prompt, then 32 or
   400 greedy steps, then 300 to 20,000 new tokens, continued by `extend` and by a token at a time. Each is compared
   against the whole sequence prefilled at once. Every case runs to the end with finite logits, the shared pair NaN
   before each run. Recorded, not judged: the largest logit difference, the next token, and how many of 16 greedy
   steps after agree.
7. **Host suite** passes, but for the one test that needs Transformers 5.x (as in every phase).

The memory a server reserves does not change: `extend` uses the runner's existing buffers, which are sized for its
capacity.

## Results (2026-11-25, job `cont-v1`, tree `dc44c9e`)

**All seven rules met.** Records: [records/](records/). `jobs/cont-v1.log` is the whole job. `cont-v1/` holds its
outputs, without the speech outputs, which stay on the card for speech-v1 rule 9's blindness. `cont-dev1/` and
`cont-dev2/` are the developments. `SHA256SUMS` covers everything.

| Rule | Result |
|---|---|
| 1. Build | All three build. The speech builds and `lfm3-continuation-check`: 0 warnings. The text-only worker: the same #177-D list as `text-v39s` |
| 2. Identity | **19 of 19** byte-identical (logits, whole cache, 8 decode steps) with `LFM3_COMPACT_WINDOWS` unset, `=1` and `=0`. Memcheck: 0 errors, unset and `=1` |
| 3. Speed | All 14 cases pass (table below) |
| 4. Served paths | Shipping worker, both builds: 1,024, 1,025 and 7,000 tokens each reuse the cached 1,000 (0.63, 0.68 and 4.3 s). Both servers: R5-1's case, 970 seeded and continued to 28,882, answers 200 in 23.5 s, reusing all 970, `/health` ready. A further turn reuses all 28,882. The batch's two lanes reuse their 970 and reach 6,089 in 6.2 s. Parity worker: 1,024 reused (16.1 s, a token at a time); 1,025 and 7,000 prefilled in full |
| 5. Nothing else | Device code identical to the baselines' in all 16 binaries, function for function. Only `convolution7_continue` and `convolution7_continue_state` are new, in each binary but `lfm3-vision-check`, which has neither. Parity gate 7,826 of 7,826. Fixtures pass with compact windows unset and 1, and under memcheck. Scorer byte-identical to v40 at 256, 257 and 4,096, both builds. The A1 prompts' logits identical, both builds. Speech: 16 of 16 Thor clips, the rendering 15 of 15; the shipping seed-1 clips (10 of 10), the split lighthouse, and the 19 LibriSpeech utterances on both builds identical to speech-v1's. Paired decode passes |
| 6. After decoded positions | 8 of 8 cases run, unset and `=1`, the two runs' records identical byte for byte (table below) |
| 7. Host suite | 2,398 passed, 5 skipped, 1 failed: the test that needs Transformers 5.x, as in every phase (`records/host-suite.txt`). It ran on the tree before the review's changes, which touched only the GPU check program and the documents |

Rule 3: seconds to continue a session of L cached tokens by N; the whole prompt's prefill and the token-at-a-time
continuation (the parity build's path, and the shipping build's before this change) on the same build:

| Capacity | L | N | `extend` | Whole prefill | A token at a time |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 3,072 | 1 | 0.062 | 2.144 | 0.016 |
| 4,096 | 3,072 | 8 | 0.069 | 2.160 | 0.127 |
| 4,096 | 3,072 | 64 | 0.079 | 2.159 | 1.010 |
| 4,096 | 3,072 | 1,024 | 0.671 | 2.805 | 16.169 |
| 4,096 | 1,024 | 1,024 | 0.616 | 1.378 | 16.120 |
| 4,096 | 1 | 4,095 | 2.605 | 2.792 | |
| 32,768 | 31,744 | 1 | 0.078 | 27.474 | 0.018 |
| 32,768 | 31,744 | 8 | 0.080 | 27.639 | 0.141 |
| 32,768 | 31,744 | 64 | 0.100 | 27.633 | 1.121 |
| 32,768 | 31,744 | 1,024 | 1.299 | 28.747 | 17.944 |
| 32,768 | 8,192 | 1,024 | 0.725 | 5.721 | 16.528 |
| 32,768 | 16,384 | 16,384 | 17.477 | 28.624 | |
| 32,768 | 1,024 | 31,744 | 28.132 | 28.789 | |
| 32,768 | 1 | 32,767 | 28.640 | 28.832 | |

A continuation of 1 token is the one case slower than before (62-78 ms against 16-18 ms): one row through the
prefill kernels. At 8 tokens and above it is faster, and a served chat turn adds more than 8.

Rule 6: L prompt tokens, D greedy steps, then N new tokens. The largest logit difference between each pair of `extend`,
the token-at-a-time continuation and the whole sequence prefilled at once; the next token by each (the three
agreed in every case); and how many of the 16 greedy steps after agree:

| Capacity | L | D | N | extend / tokens | extend / whole | tokens / whole | Next token | Steps agreeing (e/t, e/w, t/w) |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4,096 | 1,000 | 32 | 500 | 0.134 | 0.137 | 0.131 | 3927 | 16, 3, 3 |
| 4,096 | 300 | 32 | 2,000 | 0.157 | 0.128 | 0.140 | 643 | 16, 16, 16 |
| 4,096 | 1,000 | 400 | 600 | 0.155 | 0.185 | 0.154 | 1200 | 16, 16, 16 |
| 32,768 | 8,000 | 32 | 1,024 | 0.146 | 0.140 | 0.136 | 467 | 16, 16, 16 |
| 32,768 | 5,000 | 32 | 20,000 | 0.160 | 0.155 | 0.101 | 25 | 16, 16, 16 |
| 32,768 | 30,000 | 32 | 1,000 | 0.120 | 0.139 | 0.155 | 302 | 16, 16, 16 |
| 32,768 | 8,000 | 400 | 1,024 | 0.142 | 0.144 | 0.147 | 467 | 16, 16, 16 |
| 32,768 | 4,500 | 400 | 300 | 0.135 | 0.173 | 0.134 | 1033 | 16, 12, 12 |

`extend` is as close to the whole sequence's prefill as the token-at-a-time path is. In the two cases where the
greedy steps part from the prefill's, `extend` and the token-at-a-time path part at the same step, and agree with
each other throughout.

The owner's item 3 in red-team-2 (the 1,024-token cap, a departure from the Thor engine's server) is closed for
the shipping build: it reuses every cached prefix, and a continuation costs at most what the whole prompt's prefill
costs. The parity build keeps the cap, as the Thor engine's path, and its exposure noted there (R6-3) is unchanged.
