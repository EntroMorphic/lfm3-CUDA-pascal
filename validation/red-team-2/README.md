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

## Findings and what was done (round 1)

Five reviews (A to E) found 50. None was found in what the shipping build computes. One was in what the parity
build computes (A1), one could shut the server down (B1), one may change a result recorded as passed (D1), and one
result recorded as passed was never shown by its run (C1); the rest are tools that could pass without showing, server
paths that advertise what they cannot serve, documents, and hygiene. Fixes: commit `396c578`.

**Engine and server**

| # | Finding | Fix |
|---|---|---|
| A1 | The parity build stopped being the Thor engine's arithmetic above 4,096 tokens of capacity: red-team-1's F1/F2 fix (a last attention tile, or scorer tile, of 1 to 8 rows takes the prefill kernel) was applied to both builds, but the Thor engine sends such tiles to its decode kernels, and in the parity build the prefill kernel is the tensor-core model. No gate saw it: the pins run at 4,096 or less | In the parity build, attention and scorer tiles do not pass `wide` (the Thor engine's tiles); the MLP's blocks, which this port added to what the Thor engine ran as one call, keep it. The shipping build is unchanged |
| A2, A4 | Comments on the position-0 launch (8 rows; it is one), on who sets `row0`, and on the scorer's `wide` | Corrected |
| A3 | `LFM_FP32_STAGED=on` built as 0 (red-team-1's R1 failure, in a switch added later) | The value must paste to a listed digit |
| A5 | `linear_block` passed `row0` = the block's first row, right only because a multi-row call starts at cursor 0 | `cursor + first` |
| B1 | `/generate_batch` runs both lanes, prefill and every step, as one engine call; a lane may ask for up to its session's capacity of new tokens, and past `--engine-timeout` (300 s) the watchdog fails the server closed. About 14,000 steps exceed it on this card (imported code; the Thor engine has the same exposure) | At most 4,096 `max_new_tokens` a lane, refused 400 before any lease is taken, in the native server and the Python one alike; the worst case measured (verification) |
| B2 | With `LFM3_COMPACT_WINDOWS=1` or `LFM3_LOSSLESS=1` in its environment the server advertised speech and failed closed at the first `/speech` | Speech is then not served (and the defaults are those without speech) |
| B3, E5-E8 | `docs/server-api.md` was the Thor engine's: ready line, `--speech-rows` default 8, a `capacity: 32768` example the speech build refuses, images described as served, a `SECURITY.md` not in this repository | A section for this port: which build serves what, the defaults and why, the bound on the batch |
| B4 | With `--engine-timeout` below 300 the ready line said speech was served and the capacity was lowered for it, while every `/speech` got 501 | Both follow what is served |
| B5, E3 | A build directory reused by the other build script kept that script's records beside the new binaries; `mkdir -p "$out" && cd "$out"` under `set -e` carried on in the current directory when `/workspace` was missing (reproduced: records written into the repository) | Each script removes both scripts' records; the two commands separate |
| B6, B7, B8, E12 | A codec memory figure counted the shared checkpoint once per decoder; refusals named the wrong cause (`LIQUID_LFM3_NATIVE_GEMM` in a text-only build, "this checkpoint" when the build has no vision); stale comments on the speech-row default and the text-only build | Reworded |

**Tools: a pass without showing**

| # | Finding | Fix |
|---|---|---|
| D1 | speech-v1 rule 12's "audio in at the request limits" ran an 86.9 s clip; the 4 MiB body holds 98.2 s of 16 kHz 16-bit audio, the densest format the server takes. The encoder's scratch grows with the clip, and the recorded margin was 43 MiB | The clip fills the body (asserted); rule 12 measured again at 1, 8 and 16 rows (verification) |
| D2 | Two of rule 11's five checks were carried from development 11 and 12, run on servers whose default capacity has since changed; the judge accepts any listed build's hash | Rerun on the final servers (verification) |
| D3 | `serve_gates.sh` with a mistyped `STAGES` ran nothing and printed PASS (reproduced) | Unknown stages refused; no stage run fails |
| D6 | Every exception is a 400 `invalid_request`, so rule 13's 65-clip refusal did not show the count refused it | A 64-clip request must answer 200 first |
| D7 | `speech_memory.py`'s idle sample was taken before the weights loaded (2 MiB), so "peak above idle" could not fail | Sampled after the warm requests, at least 3 samples and above 1 GiB |
| D10, C7 | Job scripts copy carried records, or print a comparison, whatever the step before returned | Not changed in the scripts that ran (each is the record of what ran, and every log shows the steps' rc=0); the verification job gates each result on its step |
| C1 | text-correct-v2 rule 2 recorded fixtures "with compact windows off and on"; its job never ran the "on" leg on `text-v6` | Run now on `text-v6`'s binary (verification); the README corrected |
| C2 | `speed_report.py X X` passed every relative rule (reproduced) | Refuses one directory twice, one binary twice, fewer than 3 runs, a median not its runs'; the three recorded reports reproduce unchanged |
| C3, C4 | `verify_memory.sh`: rule 4 hashed every trace file and, with no logits files, compared two empty inputs; its self-comparison guard compared strings, so `ppl/` or `./ppl` deleted the reference and compared a file with itself (both reproduced) | Logits files only, none fails; paths resolved before anything is removed |
| C8 | `boundary_sweep.sh` judged the sanitizer by its exit code and deleted its log; `run_text_fixtures.sh` missed a sanitizer behind `env` | Both read the sanitizer's own summary; the sweep keeps the log |
| C9 | `round_once_report.py` did not check a file against its recorded hash; `arith_report.py` read no metadata and accepted non-finite values | Both check |
| C10 | text-correct-v2 rule 4 (position 1 equal to the Thor engine's in 1,160, 1,155, 72 and 9 windows) and prefill-arithmetic-v1's position-1 columns had no committed tool | `analysis/position1_check.py` reproduces rule 4 from the committed NLL files and the Thor engine's (hash-checked): all windows equal (`text-correct-v2/records/position1.txt`); `arith_report.py` prints the position-1 columns (verification) |
| C12 | `remote.sh detach` checked only `NAME.log`, and took any name | Both files; `[A-Za-z0-9._-]` |
| nit | `arith_experiment.sh` let a caller's `EXTRA_FLAGS` into its plain build | Unset |

**Privacy, blindness, provenance**

| # | Finding | Fix |
|---|---|---|
| C6 | `remote.sh` synced git-ignored files: `outputs/`, with the owner's listening set and its key, was on the card's host | Removed from the host (no other copy found there); the sync excludes every ignored path and refuses to run if git ignores anything it does not exclude |
| C5 | A job started from another tree (`PASCAL_TREE`) ran the main tree's tools, and a sync during a job changed its tools under it; the job log did not say which tree it ran | Detach refuses another tree; no sync while a job runs; the job's first line is its tree's hash and commit |
| C11 | Only the exact `target.env` was ignored; an editor's copy would be committed | `target.env*` ignored, the example kept |
| D4 | The listening set's shipping clips are byte-identical to the final build's, whose hashes were committed: hashing A.wav and B.wav names the shipping side | The hashes out of the tree (still in its history, commit `834bc1c`); `rule9_key_check.py` checks case by case and prints a count only. Also: `tts-long-male`'s shipping clip is split by sentence and HF's is not, as in listening-c3-v1 (the rule as set) |
| D8, D9 | The rule 9 check was not committed; a comment misdescribed rule 6's clips | Committed; corrected |
| D5 | PLAN said the parity build's audio in matched the Thor engine's "word for word (2,620 utterances)": shown are the corpus's error count and the 19 committed transcripts | Reworded |
| E10 | Two provenance files held the owner's home directory path | Redacted in the tree (still in its history, from the import) |
| E11 | The notices named the Thor repository only; the checkpoint's `config.json` sits in some records | Both stated |

**For a newcomer**

| # | Finding | Fix |
|---|---|---|
| E1 | The README was one line | Written: goal, status, the two builds, building, running, layout |
| E2, E4 | `docs/building.md` described only the Thor engine's CMake build (`sm_110`, CUDA 13), which cannot build for this card | A section for this port; CMake stated as the Thor engine's (not made to build for `sm_61`: the build scripts are the validated path) |
| E9 | Links to a `validation/README.md` that did not exist, and to a round not imported | The index written; the link replaced |
| E13 | `speed-v1/dev19.sh` compiled a test never committed | A note: a development experiment not adopted, which no result depends on (its log on the card) |
| E14 | PLAN linked "liquid-CUDA" to an organisation; a document cited a workspace path | Reworded |

## Development runs

- **Development 1** (`dev1.sh`, tree `396c578`; builds `text-v36`, `parity-v36`). Device code of all ten binaries
  identical to v35's. The CLI at capacity 4,097, 16 teacher-forced steps: the shipping build unchanged at prompts 129,
  136, 137 and 300; the parity build changed at 136 (a last tile of 8 rows) and not at 137 or 300 (none), nor at 129.
  speech-v1 rule 12 with the clip at the request limit: one row at 30,720 peaked at **7,759 MiB, over the 7,740 limit**
  (the audio-in configuration; every other configuration as recorded), 8 rows at 16,384 at 7,523, 16 rows at 2,048 at
  **7,771, over**. `batch_budget.py` failed by its own fault (its small session could not hold 4,096 new tokens).
- **Development 2** (`dev2.sh`, tree `fc2271e`; `text-v37`, `parity-v37`): the defaults lowered as rule 12 says, to
  29,696 at one row, 2,048 less for each further row, at least 1,024; ready lines at 1, 8 and 16 rows as designed.
  Rule 12 at those defaults passes: 7,707 MiB at one row (the audio-in configuration), 7,471 at 8 rows, 7,417 at 16.
  A1 in the attention itself (the parity build's probabilities of the last tile, v37 against v35): at 136, 11 of 11
  layers' traces differ and so do the logits; at 129, 2 of 11 differ and the logits do not (a 1-row tile's
  difference rounds away by the logits). The batch: one lane stopped at 200 tokens, so the worst case was not
  reached; extrapolated, 80 s against the 240 s limit. The prompts are now lists of numbers to continue
  (development 3).

## Round 2: the fixes reviewed

Two independent reviews of commit `396c578` (R2A: engine and server; R2B: tools and documents). Fixes: `fc2271e`.

| # | Finding | Fix |
|---|---|---|
| R2A-1 | The new native test of the batch bound could not fail: its sessions of 1,024 refused 4,097 tokens by capacity, and its 4,096 case failed for the same reason | Its own test on sessions of 8,192, where only the bound refuses; each assertion fails with its fix removed (checked by the round 3 reviewer) |
| R2A-2, R2B-3 | The bound holds only at the default engine timeout; `--engine-timeout 60` let a legal batch fail the server closed | The native server refuses the batch (501 `batching_unavailable`) under 300 s, as `/speech`; the Python server, the reference's check harness here, states that its worker deadline is the caller's |
| R2A-3 | The Python server's `/generate` is one engine call too, unbounded | Stated: on this card it is the reference's check harness, not the served path |
| R2A-4 | A comment said only the MLP blocks set `wide` in the parity build; the batch's projections do too | Corrected |
| R2A-5 | A5's change has no effect (a multi-row call always starts at cursor 0) | Recorded as cosmetic |
| R2A-6 | The default handler count followed the speech rows when speech is not served | 8 then |
| R2A-7 | The new `LFM_FP32_STAGED` check accepted `1+0` | The worker's three-way paste |
| R2B-1, R2B-2 | `batch_budget.py` could not pass (the small session), its refusal case could not show the bound, and its rate ignored a lane's single steps | Rewritten; then made to judge the worst case itself (round 3) |
| R2B-4 | One listening-set clip's hash remained in a committed development log | Withheld there too |
| R2B-5 | The withheld hashes and the provenance paths remain in the published history | For the owner (below) |
| R2B-6 | The ignored-file check named directories holding an ignored file, blocking syncs of new rounds | File by file; a git failure stops the sync |
| R2B-7 | `arith_report.py`'s new table shared one generator with the old, so its intervals did not reproduce | Each interval its own generator; four intervals' last digits in prefill-arithmetic-v1 moved, no reading changed |
| R2B-8 | `dev1.sh`'s A1 hashing repeated C3's empty-input fault | The verification job counts the files (17 each) |
| R2B-9 to 15 | The license holder misquoted; B2 missing from the server documentation; `serve_gates.sh` made its output before checking `STAGES`; a comment; `remote.sh`'s usage and job-name order; `position1_check.py` not comparing windows | Fixed |

## Round 3

One independent review of `fc2271e`. Nothing in what either build computes. Fixes: this commit.

| # | Finding | Fix |
|---|---|---|
| R3-1 | The listening set can still be unblinded from public records: development 5's ASR records, rule 14's records and the reference's HF records give each case's clip duration (and rule 14's the fox clip's codec codes), and every pair differs in length | For the owner (below) |
| R3-2 | The batch bound was measured on one server; the parity server is slower and the text-only server's total is 32,768 | Both shipping servers measured in the verification; the parity build is the validation build, whose single prefill near capacity alone (about 430 s, serve-v1) is past the default engine timeout for `/generate` as for the batch: stated |
| R3-3, R3-4 | `batch_budget.py` left the small lane 2,000 tokens short, and could pass on an extrapolation from 25% of the steps | Both lanes filled; the call itself is the worst case, both lanes 4,096 tokens, or inconclusive |
| R3-5 | Development 1's failure was recorded nowhere but a commit message; PLAN and speech-v1 still said rule 12 passed | This section; speech-v1's status and rule 12 row, PLAN |
| R3-6 | `git ls-files` quotes unusual paths, which the ignored-file patterns then miss (fails safe) | `core.quotePath=false` |
| R3-7 | `dev2.sh`'s A1 lines had no expectation | The verification job states it |
| R3-8 | The handler comment; two identical configurations at a total of 1,024; a trailing blank line | Fixed |
