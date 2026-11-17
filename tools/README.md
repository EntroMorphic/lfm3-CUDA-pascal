# tools/

Python and shell tooling: Hugging Face reference runners, native-against-reference comparisons, the gates of each
validation round, and the job scripts that run them on the target machine. Python tools run as scripts
(`python3 tools/<name>.py --help` or the usage in each docstring); several import the shared libraries below.

| Folder | Contents |
|---|---|
| `thor/` | Job scripts for the target machine: builds in the pinned image, validation rounds, regressions, sanitizers ([machine layout](../docs/building.md#machine-layout)) |
| `thor/legacy/` | The upstream-era build and sanitizer scripts, run from the repository root |
| `upstream/` | Tooling inherited from the `liquid-native` (LFM2.5) runtime this engine started from |

## Shared libraries

| Tool | What it does |
|---|---|
| [`lfm3_owned_process.py`](lfm3_owned_process.py) | Bounded Linux process-group cleanup for GPU acceptance jobs. |
| [`lfm3_worker_client.py`](lfm3_worker_client.py) | Bounded local resident-worker transport, independent of the GPU gate tool. |
| [`lfm3_text_frontend.py`](lfm3_text_frontend.py) | Tokenizer-only LFM3 text frontend; no processor, codec, model, or CUDA import. |
| [`lfm3_token.py`](lfm3_token.py) | The bearer token for liquid-lfm3-server, for clients: LFM3_SERVER_TOKEN if set, else the server's default token file, $XDG_CONFIG_HOME/liquid-lfm3/token or ~/.config/liquid-lfm3/token (lfm3_http.hpp default_token_path; the server creates it on its first start). |
| [`thor_remote.py`](thor_remote.py) | Workstation-side Thor operations that fail loudly instead of silently. |
| [`lfm3_reference.py`](lfm3_reference.py) | Pinned Transformers oracle and explicit token-ID requests for native validation. |
| [`lfm3_bridge.py`](lfm3_bridge.py) | Checked, synchronous, inference-only boundary for the isolated CUDA test ABI. |
| [`lfm3_server.py`](lfm3_server.py) | Experimental loopback-only HTTP text serving with explicit two-lane decode batches. |
| [`lfm3_chat.py`](lfm3_chat.py) | Local tokenizer-only frontend -> experimental native CUDA -> decoded text. |
| [`inspect_checkpoint.py`](inspect_checkpoint.py) | Validate LFM3 text tensor names/shapes without loading weights or importing torch. |

## Text: correctness and quality

| Tool | What it does |
|---|---|
| [`check_lfm3_forward.py`](check_lfm3_forward.py) | Checkpoint integration gates. |
| [`check_lfm3_32k.py`](check_lfm3_32k.py) | Long-context checkpoint validation against memory-efficient Transformers SDPA. |
| [`check_lfm3_context.py`](check_lfm3_context.py) | Logits-only-on-disk context gates; run in the pinned CUDA reference container. |
| [`check_lfm3_convolution.py`](check_lfm3_convolution.py) | Verify distinct checkpoint prefill/decode contracts across lengths and magnitudes. |
| [`check_lfm3_ops.py`](check_lfm3_ops.py) | GPU numerical checks for isolated LFM3 kernels (not end-to-end inference). |
| [`check_lfm3_sampling.py`](check_lfm3_sampling.py) | Native sampling integration gates; not a PyTorch RNG parity claim. |
| [`check_lfm3_native_text.py`](check_lfm3_native_text.py) | Gate: the engine binaries' native text frontend ("messages" in, "text" out) is identical to the Python serving path (tools/lfm3_chat.make_request + Transformers frontend, then frontend.decode). |
| [`check_gpu_select.py`](check_gpu_select.py) | Chi-squared test of the device sampler's draws (native/tests/lfm3_gpu_select_check.cu) against the host sampler's exact distribution. |
| [`compare_lfm3_traces.py`](compare_lfm3_traces.py) | Compare actual checkpoint activations; tolerances are explicit, never auto-relaxed. |
| [`compare_trace_spread.py`](compare_trace_spread.py) | Distance of BF16 implementations from an anchor forward pass, per traced activation and per logit step. |
| [`lfm3_trace_exact.py`](lfm3_trace_exact.py) | Exact FP32 logit traces (G4): STEP.logits.f32 written next to STEP.logits.bf16 by an FP32-logit engine. |
| [`lfm3_oracle_live.py`](lfm3_oracle_live.py) | Read the live token count the sampling oracle used (it counts with the engine's own C++ lfm::tokenizer_token_count, so gates never re-implement the tokenizer rules in Python). |
| [`trace_hf_text_path.py`](trace_hf_text_path.py) | Trace the Hugging Face text-generation path for LFM3-Omni-2B, end to end, as ground truth. |
| [`lfm3_text_diff.py`](lfm3_text_diff.py) | Differential test of the native LFM3 text frontend (native/include/lfm/lfm3_text.hpp) against the reference stack: Rust `tokenizers` (encode/decode) and Transformers' chat-template render, through tools/lfm3_text_frontend.py exactly as text serving uses them. |
| [`gen_lfm3_text_classes.py`](gen_lfm3_text_classes.py) | Generate native/include/lfm/lfm3_text_classes.hpp from the measured pre-tokenizer classes. |
| [`probe_pretokenizer_classes.py`](probe_pretokenizer_classes.py) | Measure the character classes LFM3's pre-tokenizer regex actually uses, through the real tokenizer. |
| [`measure_dead_vocab.py`](measure_dead_vocab.py) | Measure how LFM3-Omni-2B treats output ids that have no token (G6), with the HF SDPA reference. |
| [`lfm3_coherency.py`](lfm3_coherency.py) | Coherency check for the LFM3 text engines: real chat generations, read by a person, plus mechanical flags. |
| [`ppl_reference.py`](ppl_reference.py) | Reference perplexity for LFM3-Omni-2B (or, with --model-class causal, the text-only LFM3-1.7B) with the standard Hugging Face stack (teacher forced). |
| [`ppl_compare.py`](ppl_compare.py) | Compare engine perplexity against the HF SDPA reference (the agreed quality bar). |
| [`ppl_fidelity.py`](ppl_fidelity.py) | G4 tier 2b: is an engine's per-token NLL as close to an FP32 model's as HF SDPA's is? |
| [`pg19_ppl_report.py`](pg19_ppl_report.py) | Report of a PG-19 long-context perplexity round (tools/thor/pg19_ppl_round.sh): ROUND/onepass and ROUND/chunked, each a ppl_reference.py output (one prefix window per book: [BOS] + its first L - 1 tokens). |
| [`g4_paired_stats.py`](g4_paired_stats.py) | Paired per-window NLL differences between engines (G4 evidence). |

## Text: speed

| Tool | What it does |
|---|---|
| [`benchmark_lfm3.py`](benchmark_lfm3.py) | Matched batch-one greedy measurements; exact gates, no adaptive tolerances. |
| [`batch_decode_fidelity.py`](batch_decode_fidelity.py) | Decode-path fidelity of the batched decoder (P0 speed gate), read like G4 tier 2b but on decode positions. |
| [`check_lfm3_decode_batch.py`](check_lfm3_decode_batch.py) | Fail-fast standalone paired-decode fixture runner with owned-group cleanup. |
| [`lfm3_acceptance.py`](lfm3_acceptance.py) | Offline acceptance of zero-cost drafts (n-gram / prompt lookup) for tree speculation on LFM3's own greedy outputs (journal/lfm3-single-stream_synth.md, step 1). |
| [`head_data.py`](head_data.py) | Draft-head data, step 1: chat prompts for self-distillation (journal/lfm3-single-stream_synth.md, decision 4). |
| [`head_train.py`](head_train.py) | Draft head for tree speculation (journal/lfm3-single-stream_synth.md, decision 4): an EAGLE-style head trained on the frozen LFM3-Omni-2B's own features and outputs. |
| [`head_eval.py`](head_eval.py) | Offline acceptance of the trained draft head (tools/head_train.py) for tree speculation, teacher forced on held-out greedy LFM3 outputs: the go/no-go before the native head (journal/lfm3-single-stream_synth.md, decision 4). |
| [`ternary_draft_demo.py`](ternary_draft_demo.py) | Ternary drafts of LFM3-Omni-2B for exact tree/chain speculation: a small, CPU-runnable demonstration. |

## Kernels, libraries and sanitizers

| Tool | What it does |
|---|---|
| [`check_lfm3_blas_policy.py`](check_lfm3_blas_policy.py) | Screen isolated policies on the measured small GEMM; never full acceptance. |
| [`prepare_lfm3_blas_policy.py`](prepare_lfm3_blas_policy.py) | Prepare an isolated cuBLAS policy candidate; never modifies production or goldens. |
| [`check_lfm3_cublas_modes.py`](check_lfm3_cublas_modes.py) | Exclusive-owner, checkpoint-free cuBLAS mode diagnostic; never acceptance. |
| [`check_lfm3_lt_candidates.py`](check_lfm3_lt_candidates.py) | Screen explicit cuBLASLt algorithms on real layer-0 inputs; no production adoption. |
| [`check_lfm3_wmma_candidate.py`](check_lfm3_wmma_candidate.py) | Screen native WMMA split orders against real outputs, then sanitize exact cases. |
| [`prepare_lfm3_wmma_runtime.py`](prepare_lfm3_wmma_runtime.py) | Generate an isolated WMMA runtime candidate; never change production or goldens. |
| [`prepare_lfm3_thin_hybrid_runtime.py`](prepare_lfm3_thin_hybrid_runtime.py) | Generate an isolated v5/v6 thin-hybrid runtime candidate; never change production or goldens. |
| [`run_lfm3_thin_hybrid_gates.py`](run_lfm3_thin_hybrid_gates.py) | Run the thin-hybrid candidate gate matrix sequentially; stop at the first failure. |
| [`prepare_lfm3_pedantic_diagnostic.py`](prepare_lfm3_pedantic_diagnostic.py) | Generate an isolated, unaccepted PEDANTIC runtime mirror; never build or run it. |
| [`check_lfm3_native_policy_benchmark.py`](check_lfm3_native_policy_benchmark.py) | Balanced warmed scalar baseline/candidate measurements; not release acceptance. |
| [`check_lfm3_projection_bedrock.py`](check_lfm3_projection_bedrock.py) | Whole-output GEMM measurement, not production or sanitizer acceptance. |
| [`analyze_lfm3_projection_bedrock.py`](analyze_lfm3_projection_bedrock.py) | Replay saved bedrock measurements without GPU access; no acceptance inference. |
| [`check_lfm3_mma_tails.py`](check_lfm3_mma_tails.py) | Exact bounded tail/warp-retirement tests for the native GEMM candidate. |
| [`check_lfm3_tma_store.py`](check_lfm3_tma_store.py) | Bounded no-cuBLAS TMA/ordinary-store sanitizer diagnostic; never acceptance. |
| [`check_lfm3_boundary.py`](check_lfm3_boundary.py) | Negative tests for the checked tensor wrapper and private raw-ABI metadata guard. |
| [`check_lfm3_launch_boundary.py`](check_lfm3_launch_boundary.py) | Calibrate and measure synchronized, read-only Sanitizer SDK launch snapshots. |
| [`check_lfm3_sanitizer.py`](check_lfm3_sanitizer.py) | Fail-closed sanitizer preflight: never credit a library tracking failure as clean. |
| [`check_lfm3_sanitizer_repro.py`](check_lfm3_sanitizer_repro.py) | Host-only, checkpoint-free evidence packet builder. |
| [`check_lfm3_candidate_sanitizers.py`](check_lfm3_candidate_sanitizers.py) | Unexcluded integrated candidate checks with exact application-output validation. |
| [`check_lfm3_initcheck_matrix.py`](check_lfm3_initcheck_matrix.py) | Checkpoint-free sanitizer-only version/shape matrix. |
| [`check_lfm3_racecheck_diagnostic.py`](check_lfm3_racecheck_diagnostic.py) | All-launch32K racecheck with launch logs and bounded CPU/RSS observations. |
| [`check_lfm3_racecheck_guarded.py`](check_lfm3_racecheck_guarded.py) | Resource-bounded, all-launch 32768-token racecheck diagnostic (not acceptance). |

## Serving

| Tool | What it does |
|---|---|
| [`check_lfm3_server.py`](check_lfm3_server.py) | Explicit loopback HTTP/native smoke; starts a temporary authenticated listener. |
| [`check_lfm3_native_server.py`](check_lfm3_native_server.py) | GPU gate for liquid-lfm3-server, the single-executable text server (G7). |
| [`check_lfm3_worker.py`](check_lfm3_worker.py) | Resident-cache protocol gates. |
| [`check_lfm3_batch_worker.py`](check_lfm3_batch_worker.py) | Native grouped-decode acceptance against serial IDs and every logit bit. |
| [`check_lfm3_http_batch.py`](check_lfm3_http_batch.py) | Authenticated explicit-group HTTP gate and fixed-budget extended resource soak. |
| [`check_lfm3_http_cancellation.py`](check_lfm3_http_cancellation.py) | Explicit authenticated loopback/native cancellation gate; no kernel preemption. |
| [`check_lfm3_cancellation.py`](check_lfm3_cancellation.py) | Cooperative native token-boundary cancellation gate, not GPU preemption. |
| [`check_lfm3_soak.py`](check_lfm3_soak.py) | Finite authenticated HTTP soak/resource/latency gate; not GPU batching. |
| [`check_lfm3_group_benchmark.py`](check_lfm3_group_benchmark.py) | Hash-pinned, paired end-to-end HTTP benchmark; no speed acceptance threshold. |
| [`check_lfm3_text_wait.py`](check_lfm3_text_wait.py) | Text under speech load, black box (validation/text-wait-v1 rules 2, 3, 4 and 6's first case; Thor, quiet). |
| [`check_lfm3_text_wait_exact.py`](check_lfm3_text_wait_exact.py) | Rule 1 of validation/text-wait-v1, over HTTP (black box): text and speech replies are the same whatever runs beside. |
| [`check_lfm3_text_wait_speed.py`](check_lfm3_text_wait_speed.py) | Rule 5 of validation/text-wait-v1: nothing slower alone (end to end, through the server; Thor quiet). |
| [`lfm3_http_diff.py`](lfm3_http_diff.py) | Request-validation parity: tools/lfm3_server.py against the native HTTP layer (lfm/lfm3_http.hpp). |
| [`diag_lfm3_server_memory.py`](diag_lfm3_server_memory.py) | Where does liquid-lfm3-server's memory grow? |

## Speech out

| Tool | What it does |
|---|---|
| [`lfm3_audio_trace.py`](lfm3_audio_trace.py) | Measured runtime trace of the HF text -> text + audio path (journal/lfm3-tts-source-map.md, open questions 1-6). |
| [`lfm3_codec_anchor.py`](lfm3_codec_anchor.py) | Codec anchors for A2 (journal/lfm3-tts-quality-bar.md): the HF codec decoder from forced codes, in four precisions. |
| [`lfm3_codec_check.py`](lfm3_codec_check.py) | A2 check of the native codec (liquid-codec) against the HF anchors of lfm3_codec_anchor.py. |
| [`lfm3_codec_widening_check.py`](lfm3_codec_widening_check.py) | Red-team of the FP64 codec anchor: do the FP32-island widening patches take effect in the FP64 run? |
| [`lfm3_depth_anchor.py`](lfm3_depth_anchor.py) | Depth-decoder anchors for A1/B1 (journal/lfm3-tts-quality-bar.md): HF's depth decoder on the reference's own passes (lfm3_audio_trace.py's frames.pt: LM hidden state, revealed mask and codes of each of a frame's 3 passes), codes forced, no RNG. |
| [`lfm3_depth_check.py`](lfm3_depth_check.py) | A1/B1 check of the native depth decoder (liquid-depth) against lfm3_depth_anchor.py's anchors. |
| [`lfm3_mode_reference.py`](lfm3_mode_reference.py) | A4 reference: HF's own generation control flow (`Lfm3OmniForConditionalGeneration._generate`) on forced inputs. |
| [`lfm3_audio_control_report.py`](lfm3_audio_control_report.py) | validation/audio-control-v1/report.json: the B2 and A4 record (#2 step 3), reproducible on any machine with g++. |
| [`lfm3_speech_hf.py`](lfm3_speech_hf.py) | #2 step 4 (C1-C4): the HF reference over the fixed prompt set, every seed, and the native engine's requests. |
| [`lfm3_speech_replay.py`](lfm3_speech_replay.py) | #2 step 4, the deterministic gate: the native speech loop with forced inputs against HF (no RNG anywhere). |
| [`lfm3_speech_cases.py`](lfm3_speech_cases.py) | Package lfm3-speech outputs as case directories for tools/lfm3_asr.py and listening: CASE/summary.json (case, mode, seed, conversation, the emitted text decoded, audio seconds, frame and slot counts, timings), CASE/response.wav, CASE/response.txt. |
| [`lfm3_speech_report.py`](lfm3_speech_report.py) | #2 step 4 report: C1 (intelligibility), C2 (duration and EOS), C4 (interleaved: speech against its own text), native against HF over the fixed prompt set and its seeds (validation/speech-e2e-v1). |
| [`lfm3_speech_alignment.py`](lfm3_speech_alignment.py) | #2 step 4, the long-form TTS failure's mechanism: where in the prompt's text the backbone attends while it speaks. |
| [`lfm3_speech_chunked_hf.py`](lfm3_speech_chunked_hf.py) | #2 step 4, long-form TTS by sentences (the fix the sliding-window finding points to), prototyped in HF. |
| [`lfm3_speech_long_form_report.py`](lfm3_speech_long_form_report.py) | validation/speech-long-form-v1/report.md: the long-form TTS investigation's tables, from the committed ASR records. |
| [`lfm3_listening_set.py`](lfm3_listening_set.py) | C3 (journal/lfm3-tts-quality-bar.md): Tripp's blind listening set, native against HF. |
| [`lfm3_asr.py`](lfm3_asr.py) | ASR round-trip for the audio quality bar (journal/lfm3-tts-quality-bar.md, C1). |
| [`lfm3_audio2b_tts_hf.py`](lfm3_audio2b_tts_hf.py) | LFM3-Audio-2B text-to-speech in HF, exactly as its model card shows it, for comparison with LFM3-Omni-2B (validation/speech-audio2b-v1). |
| [`speech_normalize.py`](speech_normalize.py) | Text normalization for speech: numbers, symbols and abbreviations written out as words, before the sentence split (validation/speech-normalize-v1, the "rules" arm). |
| [`speech_normalize_round.py`](speech_normalize_round.py) | validation/speech-normalize-v1: numbers and abbreviations, four arms (none, rules, model, hybrid), by test. |
| [`mutate_speech_normalize.py`](mutate_speech_normalize.py) | Mutation test of native/include/lfm/speech_normalize.hpp: 18 planted bugs, one per rule or helper; each must change the output on the equality corpus of tests/test_speech_normalize.py (experiment sentences, real text, fuzz). |
| [`normalizer_audit.py`](normalizer_audit.py) | Scores tests/data/normalizer_audit.json against tools/speech_normalize.py (validation/audio-assurance-v1/ normalizer-audit.md rules 2-3): fine, left as written, ambiguous, or differs (each "differs" entry is then judged wrong or not by hand and recorded in the audit's Results). |
| [`normalizer_audit_corpus.py`](normalizer_audit_corpus.py) | Writes tests/data/normalizer_audit.json: the audit corpus of validation/audio-assurance-v1/normalizer-audit.md. |
| [`normalizer_gate.py`](normalizer_gate.py) | Real-text regression gate for the speech normalizer (validation/audio-assurance-v1/normalizer-audit.md, red-team round 2). |
| [`lfm3_asr_multilingual.py`](lfm3_asr_multilingual.py) | ASR round-trip in every speech locale (validation/multilingual-speech-v1; the C1/C4 measure of journal/lfm3-tts-quality-bar.md, beyond English). |
| [`lfm3_multilingual_report.py`](lfm3_multilingual_report.py) | validation/multilingual-speech-v1: per-locale summary of a fidelity round (rules 2, 3, 5). |
| [`fleurs_cases.py`](fleurs_cases.py) | validation/multilingual-speech-v1: the fixed case lists, drawn from FLEURS test before any run. |
| [`fetch_fleurs.sh`](fetch_fleurs.sh) | validation/multilingual-speech-v1 rule 0: FLEURS test for the 18 speech languages, on the workstation (it has the internet), then copied to Thor. |
| [`check_lfm3_multilingual_asr.py`](check_lfm3_multilingual_asr.py) | validation/multilingual-speech-v1 rule 4: the native engine against HF on FLEURS speech recognition, 18 languages. |
| [`check_lfm3_multilingual_server.py`](check_lfm3_multilingual_server.py) | validation/multilingual-speech-v1 rules 6 and 7 (exact; may share Thor): POST /speech in every locale, black box. |

## Speech serving

| Tool | What it does |
|---|---|
| [`check_lfm3_speech_server.py`](check_lfm3_speech_server.py) | GPU check for POST /speech on liquid-lfm3-server (#2; Thor, lfm3-profile-v1). |
| [`check_lfm3_speech_stream.py`](check_lfm3_speech_stream.py) | GPU check for streamed POST /speech (validation/speech-stream-v1; Thor, lfm3-profile-v1). |
| [`lfm3_speech_stream_client.py`](lfm3_speech_stream_client.py) | A minimal client for streamed POST /speech (validation/speech-stream-v1): reads the reply's events as they arrive, prints the text as it comes, writes the audio to OUT.wav, and with --play pipes it to a player as it arrives (default `aplay -q -f S16_LE -r 24000 -c 1`). |
| [`check_lfm3_speech_batch_server.py`](check_lfm3_speech_batch_server.py) | GPU check for concurrent speech in liquid-lfm3-server (validation/speech-batch-v1; Thor, lfm3-profile-v1). |
| [`check_lfm3_speech_concurrent.py`](check_lfm3_speech_concurrent.py) | Concurrent speech at the engine (validation/speech-batch-v1, rule 1 below the server): every request run alone (lfm3-speech --batch) and in batches (lfm3-speech --concurrent ROWS, at once and staggered), each output compared byte for byte: OUT.wav, OUT.f32, OUT.frames.i32, OUT.raw.i32, and OUT.json without its timings. |
| [`measure_lfm3_speech_batch_throughput.py`](measure_lfm3_speech_batch_throughput.py) | Rule 2 of validation/speech-batch-v1: aggregate seconds of audio generated per wall second of the batch loop (lfm3-speech --concurrent N), the 21 TTS requests of speech-e2e-v1 (the README said 15: a miscount) looped (max(42, 4N) requests, so N rows stay busy), at N = 1, 2, 4, 8, 16; one untimed warm-up, then two runs each. |
| [`measure_lfm3_speech_stream_cost.py`](measure_lfm3_speech_stream_cost.py) | Rule 6 re-measurement (validation/speech-stream-v1): the lighthouse request, PAIRS pairs of one unstreamed and one streamed run, the order alternating pair by pair; the cost is the median over pairs of streamed / unstreamed generate_seconds (pass: <= 1.03). |
| [`compare_speech_outputs.py`](compare_speech_outputs.py) | Compare two lfm3-speech output directories (validation/audio-assurance-v1). |

## Audio in

| Tool | What it does |
|---|---|
| [`librispeech_cases.py`](librispeech_cases.py) | Audio input (#3): LibriSpeech test-clean as speech recognition cases for tools/lfm3_audio_in_hf.py and lfm3-speech. |
| [`lfm3_audio_in_hf.py`](lfm3_audio_in_hf.py) | Audio input (#3 audio in): the HF reference for speech recognition and spoken questions on LFM3-Omni-2B. |
| [`check_lfm3_audio_in.py`](check_lfm3_audio_in.py) | Audio input (#3): the native engine against HF on speech recognition (validation/audio-in-v1). |
| [`check_lfm3_audio_in_misc.py`](check_lfm3_audio_in_misc.py) | Audio input (#3), validation/audio-in-v1 rules 4-5 and the native renderer: refusals, and native chat rendering of audio parts against HF's processor. |
| [`check_lfm3_audio_in_server.py`](check_lfm3_audio_in_server.py) | GPU check for audio input on POST /speech (#3; Thor, lfm3-profile-v1; validation/audio-in-v1). |
| [`audio_prompt_cases.py`](audio_prompt_cases.py) | Audio input (#3): cases for native/tests/audio_prompt_check.cpp (the native renderer against HF's processor). |
| [`audio_in_edge_cases.py`](audio_in_edge_cases.py) | Audio input (#3): edge cases for the stage-by-stage check (validation/audio-in-v1 round 3), for tools/lfm3_audio_in_hf.py and lfm3-speech, all dumped. |
| [`lfm3_audio_in_listening.py`](lfm3_audio_in_listening.py) | Audio input (#3): the speech-to-speech listening set for Tripp (validation/audio-in-v1, rule 4). |
| [`check_lfm3_audio_fuzz.py`](check_lfm3_audio_fuzz.py) | Hostile input against a live liquid-lfm3-server (validation/audio-assurance-v1 rule 5; Thor, lfm3-profile-v1). |
| [`check_lfm3_audio_soak.py`](check_lfm3_audio_soak.py) | Long mixed audio run of liquid-lfm3-server (validation/audio-assurance-v1 rule 4; Thor, lfm3-profile-v1). |
| [`dump_lfm3_soak_replies.py`](dump_lfm3_soak_replies.py) | The soak's and fuzz run's reply contents, for reading (validation/audio-assurance-v1 rule 6). |

## Images in

| Tool | What it does |
|---|---|
| [`dump_lfm3_vision_reference.py`](dump_lfm3_vision_reference.py) | Stage 0 of the images round: Hugging Face reference dumps for the LFM3-Omni-2B vision path (CPU, no GPU needed). |
| [`make_image_variants.py`](make_image_variants.py) | The image variants of validation/images-v1 rule 1, written once as fixed files so every machine (and the native front end) decodes the same bytes. |
| [`make_image_mutants.py`](make_image_mutants.py) | Hostile inputs for the image front end (validation/images-v1 rules 1 and 5): COUNT files made from the variant set by seeded mutations, so HF (Pillow) and the native decoder can be compared file by file. |
| [`make_png_integrity_cases.py`](make_png_integrity_cases.py) | PNGs whose chunk CRCs are all valid but whose zlib stream is cut, altered or unusually laid out, each named with the outcome the native decoder must give (validation/images-v1, decision of 2026-11-15; the stb Adler-32 patch): ok-TAG-*  … |
| [`make_exif_orientation_grid.py`](make_exif_orientation_grid.py) | JPEGs whose EXIF Orientation entry is well formed or not, to compare the native reading with Pillow's (validation/images-v1; the m02528 finding of 2026-11-15). |
| [`check_image_damage_policy.py`](check_image_damage_policy.py) | An independent oracle for the native image decoder's departures from HF (validation/images-v1, decision of 2026-11-15). |
| [`build_image_decode_check.sh`](build_image_decode_check.sh) | Build the vendored libjpeg-turbo 3.2.0 (static, Release, its default SIMD where the platform has it: NEON on aarch64, NASM on x86 if installed) and lfm3-image-decode-check against it. |
| [`hash_lfm3_image_front.py`](hash_lfm3_image_front.py) | HF's image front end, hashed per image (validation/images-v1 rule 1): what the model is given, so two machines (and later the native front end) can be compared exactly. |
| [`hash_torch_resize.py`](hash_torch_resize.py) | torchvision's uint8 resize as HF's image processor calls it (BICUBIC, antialias=True, CPU), hashed per case, so the native port (lfm::image::resize_bicubic_aa) can be compared exactly (validation/images-v1 rule 1). |
| [`compare_lfm3_image_front.py`](compare_lfm3_image_front.py) | Compare the native front end (lfm3-image-decode-check [--preprocess]) with HF's (tools/hash_lfm3_image_front.py), image by image (validation/images-v1 rule 1). |
| [`extract_vision_benchmarks.py`](extract_vision_benchmarks.py) | Unpack MMStar and POPE (validation/images-v1) from their parquet files into each image's original bytes and a JSONL manifest, so the HF run and the native engine read the same files and Thor needs no pyarrow. |
| [`run_lfm3_vision_bench_hf.py`](run_lfm3_vision_bench_hf.py) | HF's answers to MMStar and POPE (validation/images-v1 rule 3's reference): LFM3-Omni-2B in BF16 on the GPU (attn_implementation "sdpa"), greedy, one request at a time, through the checkpoint's own chat template. |
| [`score_lfm3_vision_bench.py`](score_lfm3_vision_bench.py) | Score MMStar and POPE runs (tools/run_lfm3_vision_bench_hf.py's JSONL, or the native engine's in the same form). |
| [`build_libjpeg_turbo.sh`](build_libjpeg_turbo.sh) | Build the vendored libjpeg-turbo 3.2.0 (native/third_party/libjpeg-turbo; static, Release, its default SIMD where the platform has it: NEON on aarch64, NASM on x86 if installed) into OUT_DIR/ljt/{include,lib/libjpeg.a}, once: an existing OUT_DIR/ljt/lib/libjpeg.a is kept. |
| [`lfm3_vision_bench_native.py`](lfm3_vision_bench_native.py) | The native engine's answers to MMStar and POPE (validation/images-v1 rule 3), the same requests as HF's (tools/run_lfm3_vision_bench_hf.py): no system turn; the user turn is the image (its file path) then the question with the lmms-eval instruction; greedy, text mode, max_new_tokens 512 (MMStar) or 16 (POPE). |
| [`run_lfm3_vision_replies_hf.py`](run_lfm3_vision_replies_hf.py) | HF's replies to lfm3-speech image requests (validation/images-v1 rules 3-4): each request's "messages" (image parts {"type": "image", "image": path}, as tools/thor/vision_e2e_round.sh writes them) through the checkpoint's chat template, … |
| [`score_lfm3_vision_fp32_agreement.py`](score_lfm3_vision_fp32_agreement.py) | validation/images-v1 rule 3b: which engine's MMStar answers are closer to the model's own FP32 answers. |
| [`check_lfm3_image_server.py`](check_lfm3_image_server.py) | Image input through liquid-lfm3-server (#4; validation/images-v1; Thor, lfm3-profile-v1). |
| [`vision_listening_set.py`](vision_listening_set.py) | validation/images-v1 rule 4: images described in speech, for Tripp's listening. |
| [`time_lfm3_vision_hf.py`](time_lfm3_vision_hf.py) | validation/images-v1, speed against HF: HF's time per stage on lfm3-speech image requests, BF16 on the GPU (rule 3's reference configuration). |
| [`compare_lfm3_vision_stages.py`](compare_lfm3_vision_stages.py) | validation/images-v1 rule 2: the native vision encoder's stages (lfm3-vision-check) against HF's FP32 dumps (the gold reference), next to HF's BF16 dumps against the same (the error HF itself runs with on a GPU). |
| [`image_prompt_cases.py`](image_prompt_cases.py) | Image input (#4): cases for native/tests/image_prompt_check.cpp (the native renderer against HF's processor), from the reference dumps (tools/dump_lfm3_vision_reference.py): each case's conversation as the dump tool built it, and the input_ids HF's processor gave it (input_ids.npy). |
