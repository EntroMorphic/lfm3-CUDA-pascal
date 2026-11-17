# Moved paths

Files moved in the 2026-11 reorganization. Records under `validation/`, older commit messages and the journal cite
the old paths; this table maps them. Generated from the reorganization commits' renames.

| Old path | New path | Commit |
|---|---|---|
| `CHECKPOINT_INVENTORY.json` | `docs/provenance/CHECKPOINT_INVENTORY.json` | 02a61af |
| `SOURCE_SNAPSHOT.json` | `docs/provenance/SOURCE_SNAPSHOT.json` | 02a61af |
| `build-forward-thor.sh` | `tools/thor/legacy/build-forward-thor.sh` | 02a61af |
| `build-ops-thor.sh` | `tools/thor/legacy/build-ops-thor.sh` | 02a61af |
| `run-forward-sanitizers-thor.sh` | `tools/thor/legacy/run-forward-sanitizers-thor.sh` | 02a61af |
| `run-sanitizers-thor.sh` | `tools/thor/legacy/run-sanitizers-thor.sh` | 02a61af |
| `native/tests/lfm3_batch.cu` | `native/apps/lfm3_batch.cu` | 46a27a3 |
| `native/tests/lfm3_score.cu` | `native/apps/lfm3_score.cu` | 46a27a3 |
| `native/tests/lfm3_speech.cu` | `native/apps/lfm3_speech.cu` | 46a27a3 |
| `native/tests/lfm3_tree.cu` | `native/apps/lfm3_tree.cu` | 46a27a3 |
| `native/tools/attention_bench.cu` | `native/bench/attention_bench.cu` | 46a27a3 |
| `native/tools/batch_roofline.cu` | `native/bench/batch_roofline.cu` | 46a27a3 |
| `native/tools/component_floor.cu` | `native/bench/component_floor.cu` | 46a27a3 |
| `native/tests/fp32_linear_bench.cu` | `native/bench/fp32_linear_bench.cu` | 46a27a3 |
| `native/tools/gemv.sh` | `native/bench/gemv.sh` | 46a27a3 |
| `native/tools/gemv_bench.cu` | `native/bench/gemv_bench.cu` | 46a27a3 |
| `native/tools/gemv_concurrent.cu` | `native/bench/gemv_concurrent.cu` | 46a27a3 |
| `native/tests/lfm3_decode_gemm_bench.cu` | `native/bench/lfm3_decode_gemm_bench.cu` | 46a27a3 |
| `native/tests/lfm3_decode_tile_sweep.cu` | `native/bench/lfm3_decode_tile_sweep.cu` | 46a27a3 |
| `native/tests/lfm3_gemm_sweep.cu` | `native/bench/lfm3_gemm_sweep.cu` | 46a27a3 |
| `native/tests/lfm3_mma_peak.cu` | `native/bench/lfm3_mma_peak.cu` | 46a27a3 |
| `native/tools/roofline.cu` | `native/bench/roofline.cu` | 46a27a3 |
| `native/tools/sort_bench.cu` | `native/bench/sort_bench.cu` | 46a27a3 |
| `native/tools/attention_debug.cu` | `native/probes/attention_debug.cu` | 46a27a3 |
| `native/tools/compare_attention_paths.sh` | `native/probes/compare_attention_paths.sh` | 46a27a3 |
| `native/tools/compare_hashes.sh` | `native/probes/compare_hashes.sh` | 46a27a3 |
| `native/tests/config_probe.cpp` | `native/probes/config_probe.cpp` | 46a27a3 |
| `native/tools/gemm_columns.cu` | `native/probes/gemm_columns.cu` | 46a27a3 |
| `native/tools/gemm_n_invariance.cu` | `native/probes/gemm_n_invariance.cu` | 46a27a3 |
| `native/tools/gemv_order.cu` | `native/probes/gemv_order.cu` | 46a27a3 |
| `native/tests/lfm3_boundary_control.cu` | `native/probes/lfm3_boundary_control.cu` | 46a27a3 |
| `native/tests/lfm3_boundary_observer.cpp` | `native/probes/lfm3_boundary_observer.cpp` | 46a27a3 |
| `native/tests/lfm3_config_probe.cpp` | `native/probes/lfm3_config_probe.cpp` | 46a27a3 |
| `native/tests/lfm3_decode_batch_probe.cu` | `native/probes/lfm3_decode_batch_probe.cu` | 46a27a3 |
| `native/tests/lfm3_gemm_probe.cu` | `native/probes/lfm3_gemm_probe.cu` | 46a27a3 |
| `native/tests/lfm3_lt_policy_probe.cu` | `native/probes/lfm3_lt_policy_probe.cu` | 46a27a3 |
| `native/tests/lfm3_prefill_probe.cu` | `native/probes/lfm3_prefill_probe.cu` | 46a27a3 |
| `native/tests/lfm3_projection_bedrock_probe.cu` | `native/probes/lfm3_projection_bedrock_probe.cu` | 46a27a3 |
| `native/tests/lfm3_projection_modes_probe.cu` | `native/probes/lfm3_projection_modes_probe.cu` | 46a27a3 |
| `native/tests/lfm3_projection_probe.cu` | `native/probes/lfm3_projection_probe.cu` | 46a27a3 |
| `native/tests/lfm3_serving_ops_probe.cu` | `native/probes/lfm3_serving_ops_probe.cu` | 46a27a3 |
| `native/tests/lfm3_thin_hybrid_runtime_candidate.cuh` | `native/probes/lfm3_thin_hybrid_runtime_candidate.cuh` | 46a27a3 |
| `native/tests/lfm3_thin_v5_probe.cu` | `native/probes/lfm3_thin_v5_probe.cu` | 46a27a3 |
| `native/tests/lfm3_tma_store_probe.cu` | `native/probes/lfm3_tma_store_probe.cu` | 46a27a3 |
| `native/tests/lfm3_wmma_policy_probe.cu` | `native/probes/lfm3_wmma_policy_probe.cu` | 46a27a3 |
| `native/tests/lfm3_wmma_runtime_candidate.cuh` | `native/probes/lfm3_wmma_runtime_candidate.cuh` | 46a27a3 |
| `native/tests/math_attention_dynamic.cu` | `native/probes/math_attention_dynamic.cu` | 46a27a3 |
| `native/tests/math_pack.cu` | `native/probes/math_pack.cu` | 46a27a3 |
| `native/tests/moe.cu` | `native/probes/moe.cu` | 46a27a3 |
| `native/tests/moe_accumulate.cu` | `native/probes/moe_accumulate.cu` | 46a27a3 |
| `native/tests/rms_warp.cu` | `native/probes/rms_warp.cu` | 46a27a3 |
| `native/tests/safetensors_probe.cpp` | `native/probes/safetensors_probe.cpp` | 46a27a3 |
| `native/tests/tokenizer_count_probe.cpp` | `native/probes/tokenizer_count_probe.cpp` | 46a27a3 |
| `native/tools/analyze_prefix_structure.py` | `tools/upstream/analyze_prefix_structure.py` | 46a27a3 |
| `native/tools/analyze_profile.py` | `tools/upstream/analyze_profile.py` | 46a27a3 |
| `native/tools/analyze_weight_structure.py` | `tools/upstream/analyze_weight_structure.py` | 46a27a3 |
| `native/tools/batch_table.py` | `tools/upstream/batch_table.py` | 46a27a3 |
| `native/tools/benchmark_common.py` | `tools/upstream/benchmark_common.py` | 46a27a3 |
| `native/tools/benchmark_decode.py` | `tools/upstream/benchmark_decode.py` | 46a27a3 |
| `native/tools/benchmark_text_suite.py` | `tools/upstream/benchmark_text_suite.py` | 46a27a3 |
| `native/tools/coherency_test.py` | `tools/upstream/coherency_test.py` | 46a27a3 |
| `native/tools/model_validate.py` | `tools/upstream/model_validate.py` | 46a27a3 |
| `native/tools/profile_native.py` | `tools/upstream/profile_native.py` | 46a27a3 |
| `native/tools/suite_audit.py` | `tools/upstream/suite_audit.py` | 46a27a3 |
| `native/tools/suite_stage.py` | `tools/upstream/suite_stage.py` | 46a27a3 |
| `native/tools/suite_validate.py` | `tools/upstream/suite_validate.py` | 46a27a3 |
| `native/tools/summarize_floor.py` | `tools/upstream/summarize_floor.py` | 46a27a3 |
| `native/tools/summarize_profile_suite.py` | `tools/upstream/summarize_profile_suite.py` | 46a27a3 |
| `native/tools/transport_benchmark.py` | `tools/upstream/transport_benchmark.py` | 46a27a3 |
| `native/tools/validate_batch_graphs.py` | `tools/upstream/validate_batch_graphs.py` | 46a27a3 |
