// Forced inputs compare every row's production state, not just generated text.
#include <cuda_bf16.h>
__nv_bfloat16* graph_test_trace = nullptr;
#define LFM_BATCH_GRAPH_TEST
#define main native_main_for_graph_test
#include "../src/main.cu"
#undef main

template <class T> void identical(const DeviceBuffer<T>& a, const DeviceBuffer<T>& b, const char* name) {
  require(a.size == b.size, "graph test size mismatch");
  std::vector<unsigned char> left(a.size * sizeof(T)), right(left.size());
  check_cuda(cudaMemcpy(left.data(), a.data, left.size(), cudaMemcpyDeviceToHost), "read eager state");
  check_cuda(cudaMemcpy(right.data(), b.data, right.size(), cudaMemcpyDeviceToHost), "read graph state");
  require(left == right, std::string("graph byte mismatch: ") + name);
}

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    require(options.model && !model_config.moe && !options.attention_math && !options.attention_math_dynamic,
            "graph fixture requires online dense checkpoint");
    const int batch = options.batch_size;
    lfm::Safetensors source(*options.model); validate_model(source);
    ModelWeights weights(source, kContextLength, false);
    cublasHandle_t handle;
    check_cublas(cublasCreate(&handle), "graph test handle");
    check_cublas(cublasSetStream(handle, cudaStreamPerThread), "graph test stream");
    check_cublas(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH), "graph test math");
    check_cublas(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED), "graph test atomics");
    DeviceBuffer<std::byte> workspace(kCublasWorkspaceBytes);
    check_cuda(cudaMemset(workspace.data, 0xff, workspace.size), "poison graph workspace");
    check_cublas(cublasSetWorkspace(handle, workspace.data, workspace.size), "graph workspace");
    linear_scratch = std::make_unique<LinearScratch>();
    const size_t kv = static_cast<size_t>(batch) * kKvBatchStride;
    DeviceBuffer<__nv_bfloat16> ka(kv), va(kv), kb(kv), vb(kv);
    for (auto* p : {&ka, &va, &kb, &vb})
      check_cuda(cudaMemset(p->data, 0xff, p->size * sizeof(__nv_bfloat16)), "poison graph KV");
    LayerBuffers a(batch), b(batch);
    DeviceBuffer<int> mask(batch);
    DeviceBuffer<__nv_bfloat16> ta(static_cast<size_t>(batch) * model_config.shape.layers * kHiddenSize), tb(ta.size);
    a.stopped_rows = mask.data; b.stopped_rows = mask.data;
    const int policy[] = {options.attention_window, options.attention_sinks, options.attention_sparse_after};
    for (auto* p : {&a, &b}) check_cuda(cudaMemcpy(p->attention_policy.data, policy, sizeof(policy), cudaMemcpyHostToDevice), "graph policy");
    const bool sampling = options.temperature > 0 || options.top_k > 0 || options.top_p < 1 || options.min_p > 0;
    std::unique_ptr<BatchSampler> sa, sb;
    if (sampling) {
      sa = std::make_unique<BatchSampler>(options, batch); sb = std::make_unique<BatchSampler>(options, batch);
      std::vector<uint64_t> seeds(batch);
      for (int row = 0; row < batch; ++row) seeds[row] = options.seed + static_cast<uint64_t>(row) * kRowSeedStride;
      sa->set_row_seeds(seeds); sb->set_row_seeds(seeds);
    }
    std::vector<int> tokens(batch), positions(batch), stopped(batch);
    auto input = [&](int step) {
      for (int row = 0; row < batch; ++row) {
        tokens[row] = (row * 11 + step * 13 + 1) % kVocabularySize;
        stopped[row] = step >= 9 && row % 3 == 0;
        positions[row] = stopped[row] ? 8 : step;
      }
      check_cuda(cudaMemcpy(mask.data, stopped.data(), batch * sizeof(int), cudaMemcpyHostToDevice), "graph stop mask");
      for (auto* p : {&a, &b}) {
        check_cuda(cudaMemcpy(p->input_token.data, tokens.data(), batch * sizeof(int), cudaMemcpyHostToDevice), "forced tokens");
        check_cuda(cudaMemcpy(p->position.data, positions.data(), batch * sizeof(int), cudaMemcpyHostToDevice), "forced positions");
      }
    };
    auto forward = [&](LayerBuffers& buffers, DeviceBuffer<__nv_bfloat16>& keys,
                       DeviceBuffer<__nv_bfloat16>& values, DeviceBuffer<__nv_bfloat16>& trace, bool reset) {
      graph_test_trace = trace.data;
      run_full_token_device(handle, weights, keys.data, values.data, buffers,
                            reset, false, nullptr, nullptr, true, !sampling);
    };
    auto advance = [&](LayerBuffers& buffers) {
      increment_position_batch<<<(batch + 255) / 256, 256>>>(buffers.position.data, batch);
      advance_selected_tokens<<<(batch + 255) / 256, 256>>>(buffers.input_token.data, buffers.selected_token.data, batch);
    };
    input(0); forward(a, ka, va, ta, true); forward(b, kb, vb, tb, true);
    if (sampling) { sa->select_device(a.logits.data, batch, 0, a.selected_token.data); sb->select_device(b.logits.data, batch, 0, b.selected_token.data); }
    check_cuda(cudaDeviceSynchronize(), "graph fixture warmup");
    graph_test_trace = tb.data;
    {
      BatchDecodeGraph graph(handle, weights, kb.data, vb.data, b, !sampling);
      const int steps = std::min(kContextLength - 1, options.max_new_tokens > 0 ? options.max_new_tokens : 66);
      for (int step = 1; step <= steps; ++step) {
        input(step); forward(a, ka, va, ta, false);
        if (sampling) sa->select_device(a.logits.data, batch, step, a.selected_token.data);
        advance(a); graph.launch();
        if (sampling) sb->select_device(b.logits.data, batch, step, b.selected_token.data);
        advance(b);
        identical(ta, tb, "every layer/all rows");
        identical(a.hidden, b.hidden, "hidden"); identical(a.normalized, b.normalized, "normalized");
        identical(a.logits, b.logits, "logits"); identical(a.convolution_states, b.convolution_states, "convolution state");
        identical(ka, kb, "keys and guards"); identical(va, vb, "values and guards");
        identical(a.selected_token, b.selected_token, "draws"); identical(a.input_token, b.input_token, "advanced tokens");
        identical(a.position, b.position, "positions");
      }
      std::cout << "PASS forward graph: " << steps << " forced positions, batch " << batch
                << ", all-row layers/logits/KV/conv/draw bytes\n";
    }
    graph_test_trace = nullptr; linear_scratch.reset(); cublasDestroy(handle);
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
