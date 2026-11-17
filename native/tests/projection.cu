// Compile the production translation unit, not a copied projection algorithm.
#define main liquid_engine_entry_for_projection_test
#include "../src/main.cu"
#undef main

int main(int argc, char**) {
  try {
    kHiddenSize = 128; kMlpWidth = 256; kVocabularySize = 259; kQkvWidth = 384;
    linear_scratch = std::make_unique<LinearScratch>();
    cublasHandle_t handle;
    check_cublas(cublasCreate(&handle), "projection test handle");
    check_cublas(cublasSetStream(handle, cudaStreamPerThread), "projection test stream");
    DeviceBuffer<__nv_bfloat16> weight(259 * 128), alias(32 * 128);
    std::vector<__nv_bfloat16> host_weight(weight.size), host_alias(alias.size);
    for (size_t i = 0; i < host_weight.size(); ++i)
      host_weight[i] = __float2bfloat16_rn((static_cast<int>(i * 17 % 101) - 50) / 64.f);
    for (size_t i = 0; i < host_alias.size(); ++i)
      host_alias[i] = __float2bfloat16_rn((static_cast<int>(i * 13 % 97) - 48) / 64.f);
    check_cuda(cudaMemcpy(weight.data, host_weight.data(), weight.size * 2, cudaMemcpyHostToDevice), "projection test weights");
    check_cuda(cudaMemcpy(alias.data, host_alias.data(), alias.size * 2, cudaMemcpyHostToDevice), "projection alias input");
    bool rejected = false;
    try { linear_bf16(handle, alias.data, weight.data, alias.data, 128, 128, 32); }
    catch (const std::invalid_argument&) { rejected = true; }
    std::vector<__nv_bfloat16> after_alias(alias.size);
    check_cuda(cudaMemcpy(after_alias.data(), alias.data, alias.size * 2, cudaMemcpyDeviceToHost), "projection alias result");
    if (!rejected || std::memcmp(after_alias.data(), host_alias.data(), alias.size * 2))
      throw std::runtime_error("overlapping projection accepted or modified input before rejection");
    if (argc > 1) { cublasDestroy(handle); return 0; }
    DeviceBuffer<__nv_bfloat16> invalid_output(32 * 259);
    check_cuda(cudaMemset(invalid_output.data, 0xa5, invalid_output.size * 2), "invalid projection sentinel");
    auto invalid = [&](const __nv_bfloat16* in, const __nv_bfloat16* w, __nv_bfloat16* out,
                       int n, int m, int batch, int stride = 0) {
      bool caught = false;
      try { linear_bf16(handle, in, w, out, n, m, batch, stride); }
      catch (const std::invalid_argument&) { caught = true; }
      if (!caught) throw std::runtime_error("invalid projection accepted");
    };
    invalid(alias.data, weight.data, alias.data + 1, 128, 128, 31);
    invalid(alias.data, weight.data, weight.data, 128, 128, 32);
    invalid(nullptr, weight.data, invalid_output.data, 128, 128, 32);
    invalid(alias.data, nullptr, invalid_output.data, 128, 128, 32);
    invalid(alias.data, weight.data, nullptr, 128, 128, 32);
    invalid(alias.data, weight.data, invalid_output.data, 0, 128, 32);
    invalid(alias.data, weight.data, invalid_output.data, 128, 0, 32);
    invalid(alias.data, weight.data, invalid_output.data, 128, 128, 0);
    invalid(alias.data, weight.data, invalid_output.data, 128, 128, 65536);
    invalid(alias.data, weight.data, invalid_output.data, 128, 128, 32, 127);
    invalid(alias.data, weight.data, invalid_output.data, 257, 128, 32);
    invalid(alias.data, weight.data, invalid_output.data, 128, 385, 32);
    std::vector<uint16_t> untouched(invalid_output.size);
    check_cuda(cudaMemcpy(untouched.data(), invalid_output.data, untouched.size() * 2, cudaMemcpyDeviceToHost), "invalid projection result");
    for (uint16_t bits : untouched) if (bits != 0xa5a5) throw std::runtime_error("invalid projection wrote output");

    int cases = 0;
    for (int batch : {1, 2, 31, 32, 33, 63, 64, 65}) for (int width : {64, 259})
      for (int offset : {0, 1}) for (int padding : {0, 7}) {
        const int guard = 128 + offset, stride = width + padding;
        DeviceBuffer<__nv_bfloat16> input(guard + batch * 128 + 128), output(guard + batch * stride + 128);
        DeviceBuffer<__nv_bfloat16> expected(batch * width);
        std::vector<__nv_bfloat16> host(input.size), observed(output.size), target(expected.size);
        for (size_t i = 0; i < host.size(); ++i)
          host[i] = __float2bfloat16_rn((static_cast<int>(i * 13 % 103) - 51) / 64.f);
        check_cuda(cudaMemcpy(input.data, host.data(), input.size * 2, cudaMemcpyHostToDevice), "projection input");
        check_cuda(cudaMemset(output.data, 0xa5, output.size * 2), "projection guards");
        linear_bf16(handle, input.data + guard, weight.data, output.data + guard, 128, width, batch, stride);
        // Canonical padded one-row GEMMs independently check every output row,
        // including later tiles and odd-vocabulary/misaligned fallback paths.
        for (int row = 0; row < batch; ++row)
          linear_bf16(handle, input.data + guard + row * 128, weight.data,
                      expected.data + row * width, 128, width, 1);
        check_cuda(cudaMemcpy(observed.data(), output.data, output.size * 2, cudaMemcpyDeviceToHost), "projection output");
        check_cuda(cudaMemcpy(target.data(), expected.data, expected.size * 2, cudaMemcpyDeviceToHost), "projection reference");
        for (size_t i = 0; i < observed.size(); ++i) {
          const bool active = i >= static_cast<size_t>(guard) && i < static_cast<size_t>(guard + batch * stride) &&
              (i - guard) % stride < static_cast<size_t>(width);
          if (active) {
            const size_t index = ((i - guard) / stride) * width + (i - guard) % stride;
            if (!std::isfinite(__bfloat162float(observed[i])) || std::memcmp(&observed[i], &target[index], 2))
              throw std::runtime_error("projection row mismatch");
          } else {
            uint16_t bits; std::memcpy(&bits, &observed[i], 2);
            if (bits != 0xa5a5) throw std::runtime_error("projection guard or row padding overwritten");
          }
        }
        ++cases;
      }
    std::printf("%d all-row projection cases passed; overlapping input rejected before writes\n", cases);
    cublasDestroy(handle);
  } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
