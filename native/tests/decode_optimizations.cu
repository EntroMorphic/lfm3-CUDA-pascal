// Exercise the production translation unit, not a copy of its projection code.
#define main liquid_engine_entry_for_decode_test
#include "../src/main.cu"
#undef main

namespace {
void same_bytes(const __nv_bfloat16* expected, const __nv_bfloat16* actual,
                size_t count, const char* operation) {
  std::vector<__nv_bfloat16> a(count), b(count);
  if (!count) return;
  check_cuda(cudaMemcpy(a.data(), expected, count * 2, cudaMemcpyDeviceToHost), "read oracle");
  check_cuda(cudaMemcpy(b.data(), actual, count * 2, cudaMemcpyDeviceToHost), "read candidate");
  for (size_t i = 0; i < count; ++i)
    require(std::isfinite(__bfloat162float(b[i])) && std::memcmp(&a[i], &b[i], 2) == 0,
            std::string(operation) + " differs at " + std::to_string(i));
}

void initialize(DeviceBuffer<__nv_bfloat16>& buffer, int seed) {
  std::vector<__nv_bfloat16> host(buffer.size);
  for (size_t i = 0; i < host.size(); ++i)
    host[i] = __float2bfloat16_rn(sinf(static_cast<float>((i * seed) % 997) * .17f) * .125f);
  check_cuda(cudaMemcpy(buffer.data, host.data(), host.size() * 2, cudaMemcpyHostToDevice), "initialize fixture");
}

void attention_case(int dim, int context, int kv_heads, int policy_index, int mask, bool math) {
  constexpr int batch = 5, heads = 4;
  lfm::RuntimeShape s{};
  s.head_dim = dim; s.query_heads = heads; s.kv_heads = kv_heads;
  s.hidden = heads * dim; s.context = context; s.attention_layers = 2;
  DeviceBuffer<__nv_bfloat16> q(batch * s.qkv_width()), keys(batch * s.kv_batch_stride()), values(keys.size);
  DeviceBuffer<__nv_bfloat16> expected(batch * s.hidden), actual(expected.size);
  DeviceBuffer<int> positions(batch), stopped(batch), policy(3);
  const int policies[][3] = {{0, 0, 0}, {33, 4, 32}, {1, 0, 0}, {64, 100, 128}};
  std::vector<int> hp{0, 31, 32, std::min(context - 1, 65), context - 1}, hm(batch);
  for (int row = 0; row < batch; ++row) hm[row] = mask == 2 || (mask == 1 && row % 2 == 0);
  initialize(q, 3);
  std::vector<__nv_bfloat16> hk(keys.size), hv(values.size);
  for (size_t i = 0; i < hk.size(); ++i) {
    const int row = i / s.kv_batch_stride(), token = (i / dim) % context;
    hk[i] = __float2bfloat16_rn(token > hp[row] ? nanf("") : cosf((i % 991) * .17f) * 2.f);
    hv[i] = __float2bfloat16_rn(token > hp[row] ? nanf("") : sinf((i % 983) * .13f));
  }
  check_cuda(cudaMemcpy(keys.data, hk.data(), hk.size() * 2, cudaMemcpyHostToDevice), "keys");
  check_cuda(cudaMemcpy(values.data, hv.data(), hv.size() * 2, cudaMemcpyHostToDevice), "values");
  check_cuda(cudaMemcpy(positions.data, hp.data(), batch * sizeof(int), cudaMemcpyHostToDevice), "positions");
  check_cuda(cudaMemcpy(stopped.data, hm.data(), batch * sizeof(int), cudaMemcpyHostToDevice), "stopped");
  check_cuda(cudaMemcpy(policy.data, policies[policy_index], 3 * sizeof(int), cudaMemcpyHostToDevice), "policy");
  std::unique_ptr<lfm::MathAttentionDynamic> math_attention;
  if (math) math_attention = std::make_unique<lfm::MathAttentionDynamic>(s);
  auto production = [&](auto* out, const int* stop) {
    if (math) math_attention->run(q.data, keys.data, values.data, out, positions.data, batch,
                                 stop, stop ? hm.data() : nullptr);
    else if (dim == 64)
      lfm::online_softmax_gqa_attention_staged<<<dim3(heads, batch), 1024>>>(
          q.data, keys.data, values.data, out, positions.data, policy.data,
          int(s.kv_batch_stride()), s.hidden, heads, kv_heads, context, stop);
    else lfm::online_attention_dynamic<<<dim3(heads, batch), 1024>>>(
        q.data, keys.data, values.data, out, positions.data, policy.data, s, stop);
  };
  production(expected.data, nullptr);
  std::vector<__nv_bfloat16> reference(expected.size);
  check_cuda(cudaMemcpy(reference.data(), expected.data, reference.size() * 2, cudaMemcpyDeviceToHost), "attention reference");
  for (size_t i = 0; i < reference.size(); ++i)
    if (hm[i / s.hidden]) reference[i] = __float2bfloat16_rn(0.f);
  check_cuda(cudaMemcpy(expected.data, reference.data(), reference.size() * 2, cudaMemcpyHostToDevice), "masked oracle");
  check_cuda(cudaMemset(actual.data, 0xff, actual.size * 2), "poison attention output");
  production(actual.data, mask ? stopped.data : nullptr);
  same_bytes(expected.data, actual.data, actual.size, "production attention mask");
  if (!math) {
    check_cuda(cudaMemset(actual.data, 0xff, actual.size * 2), "poison compact output");
    lfm::online_attention_compact<256><<<dim3(heads, batch), 256, 32 * dim * 2>>>(
        q.data, keys.data, values.data, actual.data, positions.data, policy.data,
        s, mask ? stopped.data : nullptr);
    same_bytes(expected.data, actual.data, actual.size, "compact attention");
  }
}

int expert_cases() {
  kHiddenSize = 256; kMlpWidth = 1024; kVocabularySize = 1024; kQkvWidth = 768;
  linear_scratch = std::make_unique<LinearScratch>();
  cublasHandle_t handle;
  check_cublas(cublasCreate(&handle), "expert handle");
  check_cublas(cublasSetStream(handle, cudaStreamPerThread), "expert stream");
  check_cublas(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED), "expert atomics");
  DeviceBuffer<std::byte> workspace(kCublasWorkspaceBytes);
  check_cuda(cudaMemset(workspace.data, 0xff, workspace.size), "initialize expert workspace");
  check_cublas(cublasSetWorkspace(handle, workspace.data, workspace.size), "expert workspace");
  int cases = 0;
  for (int hidden : {64, 256}) for (int width : {128, 512}) {
    DeviceBuffer<__nv_bfloat16> input(32 * hidden), packed(32 * hidden);
    DeviceBuffer<__nv_bfloat16> w1(2 * width * hidden), w2(hidden * width);
    DeviceBuffer<__nv_bfloat16> gate(32 * 2 * width), product(32 * width), output(32 * hidden);
    DeviceBuffer<__nv_bfloat16> accumulated(32 * hidden), fused(accumulated.size), weights(32 * 4);
    DeviceBuffer<int> rows(32), ranks(32);
    initialize(input, 3); initialize(w1, 7); initialize(w2, 11); initialize(weights, 13);
    std::vector<int> hr(32), hk(32);
    for (int row = 0; row < 32; ++row) { hr[row] = 31 - row; hk[row] = row % 4; }
    check_cuda(cudaMemcpy(rows.data, hr.data(), 32 * sizeof(int), cudaMemcpyHostToDevice), "expert rows");
    check_cuda(cudaMemcpy(ranks.data, hk.data(), 32 * sizeof(int), cudaMemcpyHostToDevice), "expert ranks");
    check_cuda(cudaMemset(accumulated.data, 0, accumulated.size * 2), "zero reference accumulation");
    check_cuda(cudaMemset(fused.data, 0, fused.size * 2), "zero fused accumulation");
    for (int count : {1, 2, 3, 7, 16, 31, 32}) {
      lfm::moe_gather<<<(count * hidden + 255) / 256, 256>>>(input.data, packed.data, rows.data, count, hidden);
      linear_bf16(handle, packed.data, w1.data, gate.data, hidden, 2 * width, count);
      lfm::moe_swiglu<<<dim3((width + 255) / 256, count), 256>>>(gate.data, product.data, width);
      linear_bf16(handle, product.data, w2.data, output.data, width, hidden, count);
      lfm::moe_scatter<<<(count * hidden + 255) / 256, 256>>>(output.data, accumulated.data, rows.data, ranks.data, weights.data, count, hidden, 4);
      lfm::moe_gather_canonical<<<(32 * hidden + 255) / 256, 256>>>(input.data, linear_scratch->input.data, rows.data, count, hidden);
      linear_bf16_canonical(handle, linear_scratch->input.data, w1.data, hidden, 2 * width);
      same_bytes(gate.data, linear_scratch->output.data, count * 2 * width, "expert gate");
      lfm::moe_swiglu_canonical<<<dim3((width + 255) / 256, 32), 256>>>(linear_scratch->output.data, linear_scratch->input.data, count, width);
      same_bytes(product.data, linear_scratch->input.data, count * width, "expert product");
      std::vector<__nv_bfloat16> padding((32 - count) * width);
      if (!padding.empty())
        check_cuda(cudaMemcpy(padding.data(), linear_scratch->input.data + count * width, padding.size() * 2, cudaMemcpyDeviceToHost), "expert padding");
      for (auto value : padding) require(__bfloat162float(value) == 0.f, "nonzero canonical padding");
      linear_bf16_canonical(handle, linear_scratch->input.data, w2.data, width, hidden);
      same_bytes(output.data, linear_scratch->output.data, count * hidden, "expert output");
      lfm::moe_scatter<<<(count * hidden + 255) / 256, 256>>>(linear_scratch->output.data, fused.data, rows.data, ranks.data, weights.data, count, hidden, 4);
      same_bytes(accumulated.data, fused.data, fused.size, "ordered expert scatter");
      ++cases;
    }
  }
  cublasDestroy(handle);
  return cases;
}
}  // namespace

int main() {
  try {
    int cases = 0;
    for (int dim : {32, 64, 96, 128}) for (int context : {66, 513, 2049})
      for (int policy = 0; policy < 4; ++policy) for (int mask = 0; mask < 3; ++mask) {
        attention_case(dim, context, 2, policy, mask, false); ++cases;
      }
    for (int kv : {1, 4}) for (int policy : {0, 1}) for (int mask = 0; mask < 3; ++mask) {
      attention_case(64, 8192, kv, policy, mask, false); ++cases;
    }
    for (int dim : {32, 64, 128}) for (int context : {66, 513, 2049})
      for (int mask = 0; mask < 3; ++mask) {
        attention_case(dim, context, 2, 0, mask, true); ++cases;
      }
    require(cases == 183, "incomplete attention test matrix");
    const int experts = expert_cases();
    check_cuda(cudaGetLastError(), "decode optimization launches");
    std::cout << "PASS: " << cases << " attention and " << experts << " expert cases\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
