#include <lfm/math_attention.cuh>
#include <cstdio>
#include <cstring>
#include <vector>

using BF = __nv_bfloat16;
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template <typename T> struct Buffer {
  T* p = nullptr;
  explicit Buffer(size_t n) { check(cudaMalloc(&p, n * sizeof(T))); }
  ~Buffer() { cudaFree(p); }
};

int main() {
  try {
    // Include partial sequence tiles, reordered rows, GQA, non-power-of-two
    // dimensions, a dimension wider than a block, and token/block tails.
    auto run = [](int dim, int tokens, int heads = 6, int kv_heads = 2, int sequences = 32, int rows = 5) {
      const int capacity = tokens + 7;
      const int stride = (heads + 2 * kv_heads) * dim;
      const size_t cache_stride = static_cast<size_t>(kv_heads) * capacity * dim;
      std::vector<BF> query(rows * stride), keys(rows * cache_stride), values(keys.size());
      for (size_t i = 0; i < query.size(); ++i) query[i] = __float2bfloat16_rn(static_cast<int>(i % 101) * .03125f - 1.f);
      for (size_t i = 0; i < keys.size(); ++i) {
        keys[i] = __float2bfloat16_rn(static_cast<int>(i % 73) * .0625f - 2.f);
        values[i] = __float2bfloat16_rn(static_cast<int>(i % 97) * .125f - 3.f);
      }
      auto edge = [](size_t i, __nv_bfloat16 fallback) {
        const uint16_t bits[] = {0x8000, 0x0001, 0x8001, 0x7f7f, 0xff7f};
        if (i % 17 >= 5) return fallback;
        __nv_bfloat16 value; std::memcpy(&value, &bits[i % 17], 2); return value;
      };
      for (size_t i = 0; i < query.size(); ++i) query[i] = edge(i, query[i]);
      for (size_t i = 0; i < keys.size(); ++i) { keys[i] = edge(i, keys[i]); values[i] = edge(i + 3, values[i]); }
      const uint16_t nan_bits = 0x7fc0;
      __nv_bfloat16 nan; std::memcpy(&nan, &nan_bits, 2);
      for (int row = 0; row < rows; ++row) {
        for (int d = heads * dim; d < stride; ++d) query[row * stride + d] = nan;
        for (int h = 0; h < kv_heads; ++h) for (int t = tokens; t < capacity; ++t) for (int d = 0; d < dim; ++d) {
          const size_t i = row * cache_stride + (h * capacity + t) * dim + d;
          keys[i] = nan; values[i] = nan;
        }
      }
      std::vector<int> mapping(sequences, -1);
      for (int i = 0; i < rows; ++i) mapping[i] = rows - 1 - i;
      const size_t qn = static_cast<size_t>(sequences) * heads * dim;
      const size_t kn = qn * tokens;
      Buffer<BF> dq(query.size()), dk(keys.size()), dv(values.size());
      Buffer<int> dm(sequences);
      Buffer<float> output(qn + 2 * kn + 16);
      check(cudaMemcpy(dq.p, query.data(), query.size() * sizeof(BF), cudaMemcpyHostToDevice));
      check(cudaMemcpy(dk.p, keys.data(), keys.size() * sizeof(BF), cudaMemcpyHostToDevice));
      check(cudaMemcpy(dv.p, values.data(), values.size() * sizeof(BF), cudaMemcpyHostToDevice));
      check(cudaMemcpy(dm.p, mapping.data(), sequences * sizeof(int), cudaMemcpyHostToDevice));
      check(cudaMemset(output.p, 0xff, (qn + 2 * kn + 16) * sizeof(float)));
      const float scale = .3535533845424652f;
      lfm::math_detail::pack<<<dim3((tokens * dim + 255) / 256, std::min(heads, 65535),
                                   sequences * ((heads + 65534) / 65535)), 256>>>(
          dq.p, dk.p, dv.p, output.p, output.p + qn, output.p + qn + kn,
          heads, kv_heads, dim, tokens, capacity, scale, dm.p, cache_stride, stride);
      check(cudaGetLastError());
      std::vector<float> actual(qn + 2 * kn + 16);
      check(cudaMemcpy(actual.data(), output.p, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
      for (int s = 0; s < sequences; ++s) for (int h = 0; h < heads; ++h) {
        const int row = mapping[s];
        for (int d = 0; d < dim; ++d) {
          const float expected = row < 0 ? 0 : __bfloat162float(query[row * stride + h * dim + d]) * scale;
          if (std::memcmp(&actual[(s * heads + h) * dim + d], &expected, sizeof(float))) throw std::runtime_error("query bits mismatch");
        }
        for (int t = 0; t < tokens; ++t) for (int d = 0; d < dim; ++d) {
          const size_t source = row < 0 ? 0 : row * cache_stride + (h / (heads / kv_heads)) * capacity * dim + t * dim + d;
          const size_t dest = (static_cast<size_t>(s) * heads + h) * tokens * dim + t * dim + d;
          const float k = row < 0 ? 0 : __bfloat162float(keys[source]) * scale;
          const float v = row < 0 ? 0 : __bfloat162float(values[source]);
          if (std::memcmp(&actual[qn + dest], &k, sizeof(float)) || std::memcmp(&actual[qn + kn + dest], &v, sizeof(float)))
            throw std::runtime_error("KV bits mismatch");
        }
      }
      for (size_t i = qn + 2 * kn; i < actual.size(); ++i)
        if (!std::isnan(actual[i])) throw std::runtime_error("guard overwritten");
    };
    for (int dim : {32, 64, 96, 320}) for (int tokens : {1, 3, 129}) run(dim, tokens);
    run(2, 1, 65536, 2, 2, 1);  // Cross CUDA's grid-y limit without huge KV storage.
    std::puts("13 attention packing cases passed");
  } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
