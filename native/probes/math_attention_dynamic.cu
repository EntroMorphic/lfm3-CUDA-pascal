#include <lfm/math_attention.cuh>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template <typename T> struct Buffer {
  T* p = nullptr;
  explicit Buffer(size_t n) { check(cudaMalloc(&p, n * sizeof(T))); }
  ~Buffer() { cudaFree(p); }
};

int main() {
  try {
    lfm::RuntimeShape s{};
    s.hidden = 384; s.query_heads = 6; s.kv_heads = 2; s.head_dim = 64;
    s.context = 9; s.attention_layers = 2;
    lfm::MathAttentionDynamic batched(s), isolated(s);
    for (int batch : {1, 31, 32, 33, 65}) for (bool ragged : {false, true}) {
      std::vector<int> positions(batch, 8);
      if (ragged) for (int row = 0; row < batch; ++row)
        positions[row] = batch >= 34 && row < 34 ? 8 : (row * 7) % 9;
      std::vector<__nv_bfloat16> query(static_cast<size_t>(batch) * s.qkv_width());
      std::vector<__nv_bfloat16> keys(static_cast<size_t>(batch) * s.kv_batch_stride());
      auto values = keys;
      const auto nan = __float2bfloat16_rn(std::numeric_limits<float>::quiet_NaN());
      std::fill(query.begin(), query.end(), nan);
      std::fill(keys.begin(), keys.end(), nan); std::fill(values.begin(), values.end(), nan);
      for (int row = 0; row < batch; ++row) {
        for (int d = 0; d < s.hidden; ++d)
          query[static_cast<size_t>(row) * s.qkv_width() + d] = __float2bfloat16_rn(((row * 13 + d * 7) % 101 - 50) / 128.f);
        for (int h = 0; h < s.kv_heads; ++h) for (int t = 0; t <= positions[row]; ++t) for (int d = 0; d < s.head_dim; ++d) {
          const size_t i = static_cast<size_t>(row) * s.kv_batch_stride() + s.kv_layer_stride() + (h * s.context + t) * s.head_dim + d;
          keys[i] = __float2bfloat16_rn(((row * 11 + h * 17 + t * 3 + d * 7) % 97 - 48) / 128.f);
          values[i] = __float2bfloat16_rn(((row * 19 + h * 13 + t * 5 + d * 3) % 89 - 44) / 128.f);
        }
      }
      Buffer<__nv_bfloat16> q(query.size()), k(keys.size()), v(values.size());
      Buffer<int> p(batch);
      Buffer<__nv_bfloat16> out(static_cast<size_t>(batch) * s.hidden + 2), ref(static_cast<size_t>(batch) * s.hidden);
      check(cudaMemcpy(q.p, query.data(), query.size() * 2, cudaMemcpyHostToDevice));
      check(cudaMemcpy(k.p, keys.data(), keys.size() * 2, cudaMemcpyHostToDevice));
      check(cudaMemcpy(v.p, values.data(), values.size() * 2, cudaMemcpyHostToDevice));
      check(cudaMemcpy(p.p, positions.data(), batch * sizeof(int), cudaMemcpyHostToDevice));
      check(cudaMemset(out.p, 0xa5, (static_cast<size_t>(batch) * s.hidden + 2) * 2));
      batched.run(q.p, k.p + s.kv_layer_stride(), v.p + s.kv_layer_stride(), out.p + 1, p.p, batch);
      for (int row = 0; row < batch; ++row)
        isolated.run(q.p + static_cast<size_t>(row) * s.qkv_width(),
            k.p + static_cast<size_t>(row) * s.kv_batch_stride() + s.kv_layer_stride(),
            v.p + static_cast<size_t>(row) * s.kv_batch_stride() + s.kv_layer_stride(),
            ref.p + static_cast<size_t>(row) * s.hidden, p.p + row, 1);
      std::vector<__nv_bfloat16> actual(static_cast<size_t>(batch) * s.hidden + 2), expected(static_cast<size_t>(batch) * s.hidden);
      check(cudaMemcpy(actual.data(), out.p, actual.size() * 2, cudaMemcpyDeviceToHost));
      check(cudaMemcpy(expected.data(), ref.p, expected.size() * 2, cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < expected.size(); ++i)
        if (!std::isfinite(__bfloat162float(actual[i + 1])) || std::memcmp(&actual[i + 1], &expected[i], 2))
          throw std::runtime_error("all-row attention mismatch");
      for (size_t i : {size_t(0), actual.size() - 1}) {
        uint16_t bits; std::memcpy(&bits, &actual[i], 2);
        if (bits != 0xa5a5) throw std::runtime_error("attention output guard overwritten");
      }
      for (int invalid : {-1, 9, std::numeric_limits<int>::max()}) {
        check(cudaMemcpy(p.p, &invalid, sizeof(int), cudaMemcpyHostToDevice));
        bool rejected = false;
        try { batched.run(q.p, k.p, v.p, out.p, p.p, batch); }
        catch (const std::runtime_error&) { rejected = true; }
        if (!rejected) throw std::runtime_error("invalid attention position accepted");
      }
    }
    std::puts("10 all-row attention grouping cases passed, including poisoned unused KV and invalid positions");
  } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
