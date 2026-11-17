#include <lfm/moe.cuh>
#include <cstdio>
#include <fstream>
#include <vector>
#include <stdexcept>

void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template <typename T> struct Buffer {
  T* p;
  explicit Buffer(size_t count) { check(cudaMalloc(&p, count * sizeof(T))); }
  ~Buffer() { cudaFree(p); }
};

int main(int argc, char** argv) {
  try {
    if (argc != 3) return 2;
    std::ifstream in(argv[1], std::ios::binary);
    int rows, experts, k, norm, biased;
    float scaling;
    for (int* value : {&rows, &experts, &k, &norm, &biased}) in.read(reinterpret_cast<char*>(value), 4);
    in.read(reinterpret_cast<char*>(&scaling), 4);
    if (!in || rows < 1 || rows > 64 || experts < 1 || experts > 256 || k < 1 || k > experts) return 2;
    std::vector<__nv_bfloat16> logits(rows * experts), weights(rows * k);
    std::vector<float> bias(experts);
    in.read(reinterpret_cast<char*>(logits.data()), logits.size() * 2);
    in.read(reinterpret_cast<char*>(bias.data()), bias.size() * 4);
    if (!in) return 2;
    Buffer<__nv_bfloat16> x(logits.size()), p(logits.size()), w(weights.size());
    Buffer<float> b(experts);
    Buffer<int> selected(rows * k), counts(experts + 1), group_rows(rows * experts), group_ranks(rows * experts);
    check(cudaMemcpy(x.p, logits.data(), logits.size() * 2, cudaMemcpyHostToDevice));
    check(cudaMemcpy(b.p, bias.data(), bias.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemset(counts.p, 0, (experts + 1) * 4));
    lfm::moe_sigmoid<<<(logits.size() + 255) / 256, 256>>>(x.p, p.p, logits.size());
    lfm::moe_select<<<rows, 1>>>(x.p, p.p, biased ? b.p : nullptr, selected.p, w.p, counts.p, experts, k, norm, scaling);
    lfm::moe_groups<<<(experts + 255) / 256, 256>>>(selected.p, group_rows.p, group_ranks.p, counts.p, rows, experts, k);
    check(cudaGetLastError());
    std::vector<int> chosen(rows * k), host_counts(experts + 1);
    check(cudaMemcpy(host_counts.data(), counts.p, host_counts.size() * 4, cudaMemcpyDeviceToHost));
    if (host_counts[experts]) { std::fprintf(stderr, "nonfinite routing\n"); return 1; }
    check(cudaMemcpy(chosen.data(), selected.p, chosen.size() * 4, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(weights.data(), w.p, weights.size() * 2, cudaMemcpyDeviceToHost));
    std::ofstream out(argv[2], std::ios::binary);
    out.write(reinterpret_cast<const char*>(chosen.data()), chosen.size() * 4);
    out.write(reinterpret_cast<const char*>(weights.data()), weights.size() * 2);
    for (int e = 0; e < experts; ++e) {
      const int count = host_counts[e];
      if (count < 0 || count > rows) return 1;
      std::vector<int> indices(count), ranks(count);
      if (count) {
        check(cudaMemcpy(indices.data(), group_rows.p + e * rows, count * 4, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(ranks.data(), group_ranks.p + e * rows, count * 4, cudaMemcpyDeviceToHost));
      }
      out.write(reinterpret_cast<const char*>(&count), 4);
      out.write(reinterpret_cast<const char*>(indices.data()), count * 4);
      out.write(reinterpret_cast<const char*>(ranks.data()), count * 4);
    }
    return out ? 0 : 2;
  } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
