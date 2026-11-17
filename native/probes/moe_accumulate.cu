#include <lfm/moe.cuh>
#include <fstream>
#include <vector>
#include <stdexcept>

void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
__global__ void contracted(const __nv_bfloat16* x, const __nv_bfloat16* w,
                            __nv_bfloat16* out, int count, int width, int experts, int e) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count * width) out[i] = __hadd(out[i], __hmul(x[i], w[(i / width) * experts + e]));
}
int main(int argc, char** argv) {
  try {
    if (argc != 3) return 2;
    std::ifstream file(argv[1], std::ios::binary);
    int rows, width, experts;
    for (int* n : {&rows, &width, &experts}) file.read(reinterpret_cast<char*>(n), 4);
    if (!file || rows < 1 || rows > 64 || width < 1 || width > 4096 || experts < 1 || experts > 32) return 2;
    const int count = rows * width;
    std::vector<__nv_bfloat16> x(experts * count), w(rows * experts), result(2 * count);
    file.read(reinterpret_cast<char*>(x.data()), x.size() * 2);
    file.read(reinterpret_cast<char*>(w.data()), w.size() * 2);
    if (!file) return 2;
    __nv_bfloat16 *dx, *dw, *good, *bad;
    int *row_ids, *ranks;
    check(cudaMalloc(&dx, x.size() * 2)); check(cudaMalloc(&dw, w.size() * 2));
    check(cudaMalloc(&good, count * 2)); check(cudaMalloc(&bad, count * 2));
    check(cudaMalloc(&row_ids, rows * 4)); check(cudaMalloc(&ranks, rows * 4));
    check(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
    check(cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
    check(cudaMemset(good, 0, count * 2)); check(cudaMemset(bad, 0, count * 2));
    std::vector<int> ids(rows), slots(rows);
    for (int row = 0; row < rows; ++row) ids[row] = row;
    check(cudaMemcpy(row_ids, ids.data(), rows * 4, cudaMemcpyHostToDevice));
    for (int e = 0; e < experts; ++e) {
      std::fill(slots.begin(), slots.end(), e);
      check(cudaMemcpy(ranks, slots.data(), rows * 4, cudaMemcpyHostToDevice));
      lfm::moe_scatter<<<(count + 255) / 256, 256>>>(dx + e * count, good, row_ids, ranks, dw, rows, width, experts);
      contracted<<<(count + 255) / 256, 256>>>(dx + e * count, dw, bad, rows, width, experts, e);
    }
    check(cudaGetLastError());
    check(cudaMemcpy(result.data(), good, count * 2, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(result.data() + count, bad, count * 2, cudaMemcpyDeviceToHost));
    std::ofstream output(argv[2], std::ios::binary);
    output.write(reinterpret_cast<const char*>(result.data()), result.size() * 2);
    cudaFree(dx); cudaFree(dw); cudaFree(good); cudaFree(bad); cudaFree(row_ids); cudaFree(ranks);
    return output ? 0 : 2;
  } catch (...) { return 1; }
}
