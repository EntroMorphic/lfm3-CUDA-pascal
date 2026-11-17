#include <lfm/rmsnorm.cuh>
#include <fstream>
#include <vector>
#include <stdexcept>

void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
int main(int argc, char** argv) {
  try {
    if (argc != 3) return 2;
    std::ifstream in(argv[1], std::ios::binary);
    int rows, width;
    in.read(reinterpret_cast<char*>(&rows), 4); in.read(reinterpret_cast<char*>(&width), 4);
    if (!in || rows != 32 || width < 128 || width >= 8192 || width % 4) return 2;
    std::vector<__nv_bfloat16> x(rows * width), weight(width), result(x.size());
    in.read(reinterpret_cast<char*>(x.data()), x.size() * 2);
    in.read(reinterpret_cast<char*>(weight.data()), width * 2);
    if (!in) return 2;
    __nv_bfloat16 *dx, *dw, *out;
    check(cudaMalloc(&dx, x.size() * 2)); check(cudaMalloc(&dw, width * 2)); check(cudaMalloc(&out, x.size() * 2));
    check(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
    check(cudaMemcpy(dw, weight.data(), width * 2, cudaMemcpyHostToDevice));
    lfm::rms_norm_vectorized_warp<<<rows, 32>>>(dx, dw, out, width, 1e-5f);
    check(cudaGetLastError());
    check(cudaMemcpy(result.data(), out, x.size() * 2, cudaMemcpyDeviceToHost));
    std::ofstream file(argv[2], std::ios::binary);
    file.write(reinterpret_cast<const char*>(result.data()), result.size() * 2);
    cudaFree(dx); cudaFree(dw); cudaFree(out);
    return file ? 0 : 2;
  } catch (const std::exception& e) { return 1; }
}
