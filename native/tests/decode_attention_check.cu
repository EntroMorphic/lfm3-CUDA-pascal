// Decode attention's two products against the kernels as they were before phase 2 (copied below from commit a5bcd4a,
// unchanged but for their names), on the card: every output bit for bit, for each way the engine can launch them
// (G = 1, 2 or 4 query heads per read of the keys and values), over cache lengths from 1 to 31,000 keys, one and two
// rows, and 2 or 4 query heads per key/value head (the model has 4).
//   decode_attention_check        exit 0 if every case is identical
// Build: nvcc [the engine's flags] decode_attention_check.cu
#include <lfm/gemm.cuh>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace lfm::gemm::detail {
// ---- before phase 2 (a5bcd4a) ----
template <int R>
__global__ void __launch_bounds__(128) old_split_kn_partial(Problem p, int splits) {
  const int split = blockIdx.x, z = blockIdx.y;
  const bf16* A = p.a + z * p.stride_a;
  const bf16* B = p.b + (z / p.group) * p.stride_b;
  const int col = 2 * (blockIdx.z * blockDim.x + threadIdx.x);
  if (col >= p.n) return;
  const int k0 = split * kSplitKeys, k1 = min(k0 + kSplitKeys, p.b_rows);
  float acc[R][2];
#pragma unroll
  for (int r = 0; r < R; ++r) acc[r][0] = acc[r][1] = 0.f;
  for (int kk = k0; kk < k1; ++kk) {
    const float2 v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(B + kk * p.ldb + col));
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const float a = __bfloat162float(A[r * p.lda + kk]);
      acc[r][0] = fmaf(a, v.x, acc[r][0]);
      acc[r][1] = fmaf(a, v.y, acc[r][1]);
    }
  }
  float* out = p.workspace + ((size_t(z) * splits + split) * R) * p.n + col;
#pragma unroll
  for (int r = 0; r < R; ++r) { out[r * p.n] = acc[r][0]; out[r * p.n + 1] = acc[r][1]; }
}

template <int R>
__global__ void __launch_bounds__(256) old_thin_k128_kernel(Problem p) {
  const int z = blockIdx.y;
  const bf16* A = p.a + z * p.stride_a;
  const bf16* B = p.b + (z / p.group) * p.stride_b;
  const long long c0 = z * p.stride_c;
  const int lane = threadIdx.x % 32, sub = lane % 8, slot = lane / 8;
  const int warps = gridDim.x * (blockDim.x / 32);
  const int warp = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  float q[R][16];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int4 lo = __ldg(reinterpret_cast<const int4*>(A + r * p.lda + sub * 16));
    const int4 hi = __ldg(reinterpret_cast<const int4*>(A + r * p.lda + sub * 16 + 8));
    const __nv_bfloat162* l2 = reinterpret_cast<const __nv_bfloat162*>(&lo);
    const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&hi);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 a = __bfloat1622float2(l2[e]), b = __bfloat1622float2(h2[e]);
      q[r][2 * e] = a.x; q[r][2 * e + 1] = a.y; q[r][8 + 2 * e] = b.x; q[r][8 + 2 * e + 1] = b.y;
    }
  }
  for (int row0 = warp * 4; row0 < p.n; row0 += warps * 4) {
    const int row = row0 + slot;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;
    if (row < p.n) {
      const int4 lo = __ldg(reinterpret_cast<const int4*>(B + row * p.ldb + sub * 16));
      const int4 hi = __ldg(reinterpret_cast<const int4*>(B + row * p.ldb + sub * 16 + 8));
      const __nv_bfloat162* l2 = reinterpret_cast<const __nv_bfloat162*>(&lo);
      const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&hi);
      float k[16];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 a = __bfloat1622float2(l2[e]), b = __bfloat1622float2(h2[e]);
        k[2 * e] = a.x; k[2 * e + 1] = a.y; k[8 + 2 * e] = b.x; k[8 + 2 * e + 1] = b.y;
      }
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int e = 0; e < 16; ++e) acc[r] = fmaf(q[r][e], k[e], acc[r]);
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      float v = acc[r];
      v += __shfl_xor_sync(0xffffffffu, v, 4);
      v += __shfl_xor_sync(0xffffffffu, v, 2);
      v += __shfl_xor_sync(0xffffffffu, v, 1);
      if (sub == r && row < p.n) put(p, c0 + r * p.ldc + row, v);
    }
  }
}

}  // namespace lfm::gemm::detail

namespace {
using u16 = unsigned short;
using lfm::gemm::Problem;
using lfm::gemm::bf16;
void check(cudaError_t e) { if (e != cudaSuccess) { std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); std::exit(2); } }
unsigned long long state = 0x9E3779B97F4A7C15ull;
unsigned next() { state = state * 6364136223846793005ull + 1442695040888963407ull; return unsigned(state >> 33); }
u16 random_bf16() { const unsigned r = next(); return u16((r >> 8 & 1) << 15 | (127 - 6 + (r >> 9) % 8) << 7 | (r >> 16 & 0x7f)); }
template <class T> T* upload(const std::vector<T>& h) { T* d; check(cudaMalloc(&d, h.size() * sizeof(T))); check(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); return d; }
template <class T> std::vector<T> download(const T* d, size_t n) { std::vector<T> h(n); check(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }

// G = 0: the kernels before phase 2.
template <int R, int G> void scores(const Problem& p) {
  using namespace lfm::gemm::detail;
  const int blocks = std::min((p.n + 31) / 32, 1024);
  if constexpr (G == 0) old_thin_k128_kernel<R><<<dim3(blocks, p.batch), 256>>>(p);
  else thin_k128_kernel<R, G><<<dim3(blocks, p.batch / G), 256>>>(p);
}
// WIDE: split_kn_partial_wide (one row only).
template <int R, int G, bool WIDE = false> void values(const Problem& p) {
  using namespace lfm::gemm; using namespace lfm::gemm::detail;
  const int splits = (p.k + kSplitKeys - 1) / kSplitKeys, threads = 64, column_blocks = (p.n / 2 + threads - 1) / threads;
  if constexpr (G == 0) old_split_kn_partial<R><<<dim3(splits, p.batch, column_blocks), threads>>>(p, splits);
  else if constexpr (WIDE) split_kn_partial_wide<G><<<dim3((splits + kWideSplitsPerBlock - 1) / kWideSplitsPerBlock, p.batch / G, (p.n + 127) / 128), 16 * kWideSplitsPerBlock>>>(p, splits);
  else split_kn_partial<R, G><<<dim3(splits, p.batch / G, column_blocks), threads>>>(p, splits);
  const size_t outputs = size_t(R) * p.n * p.batch;
  split_kn_finish<R><<<unsigned((outputs + 255) / 256), 256>>>(p, splits);
}

int cases = 0, failures = 0;
template <int R> void one(int keys, int heads, int group) {
  const int head_dim = 128, kv = heads / group, context = keys + 16, pitch = (keys + 7) / 8 * 8;
  std::vector<u16> q(size_t(R) * heads * head_dim), cache(size_t(kv) * context * head_dim), probs(size_t(heads) * R * pitch);
  for (auto& v : q) v = random_bf16();
  for (auto& v : cache) v = random_bf16();
  for (auto& v : probs) v = random_bf16();
  auto *dq = upload(q), *dk = upload(cache), *dp = upload(probs);
  const size_t s_elems = size_t(heads) * R * pitch, v_elems = size_t(R) * heads * head_dim, guard = 4096;
  float* ds; check(cudaMalloc(&ds, (s_elems + guard) * 4));
  bf16* dout; check(cudaMalloc(&dout, (v_elems + guard) * 2));
  const size_t ws = lfm::gemm::workspace_bytes(R, head_dim, pitch, heads);
  float* dw; check(cudaMalloc(&dw, ws));
  Problem s;  // scores, as lfm3_main.cu head_scores
  s.a = reinterpret_cast<const bf16*>(dq); s.b = reinterpret_cast<const bf16*>(dk); s.c_f32 = ds;
  s.m = R; s.n = keys; s.k = head_dim; s.lda = heads * head_dim; s.ldb = head_dim; s.ldc = pitch;
  s.stride_a = head_dim; s.stride_b = (long long)context * head_dim; s.stride_c = (long long)R * pitch; s.batch = heads; s.group = group;
  Problem v;  // values, as head_values
  v.a = reinterpret_cast<const bf16*>(dp); v.b = reinterpret_cast<const bf16*>(dk); v.c = dout;
  v.m = R; v.n = head_dim; v.k = pitch; v.b_rows = keys; v.b_k_contiguous = false;
  v.lda = pitch; v.ldb = head_dim; v.ldc = heads * head_dim; v.stride_a = (long long)R * pitch; v.stride_b = (long long)context * head_dim;
  v.stride_c = head_dim; v.batch = heads; v.group = group; v.workspace = dw; v.workspace_size = ws;
  std::vector<float> s0; std::vector<bf16> v0;
  for (int g : {0, 1, 2, 4, 11, 12, 14}) {  // 11, 12, 14: split_kn_partial_wide with 1, 2, 4 heads (one row only)
    if (g % 10 > 1 && group % (g % 10)) continue;
    if (g > 10 && R != 1) continue;
    check(cudaMemset(ds, 0x5a, (s_elems + guard) * 4)); check(cudaMemset(dout, 0x5a, (v_elems + guard) * 2));
    switch (g) {
      case 0: scores<R, 0>(s); values<R, 0>(v); break;
      case 1: scores<R, 1>(s); values<R, 1>(v); break;
      case 2: scores<R, 2>(s); values<R, 2>(v); break;
      case 4: scores<R, 4>(s); values<R, 4>(v); break;
      case 11: scores<R, 1>(s); values<R, 1, true>(v); break;
      case 12: scores<R, 2>(s); values<R, 2, true>(v); break;
      default: scores<R, 4>(s); values<R, 4, true>(v); break;
    }
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
    const auto s1 = download(ds, s_elems + guard); const auto v1 = download(dout, v_elems + guard);
    if (g == 0) { s0 = s1; v0 = v1; continue; }
    const bool same_s = std::memcmp(s0.data(), s1.data(), s1.size() * 4) == 0, same_v = std::memcmp(v0.data(), v1.data(), v1.size() * 2) == 0;
    cases += 2; failures += !same_s + !same_v;
    std::printf("%s rows %d keys %5d heads %d group %d, %d heads per read%s: scores %s, values %s\n", same_s && same_v ? "ok  " : "FAIL",
                R, keys, heads, group, g % 10, g > 10 ? " (16-byte values)" : "", same_s ? "identical" : "DIFFER", same_v ? "identical" : "DIFFER");
  }
  check(cudaFree(dq)); check(cudaFree(dk)); check(cudaFree(dp)); check(cudaFree(ds)); check(cudaFree(dout)); check(cudaFree(dw));
}
}  // namespace

int main() {
  for (int keys : {1, 5, 8, 255, 256, 257, 1000, 4097, 31000}) {
    one<1>(keys, 16, 4);
    one<2>(keys, 16, 4);
    one<1>(keys, 16, 2);
  }
  std::printf("decode_attention_check: %d cases, %d failed: %s\n", cases, failures, failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
