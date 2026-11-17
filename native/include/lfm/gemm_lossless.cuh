#pragma once
// Lossless BF16 weights for lfm::gemm's wide kernel (journal/lfm3-single-stream_synth.md, decision 5).
//
// A weight matrix W [N][K] (K % 64 == 0) is stored as three streams, per 64-weight group along K (one BK = 64 tile):
//   man  one byte per weight: sign (bit 7) and the 7 mantissa bits
//   code one nibble per weight: exponent - base for 0..14; 15 marks an exception
// man and code are tile-major, [N / 64][K / 64][64 rows][64 or 32 bytes] (N padded to 64), so a tile's bytes are one
// contiguous read; row-major fragments of 32 and 64 bytes measured at half the plain kernel's DRAM throughput.
//   hdr  [K / 64][N] 8 bytes (transposed, so a tile's headers are contiguous): base exponent (= group max - 14), exception count (<= 3), then (index, exponent) x 3
// 104 bytes per group instead of 128: 81.25%. LFM3-Omni-2B's weights never need more than 3 exceptions per group
// (measured); pack() reports a tensor it cannot pack, which then stays uncompressed.
// The kernel decodes each B fragment of the wide kernel's mma steps to the exact BF16 values, straight into registers,
// so its outputs are bitwise those of the uncompressed kernel.
#include <lfm/gemm.cuh>

#include <unordered_map>

namespace lfm::gemm {

struct Packed {
  const unsigned char* man = nullptr;
  const unsigned char* code = nullptr;
  const unsigned long long* hdr = nullptr;
  int n = 0, k = 0;
  explicit operator bool() const { return man != nullptr; }
};

namespace detail {

__device__ __forceinline__ void cp_async8(unsigned dst, const void* src, bool valid) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(dst), "l"(src), "r"(valid ? 8 : 0));
}

// One thread per 64-weight group. bad[0] counts groups with more than 3 exceptions.
__global__ void pack_groups(const bf16* w, long long groups, int n, int groups_per_row, unsigned char* man,
                            unsigned char* code, unsigned long long* hdr, int* bad) {
  const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= groups) return;
  const unsigned short* x = reinterpret_cast<const unsigned short*>(w) + g * 64;
  const long long row = g / groups_per_row, kg = g % groups_per_row;
  const long long t = ((row / 64) * groups_per_row + kg) * 64 + row % 64;  // tile-major slot of this group
  int mx = 0;
  for (int i = 0; i < 64; ++i) mx = max(mx, (x[i] >> 7) & 0xFF);
  const int base = max(0, mx - 14);
  unsigned long long h = (unsigned long long)base;
  int count = 0;
  for (int i = 0; i < 64; i += 2) {
    unsigned char nib[2];
    for (int j = 0; j < 2; ++j) {
      const unsigned short v = x[i + j];
      const int e = (v >> 7) & 0xFF;
      man[t * 64 + i + j] = (unsigned char)(((v >> 8) & 0x80) | (v & 0x7F));
      if (e >= base && e - base <= 14) nib[j] = (unsigned char)(e - base);
      else {
        nib[j] = 15;
        if (count < 3) h |= (unsigned long long)(((unsigned)(i + j)) | ((unsigned)e << 8)) << (16 + 16 * count);
        ++count;
      }
    }
    code[t * 32 + i / 2] = (unsigned char)(nib[0] | (nib[1] << 4));
  }
  h |= (unsigned long long)min(count, 255) << 8;
  hdr[(g % groups_per_row) * n + g / groups_per_row] = h;  // [kg][row]
  if (count > 3) atomicAdd(bad, 1);
}

// The exact BF16 bits of weights [8 ch, 8 ch + 8) of a group, from its streams in shared memory. Branch-free: two
// weights per 32-bit word as sign | (base + nibble) << 7 | mantissa; then the (at most 3) exceptions of the group that
// fall in this chunk overwrite their exponent fields.
__device__ __forceinline__ int4 decode8(const unsigned char* man, const unsigned char* code, unsigned long long h, int ch) {
  const unsigned base = unsigned(h & 0xFF), count = unsigned((h >> 8) & 0xFF);
  const uint2 m = *reinterpret_cast<const uint2*>(man + ch * 8);
  const unsigned c = *reinterpret_cast<const unsigned*>(code + ch * 4);
  unsigned out[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const unsigned mb = ((j < 2 ? m.x : m.y) >> (16 * (j & 1))) & 0xFFFFu;  // bytes of weights 2j (low), 2j+1
    const unsigned e0 = base + ((c >> (8 * j)) & 0xFu), e1 = base + ((c >> (8 * j + 4)) & 0xFu);
    const unsigned lo = ((mb & 0x80u) << 8) | (e0 << 7) | (mb & 0x7Fu);
    const unsigned hi = (mb & 0x8000u) | (e1 << 7) | ((mb >> 8) & 0x7Fu);
    out[j] = lo | (hi << 16);
  }
  if (count) {
#pragma unroll
    for (int x = 0; x < 3; ++x) {
      const unsigned pair = unsigned(h >> (16 + 16 * x)) & 0xFFFFu;
      const int idx = int(pair & 0xFF) - ch * 8;
      if (unsigned(x) < count && idx >= 0 && idx < 8) {
        const unsigned shift = 16u * unsigned(idx & 1);
        const unsigned mask = ~(0x7F80u << shift), bits = (pair >> 8) << (7 + shift);
        const int word = idx / 2;  // explicit selects: no dynamically indexed (local-memory) array
        if (word == 0) out[0] = (out[0] & mask) | bits;
        if (word == 1) out[1] = (out[1] & mask) | bits;
        if (word == 2) out[2] = (out[2] & mask) | bits;
        if (word == 3) out[3] = (out[3] & mask) | bits;
      }
    }
  }
  return make_int4(int(out[0]), int(out[1]), int(out[2]), int(out[3]));
}

// wide_kernel (B stored N x K) with B read from the packed streams. Same tiles, same mma steps, same epilogue.
template <class T>
__global__ void __launch_bounds__(T::THREADS) wide_lossless_kernel(Problem p, Packed w) {
  constexpr int BM = T::BM, BN = T::BN, BK = T::BK, MI = T::MI, NI = T::NI, KCH = T::KCH, STAGES = T::STAGES;
  static_assert(BK == 64 && KCH == 8, "lossless tiles are one 64-weight group deep");
  static_assert(STAGES >= 3, "the decode of k-tile kt + 1 runs in iteration kt: k-tile kt + 1 must already be in flight");
  // per stage: A (BM x 64 bf16), man (BN x 64 B), code (BN x 32 B), hdr (BN x 8 B); B fragments decode to registers
  constexpr int MAN_CH = BN * 4, CODE_CH = BN * 2, HDR_CH = (BN + 1) / 2;
  constexpr int STAGE_CH = T::A_CHUNKS + MAN_CH + CODE_CH + HDR_CH;
  extern __shared__ __align__(128) unsigned char smem[];
  int4* stage0 = reinterpret_cast<int4*>(smem);

  int4* sbd = stage0 + STAGES * STAGE_CH;  // two decoded B tiles, swizzled as wide_kernel's B
  const int z = blockIdx.z;
  const bf16* A = p.a + z * p.stride_a;
  const long long c0 = z * p.stride_c;
  const long long boff = (z / p.group) * p.stride_b;  // element offset of this batch item's B (along K for split-K)
  const int n0 = blockIdx.x * BN, m0 = blockIdx.y * BM;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int wm = warp / T::WARPS_N, wn = warp % T::WARPS_N;
  const int ktiles = (p.k + BK - 1) / BK;
  const long long groups_per_row = w.k / 64;

  auto load = [&](int slot, int kt) {
    const int k0 = kt * BK;
    int4* st = stage0 + slot * STAGE_CH;
#pragma unroll
    for (int i = tid; i < T::A_CHUNKS; i += T::THREADS) {
      const int row = i / KCH, ch = i % KCH;
      const int gm = m0 + row, gk = k0 + ch * 8;
      const bool ok = gm < p.m && gk < p.k;
      cp_async16(smem_u32(st + swz_k<KCH>(row, ch)), ok ? A + gm * p.lda + gk : A, ok);
    }
    const long long kg = (boff + k0) / 64;  // the group along K
    int4* man = st + T::A_CHUNKS;
    int4* code = man + MAN_CH;
    int4* hdr = code + CODE_CH;
    for (int i = tid; i < MAN_CH; i += T::THREADS) {
      const int row = i / 4, ch = i % 4, gn = n0 + row;
      const bool ok = gn < p.n;
      cp_async16(smem_u32(man + i), ok ? w.man + (((gn / 64) * groups_per_row + kg) * 64 + gn % 64) * 64 + ch * 16 : w.man, ok);
    }
    for (int i = tid; i < CODE_CH; i += T::THREADS) {
      const int row = i / 2, ch = i % 2, gn = n0 + row;
      const bool ok = gn < p.n;
      cp_async16(smem_u32(code + i), ok ? w.code + (((gn / 64) * groups_per_row + kg) * 64 + gn % 64) * 32 + ch * 16 : w.code, ok);
    }
    for (int i = tid; i < BN; i += T::THREADS) {  // 8-byte headers, asynchronous like the other streams
      const int gn = n0 + i;
      const bool ok = gn < p.n;
      cp_async8(smem_u32(reinterpret_cast<unsigned long long*>(hdr) + i), ok ? w.hdr + kg * w.n + gn : w.hdr, ok);
    }
  };

  float acc[MI][NI][4];
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NI; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

  // The decode of k-tile kt + 1 (into sbd[(kt + 1) & 1]) runs in iteration kt, alongside the mma steps of k-tile kt
  // (from sbd[kt & 1]): one barrier per k-tile, as in wide_kernel.
  auto decode_stage = [&](int kt, int4* dst) {
    const int4* st = stage0 + (kt % STAGES) * STAGE_CH;
    const unsigned char* man = reinterpret_cast<const unsigned char*>(st + T::A_CHUNKS);
    const unsigned char* code = man + MAN_CH * 16;
    const unsigned long long* hdr = reinterpret_cast<const unsigned long long*>(code + CODE_CH * 16);
    for (int i = tid; i < BN * KCH; i += T::THREADS) {
      const int row = i / KCH, ch = i % KCH;
#ifdef LFM3_LOSSLESS_NODECODE  // diagnostic: the copy into the decoded tile without the decode (wrong values)
      dst[swz_k<KCH>(row, ch)] = *reinterpret_cast<const int4*>(man + row * 64 + (ch % 4) * 16);
#else
      dst[swz_k<KCH>(row, ch)] = decode8(man + row * 64, code + row * 32, hdr[row], ch);
#endif
    }
  };
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < ktiles) load(s, s);
    cp_commit();
  }
  cp_wait<STAGES - 2>();  // k-tile 0 is in
  __syncthreads();
  decode_stage(0, sbd);
  for (int kt = 0; kt < ktiles; ++kt) {
    // k-tile kt + 1 must be in before it is decoded; then the barrier also publishes sbd[kt & 1]
    cp_wait<STAGES - 3>();
    __syncthreads();
    const int next = kt + STAGES - 1;
    if (next < ktiles) load(next % STAGES, next);
    cp_commit();
    const int4* ta = stage0 + (kt % STAGES) * STAGE_CH;
    const int4* tb = sbd + (kt & 1) * BN * KCH;
    if (kt + 1 < ktiles) decode_stage(kt + 1, sbd + ((kt + 1) & 1) * BN * KCH);
#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      unsigned af[MI][4];
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) {
        const int row = wm * T::WM + mi * 16 + (lane % 16);
        ldmatrix_x4(smem_u32(ta + swz_k<KCH>(row, kk * 2 + lane / 16)), af[mi][0], af[mi][1], af[mi][2], af[mi][3]);
      }
      unsigned bfr[NI][2];
#pragma unroll
      for (int pair = 0; pair < NI / 2; ++pair) {
        unsigned r0, r1, r2, r3;
        const int row = wn * T::WN + pair * 16 + (lane % 8) + (lane / 16) * 8;
        ldmatrix_x4(smem_u32(tb + swz_k<KCH>(row, kk * 2 + (lane / 8) % 2)), r0, r1, r2, r3);
        bfr[pair * 2][0] = r0; bfr[pair * 2][1] = r1;
        bfr[pair * 2 + 1][0] = r2; bfr[pair * 2 + 1][1] = r3;
      }
#pragma unroll
      for (int mi = 0; mi < MI; ++mi)
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) mma_bf16(acc[mi][ni], af[mi], bfr[ni][0], bfr[ni][1]);
    }
  }
  cp_wait<0>();

  const int g = lane / 4, t = lane % 4;
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int ni = 0; ni < NI; ++ni)
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int row = m0 + wm * T::WM + mi * 16 + g + half * 8;
        const int col = n0 + wn * T::WN + ni * 8 + 2 * t;
        if (row >= p.m) continue;
        const long long at = c0 + row * p.ldc + col;
        const float x = acc[mi][ni][half * 2], y = acc[mi][ni][half * 2 + 1];
        if (p.c_f32) {
          if (col < p.n) p.c_f32[at] = x;
          if (col + 1 < p.n) p.c_f32[at + 1] = y;
          continue;
        }
        bf16* out = p.c + at;
        if (col + 1 < p.n && (reinterpret_cast<std::uintptr_t>(out) & 3) == 0) *reinterpret_cast<__nv_bfloat162*>(out) = __floats2bfloat162_rn(x, y);
        else {
          if (col < p.n) out[0] = __float2bfloat16_rn(x);
          if (col + 1 < p.n) out[1] = __float2bfloat16_rn(y);
        }
      }
}

template <class T>
inline void launch_wide_lossless(const Problem& p, const Packed& w, cudaStream_t stream) {
  constexpr int STAGE_CH = T::A_CHUNKS + T::BN * 4 + T::BN * 2 + (T::BN + 1) / 2;
  constexpr int SMEM = (T::STAGES * STAGE_CH + 2 * T::BN * T::KCH) * 16;
  static const bool configured =
      cudaFuncSetAttribute(wide_lossless_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) == cudaSuccess;
  if (!configured) throw std::runtime_error("gemm: lossless shared memory configuration failed");
  wide_lossless_kernel<T><<<dim3((p.n + T::BN - 1) / T::BN, (p.m + T::BM - 1) / T::BM, p.batch), T::THREADS, SMEM, stream>>>(p, w);
}

}  // namespace detail

// Packs a device BF16 matrix [n][k] (k % 64 == 0) into newly allocated device streams. Returns an empty Packed if any
// group has more than 3 exceptions (the caller keeps the uncompressed weights).
inline Packed pack(const bf16* w, int n, int k) {
  if (k % 64) return {};
  const long long groups = (long long)n * k / 64, padded = (long long)((n + 63) / 64 * 64) * k / 64;
  unsigned char *man = nullptr, *code = nullptr;
  unsigned long long* hdr = nullptr;
  int* bad = nullptr;
  if (cudaMalloc(&man, padded * 64) != cudaSuccess || cudaMalloc(&code, padded * 32) != cudaSuccess ||
      cudaMalloc(&hdr, groups * 8) != cudaSuccess || cudaMalloc(&bad, sizeof(int)) != cudaSuccess)
    throw std::runtime_error("gemm: lossless pack allocation failed");
  cudaMemset(bad, 0, sizeof(int));
  cudaMemset(man, 0, padded * 64);
  cudaMemset(code, 0, padded * 32);
  detail::pack_groups<<<unsigned((groups + 255) / 256), 256>>>(w, groups, n, k / 64, man, code, hdr, bad);
  int failed = 0;
  if (cudaMemcpy(&failed, bad, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
    throw std::runtime_error("gemm: lossless pack failed");
  cudaFree(bad);
  if (failed) { cudaFree(man); cudaFree(code); cudaFree(hdr); return {}; }
  Packed p;
  p.man = man, p.code = code, p.hdr = hdr, p.n = n, p.k = k;
  return p;
}

// Weights packed so far, by the address of their BF16 data.
inline std::unordered_map<const bf16*, Packed>& packed_weights() {
  static std::unordered_map<const bf16*, Packed> table;
  return table;
}

}  // namespace lfm::gemm
