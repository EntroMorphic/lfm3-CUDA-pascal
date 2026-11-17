// Peak mma.sync m16n8k16 BF16->FP32 throughput: register-only chains, no memory traffic.
#include <cuda_bf16.h>
#include <cstdio>
__global__ void peak(float* out, int iters) {
  float d[16][4] = {};
  unsigned a[4] = {threadIdx.x, threadIdx.x * 3u, threadIdx.x * 5u, threadIdx.x * 7u}, b0 = threadIdx.x * 11u, b1 = threadIdx.x * 13u;
  for (int i = 0; i < iters; ++i)
#pragma unroll
    for (int j = 0; j < 16; ++j)
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                   : "+f"(d[j][0]), "+f"(d[j][1]), "+f"(d[j][2]), "+f"(d[j][3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  float s = 0; for (int j = 0; j < 16; ++j) s += d[j][0] + d[j][1] + d[j][2] + d[j][3];
  if (s == 1.2345f) out[0] = s;
}
int main() {
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  int clock = 0; cudaDeviceGetAttribute(&clock, cudaDevAttrClockRate, 0);
  float* o; cudaMalloc(&o, 4);
  for (int warps : {4, 8, 16}) {
    const int iters = 20000;
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    peak<<<p.multiProcessorCount * 2, warps * 32>>>(o, 100);
    cudaEventRecord(s); peak<<<p.multiProcessorCount * 2, warps * 32>>>(o, iters); cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);
    double flops = double(p.multiProcessorCount) * 2 * warps * iters * 16 * 2.0 * 16 * 8 * 16;
    printf("sms=%d clock=%d kHz warps/block=%d: %.1f TFLOPS (%.0f FLOP/clk/SM)\n", p.multiProcessorCount, clock, warps,
           flops / ms / 1e9, flops / (ms * 1e-3) / p.multiProcessorCount / (clock * 1e3));
  }
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));
}
