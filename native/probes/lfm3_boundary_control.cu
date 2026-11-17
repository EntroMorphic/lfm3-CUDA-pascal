// Known producer/reducer calibration for the read-only launch-boundary observer.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
void check(cudaError_t s) {if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
__global__ void boundary_control_fill(float* scratch,int n) {
  int i=int(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)scratch[i]=float(i%37-18)*0.25f;
}
__global__ void boundary_control_reduce(const float* scratch,float* out,int n) {
  int i=int(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)out[i]=scratch[i]*2;
}
__global__ void boundary_control_consume(const float* out,float* sink,int n) {
  int i=int(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)sink[i]=out[i];
}
int main(int argc,char** argv) try {
  if(argc!=2||(std::strcmp(argv[1],"positive")&&std::strcmp(argv[1],"missing")))throw std::runtime_error("usage: control positive|missing");
  constexpr int n=12288;float *scratch,*out,*sink;
  check(cudaMalloc(&scratch,8519680));check(cudaMalloc(&out,n*4));check(cudaMalloc(&sink,n*4));
  check(cudaMemset(scratch,0xa5,8519680));
  check(cudaMemset(out,0x5a,n*4));check(cudaMemset(sink,0x3c,n*4));
  if(!std::strcmp(argv[1],"positive")) {boundary_control_fill<<<48,256>>>(scratch,n);check(cudaGetLastError());}
  boundary_control_reduce<<<48,256>>>(scratch,out,n);check(cudaGetLastError());
  boundary_control_consume<<<48,256>>>(out,sink,n);check(cudaGetLastError());
  std::vector<float> host(n);check(cudaMemcpy(host.data(),sink,n*4,cudaMemcpyDeviceToHost));
  unsigned mismatches=0;for(int i=0;i<n;++i)if(host[i]!=float(i%37-18)*0.5f)++mismatches;
  check(cudaFree(scratch));check(cudaFree(out));check(cudaFree(sink));
  std::printf("{\"control\":\"%s\",\"elements\":12288,\"mismatches\":%u}\n",argv[1],mismatches);
  return mismatches?2:0;
} catch(const std::exception& e) {std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
