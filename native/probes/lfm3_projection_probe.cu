// Isolate cuBLAS output initialization tracking; never substitutes for model tests.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>
using bf16=__nv_bfloat16;
void check(cudaError_t s) { if(s!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
void check(cublasStatus_t s) { if(s!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS failed"); }
__global__ void fill(bf16* out,size_t n,float value) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n) out[i]=__float2bfloat16_rn(value);
}
__global__ void read_output(const bf16* in,bf16* out,size_t n) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n) out[i]=in[i];
}
__global__ void varied_weights(bf16* out,size_t n,int width) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n) out[i]=__float2bfloat16_rn(float(int((i/width)%13)-6)/8.0f);
}
int main(int argc,char** argv) try {
  if(argc!=4) throw std::runtime_error("usage: probe gemm|kernel|missing ROWS WIDTH");
  bool gemm=std::string(argv[1])=="gemm";
  bool missing=std::string(argv[1])=="missing";
  if(!gemm && !missing && std::string(argv[1])!="kernel") throw std::runtime_error("bad producer");
  int rows=std::stoi(argv[2]),width=std::stoi(argv[3]),output=3*width;
  if(rows<1 || rows>257 || (width!=128 && width!=2048)) throw std::runtime_error("unsupported probe shape");
  size_t n=size_t(rows)*output;
  bf16 *a,*b,*c,*d; void* workspace;
  check(cudaMalloc(&a,size_t(output)*width*2)); check(cudaMalloc(&b,size_t(rows)*width*2));
  check(cudaMalloc(&c,n*2)); check(cudaMalloc(&d,n*2)); check(cudaMalloc(&workspace,8519680));
  // Nonuniform expected GEMM output cannot be mistaken for the control's stale
  // constant-filled allocation if a library failed to write the output at all.
  varied_weights<<<(size_t(output)*width+255)/256,256>>>(a,size_t(output)*width,width);
  fill<<<(size_t(rows)*width+255)/256,256>>>(b,size_t(rows)*width,0.5f);
  cublasHandle_t handle; check(cublasCreate(&handle)); check(cublasSetWorkspace(handle,workspace,8519680));
  if(gemm) {
    float alpha=1,beta=0;
    check(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,output,rows,width,&alpha,a,CUDA_R_16BF,width,
                      b,CUDA_R_16BF,width,&beta,c,CUDA_R_16BF,output,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  } else if(!missing) fill<<<(n+255)/256,256>>>(c,n,float(width));
  const size_t sample=std::min(n,size_t(128));
  read_output<<<1,256>>>(c,d,sample);
  std::vector<bf16> host(sample); check(cudaMemcpy(host.data(),d,sample*2,cudaMemcpyDeviceToHost));
  if(!missing) for(size_t i=0;i<sample;++i) {
    const float expected=gemm?float(width)*float(int(i%13)-6)/16.0f:float(width);
    if(__bfloat162float(host[i])!=expected) throw std::runtime_error("incorrect projection output");
  }
  int version; check(cublasGetVersion(handle,&version));
  std::cout<<(missing?"INTENTIONAL UNINITIALIZED CONTROL: ":"PASS: ")<<sample
           <<" sampled outputs of "<<n<<"; producer="<<argv[1]<<"; cublas="<<version<<"\n";
  check(cublasDestroy(handle)); check(cudaFree(a)); check(cudaFree(b)); check(cudaFree(c)); check(cudaFree(d)); check(cudaFree(workspace));
  return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<"\n"; return 1;}
