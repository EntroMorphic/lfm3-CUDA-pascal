// Supported cuBLAS mode diagnostic only; not a runtime or acceptance change.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using bf16=__nv_bfloat16;
void check(cudaError_t s) { if(s!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
void check(cublasStatus_t s) {
  if(s!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS status="+std::to_string(int(s))+
    " (unsupported/error is failure, never sanitizer clean)");
}
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
int dimension(const char* text) {
  std::string s(text);
  if(s.empty() || s.find_first_not_of("0123456789")!=std::string::npos)
    throw std::runtime_error("bad dimension");
  return std::stoi(s);
}
int main(int argc,char** argv) try {
  if(argc!=5) throw std::runtime_error("usage: probe kernel|missing|gemm ROWS WIDTH default|pedantic|default-algo");
  std::string producer(argv[1]),mode(argv[4]);
  bool gemm=producer=="gemm",missing=producer=="missing";
  if(!gemm && !missing && producer!="kernel") throw std::runtime_error("bad producer");
  if(mode!="default" && mode!="pedantic" && mode!="default-algo") throw std::runtime_error("bad mode");
  int rows=dimension(argv[2]),width=dimension(argv[3]);
  if(rows<1 || rows>257 || (width!=128 && width!=2048)) throw std::runtime_error("unsupported probe shape");
  int output=3*width;
  auto compute=mode=="pedantic"?CUBLAS_COMPUTE_32F_PEDANTIC:CUBLAS_COMPUTE_32F;
  auto algo=mode=="default"?CUBLAS_GEMM_DEFAULT_TENSOR_OP:CUBLAS_GEMM_DEFAULT;
  std::cout<<"CONFIG producer="<<producer<<"; rows="<<rows<<"; width="<<width<<"; mode="<<mode
           <<"; compute="<<int(compute)<<"; algo="<<int(algo)<<"; nvcc="
           <<__CUDACC_VER_MAJOR__<<'.'<<__CUDACC_VER_MINOR__<<'.'<<__CUDACC_VER_BUILD__
           <<"; cudart_build="<<CUDART_VERSION<<"; cublas_build="<<CUBLAS_VER_MAJOR<<'.'
           <<CUBLAS_VER_MINOR<<'.'<<CUBLAS_VER_PATCH<<'.'<<CUBLAS_VER_BUILD<<std::endl;
  size_t n=size_t(rows)*output;
  bf16 *a,*b,*c,*d; void* workspace;
  check(cudaMalloc(&a,size_t(output)*width*2)); check(cudaMalloc(&b,size_t(rows)*width*2));
  check(cudaMalloc(&c,n*2)); check(cudaMalloc(&d,n*2)); check(cudaMalloc(&workspace,8519680));
  varied_weights<<<(size_t(output)*width+255)/256,256>>>(a,size_t(output)*width,width);
  fill<<<(size_t(rows)*width+255)/256,256>>>(b,size_t(rows)*width,0.5f);
  cublasHandle_t handle; check(cublasCreate(&handle)); check(cublasSetWorkspace(handle,workspace,8519680));
  int version; check(cublasGetVersion(handle,&version));
  std::cout<<"LIBRARY cublas="<<version<<"; mode="<<mode<<std::endl;
  if(mode=="pedantic") check(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH));
  cublasMath_t math; check(cublasGetMathMode(handle,&math));
  std::cout<<"MATH actual="<<int(math)<<"; mode="<<mode<<std::endl;
  if(gemm) {
    float alpha=1,beta=0;
    check(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,output,rows,width,&alpha,a,CUDA_R_16BF,width,
                      b,CUDA_R_16BF,width,&beta,c,CUDA_R_16BF,output,compute,algo));
  } else if(!missing) fill<<<(n+255)/256,256>>>(c,n,float(width));
  const size_t sample=std::min(n,size_t(128));
  read_output<<<1,256>>>(c,d,sample);
  std::vector<bf16> host(sample); check(cudaMemcpy(host.data(),d,sample*2,cudaMemcpyDeviceToHost));
  if(!missing) for(size_t i=0;i<sample;++i) {
    const float expected=gemm?float(width)*float(int(i%13)-6)/16.0f:float(width);
    if(__bfloat162float(host[i])!=expected) throw std::runtime_error("incorrect projection output");
  }
  check(cublasDestroy(handle)); check(cudaFree(a)); check(cudaFree(b)); check(cudaFree(c)); check(cudaFree(d)); check(cudaFree(workspace));
  std::cout<<(missing?"INTENTIONAL UNINITIALIZED CONTROL: ":"PASS: ")<<sample
           <<" sampled outputs of "<<n<<"; producer="<<producer<<"; mode="<<mode
           <<"; cublas="<<version<<"; numerical-only (not sanitizer acceptance)\n";
  return 0;
} catch(const std::exception& e) {std::cerr<<"FAIL: "<<e.what()<<"\n"; return 1;}
