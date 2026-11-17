#pragma once
// Isolated candidate only. Production never includes this header.
// Same as lfm3_wmma_runtime_candidate.cuh (v4) except the thin (rows<=8) producer:
// v6 (shared-memory staged, 16-byte row-contiguous loads), or v7 (same loads, quad
// rotation transpose) or v8 (butterfly transpose) -- both without shared memory -- per
// LFM3_THIN_POLICY, for large weights and
// v5 (register-prefetched) otherwise. Both are bitwise-identical to v4's thin
// producer (validation/thin-probe-v1); the choice affects speed, never arithmetic.
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <map>
#include <stdexcept>
#include <tuple>

namespace lfm3_wmma_candidate {
// The generator places the exactly tested WMMA kernels before this class.
class Linear {
  struct Plan {bool native;unsigned parts;};
  std::map<std::tuple<int,int,int>,Plan> plans;
  cublasLtHandle_t handle=nullptr;
  float* scratch=nullptr;size_t capacity=0;
  static void cuda(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
  static void blas(cublasStatus_t s){if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("candidate BLAS status="+std::to_string(int(s)));}
  // Thin-producer selection, measured cold-cache on Thor (thin-probe-v1): v6 wins once a
  // linear streams >=16 MiB of weights; below that v5 has more useful warp parallelism.
  // LFM3_THIN_POLICY (set by the generator) picks the large-weight producer: 6 = v6 (shared-memory
  // staging), 7 = v7 (quad rotation transpose), 8 = v8 (quad XOR-butterfly transpose). All are
  // bitwise-identical to v4; choices follow interleaved cold-cache runs in validation/thin-probe-v1.
  enum class Thin {v5u4,v5u16,v6s8,v7s4,v7s8,v8s8};
  static Thin thin_kind(int m,int k){
#if LFM3_THIN_POLICY==8
    return m>=2048?Thin::v8s8:Thin::v5u16;
#else
    if(size_t(m)*k*2>=(size_t(16)<<20)){
#if LFM3_THIN_POLICY==7
      return (k>=4096||(m>=7000&&m<8000))?Thin::v7s4:Thin::v7s8;
#else
      return Thin::v6s8;
#endif
    }
    return m>=1024?Thin::v5u4:Thin::v5u16;
#endif
  }
  static const char* thin_name(Thin t){
    switch(t){case Thin::v5u4:return "v5u4";case Thin::v5u16:return "v5u16";case Thin::v6s8:return "v6s8";case Thin::v7s4:return "v7s4";case Thin::v7s8:return "v7s8";default:return "v8s8";}
  }
  Plan plan(int m,int n,int k){
    auto key=std::make_tuple(m,n,k);auto found=plans.find(key);if(found!=plans.end())return found->second;
    if(!handle)blas(cublasLtCreate(&handle));
    cublasLtMatmulDesc_t op;blas(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
    cublasOperation_t trans=CUBLAS_OP_T;blas(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
    cublasLtMatrixLayout_t a,b,c;blas(cublasLtMatrixLayoutCreate(&a,CUDA_R_16BF,k,m,k));
    blas(cublasLtMatrixLayoutCreate(&b,CUDA_R_16BF,k,n,k));blas(cublasLtMatrixLayoutCreate(&c,CUDA_R_16BF,m,n,m));
    cublasLtMatmulPreference_t preference;blas(cublasLtMatmulPreferenceCreate(&preference));size_t budget=8519680;
    blas(cublasLtMatmulPreferenceSetAttribute(preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&budget,sizeof(budget)));
    cublasLtMatmulHeuristicResult_t result{};int returned=0;
    blas(cublasLtMatmulAlgoGetHeuristic(handle,op,a,b,c,c,preference,1,&result,&returned));
    if(returned!=1||result.state!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("candidate plan unavailable");
    unsigned id=0,parts=0;size_t bytes=0;
    blas(cublasLtMatmulAlgoConfigGetAttribute(&result.algo,CUBLASLT_ALGO_CONFIG_ID,&id,sizeof(id),&bytes));
    blas(cublasLtMatmulAlgoConfigGetAttribute(&result.algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&parts,sizeof(parts),&bytes));
    if(!parts)parts=1;
    blas(cublasLtMatmulPreferenceDestroy(preference));blas(cublasLtMatrixLayoutDestroy(a));
    blas(cublasLtMatrixLayoutDestroy(b));blas(cublasLtMatrixLayoutDestroy(c));blas(cublasLtMatmulDescDestroy(op));
    Plan p{id==66,parts};
    if(p.native&&(parts>16||k%parts||(k/parts)%16))throw std::runtime_error("candidate partition unsupported");
    const char* thin=n>8?"none":thin_name(thin_kind(m,k));
    std::fprintf(stderr,"{\"candidate_linear_plan\":true,\"m\":%d,\"n\":%d,\"k\":%d,\"heuristic_id\":%u,\"parts\":%u,\"native_wmma\":%s,\"thin\":\"%s\"}\n",m,n,k,id,parts,p.native?"true":"false",thin);
    plans.emplace(key,p);return p;
  }
 public:
  Linear()=default;Linear(const Linear&)=delete;Linear& operator=(const Linear&)=delete;
  ~Linear(){if(scratch)cudaFree(scratch);if(handle)cublasLtDestroy(handle);}
  void run(cublasHandle_t original,const __nv_bfloat16* w,const __nv_bfloat16* x,__nv_bfloat16* out,int n,int k,int m){
    const auto p=plan(m,n,k);
    if(!p.native){float alpha=1,beta=0;blas(cublasGemmEx(original,CUBLAS_OP_T,CUBLAS_OP_N,m,n,k,&alpha,w,CUDA_R_16BF,k,x,CUDA_R_16BF,k,&beta,out,CUDA_R_16BF,m,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));return;}
    int pm=(m+15)/16*16,pn=n<=8?8:(n+15)/16*16;size_t required=size_t(pm)*pn*p.parts;
    if(required>capacity){if(scratch){cuda(cudaFree(scratch));scratch=nullptr;capacity=0;}cuda(cudaMalloc(&scratch,required*sizeof(float)));capacity=required;}
    if(n<=8){
      int warps=std::min(4,pm/16);dim3 grid((pm/16+warps-1)/warps,1,p.parts),block(32*warps);
      // Fail closed: v6 issues 16-byte loads (rows and partition starts must stay 16-byte aligned);
      // v5 keeps v4's 4-byte operand loads.
      if(reinterpret_cast<uintptr_t>(w)%16||k%8)throw std::runtime_error("candidate thin alignment unsupported");
      switch(thin_kind(m,k)){
        case Thin::v6s8:v6::mma_thin_partials<8><<<grid,block>>>(w,x,scratch,m,n,k,pn,pm,p.parts);break;
        case Thin::v5u4:v5::mma_thin_partials<4><<<grid,block>>>(w,x,scratch,m,n,k,pn,pm,p.parts);break;
        case Thin::v5u16:v5::mma_thin_partials<16><<<grid,block>>>(w,x,scratch,m,n,k,pn,pm,p.parts);break;
        case Thin::v7s4:v7::mma_thin_partials<4><<<grid,block>>>(w,x,scratch,m,n,k,pn,pm,p.parts);break;
        case Thin::v7s8:v7::mma_thin_partials<8><<<grid,block>>>(w,x,scratch,m,n,k,pn,pm,p.parts);break;
        case Thin::v8s8:v8::mma_thin_partials<8><<<grid,block>>>(w,x,scratch,m,n,k,pn,pm,p.parts);break;
      }
    }
    else wmma_partials<<<dim3(pm/16,pn/16,p.parts),32>>>(w,x,scratch,m,n,k,pn,pm,p.parts);
    cuda(cudaGetLastError());
    size_t count=size_t(m)*n;wmma_finish<<<(count+255)/256,256>>>(scratch,out,count,m,n,pm,pn,p.parts);cuda(cudaGetLastError());
  }
};
} // namespace lfm3_wmma_candidate
