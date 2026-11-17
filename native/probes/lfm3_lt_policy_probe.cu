// Explicit-algorithm screening against complete real checkpoint projection outputs.
// No scratch/output initialization, suppressions, model mutation or production use.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
using bf16=__nv_bfloat16;
void check(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
void check(cublasStatus_t s){if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("BLAS status="+std::to_string(int(s)));}
std::vector<char> read(const std::string& path,size_t bytes){
  std::ifstream f(path,std::ios::binary|std::ios::ate);
  if(!f||f.tellg()!=std::streamoff(bytes))throw std::runtime_error("file size: "+path);
  std::vector<char> b(bytes);f.seekg(0);f.read(b.data(),b.size());if(!f)throw std::runtime_error("file read");return b;
}
void write(const std::string& path,const void* data,size_t bytes){
  std::ofstream f(path,std::ios::binary);f.write(static_cast<const char*>(data),bytes);if(!f)throw std::runtime_error("file write");
}
__global__ void lt_consume(const bf16* in,bf16* out,size_t n){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)out[i]=in[i];}
unsigned attr(const cublasLtMatmulAlgo_t& a,cublasLtMatmulAlgoConfigAttributes_t k){
  unsigned v=0;size_t bytes=0;check(cublasLtMatmulAlgoConfigGetAttribute(&a,k,&v,sizeof(v),&bytes));
  if(bytes!=sizeof(v))throw std::runtime_error("attribute size");return v;
}
int number(const char* text){std::string s=text;if(s.empty()||s.find_first_not_of("0123456789")!=std::string::npos)throw std::runtime_error("invalid dimension");return std::stoi(s);}
int main(int argc,char** argv) try {
  if(argc!=8&&argc!=9)throw std::runtime_error("usage: probe ROWS INPUT OUTPUT W.bf16 X.bf16 EXPECT.bf16 PREFIX [ALGO.bin]");
  int n=number(argv[1]),k=number(argv[2]),m=number(argv[3]);
  if(n<1||n>4096||k<1||k>32768||m<1||m>128000||uint64_t(k)*m>300000000)throw std::runtime_error("bounded dimensions required");
  auto w=read(argv[4],size_t(k)*m*2),x=read(argv[5],size_t(k)*n*2),expected=read(argv[6],size_t(m)*n*2);
  std::string prefix=argv[7];bf16 *a,*b,*c,*sink;void* workspace;constexpr size_t budget=8519680;size_t elements=size_t(m)*n;
  check(cudaMalloc(&a,w.size()));check(cudaMalloc(&b,x.size()));check(cudaMalloc(&c,expected.size()));
  check(cudaMalloc(&sink,expected.size()));check(cudaMalloc(&workspace,budget));
  check(cudaMemcpy(a,w.data(),w.size(),cudaMemcpyHostToDevice));check(cudaMemcpy(b,x.data(),x.size(),cudaMemcpyHostToDevice));
  cublasLtHandle_t handle;check(cublasLtCreate(&handle));cublasLtMatmulDesc_t op;
  check(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
  cublasOperation_t trans=CUBLAS_OP_T;check(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
  cublasLtMatrixLayout_t la,lb,lc;check(cublasLtMatrixLayoutCreate(&la,CUDA_R_16BF,k,m,k));
  check(cublasLtMatrixLayoutCreate(&lb,CUDA_R_16BF,k,n,k));check(cublasLtMatrixLayoutCreate(&lc,CUDA_R_16BF,m,n,m));
  std::vector<cublasLtMatmulAlgo_t> algorithms;
  auto add=[&](const cublasLtMatmulAlgo_t& candidate){for(const auto& a:algorithms)if(!std::memcmp(&a,&candidate,sizeof(a)))return;algorithms.push_back(candidate);};
  if(argc==9){auto raw=read(argv[8],sizeof(cublasLtMatmulAlgo_t));cublasLtMatmulAlgo_t a{};std::memcpy(&a,raw.data(),raw.size());add(a);}
  else {
    cublasLtMatmulPreference_t preference;check(cublasLtMatmulPreferenceCreate(&preference));
    check(cublasLtMatmulPreferenceSetAttribute(preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&budget,sizeof(budget)));
    cublasLtMatmulHeuristicResult_t result[64]{};int returned=0;
    check(cublasLtMatmulAlgoGetHeuristic(handle,op,la,lb,lc,lc,preference,64,result,&returned));
    for(int i=0;i<returned;++i)if(result[i].state==CUBLAS_STATUS_SUCCESS)add(result[i].algo);
    check(cublasLtMatmulPreferenceDestroy(preference));
    int ids[128],count=0;check(cublasLtMatmulAlgoGetIds(handle,CUBLAS_COMPUTE_32F,CUDA_R_32F,CUDA_R_16BF,CUDA_R_16BF,CUDA_R_16BF,CUDA_R_16BF,128,ids,&count));
    for(int i=0;i<count;++i){cublasLtMatmulAlgo_t candidate{};auto status=cublasLtMatmulAlgoInit(handle,CUBLAS_COMPUTE_32F,CUDA_R_32F,CUDA_R_16BF,CUDA_R_16BF,CUDA_R_16BF,CUDA_R_16BF,ids[i],&candidate);if(status==CUBLAS_STATUS_SUCCESS)add(candidate);}
  }
  if(argc==8){
    const auto bases=algorithms;
    for(const auto& base:bases) {
      int support=0;size_t size=0;
      if(cublasLtMatmulAlgoCapGetAttribute(&base,CUBLASLT_ALGO_CAP_SPLITK_SUPPORT,&support,sizeof(support),&size)!=CUBLAS_STATUS_SUCCESS||!support)continue;
      for(unsigned split:{2u,4u,8u,16u})for(unsigned reduction:{1u,2u,4u}) {
        auto candidate=base;
        if(cublasLtMatmulAlgoConfigSetAttribute(&candidate,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split,sizeof(split))!=CUBLAS_STATUS_SUCCESS)continue;
        if(cublasLtMatmulAlgoConfigSetAttribute(&candidate,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,&reduction,sizeof(reduction))!=CUBLAS_STATUS_SUCCESS)continue;
        add(candidate);
      }
    }
    if(algorithms.size()>1024)throw std::runtime_error("candidate budget exceeded");
  }
  std::printf("{\"configuration\":true,\"rows\":%d,\"input\":%d,\"output\":%d,\"algorithms\":%zu,\"workspace_bytes\":%zu,\"output_initialized\":false,\"full_acceptance\":false}\n",n,k,m,algorithms.size(),budget);std::fflush(stdout);
  int executed=0,exact=0;unsigned index=0;
  for(const auto& candidate:algorithms){
    cublasLtMatmulHeuristicResult_t validated{};auto status=cublasLtMatmulAlgoCheck(handle,op,la,lb,lc,lc,&candidate,&validated);
    if(status!=CUBLAS_STATUS_SUCCESS||validated.workspaceSize>budget){std::printf("{\"candidate\":%u,\"supported\":false,\"status\":%d}\n",index++,int(status));continue;}
    std::printf("{\"launch_candidate\":%u,\"id\":%u,\"split_k\":%u,\"tile\":%u,\"stages\":%u,\"reduction\":%u}\n",index,attr(candidate,CUBLASLT_ALGO_CONFIG_ID),attr(candidate,CUBLASLT_ALGO_CONFIG_SPLITK_NUM),attr(candidate,CUBLASLT_ALGO_CONFIG_TILE_ID),attr(candidate,CUBLASLT_ALGO_CONFIG_STAGES_ID),attr(candidate,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME));std::fflush(stdout);
    float alpha=1,beta=0;check(cublasLtMatmul(handle,op,&alpha,a,la,b,lb,&beta,c,lc,c,lc,&candidate,workspace,budget,nullptr));
    lt_consume<<<(elements+255)/256,256>>>(c,sink,elements);check(cudaGetLastError());
    std::vector<char> out(expected.size()),consumed(expected.size());check(cudaMemcpy(out.data(),c,out.size(),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(consumed.data(),sink,consumed.size(),cudaMemcpyDeviceToHost));
    size_t mismatches=0,consumer_mismatches=0;for(size_t i=0;i<elements;++i){if(std::memcmp(out.data()+2*i,expected.data()+2*i,2))++mismatches;if(std::memcmp(out.data()+2*i,consumed.data()+2*i,2))++consumer_mismatches;}
    if(consumer_mismatches)throw std::runtime_error("consumer differs from producer");
    std::string base=prefix+"."+std::to_string(index);write(base+".algo.bin",&candidate,sizeof(candidate));
    write(base+".producer.bf16",out.data(),out.size());write(base+".consumer.bf16",consumed.data(),consumed.size());
    std::printf("{\"candidate\":%u,\"supported\":true,\"checked\":%zu,\"mismatches\":%zu,\"consumer_mismatches\":%zu}\n",index++,elements,mismatches,consumer_mismatches);std::fflush(stdout);
    ++executed;if(!mismatches)++exact;
  }
  check(cublasLtMatrixLayoutDestroy(la));check(cublasLtMatrixLayoutDestroy(lb));check(cublasLtMatrixLayoutDestroy(lc));
  check(cublasLtMatmulDescDestroy(op));check(cublasLtDestroy(handle));
  check(cudaFree(a));check(cudaFree(b));check(cudaFree(c));check(cudaFree(sink));check(cudaFree(workspace));
  std::printf("{\"complete\":true,\"executed\":%d,\"exact\":%d,\"full_acceptance\":false}\n",executed,exact);
  return executed?0:2;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
