// Diagnostic only. All outputs are consumed; no output initialization or runtime changes.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using bf16=__nv_bfloat16;
void check(cudaError_t s) {if(s!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));}
void check(cublasStatus_t s) {if(s!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS status="+std::to_string(int(s)));}
void stage(const char* name) {
  auto us=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
  std::cout<<"{\"stage\":\""<<name<<"\",\"host_monotonic_us\":"<<us<<"}\n"<<std::flush;
}
std::vector<bf16> read(const std::string& path,size_t n) {
  std::ifstream f(path,std::ios::binary|std::ios::ate);
  if(!f || f.tellg()!=std::streamoff(n*sizeof(bf16))) throw std::runtime_error("input size mismatch");
  std::vector<bf16> v(n);f.seekg(0);f.read(reinterpret_cast<char*>(v.data()),n*sizeof(bf16));
  if(!f) throw std::runtime_error("input read failed");return v;
}
void write(const std::string& path,const std::vector<bf16>& v) {
  std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(bf16));
  if(!f) throw std::runtime_error("output write failed");
}
__global__ void fill_output(bf16* c,size_t n,float value) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)c[i]=__float2bfloat16_rn(value);
}
__global__ void consume_all(const bf16* c,bf16* d,size_t n) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)d[i]=c[i];
}
int main(int argc,char** argv) try {
  if(argc!=11&&argc!=13) throw std::runtime_error("usage: probe PRODUCER ROWS WIDTH COMPUTE ALGO MATH A.bf16 B.bf16 OUTPUT_PREFIX SYNC [WORKSPACE_SEED DUMP_BYTES]");
  std::string producer=argv[1],compute=argv[4],algo=argv[5],math=argv[6],prefix=argv[9],sync=argv[10];
  auto number=[](const char* text){std::string s(text);if(s.empty()||s.find_first_not_of("0123456789")!=std::string::npos)throw std::runtime_error("bad dimension");return std::stoi(s);};
  int rows=number(argv[2]),width=number(argv[3]);
  std::string workspace_seed=argc==13?argv[11]:"none";
  int dump_bytes=argc==13?number(argv[12]):0;
  if((producer!="gemm"&&producer!="kernel"&&producer!="missing") || (rows!=1&&rows!=257) || width!=2048 ||
     (compute!="default"&&compute!="pedantic") || (algo!="tensor"&&algo!="default") ||
     (math!="default"&&math!="pedantic") || (sync!="none"&&sync!="producer") ||
     (workspace_seed!="none"&&workspace_seed!="a5"&&workspace_seed!="5a") || dump_bytes>8519680 || dump_bytes%4) throw std::runtime_error("unsupported diagnostic configuration");
  const int output=3*width;
  auto av=read(argv[7],size_t(output)*width),bv=read(argv[8],size_t(rows)*width);
  stage("before_cuda_setup");
  int device,driver,runtime;check(cudaGetDevice(&device));check(cudaDriverGetVersion(&driver));check(cudaRuntimeGetVersion(&runtime));
  cudaDeviceProp props{};check(cudaGetDeviceProperties(&props,device));
  bf16 *a,*b,*c,*d;void* workspace;size_t n=size_t(rows)*output;
  check(cudaMalloc(&a,av.size()*2));check(cudaMalloc(&b,bv.size()*2));check(cudaMalloc(&c,n*2));check(cudaMalloc(&d,n*2));check(cudaMalloc(&workspace,8519680));
  check(cudaMemcpy(a,av.data(),av.size()*2,cudaMemcpyHostToDevice));check(cudaMemcpy(b,bv.data(),bv.size()*2,cudaMemcpyHostToDevice));
  cublasHandle_t handle;check(cublasCreate(&handle));check(cublasSetWorkspace(handle,workspace,8519680));
  // Explicitly preserve the baseline's default handle mode; change only requested knobs.
  if(math=="pedantic")check(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH));
  cublasMath_t actual;cudaStream_t stream;int version;
  check(cublasGetMathMode(handle,&actual));check(cublasGetStream(handle,&stream));check(cublasGetVersion(handle,&version));
  // A separately labeled scratch-write diagnostic, never an acceptance workaround.
  // Neither c nor d is initialized. The default remains untouched workspace.
  if(workspace_seed!="none") {check(cudaMemset(workspace,workspace_seed=="a5"?0xa5:0x5a,8519680));stage("diagnostic_workspace_seed_submitted");}
  auto ct=compute=="default"?CUBLAS_COMPUTE_32F:CUBLAS_COMPUTE_32F_PEDANTIC;
  auto algorithm=algo=="tensor"?CUBLAS_GEMM_DEFAULT_TENSOR_OP:CUBLAS_GEMM_DEFAULT;
  std::cout<<"{\"configuration\":true,\"producer\":\""<<producer<<"\",\"rows\":"<<rows<<",\"width\":"<<width
    <<",\"compute\":\""<<compute<<"\",\"algo\":\""<<algo<<"\",\"math\":\""<<math<<"\",\"sync\":\""<<sync
    <<"\",\"compute_enum\":"<<int(ct)<<",\"algo_enum\":"<<int(algorithm)<<",\"actual_math_enum\":"<<int(actual)
    <<",\"default_stream\":"<<(stream==nullptr?"true":"false")<<",\"cublas\":"<<version<<",\"driver\":"<<driver
    <<",\"runtime\":"<<runtime<<",\"sm_major\":"<<props.major<<",\"sm_minor\":"<<props.minor<<",\"workspace_bytes\":8519680,\"workspace_seed\":\""<<workspace_seed<<"\",\"workspace_dump_bytes\":"<<dump_bytes<<"}\n"<<std::flush;
  if(stream!=nullptr)throw std::runtime_error("unexpected nondefault producer stream");
  auto allocation=[](const char* name,const void* ptr,size_t bytes) {
    std::cout<<"{\"allocation\":\""<<name<<"\",\"base\":"<<reinterpret_cast<std::uintptr_t>(ptr)<<",\"bytes\":"<<bytes<<"}\n"<<std::flush;
  };
  allocation("a",a,av.size()*2);allocation("b",b,bv.size()*2);allocation("c",c,n*2);allocation("d",d,n*2);allocation("workspace",workspace,8519680);
  if(dump_bytes) {
    std::vector<char> before(dump_bytes);stage("before_workspace_preimage_download");
    check(cudaMemcpy(before.data(),workspace,dump_bytes,cudaMemcpyDeviceToHost));
    std::ofstream f(prefix+".workspace-before.bin",std::ios::binary);f.write(before.data(),dump_bytes);
    if(!f)throw std::runtime_error("workspace preimage write failed");stage("workspace_preimage_download_returned");
  }
  stage("before_producer_submission");
  if(producer=="gemm") {
    float alpha=1,beta=0;
    check(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,output,rows,width,&alpha,a,CUDA_R_16BF,width,b,CUDA_R_16BF,width,&beta,c,CUDA_R_16BF,output,ct,algorithm));
  } else if(producer=="kernel") {fill_output<<<(n+255)/256,256>>>(c,n,float(width));check(cudaGetLastError());}
  stage("producer_submission_returned");
  if(sync=="producer") {check(cudaDeviceSynchronize());stage("producer_device_synchronize_returned");}
  stage("before_consumer_submission");consume_all<<<(n+255)/256,256>>>(c,d,n);check(cudaGetLastError());stage("consumer_submission_returned");
  std::vector<bf16> hc(n),hd(n);
  stage("before_consumer_blocking_download");check(cudaMemcpy(hd.data(),d,n*2,cudaMemcpyDeviceToHost));stage("consumer_blocking_download_returned");
  check(cudaMemcpy(hc.data(),c,n*2,cudaMemcpyDeviceToHost));stage("producer_blocking_download_returned");
  write(prefix+".producer.bf16",hc);write(prefix+".consumer.bf16",hd);
  if(dump_bytes) {
    std::vector<char> scratch(dump_bytes);check(cudaMemcpy(scratch.data(),workspace,dump_bytes,cudaMemcpyDeviceToHost));
    std::ofstream f(prefix+".workspace.bin",std::ios::binary);f.write(scratch.data(),dump_bytes);
    if(!f)throw std::runtime_error("workspace diagnostic write failed");stage("workspace_postcompletion_download_returned");
  }
  check(cublasDestroy(handle));check(cudaFree(a));check(cudaFree(b));check(cudaFree(c));check(cudaFree(d));check(cudaFree(workspace));
  std::cout<<"{\"complete\":true,\"outputs\":"<<n<<",\"consumer_elements\":"<<n<<"}\n";
  return 0;
} catch(const std::exception& e) {std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}
