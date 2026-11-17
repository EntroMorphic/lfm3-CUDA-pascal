// Diagnostic microbenchmarks, not a full-inference ceiling. All inputs are
// initialized; constant exact products and output guards check missing stores.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void blas(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS status " + std::to_string(status));
}
template <class T> struct Buffer {
  T* data = nullptr;
  size_t size;
  explicit Buffer(size_t n) : size(n) { check(cudaMalloc(&data, n*sizeof(T))); }
  ~Buffer() { cudaFree(data); }
  Buffer(const Buffer&) = delete;
};
__global__ void fill_bf16(__nv_bfloat16* data, size_t size, float value) {
  for (size_t i=blockIdx.x*blockDim.x+threadIdx.x; i<size; i+=static_cast<size_t>(gridDim.x)*blockDim.x)
    data[i]=__float2bfloat16_rn(value);
}
__global__ void launch_probe(unsigned* output) { if (!threadIdx.x) *output=0x12345678u; }
__global__ void read_stream(const uint4* input, uint4* output, size_t size) {
  uint4 value{};
  for (size_t i=blockIdx.x*blockDim.x+threadIdx.x; i<size; i+=static_cast<size_t>(gridDim.x)*blockDim.x) {
    const auto x=input[i]; value.x+=x.x; value.y+=x.y; value.z+=x.z; value.w+=x.w;
  }
  // Each thread has its own observable sink. Every initialized input is read.
  output[blockIdx.x*blockDim.x+threadIdx.x]=value;
}
double measure(const std::function<void()>& operation, int iterations) {
  for (int i=0;i<3;++i) operation();
  check(cudaDeviceSynchronize());
  cudaEvent_t start,stop; check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
  check(cudaEventRecord(start,cudaStreamPerThread));
  for (int i=0;i<iterations;++i) operation();
  check(cudaEventRecord(stop,cudaStreamPerThread)); check(cudaEventSynchronize(stop));
  float ms=0; check(cudaEventElapsedTime(&ms,start,stop));
  check(cudaEventDestroy(start)); check(cudaEventDestroy(stop)); check(cudaGetLastError());
  return ms*1000./iterations;
}
void record(const char* operation,int trial,double us,size_t bytes=0,double flops=0,int in=0,int out=0) {
  if (!(std::isfinite(us)&&us>0)) throw std::runtime_error("invalid measured time");
  std::printf("{\"operation\":\"%s\",\"trial\":%d,\"us\":%.9f,\"logical_bytes\":%zu,\"scheduled_flops\":%.0f,\"input_width\":%d,\"output_width\":%d,\"physical_n\":%d}\n",
      operation,trial,us,bytes,flops,in,out,in ? 32 : 0);
  std::fflush(stdout);
}
}  // namespace

int main(int argc,char** argv) {
  try {
    bool small=false;
    std::vector<std::pair<int,int>> shapes;
    for (int i=1;i<argc;++i) {
      const std::string arg=argv[i];
      if (arg=="--verify-small") small=true;
      else if (arg=="--shape"&&i+2<argc) {
        const int in=std::stoi(argv[++i]),out=std::stoi(argv[++i]);
        if (in<1||out<1||in>1048576||out>1048576) throw std::runtime_error("shape out of bounds");
        shapes.emplace_back(in,out);
      } else throw std::runtime_error("usage: component_floor [--verify-small] [--shape INPUT OUTPUT]...");
    }
    if (shapes.empty()) shapes=small?std::vector<std::pair<int,int>>{{128,256},{256,128}}:
        std::vector<std::pair<int,int>>{{1024,4608},{4608,1024},{1024,3072},{1024,2048},{1024,1024},{1024,65536}};
    cudaDeviceProp properties{}; check(cudaGetDeviceProperties(&properties,0));
    const size_t stream_bytes=small?4ULL<<20:512ULL<<20;
    const int blocks=properties.multiProcessorCount*16;
    Buffer<uint4> input(stream_bytes/sizeof(uint4)),sink(static_cast<size_t>(blocks)*256);
    check(cudaMemset(input.data,1,stream_bytes));
    for (int trial=1;trial<=5;++trial)
      record("stream_read",trial,measure([&]{read_stream<<<blocks,256,0,cudaStreamPerThread>>>(input.data,sink.data,input.size);},small?3:20),stream_bytes);
    std::vector<uint4> checksums(sink.size); check(cudaMemcpy(checksums.data(),sink.data,sink.size*sizeof(uint4),cudaMemcpyDeviceToHost));
    for (size_t thread=0;thread<sink.size;++thread) {
      const size_t count=thread<input.size?(input.size-1-thread)/sink.size+1:0;
      const unsigned expected=static_cast<unsigned>(count*0x01010101ULL);
      const auto got=checksums[thread];
      if (got.x!=expected||got.y!=expected||got.z!=expected||got.w!=expected) throw std::runtime_error("stream checksum mismatch");
    }
    Buffer<unsigned> probe(1);
    for (int trial=1;trial<=5;++trial) {
      record("eager_launch_event",trial,measure([&]{launch_probe<<<1,32,0,cudaStreamPerThread>>>(probe.data);},small?10:1000));
      cudaGraph_t graph; cudaGraphExec_t executable;
      check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
      for(int i=0;i<100;++i) launch_probe<<<1,32,0,cudaStreamPerThread>>>(probe.data);
      check(cudaStreamEndCapture(cudaStreamPerThread,&graph)); check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
      record("graph_launch_per_node_event",trial,measure([&]{check(cudaGraphLaunch(executable,cudaStreamPerThread));},small?3:100)/100.);
      check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
      unsigned observed=0; check(cudaMemcpy(&observed,probe.data,sizeof(observed),cudaMemcpyDeviceToHost));
      if (observed!=0x12345678u) throw std::runtime_error("launch probe mismatch");
      const auto begin=std::chrono::steady_clock::now();
      for(int i=0;i<100;++i) check(cudaMemcpy(&observed,probe.data,sizeof(observed),cudaMemcpyDeviceToHost));
      const auto end=std::chrono::steady_clock::now();
      record("pageable_d2h_word_wall",trial,std::chrono::duration<double,std::micro>(end-begin).count()/100.,sizeof(unsigned));
    }
    cublasHandle_t handle; blas(cublasCreate(&handle)); blas(cublasSetStream(handle,cudaStreamPerThread));
    Buffer<unsigned char> workspace(32ULL<<20); check(cudaMemset(workspace.data,0xff,workspace.size));
    blas(cublasSetWorkspace(handle,workspace.data,workspace.size));
    for (auto [in,out]:shapes) {
      const size_t elements=static_cast<size_t>(in)*out;
      const size_t bytes=elements*sizeof(__nv_bfloat16);
      const size_t copies=std::max<size_t>(2,(small?4ULL<<20:768ULL<<20)/bytes);
      Buffer<__nv_bfloat16> weights(elements*copies),x(static_cast<size_t>(in)*32),y(static_cast<size_t>(out)*32+128);
      fill_bf16<<<blocks,256,0,cudaStreamPerThread>>>(weights.data,weights.size,1.f/256);
      fill_bf16<<<blocks,256,0,cudaStreamPerThread>>>(x.data,x.size,1.f/128);
      check(cudaMemsetAsync(y.data,0xff,y.size*sizeof(__nv_bfloat16),cudaStreamPerThread));
      const float alpha=1,beta=0; size_t cursor=0;
      auto gemm=[&](bool poison){
        if(poison) check(cudaMemsetAsync(y.data,0xff,static_cast<size_t>(out)*32*sizeof(__nv_bfloat16),cudaStreamPerThread));
        const auto* w=weights.data+(cursor++%copies)*elements;
        blas(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,out,32,in,&alpha,w,CUDA_R_16BF,in,
            x.data,CUDA_R_16BF,in,&beta,y.data,CUDA_R_16BF,out,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      };
      for(int trial=1;trial<=5;++trial) {
        for(int variant=0;variant<2;++variant) {
          const bool poison=(variant+(trial%2))%2;
          record(poison?"canonical_gemm_poison":"canonical_gemm",trial,
              measure([&]{gemm(poison);},small?3:20),bytes,2.*in*out*32,in,out);
        }
      }
      std::vector<__nv_bfloat16> actual(y.size); check(cudaMemcpy(actual.data(),y.data,y.size*sizeof(__nv_bfloat16),cudaMemcpyDeviceToHost));
      const auto expected=__float2bfloat16(static_cast<float>(in)/32768.f);
      for(size_t i=0;i<static_cast<size_t>(out)*32;++i)
        if(__bfloat162float(actual[i])!=__bfloat162float(expected)) throw std::runtime_error("GEMM value mismatch");
      for(size_t i=static_cast<size_t>(out)*32;i<actual.size();++i)
        if(!std::isnan(__bfloat162float(actual[i]))) throw std::runtime_error("GEMM output guard modified");
    }
    blas(cublasDestroy(handle));
    std::fprintf(stderr,"component_floor: finite/exact constant GEMM, stream checksum and output guard checks passed\n");
    return 0;
  } catch(const std::exception& error) {
    std::fprintf(stderr,"error: %s\n",error.what()); return 1;
  }
}
