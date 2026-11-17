// Minimal documented tensor store comparison, not a cuBLAS/runtime workaround.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
void check(cudaError_t s) {if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
void driver(CUresult s) {if(s!=CUDA_SUCCESS)throw std::runtime_error("driver status="+std::to_string(int(s)));}
__global__ void tma_or_ordinary_store(float* out,const __grid_constant__ CUtensorMap map,int tma,float bias) {
  __shared__ __align__(128) float shared[128];
  const int i=int(blockIdx.x)*128+int(threadIdx.x);
  shared[threadIdx.x]=float(i%37-18)*0.25f+bias;
  // Each writer orders its shared stores before the async proxy, then all join.
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if(tma) {
    if(threadIdx.x==0) {
      uint32_t src=uint32_t(__cvta_generic_to_shared(shared));
      uint64_t descriptor=reinterpret_cast<uint64_t>(&map);
      int zero=0,tile=int(blockIdx.x);
      asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.bulk_group [%0, {%1, %2, %3, %4}], [%5];"
                   ::"l"(descriptor),"r"(zero),"r"(zero),"r"(zero),"r"(tile),"r"(src):"memory");
      asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      // Wait for the write, not merely the shared-memory read, to complete.
      asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    }
  } else out[i]=shared[threadIdx.x];
  __syncthreads();
}
__global__ void consume_store(const float* out,float* sink) {
  const int i=int(blockIdx.x)*128+int(threadIdx.x);sink[i]=out[i];
}
void write(const std::string& path,const void* data,size_t bytes) {
  std::ofstream f(path,std::ios::binary);f.write(static_cast<const char*>(data),bytes);if(!f)throw std::runtime_error("write failed");
}
int main(int argc,char** argv) try {
  if((argc!=4&&argc!=5)||(std::strcmp(argv[1],"ordinary")&&std::strcmp(argv[1],"tma4d")&&std::strcmp(argv[1],"missing")))throw std::runtime_error("usage: probe ordinary|tma4d|missing OUTPUT_PREFIX BIAS [preimage]");
  std::string bias_text=argv[3];
  if(bias_text.empty()||bias_text.find_first_not_of("0123456789")!=std::string::npos)throw std::runtime_error("invalid bias");
  int bias=std::stoi(bias_text);if(bias<1||bias>32768)throw std::runtime_error("bias out of range");
  bool preimage=argc==5;if(preimage&&std::strcmp(argv[4],"preimage"))throw std::runtime_error("invalid diagnostic option");
  constexpr size_t n=12288;float *out,*sink;
  check(cudaMalloc(&out,n*4));check(cudaMalloc(&sink,n*4));
  alignas(64) CUtensorMap map{};
  cuuint64_t dims[4]={128,1,1,96},strides[3]={512,512,512};
  cuuint32_t box[4]={128,1,1,1},element_strides[4]={1,1,1,1};
  driver(cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,4,out,dims,strides,box,element_strides,
                             CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  int runtime_version,driver_version;check(cudaRuntimeGetVersion(&runtime_version));check(cudaDriverGetVersion(&driver_version));
  std::printf("{\"configuration\":true,\"method\":\"%s\",\"elements\":12288,\"rank\":4,\"dimensions\":[128,1,1,96],\"strides\":[512,512,512],\"box\":[128,1,1,1],\"runtime\":%d,\"driver\":%d,\"output_initialized\":false,\"bias\":%d,\"preimage\":%s}\n",argv[1],runtime_version,driver_version,bias,preimage?"true":"false");std::fflush(stdout);
  std::string prefix=argv[2];write(prefix+".tensormap.bin",&map,sizeof(map));
  if(preimage) {
    std::vector<unsigned char> before(n*4);check(cudaMemcpy(before.data(),out,n*4,cudaMemcpyDeviceToHost));
    write(prefix+".before.bin",before.data(),before.size());
  }
  if(std::strcmp(argv[1],"missing")) {tma_or_ordinary_store<<<96,128>>>(out,map,!std::strcmp(argv[1],"tma4d"),float(bias));check(cudaGetLastError());}
  consume_store<<<96,128>>>(out,sink);check(cudaGetLastError());
  std::vector<float> producer(n),consumer(n);
  check(cudaMemcpy(consumer.data(),sink,n*4,cudaMemcpyDeviceToHost));check(cudaMemcpy(producer.data(),out,n*4,cudaMemcpyDeviceToHost));
  write(prefix+".producer.f32",producer.data(),n*4);write(prefix+".consumer.f32",consumer.data(),n*4);
  check(cudaFree(out));check(cudaFree(sink));
  std::printf("{\"complete\":true,\"elements\":12288,\"method\":\"%s\"}\n",argv[1]);return 0;
} catch(const std::exception& e) {std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
