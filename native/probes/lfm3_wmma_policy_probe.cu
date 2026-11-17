// Native WMMA arithmetic candidate. No production integration or acceptance implied.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
using bf16=__nv_bfloat16;
void check(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
std::vector<char> read(const std::string& p,size_t bytes){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f||f.tellg()!=std::streamoff(bytes))throw std::runtime_error("input size");std::vector<char> b(bytes);f.seekg(0);f.read(b.data(),bytes);if(!f)throw std::runtime_error("input read");return b;}
void write(const std::string& p,const void* b,size_t n){std::ofstream f(p,std::ios::binary);f.write(static_cast<const char*>(b),n);if(!f)throw std::runtime_error("output write");}
__global__ void wmma_partials(const bf16* w,const bf16* x,float* partial,int m,int n,int k,int padded_n,int padded_m,int parts){
  namespace wm=nvcuda::wmma;
  __shared__ __align__(32) bf16 left[256],right[256];
  int lane=int(threadIdx.x),row=int(blockIdx.x)*16,col=int(blockIdx.y)*16;
  int start=int(blockIdx.z)*(k/parts),end=start+k/parts;
  const bool full_left=row+16<=m,full_right=col+16<=n;
  wm::fragment<wm::matrix_a,16,16,16,bf16,wm::row_major> a;
  wm::fragment<wm::matrix_b,16,16,16,bf16,wm::col_major> b;
  wm::fragment<wm::accumulator,16,16,16,float> sum;
  wm::fill_fragment(sum,0.0f);
  for(int base=start;base<end;base+=16){
    // Load complete aligned tiles directly into fragments. Shared memory is
    // needed only for input-tail padding, not as a compulsory staging copy.
    if(!full_left||!full_right){
      for(int i=lane;i<256;i+=32){
        int wr=row+i/16,wk=base+i%16,xc=col+i/16,xk=base+i%16;
        if(!full_left)left[i]=wr<m?w[size_t(wr)*k+wk]:__float2bfloat16_rn(0.0f);
        if(!full_right)right[i]=xc<n?x[size_t(xc)*k+xk]:__float2bfloat16_rn(0.0f);
      }
      __syncwarp();
    }
    if(full_left)wm::load_matrix_sync(a,w+size_t(row)*k+base,k);
    else wm::load_matrix_sync(a,left,16);
    if(full_right)wm::load_matrix_sync(b,x+size_t(col)*k+base,k);
    else wm::load_matrix_sync(b,right,16);
    wm::mma_sync(sum,a,b,sum);__syncwarp();
  }
  wm::store_matrix_sync(partial+size_t(blockIdx.z)*padded_m*padded_n+size_t(row)*padded_n+col,sum,padded_n,wm::mem_row_major);
}
__device__ __forceinline__ uint32_t bf16_pair(const bf16* p){
  // cudaMalloc alignment, K/partition multiples of16, and even BF16 pair
  // offsets guarantee four-byte alignment. Preserve the raw operand bits.
  uint32_t bits;asm volatile("ld.global.u32 %0, [%1];":"=r"(bits):"l"(p));return bits;
}
// PTX m16n8k16 row/column fragment layout. Each warp owns sixteen output rows;
// four independent warps share a block without shared memory or CTA barriers.
__global__ void mma_thin_partials(const bf16* w,const bf16* x,float* partial,int m,int n,int k,int padded_n,int padded_m,int parts){
  int lane=int(threadIdx.x)&31,warp=int(threadIdx.x)/32;
  int row=(int(blockIdx.x)*(int(blockDim.x)/32)+warp)*16;
  if(row>=m)return; // Entire warp retires; no block-wide synchronization follows.
  int group=lane/4,t=lane%4,start=int(blockIdx.z)*(k/parts),end=start+k/parts;
  float c0=0,c1=0,c2=0,c3=0;
  for(int base=start;base<end;base+=16){
    int offset=base+2*t;
    uint32_t a0=row+group<m?bf16_pair(w+size_t(row+group)*k+offset):0;
    uint32_t a1=row+group+8<m?bf16_pair(w+size_t(row+group+8)*k+offset):0;
    uint32_t a2=row+group<m?bf16_pair(w+size_t(row+group)*k+offset+8):0;
    uint32_t a3=row+group+8<m?bf16_pair(w+size_t(row+group+8)*k+offset+8):0;
    uint32_t b0=group<n?bf16_pair(x+size_t(group)*k+offset):0;
    uint32_t b1=group<n?bf16_pair(x+size_t(group)*k+offset+8):0;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
  }
  size_t head=size_t(blockIdx.z)*padded_m*padded_n;
  if(row+group<m){size_t offset=head+size_t(row+group)*padded_n+2*t;partial[offset]=c0;partial[offset+1]=c1;}
  if(row+group+8<m){size_t offset=head+size_t(row+group+8)*padded_n+2*t;partial[offset]=c2;partial[offset+1]=c3;}
}
__global__ void wmma_finish(const float* p,bf16* out,size_t count,int m,int n,int padded_m,int padded_n,int parts){
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
  int row=int(i%m),col=int(i/m);float sum=p[size_t(row)*padded_n+col];
  for(int part=1;part<parts;++part)sum+=p[size_t(part)*padded_m*padded_n+size_t(row)*padded_n+col];
  out[i]=__float2bfloat16_rn(sum);
}
__global__ void wmma_consume(const bf16* x,bf16* y,size_t n){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)y[i]=x[i];}
int number(const char* p){std::string s=p;if(s.empty()||s.find_first_not_of("0123456789")!=std::string::npos)throw std::runtime_error("dimension");return std::stoi(s);}
int main(int argc,char** argv)try{
  if(argc!=9)throw std::runtime_error("usage: probe ROWS INPUT OUTPUT SPLITS W.bf16 X.bf16 EXPECT.bf16 PREFIX");
  int n=number(argv[1]),k=number(argv[2]),m=number(argv[3]),parts=number(argv[4]);
  if(n<1||n>4096||m<1||m>128000||k<16||k>32768||parts<1||parts>16||k%parts||(k/parts)%16||uint64_t(m)*k>300000000)throw std::runtime_error("bounded aligned shape required");
  auto w=read(argv[5],size_t(m)*k*2),x=read(argv[6],size_t(n)*k*2),expected=read(argv[7],size_t(m)*n*2);
  int pm=(m+15)/16*16,pn=n<=8?8:(n+15)/16*16;bf16 *a,*b,*c,*sink;float* partial;
  check(cudaMalloc(&a,w.size()));check(cudaMalloc(&b,x.size()));check(cudaMalloc(&c,expected.size()));check(cudaMalloc(&sink,expected.size()));
  check(cudaMalloc(&partial,size_t(pm)*pn*parts*sizeof(float)));
  check(cudaMemcpy(a,w.data(),w.size(),cudaMemcpyHostToDevice));check(cudaMemcpy(b,x.data(),x.size(),cudaMemcpyHostToDevice));
  std::printf("{\"configuration\":true,\"rows\":%d,\"input\":%d,\"output\":%d,\"splits\":%d,\"output_initialized\":false,\"full_acceptance\":false}\n",n,k,m,parts);std::fflush(stdout);
  if(n<=8){int warps=std::min(4,pm/16);mma_thin_partials<<<dim3((pm/16+warps-1)/warps,1,parts),32*warps>>>(a,b,partial,m,n,k,pn,pm,parts);}
  else wmma_partials<<<dim3(pm/16,pn/16,parts),32>>>(a,b,partial,m,n,k,pn,pm,parts);
  check(cudaGetLastError());
  size_t count=size_t(m)*n;wmma_finish<<<(count+255)/256,256>>>(partial,c,count,m,n,pm,pn,parts);check(cudaGetLastError());
  wmma_consume<<<(count+255)/256,256>>>(c,sink,count);check(cudaGetLastError());
  std::vector<char> actual(expected.size()),consumed(expected.size());check(cudaMemcpy(actual.data(),c,actual.size(),cudaMemcpyDeviceToHost));check(cudaMemcpy(consumed.data(),sink,consumed.size(),cudaMemcpyDeviceToHost));
  size_t mismatches=0,consumer_mismatches=0;for(size_t i=0;i<count;++i){if(std::memcmp(actual.data()+i*2,expected.data()+i*2,2))++mismatches;if(std::memcmp(actual.data()+i*2,consumed.data()+i*2,2))++consumer_mismatches;}
  write(std::string(argv[8])+".producer.bf16",actual.data(),actual.size());write(std::string(argv[8])+".consumer.bf16",consumed.data(),consumed.size());
  std::printf("{\"complete\":true,\"checked\":%zu,\"mismatches\":%zu,\"consumer_mismatches\":%zu,\"full_acceptance\":false}\n",count,mismatches,consumer_mismatches);
  check(cudaFree(a));check(cudaFree(b));check(cudaFree(c));check(cudaFree(sink));check(cudaFree(partial));return consumer_mismatches?2:0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
