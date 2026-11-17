// Small maximum-dimension shared-memory probes, separate from integrated coverage.
#define LFM3_RUNTIME_ONLY
#include "../src/lfm3_main.cu"

int main() try {
  size_t checks=0,values=0;
  for(int end:{1,31,32,2048,2049,4096,8192,32768}) {
    const int rows=std::min(3,end);
    Buffer<bf16> scores(size_t(rows)*end),output(size_t(rows)*end);
    cuda_check(cudaMemset(scores.p,0,size_t(rows)*end*sizeof(bf16)));
    for(int start:{0,end-rows}) for(int window:{0,1,256}) {
      if(end<=2048) probabilities<<<rows,32>>>(scores.p,output.p,rows,end,start,window,128,0);
      else probabilities_large<<<rows,1024>>>(scores.p,output.p,rows,end,start,window,128,0);
      cuda_check(cudaGetLastError());
      std::vector<bf16> host(size_t(rows)*end);
      cuda_check(cudaMemcpy(host.data(),output.p,host.size()*sizeof(bf16),cudaMemcpyDeviceToHost));
      for(int row=0;row<rows;++row) {
        int position=start+row,begin=window?std::max(0,position-window+1):0;
        const auto probability=__float2bfloat16_rn(1.0f/float(position-begin+1));
        for(int token=0;token<end;++token) {
          float expected=token>=begin&&token<=position?__bfloat162float(probability):0.0f;
          if(__bfloat162float(host[size_t(row)*end+token])!=expected)
            throw std::runtime_error("uniform causal/window probability mismatch");
        }
      }
      ++checks;values+=host.size();
    }
  }
  for(int width:{128,2048}) for(int rows:{1,4,17}) {
    std::vector<bf16> input(size_t(rows)*width,__float2bfloat16_rn(1.0f));
    std::vector<bf16> weights(width,__float2bfloat16_rn(1.0f));
    Buffer<bf16> x(input.size()),w(width),output(input.size());
    cuda_check(cudaMemcpy(x.p,input.data(),input.size()*sizeof(bf16),cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(w.p,weights.data(),weights.size()*sizeof(bf16),cudaMemcpyHostToDevice));
    for(bool residual:{false,true}) {
      norm<<<rows,norm_threads(rows,width)>>>(x.p,w.p,output.p,width,1e-5f,residual?x.p:nullptr);
      cuda_check(cudaGetLastError());
      std::vector<bf16> host(input.size());
      cuda_check(cudaMemcpy(host.data(),output.p,host.size()*sizeof(bf16),cudaMemcpyDeviceToHost));
      for(auto value:host) if(__bfloat162float(value)!=(residual?2.0f:1.0f))
        throw std::runtime_error("normalization/residual boundary mismatch");
      ++checks;values+=host.size();
    }
  }
  std::cout<<"PASS: "<<checks<<" shared-memory operator fixtures, "<<values
           <<" exactly checked BF16 values; max softmax width32768; not integrated race coverage\n";
  return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
