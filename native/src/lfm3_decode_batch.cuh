#pragma once
// Include AFTER lfm3_main.cu with LFM3_RUNTIME_ONLY. Opt-in, one CUDA owner.
#include <tuple>

namespace lfm3_decode_batch {
inline void validate(const std::vector<Runner*>& lanes, const std::vector<int>& ids) {
  if(lanes.empty() || lanes.size()>2 || lanes.size()!=ids.size())
    throw std::invalid_argument("decode batch requires one or two token/owner pairs");
  for(size_t i=0;i<lanes.size();++i) {
    if(!lanes[i]) throw std::invalid_argument("null decode owner");
    const auto& r=*lanes[i];const auto& s=r.s;const auto& c=r.config;
    if(!r.weights || !r.trace.empty() || !c.is_lfm3() || !c.sandwich_norm || !c.tied || c.moe ||
       s.hidden!=2048 || s.head_dim!=128 || s.conv_width!=7 ||
       c.attention.size()!=size_t(s.layers) || c.layer_specs.size()!=size_t(s.layers))
      throw std::invalid_argument("unsupported decode owner or tracing enabled");
    lfm::text::scratch_rows(s.context);
    if(r.cursor<=0 || r.cursor>=s.context || r.step==std::numeric_limits<int>::max() ||
       ids[i]<0 || ids[i]>=s.vocabulary)
      throw std::invalid_argument("decode token/position/capacity invalid");
  }
  if(lanes.size()==1) return;
  const auto& a=*lanes[0];const auto& b=*lanes[1];
  if(&a==&b || a.weights.get()!=b.weights.get() || a.s.context<2)
    throw std::invalid_argument("decode owners must be distinct with shared weights and two-row scratch");
  auto geometry=[](const Runner& r) {
    const auto& s=r.s;const auto& c=r.config;
    return std::make_tuple(s.hidden,s.mlp,s.vocabulary,s.layers,s.query_heads,s.kv_heads,
      s.head_dim,s.conv_width,s.attention_layers,s.epsilon,s.rope_theta,c.architecture,
      c.sandwich_norm,c.attention_gate,c.tied,c.tensor_prefix,c.attention_output_name,
      c.query_norm_name,c.key_norm_name,c.attention);
  };
  if(geometry(a)!=geometry(b)) throw std::invalid_argument("decode geometry mismatch");
  for(int i=0;i<a.s.layers;++i) {
    const auto x=a.config.layer_specs[i],y=b.config.layer_specs[i];
    if(std::tie(x.kind,x.rope_theta,x.rotary_pairs,x.window)!=
       std::tie(y.kind,y.rope_theta,y.rotary_pairs,y.window))
      throw std::invalid_argument("decode layer specification mismatch");
  }
}

// Ragged, unpadded head GEMMs: exactly the serial decode end/key_start and lda.
inline void attention(Runner& scratch, Runner& owner, int lane, int layer,
                      Buffer<LFM3_SCORE>& scores, Buffer<bf16>& probs
#ifdef LFM3_NATIVE_GEMM
                      , Buffer<unsigned char>& work, size_t work_bytes
#endif
                      ) {
  const auto& s=owner.s;const int window=owner.config.layer_specs[layer].window;
  const int key_start=window>0?std::max(0,owner.cursor+1-window):0;
  const int end=owner.cursor+1-key_start;
#ifdef LFM3_NATIVE_GEMM
  const int pitch=key_pitch(end);
  head_scores(s,s.context,scratch.q.p+size_t(lane)*s.hidden,owner.keys[layer]->p,key_start,1,end,scores.p);
  if(end<=2048) probabilities<<<s.query_heads,32>>>(scores.p,probs.p,1,end,owner.cursor,window,s.head_dim,key_start,pitch);
  else probabilities_large<<<s.query_heads,1024>>>(scores.p,probs.p,1,end,owner.cursor,window,s.head_dim,key_start,pitch);
  head_values(s,s.context,probs.p,owner.values[layer]->p,key_start,1,end,scratch.attended.p+size_t(lane)*s.hidden,
              work.p,work_bytes);
#else
  std::vector<const bf16*> ap(s.query_heads),bp(s.query_heads);
  std::vector<bf16*> cp(s.query_heads);
  for(int h=0;h<s.query_heads;++h) {
    ap[h]=owner.keys[layer]->p+(size_t(h/(s.query_heads/s.kv_heads))*s.context+key_start)*s.head_dim;
    bp[h]=scratch.q.p+size_t(lane)*s.hidden+h*s.head_dim;
    cp[h]=scores.p+size_t(h)*end;
  }
  auto upload=[&] {
    cuda_check(cudaMemcpy(scratch.pointer_a.p,ap.data(),s.query_heads*sizeof(bf16*),cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(scratch.pointer_b.p,bp.data(),s.query_heads*sizeof(bf16*),cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(scratch.pointer_c.p,cp.data(),s.query_heads*sizeof(bf16*),cudaMemcpyHostToDevice));
  };
  upload();float alpha=1,beta=0;
  blas_check(cublasGemmBatchedEx(scratch.blas,CUBLAS_OP_T,CUBLAS_OP_N,end,1,s.head_dim,&alpha,
    reinterpret_cast<const void* const*>(scratch.pointer_a.p),CUDA_R_16BF,s.head_dim,
    reinterpret_cast<const void* const*>(scratch.pointer_b.p),CUDA_R_16BF,s.hidden,&beta,
    reinterpret_cast<void* const*>(scratch.pointer_c.p),CUDA_R_16BF,end,s.query_heads,
    CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  if(end<=2048) probabilities<<<s.query_heads,32>>>(scores.p,probs.p,1,end,owner.cursor,window,s.head_dim,key_start);
  else probabilities_large<<<s.query_heads,1024>>>(scores.p,probs.p,1,end,owner.cursor,window,s.head_dim,key_start);
  for(int h=0;h<s.query_heads;++h) {
    ap[h]=owner.values[layer]->p+(size_t(h/(s.query_heads/s.kv_heads))*s.context+key_start)*s.head_dim;
    bp[h]=probs.p+size_t(h)*end;
    cp[h]=scratch.attended.p+size_t(lane)*s.hidden+h*s.head_dim;
  }
  upload();
  blas_check(cublasGemmBatchedEx(scratch.blas,CUBLAS_OP_N,CUBLAS_OP_N,s.head_dim,1,end,&alpha,
    reinterpret_cast<const void* const*>(scratch.pointer_a.p),CUDA_R_16BF,s.head_dim,
    reinterpret_cast<const void* const*>(scratch.pointer_b.p),CUDA_R_16BF,end,&beta,
    reinterpret_cast<void* const*>(scratch.pointer_c.p),CUDA_R_16BF,s.hidden,s.query_heads,
    CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#endif
}

inline std::vector<std::vector<float>> forward(const std::vector<Runner*>& lanes,
                                              const std::vector<int>& ids) {
  validate(lanes,ids); // All metadata rejections precede any device/cache mutation.
  if(lanes.size()==1) return {lanes[0]->forward({ids[0]})};
  auto& a=*lanes[0];const auto& s=a.s;
#ifdef LFM3_NATIVE_GEMM
  const int widest=std::max(a.s.context,lanes[1]->s.context);
  const size_t scratch=size_t(key_pitch(widest))*s.query_heads;
  const size_t work_bytes=gemm_workspace_bytes(s,widest);
  Buffer<unsigned char> work(work_bytes);
#else
  const size_t scratch=size_t(std::max(a.s.context,lanes[1]->s.context))*s.query_heads;
#endif
  Buffer<LFM3_SCORE> scores(scratch);
  Buffer<bf16> probs(scratch);
  Buffer<LFM3_LOGIT> logits(size_t(2)*s.vocabulary);
  // Allocate host outputs BEFORE mutation; no allocation is needed to publish cursors.
  std::vector<LFM3_LOGIT> raw(size_t(2)*s.vocabulary);
  std::vector<std::vector<float>> result(2,std::vector<float>(s.vocabulary));
  const int positions[2]={a.cursor,lanes[1]->cursor};
  cuda_check(cudaMemcpy(a.tokens.p,ids.data(),2*sizeof(int),cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(a.positions.p,positions,2*sizeof(int),cudaMemcpyHostToDevice));
  embed<<<grid(size_t(2)*s.hidden),256>>>(a.w("embed_tokens.weight"),a.tokens.p,a.hidden.p,2,s.hidden);
  auto normalize=[&](const bf16* x,const std::string& name,bf16* out,int rows,int width,int serial_rows,
                     const bf16* residual=nullptr) {
    norm<<<rows,norm_threads(serial_rows,width)>>>(x,a.w(name),out,width,s.epsilon,residual);
  };
  for(int layer=0;layer<s.layers;++layer) {
    const std::string p="layers."+std::to_string(layer)+".";const auto spec=a.config.layer_specs[layer];
    normalize(a.hidden.p,p+"operator_norm.weight",a.normalized.p,2,s.hidden,1);
    if(!a.config.attention[layer]) {
      a.linear(p+"conv.in_proj.weight",a.normalized.p,a.projected.p,2,s.hidden,3*s.hidden);
      for(int lane=0;lane<2;++lane)
        lfm::lfm3::convolution7<<<grid(s.hidden),256>>>(a.projected.p+size_t(lane)*3*s.hidden,
          a.w(p+"conv.conv.weight"),lanes[lane]->states[layer]->p,
          a.operated.p+size_t(lane)*s.hidden,s.hidden,1);
      a.linear(p+"conv.out_proj.weight",a.operated.p,a.updated.p,2,s.hidden,s.hidden);
    } else {
      const std::string at=p+"self_attn.";
      a.linear(at+"q_proj.weight",a.normalized.p,a.q.p,2,s.hidden,s.hidden);
      a.linear(at+"k_proj.weight",a.normalized.p,a.k.p,2,s.hidden,s.kv_width());
      a.linear(at+"v_proj.weight",a.normalized.p,a.v.p,2,s.hidden,s.kv_width());
      normalize(a.q.p,at+"q_norm.weight",a.q.p,2*s.query_heads,s.head_dim,s.query_heads);
      normalize(a.k.p,at+"k_norm.weight",a.k.p,2*s.kv_heads,s.head_dim,s.kv_heads);
      rotary_from_frequencies<<<grid(size_t(2)*s.hidden/2),256>>>(a.q.p,a.positions.p,
        a.frequencies[layer]->p,2,s.query_heads,s.head_dim,spec.rotary_pairs);
      rotary_from_frequencies<<<grid(size_t(2)*s.kv_width()/2),256>>>(a.k.p,a.positions.p,
        a.frequencies[layer]->p,2,s.kv_heads,s.head_dim,spec.rotary_pairs);
      for(int lane=0;lane<2;++lane) {
        auto& r=*lanes[lane];
        scatter<<<grid(s.kv_width()),256>>>(a.k.p+size_t(lane)*s.kv_width(),a.v.p+size_t(lane)*s.kv_width(),
          r.keys[layer]->p,r.values[layer]->p,1,s.kv_heads,s.head_dim,r.s.context,r.cursor);
#ifdef LFM3_NATIVE_GEMM
        attention(a,r,lane,layer,scores,probs,work,work_bytes);
#else
        attention(a,r,lane,layer,scores,probs);
#endif
      }
      a.linear(at+"gate_proj.weight",a.normalized.p,a.gates.p,2,s.hidden,s.query_heads);
      lfm::lfm3::head_gate<<<grid(size_t(2)*s.hidden),256>>>(a.attended.p,a.gates.p,a.gated.p,
        size_t(2)*s.hidden,s.head_dim);
      a.linear(at+"o_proj.weight",a.gated.p,a.updated.p,2,s.hidden,s.hidden);
    }
    normalize(a.updated.p,p+"post_operator_norm.weight",a.hidden.p,2,s.hidden,1,a.hidden.p);
    normalize(a.hidden.p,p+"ffn_norm.weight",a.normalized.p,2,s.hidden,1);
    a.linear(p+"feed_forward.w1.weight",a.normalized.p,a.a.p,2,s.hidden,s.mlp);
    a.linear(p+"feed_forward.w3.weight",a.normalized.p,a.b.p,2,s.hidden,s.mlp);
    swiglu<<<grid(size_t(2)*s.mlp),256>>>(a.a.p,a.b.p,a.act.p,size_t(2)*s.mlp);
    a.linear(p+"feed_forward.w2.weight",a.act.p,a.updated.p,2,s.mlp,s.hidden);
    normalize(a.updated.p,p+"post_ffn_norm.weight",a.hidden.p,2,s.hidden,1,a.hidden.p);
  }
  normalize(a.hidden.p,"embedding_norm.weight",a.normalized.p,2,s.hidden,1);
#ifdef LFM3_NATIVE_GEMM
  a.lm_head(a.normalized.p,logits.p,2);
#else
  a.linear("embed_tokens.weight",a.normalized.p,logits.p,2,s.hidden,s.vocabulary);
#endif
  cuda_check(cudaGetLastError());
  cuda_check(cudaMemcpy(raw.data(),logits.p,raw.size()*sizeof(LFM3_LOGIT),cudaMemcpyDeviceToHost));
  for(int lane=0;lane<2;++lane) for(int i=0;i<s.vocabulary;++i) {
    const float value=LFM3_LOGIT_FLOAT(raw[size_t(lane)*s.vocabulary+i]);
    if(!std::isfinite(value)) throw std::runtime_error("nonfinite batched logits; discard BOTH caches");
    result[lane][i]=value;
  }
  for(auto* r:lanes) {++r->cursor;++r->step;}
  return result;
}
} // namespace lfm3_decode_batch
