// Experimental batch-one text runner. Separate from the guarded LFM2 runtime.
#include <lfm/model_config.hpp>
#include <lfm/safetensors.hpp>
#include <lfm/lfm3_ops.cuh>
#include <lfm/text_serving_limits.hpp>
#include <lfm/text_attention_tiles.hpp>
#include <lfm/text_sampling.hpp>
#include <lfm/handle_setup.hpp>
#include <lfm/lfm3_text_request.hpp>
#ifdef LFM3_NATIVE_GEMM
// G4: the engine's GEMMs are lfm::gemm (no cuBLAS). The forward pass is held to journal §9's
// perplexity bar instead of bitwise identity with the cuBLAS build.
#include <lfm/allocation_metrics.hpp>
#include <lfm/gemm.cuh>
#else
#include <cublas_v2.h>
#endif
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>

using bf16 = __nv_bfloat16;
namespace {
void cuda_check(cudaError_t s) { if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
#ifndef LFM3_NATIVE_GEMM
void blas_check(cublasStatus_t s) { if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS error " + std::to_string(s)); }
#endif
#ifdef LFM3_NATIVE_GEMM
// Engine buffers are counted when LFM_ALLOCATION_METRICS=1, so the worker can report its own live device
// bytes: on Thor's unified memory, device-wide free memory also moves with host activity.
template<class T> struct Buffer {
  T* p = nullptr;
  explicit Buffer(size_t n) { cuda_check(lfm::allocation_metrics::malloc_requested(&p, lfm::checked_product({n, sizeof(T)}))); }
  ~Buffer() { lfm::allocation_metrics::free_requested(p); }
#else
template<class T> struct Buffer {
  T* p = nullptr;
  explicit Buffer(size_t n) { cuda_check(cudaMalloc(&p, lfm::checked_product({n, sizeof(T)}))); }
  ~Buffer() { cudaFree(p); }
#endif
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};
__global__ void embed(const bf16* weights, const int* ids, bf16* out, int rows, int width) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size_t(rows) * width) out[i] = weights[size_t(ids[i / width]) * width + i % width];
}
// Vector-4 schedule from ATen Reduce.cuh; retained PyTorch license applies.
// Match the reduction's row-count-dependent block width, not just its formula.
// Audio and image input (#3, #4): row rows[j] of `hidden` becomes row j of `source` (encoded media's projected features).
__global__ void inject_rows(const bf16* source, const int* rows, bf16* hidden, int count, int width) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size_t(count) * width) hidden[size_t(rows[i / width]) * width + i % width] = source[i];
}
__global__ void norm(const bf16* x, const bf16* w, bf16* y, int width, float eps,
                     const bf16* residual = nullptr) {
  const size_t base = size_t(blockIdx.x) * width;
  float accum[4] = {0,0,0,0};
  for (int d = threadIdx.x*4; d < width; d += blockDim.x*4) {
    for(int j=0;j<4;++j) {
      float v = __bfloat162float(x[base+d+j]);
      accum[j] = __fadd_rn(accum[j],__fmul_rn(v,v));
    }
  }
  float sum=((accum[0]+accum[1])+accum[2])+accum[3];
  __shared__ float sums[512];
  sums[threadIdx.x] = sum;
  for (int s = blockDim.x/2; s>=32; s/=2) {
    __syncthreads();
    if (threadIdx.x < s) { sum+=sums[threadIdx.x+s]; sums[threadIdx.x]=sum; }
  }
  __syncthreads();
  if(threadIdx.x<32) for(int s=16;s;s/=2) sum+=__shfl_down_sync(0xffffffff,sum,s);
  __shared__ float scale;
  if(threadIdx.x==0) scale=rsqrtf(__fadd_rn(__fmul_rn(sum,1.0f/width),eps));
  __syncthreads();
  for (int d = threadIdx.x; d < width; d += blockDim.x) {
    auto value=__hmul(__float2bfloat16_rn(__fmul_rn(__bfloat162float(x[base+d]),scale)),w[d]);
    y[base+d]=residual?__hadd(value,residual[base+d]):value;
  }
}
int norm_threads(int rows,int width) {
  auto power=[](int n) {int x=1; while(x<=n/2 && x<512) x*=2; return x;};
  int dim0=power(width/4),dim1=power(rows);
  int bx=std::min(dim0,32),by=std::min(dim1,512/bx);
  return std::min(dim0,512/by);
}
__global__ void swiglu(const bf16* a, const bf16* b, bf16* out, size_t n) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = __bfloat162float(a[i]);
    // ATen SiLU multiplies by the reciprocal sigmoid, then rounds to BF16.
    const float sigmoid = 1.0f / (1.0f + expf(-x));
    out[i] = __hmul(__float2bfloat16_rn(__fmul_rn(x, sigmoid)), b[i]);
  }
}
__global__ void rotary_from_frequencies(bf16* x,const int* positions,const float* inverse,
                                        int rows,int heads,int dim,int pairs) {
  const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=size_t(rows)*heads*(dim/2)) return;
  const int d=i%(dim/2);
  if(d>=pairs) return;
  const size_t h=i/(dim/2),base=h*dim+d;
  const float angle=__fmul_rn(float(positions[h/heads]),inverse[d]);
  const float cosine=__bfloat162float(__float2bfloat16_rn(cosf(angle)));
  const float sine=__bfloat162float(__float2bfloat16_rn(sinf(angle)));
  const float a=__bfloat162float(x[base]),b=__bfloat162float(x[base+dim/2]);
  x[base]=__float2bfloat16_rn(__fadd_rn(__fmul_rn(a,cosine),-__fmul_rn(b,sine)));
  x[base+dim/2]=__float2bfloat16_rn(__fadd_rn(__fmul_rn(b,cosine),__fmul_rn(a,sine)));
}
__global__ void scatter(const bf16* k, const bf16* v, bf16* kc, bf16* vc,
                        int rows, int heads, int dim, int capacity, int start) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size_t(rows) * heads * dim) {
    int d = i % dim, h = (i / dim) % heads, t = i / (heads * dim);
    size_t dst = (size_t(h) * capacity + start + t) * dim + d;
    kc[dst] = k[i]; vc[dst] = v[i];
  }
}
#ifdef LFM3_NATIVE_GEMM
// Native GEMM build: scores and probs rows are `pitch` apart, and columns [end, pitch) of probs are zero.
#define LFM3_PITCH_PARAMETER ,int pitch
#define LFM3_ROW_STRIDE pitch
// Scores stay FP32 from the QKᵀ accumulator through the 1/sqrt(d) scale (journal §9 item 4; SDPA does the
// same). Only the probabilities are rounded to BF16, as the P·V operand. validation/g4-native-gemm-v2.
#define LFM3_SCORE float
#define LFM3_SCALED_SCORE(x) __fmul_rn((x),rsqrtf(float(dim)))
// Logits stay FP32 out of lm_head: they feed only the log-softmax and the sampler, so nothing in the model
// changes, and the convexity bias of rounding them (8e-5 nats/token, validation/g4-native-gemm-v3) is gone.
#define LFM3_LOGIT float
#define LFM3_LOGIT_FLOAT(x) (x)
#else
#define LFM3_PITCH_PARAMETER
#define LFM3_ROW_STRIDE end
#define LFM3_SCORE bf16
#define LFM3_SCALED_SCORE(x) __bfloat162float(__float2bfloat16_rn(__fmul_rn(__bfloat162float(x),rsqrtf(float(dim)))))
#define LFM3_LOGIT bf16
#define LFM3_LOGIT_FLOAT(x) __bfloat162float(x)
#endif
__global__ void probabilities(const LFM3_SCORE* scores,bf16* probs,int rows,int end,int start,int window,int dim,int key_start LFM3_PITCH_PARAMETER) {
  const int row=blockIdx.x%rows,tid=threadIdx.x;
  const size_t base=size_t(blockIdx.x)*LFM3_ROW_STRIDE;
  const int position=start+row;
  const int begin=window?max(0,position-window+1):0;
  // ATen persistent-softmax order: lane-strided accumulators, XOR butterfly.
  float elements[128];
  int power=1; while(power<end) power*=2;
  int iterations=(power+31)/32;
  float maximum=-INFINITY;
  for(int i=0;i<iterations;++i) {
    int t=tid+i*32;
    float x=-INFINITY;
    if(t<end && t+key_start>=begin && t+key_start<=position)
      x=LFM3_SCALED_SCORE(scores[base+t]);
    elements[i]=x; maximum=fmaxf(maximum,x);
  }
  for(int offset=16;offset;offset/=2) maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,offset));
  float sum=0;
  for(int i=0;i<iterations;++i) {
    elements[i]=expf(elements[i]-maximum); sum+=elements[i];
  }
  for(int offset=16;offset;offset/=2) sum+=__shfl_xor_sync(0xffffffff,sum,offset);
  for(int i=0;i<iterations;++i) {
    int t=tid+i*32;
    if(t<end) probs[base+t]=__float2bfloat16_rn(elements[i]/sum);
  }
#ifdef LFM3_NATIVE_GEMM
  for(int t=end+tid;t<pitch;t+=32) probs[base+t]=__float2bfloat16_rn(0.f);
#endif
}
template<bool Maximum> __device__ float block_softmax_reduce(float value,float* shared) {
  const int tid=threadIdx.x,lane=tid%32;
  for(int offset=16;offset;offset/=2) {
    const float other=__shfl_down_sync(0xffffffff,value,offset);
    value=Maximum?fmaxf(value,other):__fadd_rn(value,other);
  }
  __syncthreads();
  if(lane==0) shared[tid/32]=value;
  __syncthreads();
  if(tid<32) {
    value=shared[tid];
    for(int offset=16;offset;offset/=2) {
      const float other=__shfl_down_sync(0xffffffff,value,offset);
      value=Maximum?fmaxf(value,other):__fadd_rn(value,other);
    }
    if(tid==0) shared[0]=value;
  }
  __syncthreads();
  return shared[0];
}
// ATen FP32 forward switches from persistent warp to register/block at 2049.
// See native/third_party/PYTORCH_LICENSE and audit/CONTEXT.md for attribution.
__global__ void probabilities_large(const LFM3_SCORE* scores,bf16* probs,int rows,int end,int start,int window,int dim,int key_start LFM3_PITCH_PARAMETER) {
  const int tid=threadIdx.x,row=blockIdx.x%rows,position=start+row;
  const int begin=window?max(0,position-window+1):0;
  const size_t base=size_t(blockIdx.x)*LFM3_ROW_STRIDE;
  // Each pass recomputes a key's masked, scaled score (the same arithmetic every time) instead of keeping it in a
  // per-thread array: an array of 32 covered 32,768 keys, so longer contexts would have overrun it.
  auto score=[&](int t) {
    return t+key_start>=begin && t+key_start<=position ? float(LFM3_SCALED_SCORE(scores[base+t])) : -INFINITY;
  };
  float maximum=-3.402823466e38f;
  for(int t=tid;t<end;t+=1024) maximum=fmaxf(maximum,score(t));
  __shared__ float shared[32];
  maximum=block_softmax_reduce<true>(maximum,shared);
  float sum=0;
  for(int t=tid;t<end;t+=1024) sum=__fadd_rn(sum,expf(score(t)-maximum));
  sum=block_softmax_reduce<false>(sum,shared);
  for(int t=tid;t<end;t+=1024) probs[base+t]=__float2bfloat16_rn(expf(score(t)-maximum)/sum);
#ifdef LFM3_NATIVE_GEMM
  for(int t=end+tid;t<pitch;t+=1024) probs[base+t]=__float2bfloat16_rn(0.f);
#endif
}
unsigned grid(size_t n) { return (n + 255) / 256; }
lfm::Json read_json(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  f.seekg(0, std::ios::end);
  if (f.tellg() > 4 * 1024 * 1024) throw std::runtime_error("request too large");
  f.seekg(0);
  return lfm::Json::parse(std::string(std::istreambuf_iterator<char>(f), {}));
}
std::vector<int> ids(const lfm::Json& req, const std::string& key, int vocab) {
  std::vector<int> out;
  if (!req.has(key)) return out;
  for (const auto& x : req.at(key).array()) {
    int id = x.integer();
    if (id >= vocab) throw std::runtime_error("token outside vocabulary: " + key);
    out.push_back(id);
  }
  return out;
}

#ifdef LFM3_NATIVE_GEMM
// Attention rows are padded to a multiple of 8 keys (the GEMM's K granularity) with zero probability.
inline int key_pitch(int keys) { return (keys+7)/8*8; }
inline size_t gemm_workspace_bytes(const lfm::RuntimeShape& s,int context) {
  // Split-K P·V scratch for up to 8 query rows (decode, and prefill tiles of <= 8 rows).
  return lfm::gemm::workspace_bytes(8,s.head_dim,key_pitch(context),s.query_heads);
}
// Query heads per prefill attention call (decode runs all heads in one call). Bit-invariant (lfm::gemm's
// per-element arithmetic does not depend on the batch count), so it is chosen for speed under a memory budget: the
// most heads, in whole KV groups dividing the head count, whose FP32 score buffer fits kPrefillScoreBudget.
// Measured (validation/g4-prefill-heads-v1/): in 128-row tiles (capacity > 4,096) more heads per call is faster; in
// one large tile it does not matter. Result for LFM3-Omni-2B (16 heads, 4 KV heads): 16 heads at capacity <= 2,048,
// 8 at 2,049-2,896, 4 at 2,897-4,096, 16 above 4,096; never more than 256 MiB. The BF16 probability buffer has the
// same element count (half the bytes). LFM3_PREFILL_HEADS overrides it, for measurement only.
inline constexpr size_t kPrefillScoreBudget = size_t(256) << 20;
inline int prefill_heads(const lfm::RuntimeShape& s) {
  static const int requested = [] {
    const char* v = std::getenv("LFM3_PREFILL_HEADS");
    if (!v) return 0;
    char* end = nullptr; errno = 0;
    const long n = std::strtol(v, &end, 10);
    if (end == v || *end != '\0' || errno || n < 1 || n > 65535)
      throw std::runtime_error("LFM3_PREFILL_HEADS must be a positive integer");
    return int(n);
  }();
  const int group = s.query_heads / s.kv_heads;
  if (requested != 0) {
    if (requested < group || requested % group || s.query_heads % requested)
      throw std::runtime_error("LFM3_PREFILL_HEADS must be a multiple of the KV group size that divides the head count");
    return requested;
  }
  const size_t per_head = size_t(lfm::text::scratch_rows(s.context)) * key_pitch(s.context) * sizeof(float);
  int best = group;  // one KV group is the floor, even if it exceeds the budget
  for (int heads = group; heads <= s.query_heads; heads += group)
    if (s.query_heads % heads == 0 && per_head * heads <= kPrefillScoreBudget) best = heads;
  return best;
}
inline size_t padded_attention_scratch(const lfm::RuntimeShape& s) {
  const size_t pitch=key_pitch(s.context), group=prefill_heads(s);
  return std::max(size_t(lfm::text::scratch_rows(s.context))*pitch*group, pitch*s.query_heads);
}
// scores[h] (rows x pitch) = Q_h · K_{h/group}ᵀ over cache positions [key_start, key_start+end). `context` is
// the capacity of the cache that holds the keys (its per-head stride).
// `heads` query heads starting at `queries`, whose KV heads start at `key_cache` (a multiple of the group).
inline void head_scores(const lfm::RuntimeShape& s,int context,const bf16* queries,const bf16* key_cache,
                        int key_start,int rows,int end,float* scores,int heads=0) {
  lfm::gemm::Problem p;
  p.a=queries; p.b=key_cache+size_t(key_start)*s.head_dim; p.c=nullptr; p.c_f32=scores;
  p.m=rows; p.n=end; p.k=s.head_dim;
  p.lda=s.hidden; p.ldb=s.head_dim; p.ldc=key_pitch(end);
  p.stride_a=s.head_dim; p.stride_b=(long long)context*s.head_dim; p.stride_c=(long long)rows*key_pitch(end);
  p.batch=heads?heads:s.query_heads; p.group=s.query_heads/s.kv_heads;
  lfm::gemm::run(p);
}
// out_h (rows x head_dim, row stride hidden) = P_h (rows x pitch) · V_{h/group}[key_start, key_start+end).
inline void head_values(const lfm::RuntimeShape& s,int context,const bf16* probs,const bf16* value_cache,
                        int key_start,int rows,int end,bf16* out,void* workspace,size_t workspace_size,int heads=0) {
  lfm::gemm::Problem p;
  p.a=probs; p.b=value_cache+size_t(key_start)*s.head_dim; p.c=out;
  p.m=rows; p.n=s.head_dim; p.k=key_pitch(end); p.b_rows=end; p.b_k_contiguous=false;
  p.lda=key_pitch(end); p.ldb=s.head_dim; p.ldc=s.hidden;
  p.stride_a=(long long)rows*key_pitch(end); p.stride_b=(long long)context*s.head_dim; p.stride_c=s.head_dim;
  p.batch=heads?heads:s.query_heads; p.group=s.query_heads/s.kv_heads;
  p.workspace=static_cast<float*>(workspace); p.workspace_size=workspace_size;
  lfm::gemm::run(p);
}
#endif

#ifdef LFM3_NATIVE_GEMM
// Writes STEM.logits.f32 (exact) and STEM.logits.bf16 (rounded, the established trace format).
inline void write_logits_trace(const std::string& stem,const std::vector<float>& logits) {
  std::ofstream exact(stem+".logits.f32",std::ios::binary);
  exact.write(reinterpret_cast<const char*>(logits.data()),logits.size()*sizeof(float));
  std::vector<bf16> rounded(logits.size());
  for(size_t i=0;i<logits.size();++i) rounded[i]=__float2bfloat16_rn(logits[i]);
  std::ofstream f(stem+".logits.bf16",std::ios::binary);
  f.write(reinterpret_cast<const char*>(rounded.data()),rounded.size()*sizeof(bf16));
  if(!exact || !f) throw std::runtime_error("trace write failed");
}
#endif

class Runner {
 public:
  lfm::ModelConfig config;
  lfm::RuntimeShape s;
  std::shared_ptr<lfm::Safetensors> weights;
#ifndef LFM3_NATIVE_GEMM
  cublasHandle_t blas = nullptr;
#endif
  Buffer<bf16> hidden, normalized, projected, operated, updated, q, k, v, gates, attended, gated, a, b, act;
  Buffer<LFM3_LOGIT> logits;
  Buffer<int> positions, tokens;
  Buffer<unsigned char> workspace;
  Buffer<LFM3_SCORE> score_buffer;
  Buffer<bf16> probability_buffer;
  Buffer<const bf16*> pointer_a, pointer_b;
  Buffer<bf16*> pointer_c;
  std::vector<std::unique_ptr<Buffer<bf16>>> keys, values, states;
  std::vector<std::unique_ptr<Buffer<float>>> frequencies;
  std::unique_ptr<Buffer<int>> injected_rows;
  size_t injected_capacity = 0;
  std::string trace;
  bool trace_logits_only = false;
  int step = 0, cursor = 0;

  Runner(const std::string& directory, lfm::ModelConfig cfg,
         std::shared_ptr<lfm::Safetensors> shared_weights = {})
      : config(cfg), s(cfg.shape), weights(shared_weights?std::move(shared_weights):
          std::make_shared<lfm::Safetensors>(directory + "/model.safetensors")),
        hidden(size_t(s.context)*s.hidden), normalized(size_t(s.context)*s.hidden),
        projected(size_t(s.context)*3*s.hidden), operated(size_t(s.context)*s.hidden), updated(size_t(s.context)*s.hidden),
        q(size_t(s.context)*s.hidden), k(size_t(s.context)*s.kv_width()), v(size_t(s.context)*s.kv_width()),
        gates(size_t(s.context)*s.query_heads), attended(size_t(s.context)*s.hidden), gated(size_t(s.context)*s.hidden),
        a(size_t(s.context)*s.mlp), b(size_t(s.context)*s.mlp), act(size_t(s.context)*s.mlp), logits(s.vocabulary),
#ifdef LFM3_NATIVE_GEMM
        positions(s.context), tokens(s.context), workspace(gemm_workspace_bytes(s,s.context)),
        score_buffer(padded_attention_scratch(s)), probability_buffer(padded_attention_scratch(s)),
#else
        positions(s.context), tokens(s.context), workspace(4096*1024*2+16*1024*8),
        score_buffer(lfm::text::attention_scratch_elements(s.context,s.query_heads)),
        probability_buffer(lfm::text::attention_scratch_elements(s.context,s.query_heads)),
#endif
        pointer_a(s.query_heads),pointer_b(s.query_heads),pointer_c(s.query_heads) {
    // Validate the complete selected text schema before launching any forward kernels.
    weight("embed_tokens.weight", {s.vocabulary, s.hidden}); weight("embedding_norm.weight", {s.hidden});
    for (int i = 0; i < s.layers; ++i) {
      std::string p = "layers." + std::to_string(i) + ".";
      for (auto name : {"operator_norm.weight", "ffn_norm.weight", "post_operator_norm.weight", "post_ffn_norm.weight"})
        weight(p + name, {s.hidden});
      weight(p+"feed_forward.w1.weight", {s.mlp,s.hidden}); weight(p+"feed_forward.w3.weight", {s.mlp,s.hidden});
      weight(p+"feed_forward.w2.weight", {s.hidden,s.mlp});
      if (config.attention[i]) {
        for (auto name : {"q_proj.weight", "o_proj.weight"}) weight(p+"self_attn."+name, {s.hidden,s.hidden});
        for (auto name : {"k_proj.weight", "v_proj.weight"}) weight(p+"self_attn."+name, {s.kv_width(),s.hidden});
        for (auto name : {"q_norm.weight", "k_norm.weight"}) weight(p+"self_attn."+name, {s.head_dim});
        weight(p+"self_attn.gate_proj.weight", {s.query_heads,s.hidden});
        keys.emplace_back(new Buffer<bf16>(s.kv_layer_stride())); values.emplace_back(new Buffer<bf16>(s.kv_layer_stride()));
        states.emplace_back(nullptr);
        frequencies.emplace_back(new Buffer<float>(s.head_dim/2));
        std::vector<float> inverse(s.head_dim/2,0.0f);
        const auto spec=config.layer_specs[i];
        // The checkpoint builds arange/pow/reciprocal on CPU, then transfers.
        // GPU powf differs by an ULP at some frequencies (observable at pos850).
        for(int d=0;d<spec.rotary_pairs;++d)
          inverse[d]=1.0f/::powf(spec.rope_theta,float(2*d)/s.head_dim);
        cuda_check(cudaMemcpy(frequencies.back()->p,inverse.data(),inverse.size()*sizeof(float),cudaMemcpyHostToDevice));
      } else {
        weight(p+"conv.conv.weight", {s.hidden,1,7}); weight(p+"conv.in_proj.weight", {3*s.hidden,s.hidden});
        weight(p+"conv.out_proj.weight", {s.hidden,s.hidden});
        keys.emplace_back(nullptr); values.emplace_back(nullptr); states.emplace_back(new Buffer<bf16>(size_t(s.hidden)*7));
        frequencies.emplace_back(nullptr);
      }
    }
#ifdef LFM3_NATIVE_GEMM
    reset();
  }
#else
    blas_check(cublasCreate(&blas));
    lfm::finish_handle_setup(blas,[](cublasHandle_t handle) noexcept {cublasDestroy(handle);},[&] {
      // Match PyTorch 2.10's non-Hopper cuBLAS workspace budget.
      blas_check(cublasSetWorkspace(blas, workspace.p, 4096*1024*2+16*1024*8));
      // Match default PyTorch BF16 cuBLAS policy; outputs materialize in BF16.
      blas_check(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH));
      reset();
    });
  }
  ~Runner() { if (blas) cublasDestroy(blas); }
#endif
  const bf16* weight(const std::string& name, std::initializer_list<int64_t> shape) {
    const auto& t = weights->at(config.tensor_prefix + name);
    if (t.dtype != "BF16" || t.shape != std::vector<int64_t>(shape)) throw std::runtime_error("weight contract: " + name);
    return t.device_data;
  }
  const bf16* w(const std::string& name) { return weights->at(config.tensor_prefix + name).device_data; }
  void reset() {
    cuda_check(cudaDeviceSynchronize());
    cursor = 0; step = 0;
    for (int i=0;i<s.layers;++i) {
      if (states[i]) cuda_check(cudaMemset(states[i]->p,0,size_t(s.hidden)*7*sizeof(bf16)));
      if (keys[i]) {
        cuda_check(cudaMemset(keys[i]->p,0,s.kv_layer_stride()*sizeof(bf16)));
        cuda_check(cudaMemset(values[i]->p,0,s.kv_layer_stride()*sizeof(bf16)));
      }
    }
  }
  void dump(const std::string& name, const bf16* ptr, size_t n) {
    if (trace.empty() || (trace_logits_only && name!="logits")) return;
    std::vector<bf16> data(n);
    cuda_check(cudaMemcpy(data.data(),ptr,n*sizeof(bf16),cudaMemcpyDeviceToHost));
    std::ofstream f(trace+"/"+std::to_string(step)+"."+name+".bf16",std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()),n*sizeof(bf16));
    if (!f) throw std::runtime_error("trace write failed");
  }
#ifdef LFM3_NATIVE_GEMM
  // Traces rows of `width` values stored `pitch` apart, as `count` contiguous rows.
  // `append`: a later KV-head group of the same tensor (prefill), written after the earlier groups.
  void dump_rows(const std::string& name, const bf16* ptr, size_t count, int width, int pitch, bool append=false) {
    if (width == pitch && !append) return dump(name, ptr, count*width);
    if (trace.empty() || (trace_logits_only && name!="logits")) return;
    std::vector<bf16> data(count*width);
    cuda_check(cudaMemcpy2D(data.data(),width*sizeof(bf16),ptr,pitch*sizeof(bf16),width*sizeof(bf16),count,cudaMemcpyDeviceToHost));
    std::ofstream f(trace+"/"+std::to_string(step)+"."+name+".bf16",append?std::ios::binary|std::ios::app:std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(bf16));
    if (!f) throw std::runtime_error("trace write failed");
  }
#endif
#ifdef LFM3_NATIVE_GEMM
  // lm_head with the FP32 accumulator stored unrounded.
  void lm_head(const bf16* x, float* out, int rows) {
    auto p=lfm::gemm::linear(x,w("embed_tokens.weight"),nullptr,rows,s.hidden,s.vocabulary);
    p.c_f32=out; lfm::gemm::run(p);
  }
  // Logits traces: STEP.logits.f32 holds the exact FP32 values the engine returns (and samples from);
  // STEP.logits.bf16 keeps the established format, rounded, for comparisons against BF16 references.
  void dump_logits(const float* ptr, size_t n) {
    if (trace.empty()) return;
    std::vector<float> data(n);
    cuda_check(cudaMemcpy(data.data(),ptr,n*sizeof(float),cudaMemcpyDeviceToHost));
    write_logits_trace(trace+"/"+std::to_string(step),data);
  }
#endif
  void linear(const std::string& name, const bf16* x, bf16* out, int rows, int input, int output) {
#ifdef LFM3_NATIVE_GEMM
    lfm::gemm::run(lfm::gemm::linear(x,w(name),out,rows,input,output));
#else
    float alpha=1, beta=0;
    // Row-major X W^T = column-major W X^T, with W stored row-major.
    blas_check(cublasGemmEx(blas,CUBLAS_OP_T,CUBLAS_OP_N,output,rows,input,&alpha,
      w(name),CUDA_R_16BF,input,x,CUDA_R_16BF,input,&beta,out,CUDA_R_16BF,output,
      CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#endif
  }
  void normalize(const bf16* x, const std::string& weight_name, bf16* out, int rows, int width) {
    norm<<<rows,norm_threads(rows,width)>>>(x,w(weight_name),out,width,s.epsilon);
  }
  void attention_tile(int layer,int rows,int window,int first,int total_rows) {
    // Dynamic sliding cache returns only its last window for one-token decode.
    // Masking a larger product is mathematically equivalent, but not BF16-equivalent.
    const auto range=lfm::text::attention_key_range(s.context,cursor,total_rows,first,rows,window);
    const int key_start=range.begin;
    const int end=range.span();
#ifdef LFM3_NATIVE_GEMM
    const int pitch=key_pitch(end);
    std::string label="layers."+std::to_string(layer)+".self_attn.probabilities";
    if(total_rows>lfm::text::scratch_rows(s.context)) label+=".tile"+std::to_string(first);
    // Decode (one row): all heads at once. Prefill: prefill_heads() at a time (see its comment).
    const int group=rows==1?s.query_heads:prefill_heads(s);
    for(int h0=0;h0<s.query_heads;h0+=group) {
      const size_t kv=size_t(h0/(s.query_heads/s.kv_heads))*s.context*s.head_dim;
      head_scores(s,s.context,q.p+size_t(first)*s.hidden+size_t(h0)*s.head_dim,keys[layer]->p+kv,key_start,rows,end,
                  score_buffer.p,group);
      if(end<=2048) probabilities<<<rows*group,32>>>(score_buffer.p,probability_buffer.p,rows,end,cursor+first,window,s.head_dim,key_start,pitch);
      else probabilities_large<<<rows*group,1024>>>(score_buffer.p,probability_buffer.p,rows,end,cursor+first,window,s.head_dim,key_start,pitch);
      dump_rows(label,probability_buffer.p,size_t(group)*rows,end,pitch,h0>0);
      head_values(s,s.context,probability_buffer.p,values[layer]->p+kv,key_start,rows,end,
                  attended.p+size_t(first)*s.hidden+size_t(h0)*s.head_dim,workspace.p,gemm_workspace_bytes(s,s.context),group);
    }
#else
    std::vector<const bf16*> ap(s.query_heads),bp(s.query_heads);
    std::vector<bf16*> cp(s.query_heads);
    for(int h=0;h<s.query_heads;++h) {
      ap[h]=keys[layer]->p+(size_t(h/(s.query_heads/s.kv_heads))*s.context+key_start)*s.head_dim;
      bp[h]=q.p+size_t(first)*s.hidden+h*s.head_dim;
      cp[h]=score_buffer.p+size_t(h)*rows*end;
    }
    auto upload=[&] {
      cuda_check(cudaMemcpy(pointer_a.p,ap.data(),s.query_heads*sizeof(bf16*),cudaMemcpyHostToDevice));
      cuda_check(cudaMemcpy(pointer_b.p,bp.data(),s.query_heads*sizeof(bf16*),cudaMemcpyHostToDevice));
      cuda_check(cudaMemcpy(pointer_c.p,cp.data(),s.query_heads*sizeof(bf16*),cudaMemcpyHostToDevice));
    };
    upload();
    float alpha=1,beta=0;
    blas_check(cublasGemmBatchedEx(blas,CUBLAS_OP_T,CUBLAS_OP_N,end,rows,s.head_dim,&alpha,
      reinterpret_cast<const void* const*>(pointer_a.p),CUDA_R_16BF,s.head_dim,
      reinterpret_cast<const void* const*>(pointer_b.p),CUDA_R_16BF,s.hidden,&beta,
      reinterpret_cast<void* const*>(pointer_c.p),CUDA_R_16BF,end,s.query_heads,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    if(end<=2048) probabilities<<<rows*s.query_heads,32>>>(score_buffer.p,probability_buffer.p,rows,end,cursor+first,window,s.head_dim,key_start);
    else probabilities_large<<<rows*s.query_heads,1024>>>(score_buffer.p,probability_buffer.p,rows,end,cursor+first,window,s.head_dim,key_start);
    std::string label="layers."+std::to_string(layer)+".self_attn.probabilities";
    if(total_rows>lfm::text::scratch_rows(s.context)) label+=".tile"+std::to_string(first);
    dump(label,probability_buffer.p,size_t(s.query_heads)*rows*end);
    for(int h=0;h<s.query_heads;++h) {
      ap[h]=values[layer]->p+(size_t(h/(s.query_heads/s.kv_heads))*s.context+key_start)*s.head_dim;
      bp[h]=probability_buffer.p+size_t(h)*rows*end;
      cp[h]=attended.p+size_t(first)*s.hidden+h*s.head_dim;
    }
    upload();
    blas_check(cublasGemmBatchedEx(blas,CUBLAS_OP_N,CUBLAS_OP_N,s.head_dim,rows,end,&alpha,
      reinterpret_cast<const void* const*>(pointer_a.p),CUDA_R_16BF,s.head_dim,
      reinterpret_cast<const void* const*>(pointer_b.p),CUDA_R_16BF,end,&beta,
      reinterpret_cast<void* const*>(pointer_c.p),CUDA_R_16BF,s.hidden,s.query_heads,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#endif
  }
  void attention(int layer,int rows,int window) {
    const int tile=lfm::text::scratch_rows(s.context);
    for(int first=0;first<rows;first+=tile)
      attention_tile(layer,std::min(tile,rows-first),window,first,rows);
  }
  // `injected` (device, [count][hidden]): the input embeddings of the rows holding `placeholder`, in order, in place of
  // the placeholder's own embedding row (audio input; HF's masked_scatter of the audio features). Prefill only.
  std::vector<float> forward(const std::vector<int>& input, const bf16* injected = nullptr, int placeholder = -1) {
    return forward(input, injected, std::vector<int>{placeholder});
  }
  // Several placeholders (audio and image input together): the rows holding any of them, in prompt order, take the
  // rows of `injected` in that order; the media's features stacked as the prompt orders them (HF scatters each
  // modality's features into its own placeholders; with each modality's features in prompt order, the same rows).
  std::vector<float> forward(const std::vector<int>& input, const bf16* injected, const std::vector<int>& placeholders) {
    const int rows = input.size();
    if (rows < 1 || rows > s.context-cursor || (cursor && rows != 1))
      throw std::runtime_error("only empty-cache prefill or single-token decode within capacity is supported");
    for (int id : input) if (id < 0 || id >= s.vocabulary) throw std::runtime_error("invalid token");
    std::vector<int> pos(rows);
    for(int t=0;t<rows;++t) pos[t]=cursor+t;
    cuda_check(cudaMemcpy(tokens.p,input.data(),rows*sizeof(int),cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(positions.p,pos.data(),rows*sizeof(int),cudaMemcpyHostToDevice));
    embed<<<grid(size_t(rows)*s.hidden),256>>>(w("embed_tokens.weight"),tokens.p,hidden.p,rows,s.hidden);
    if(injected) {
      if(cursor) throw std::runtime_error("injected input embeddings are a prefill-only input");
      std::vector<int> at;
      for(int t=0;t<rows;++t) if(std::find(placeholders.begin(),placeholders.end(),input[t])!=placeholders.end()) at.push_back(t);
      if(at.empty()) throw std::runtime_error("injected input embeddings without placeholders");
      if(!injected_rows || injected_capacity<at.size())
        injected_rows=std::make_unique<Buffer<int>>(at.size()), injected_capacity=at.size();
      cuda_check(cudaMemcpy(injected_rows->p,at.data(),at.size()*sizeof(int),cudaMemcpyHostToDevice));
      inject_rows<<<grid(at.size()*s.hidden),256>>>(injected,injected_rows->p,hidden.p,int(at.size()),s.hidden);
    }
    dump("embed",hidden.p,size_t(rows)*s.hidden);
    for(int layer=0;layer<s.layers;++layer) {
      const std::string p="layers."+std::to_string(layer)+".";
      const auto spec=config.layer_specs[layer];
      normalize(hidden.p,p+"operator_norm.weight",normalized.p,rows,s.hidden);
      dump(p+"operator_norm",normalized.p,size_t(rows)*s.hidden);
      if(!config.attention[layer]) {
        linear(p+"conv.in_proj.weight",normalized.p,projected.p,rows,s.hidden,3*s.hidden);
        dump(p+"conv.in_proj",projected.p,size_t(rows)*3*s.hidden);
        if(cursor==0) lfm::lfm3::convolution7_prefill<<<grid(size_t(rows)*s.hidden),256>>>(projected.p,w(p+"conv.conv.weight"),states[layer]->p,operated.p,1,rows,s.hidden);
        else lfm::lfm3::convolution7<<<grid(s.hidden),256>>>(projected.p,w(p+"conv.conv.weight"),states[layer]->p,operated.p,s.hidden,1);
        dump(p+"conv.product",operated.p,size_t(rows)*s.hidden);
        linear(p+"conv.out_proj.weight",operated.p,updated.p,rows,s.hidden,s.hidden);
      } else {
        const std::string at=p+"self_attn.";
        linear(at+"q_proj.weight",normalized.p,q.p,rows,s.hidden,s.hidden);
        linear(at+"k_proj.weight",normalized.p,k.p,rows,s.hidden,s.kv_width());
        linear(at+"v_proj.weight",normalized.p,v.p,rows,s.hidden,s.kv_width());
        normalize(q.p,at+"q_norm.weight",q.p,rows*s.query_heads,s.head_dim);
        normalize(k.p,at+"k_norm.weight",k.p,rows*s.kv_heads,s.head_dim);
        rotary_from_frequencies<<<grid(size_t(rows)*s.hidden/2),256>>>(q.p,positions.p,frequencies[layer]->p,rows,s.query_heads,s.head_dim,spec.rotary_pairs);
        rotary_from_frequencies<<<grid(size_t(rows)*s.kv_width()/2),256>>>(k.p,positions.p,frequencies[layer]->p,rows,s.kv_heads,s.head_dim,spec.rotary_pairs);
        dump(at+"q_rotary",q.p,size_t(rows)*s.hidden); dump(at+"k_rotary",k.p,size_t(rows)*s.kv_width());
        scatter<<<grid(size_t(rows)*s.kv_width()),256>>>(k.p,v.p,keys[layer]->p,values[layer]->p,rows,s.kv_heads,s.head_dim,s.context,cursor);
        attention(layer,rows,spec.window);
        dump(at+"attended",attended.p,size_t(rows)*s.hidden);
        linear(at+"gate_proj.weight",normalized.p,gates.p,rows,s.hidden,s.query_heads);
        lfm::lfm3::head_gate<<<grid(size_t(rows)*s.hidden),256>>>(attended.p,gates.p,gated.p,size_t(rows)*s.hidden,s.head_dim);
        dump(at+"gated",gated.p,size_t(rows)*s.hidden);
        linear(at+"o_proj.weight",gated.p,updated.p,rows,s.hidden,s.hidden);
      }
      dump(p+"operator",updated.p,size_t(rows)*s.hidden);
      norm<<<rows,norm_threads(rows,s.hidden)>>>(updated.p,w(p+"post_operator_norm.weight"),hidden.p,s.hidden,s.epsilon,hidden.p);
      dump(p+"residual",hidden.p,size_t(rows)*s.hidden);
      normalize(hidden.p,p+"ffn_norm.weight",normalized.p,rows,s.hidden);
      dump(p+"ffn_norm",normalized.p,size_t(rows)*s.hidden);
      linear(p+"feed_forward.w1.weight",normalized.p,a.p,rows,s.hidden,s.mlp);
      linear(p+"feed_forward.w3.weight",normalized.p,b.p,rows,s.hidden,s.mlp);
      dump(p+"feed_forward.w1",a.p,size_t(rows)*s.mlp);
      dump(p+"feed_forward.w3",b.p,size_t(rows)*s.mlp);
      swiglu<<<grid(size_t(rows)*s.mlp),256>>>(a.p,b.p,act.p,size_t(rows)*s.mlp);
      dump(p+"feed_forward.product",act.p,size_t(rows)*s.mlp);
      linear(p+"feed_forward.w2.weight",act.p,updated.p,rows,s.mlp,s.hidden);
      dump(p+"feed_forward",updated.p,size_t(rows)*s.hidden);
      norm<<<rows,norm_threads(rows,s.hidden)>>>(updated.p,w(p+"post_ffn_norm.weight"),hidden.p,s.hidden,s.epsilon,hidden.p);
      dump(p+"output",hidden.p,size_t(rows)*s.hidden);
    }
    normalize(hidden.p,"embedding_norm.weight",normalized.p,rows,s.hidden);
    dump("final_norm",normalized.p,size_t(rows)*s.hidden);
#ifdef LFM3_NATIVE_GEMM
    lm_head(normalized.p+size_t(rows-1)*s.hidden,logits.p,1);
    dump_logits(logits.p,s.vocabulary);
#else
    linear("embed_tokens.weight",normalized.p+size_t(rows-1)*s.hidden,logits.p,1,s.hidden,s.vocabulary);
    dump("logits",logits.p,s.vocabulary);
#endif
    cuda_check(cudaGetLastError());
    std::vector<LFM3_LOGIT> raw(s.vocabulary);
    cuda_check(cudaMemcpy(raw.data(),logits.p,s.vocabulary*sizeof(LFM3_LOGIT),cudaMemcpyDeviceToHost));
    std::vector<float> result(s.vocabulary);
    for(int i=0;i<s.vocabulary;++i) {
      result[i]=LFM3_LOGIT_FLOAT(raw[i]);
      if(!std::isfinite(result[i])) throw std::runtime_error("nonfinite logits");
    }
    cursor+=rows; ++step;
    return result;
  }
};
void print_ids(std::ostream& f,const std::vector<int>& x) {
  f<<"["; for(size_t i=0;i<x.size();++i) { if(i) f<<","; f<<x[i]; } f<<"]";
}
} // namespace

#ifndef LFM3_RUNTIME_ONLY
int main(int argc,char** argv) try {
  if(argc!=5 || std::string(argv[1])!="--experimental")
    throw std::runtime_error("usage: liquid-lfm3 --experimental MODEL_DIRECTORY REQUEST.json OUTPUT.json");
  auto request=read_json(argv[3]);
  int capacity=request.has("capacity")?request.at("capacity").integer():4096;
  lfm::text::scratch_rows(capacity);
  const bool long_context=request.has("experimental_long_context") && request.at("experimental_long_context").boolean();
  if(capacity>4096 && !long_context)
    throw std::runtime_error("capacity must be 1..4096 unless experimental_long_context is enabled (unaccepted long-context bring-up)");
  auto cfg=lfm::ModelConfig::load(std::string(argv[2])+"/config.json",capacity);
  if(!cfg.is_lfm3() || !cfg.sandwich_norm || !cfg.tied || cfg.shape.conv_width!=7 || cfg.shape.head_dim!=128 || cfg.shape.hidden!=2048)
    throw std::runtime_error("runner requires tied, sandwich-normalized width-7 LFM3 with hidden=2048, head_dim=128");
  // "messages" (native chat template + tokenizer) replaces input_ids; see lfm/lfm3_text_request.hpp.
  const bool chat=request.has("messages");
  const bool want_text=lfm::lfm3_text::wants_text(request);
  if(chat && (request.has("input_ids") || request.has("teacher_ids")))
    throw std::runtime_error("messages cannot be combined with input_ids or teacher_ids");
  if(chat && !request.has("max_new_tokens")) throw std::runtime_error("messages require max_new_tokens");
  auto input=ids(request,"input_ids",cfg.shape.vocabulary);
  auto forced=ids(request,"teacher_ids",cfg.shape.vocabulary);
  auto suppress=ids(request,"suppress_tokens",cfg.shape.vocabulary);
  auto stop=ids(request,"stop_tokens",cfg.shape.vocabulary);
  stop.insert(stop.end(),cfg.eos.begin(),cfg.eos.end());
  if(cfg.architecture==lfm::ModelConfig::Architecture::Lfm3Omni) {
    auto omni=read_json(std::string(argv[2])+"/config.json");
    auto special=[&](const std::string& name) {
      int token=omni.at(name).integer();
      if(token>=cfg.shape.vocabulary) throw std::runtime_error("invalid Omni modality token: "+name);
      return token;
    };
    for(auto name:{"audio_token_id","audio_out_token_id","image_token_id","video_token_id","mixed_start_token_id","audio_start_token_id"})
      suppress.push_back(special(name));
    stop.push_back(special("text_end_token_id"));
  }
  int count=request.has("max_new_tokens")?request.at("max_new_tokens").integer():0;
  int warmups=request.has("warmup_runs")?request.at("warmup_runs").integer():0;
  if(warmups>10) throw std::runtime_error("at most 10 warmup runs");
  const bool check_reset=request.has("check_reset") && request.at("check_reset").boolean();
  const bool trace_logits_only=request.has("trace_logits_only") && request.at("trace_logits_only").boolean();
  auto check_geometry=[&] {
    if(input.empty() || input.size()>size_t(capacity) || count>capacity || forced.size()>size_t(capacity) ||
       input.size()+std::max(size_t(count?count-1:0),forced.size())>size_t(capacity) || (count && !forced.empty()))
      throw std::runtime_error("invalid or overflowing prefill/decode request");
    // New serving contract counts the final emitted prediction too. Legacy
    // <=4096 fixture accounting remains unchanged unless explicitly opted in.
    if(long_context) lfm::text::require_budget(input.size(),size_t(count),capacity);
  };
  if(!chat) check_geometry();  // a messages request has no ids until the tokenizer runs (below)
  // Sampling and the other scalar request checks complete before anything is read from the
  // checkpoint beyond config.json (check_lfm3_sampling.py enforces this ordering). The two checks that
  // need the tokenizer's count (stop ids, whole-vocabulary suppression) necessarily run after it.
  lfm::TextSamplingOptions sampling;
  if(request.has("do_sample")) sampling.do_sample=request.at("do_sample").boolean();
  if(request.has("temperature")) sampling.temperature=request.at("temperature").number();
  if(request.has("top_k")) sampling.top_k=request.at("top_k").integer();
  if(request.has("top_p")) sampling.top_p=request.at("top_p").number();
  if(request.has("min_p")) sampling.min_p=request.at("min_p").number();
  if(request.has("repetition_penalty")) sampling.repetition_penalty=request.at("repetition_penalty").number();
  if(request.has("seed")) sampling.seed=request.at("seed").integer();
  if(sampling.top_k>cfg.shape.vocabulary) throw std::runtime_error("top_k exceeds vocabulary size");
  { lfm::TextSampler validate(sampling); } // Validates every scalar before checkpoint access.
  // Token choice is limited to ids the tokenizer defines (verified contiguous 0..live-1); output
  // rows at or above `live` decode to nothing, so they are excluded as a prefix bound. Explicit
  // suppression ids in that range are already covered; a stop id there could never be produced.
  const int live=lfm::tokenizer_token_count(std::string(argv[2])+"/tokenizer.json",cfg.shape.vocabulary);
  for(int id:stop) if(id>=live) throw std::runtime_error("stop token outside the tokenizer vocabulary");
  std::unique_ptr<lfm::lfm3_text::Tokenizer> text_frontend;
  if(want_text) text_frontend=lfm::lfm3_text::load_checked(std::string(argv[2])+"/tokenizer.json",live);
  if(chat) {
    input=lfm::lfm3_text::message_input_ids(*text_frontend,request,capacity,count,suppress);
    check_geometry();
  }
  suppress.erase(std::remove_if(suppress.begin(),suppress.end(),[&](int id){return id>=live;}),suppress.end());
  std::vector<unsigned char> blocked(live,0);
  for(int id:suppress) blocked[id]=1;
  if(std::set<int>(suppress.begin(),suppress.end()).size()==size_t(live))
    throw std::runtime_error("cannot suppress whole vocabulary");
  // top_k is validated against the output layer (unchanged contract); k above the token count
  // meant "every candidate", which is now every token the tokenizer defines.
  if(sampling.top_k>live) sampling.top_k=live;
  lfm::TextSampler sampler(sampling);
  auto greedy=[&](const std::vector<float>& logits) {
    int best=-1;  // ascending ids with strict '>' keep the lowest-id tie, like torch.argmax
    for(int token=0;token<live;++token)
      if(!blocked[token] && (best<0 || logits[token]>logits[best])) best=token;
    if(best<0) throw std::runtime_error("cannot suppress whole vocabulary");
    return best;
  };
  auto sample=[&](lfm::TextSampler& picker,const std::vector<float>& logits,const std::vector<int>& history) {
    if(!sampling.do_sample && sampling.repetition_penalty==1.0) return greedy(logits);
    return picker.sample(logits.data(),size_t(live),history,suppress);
  };
  Runner runner(argv[2],cfg);
  runner.trace_logits_only=trace_logits_only;
  if(request.has("trace_directory")) {
    runner.trace=request.at("trace_directory").string();
    if(!runner.trace.empty()) std::filesystem::create_directories(runner.trace);
  }
  auto now=[] { return std::chrono::steady_clock::now(); };
  auto elapsed=[](auto a,auto b) { return std::chrono::duration<double>(b-a).count(); };
  const std::string trace_directory=runner.trace;
  runner.trace.clear();
  for(int run=0;run<warmups;++run) {
    lfm::TextSampler warm_sampler(sampling);
    std::vector<int> warm_history;
    auto warm_scores=runner.forward(input);
    for(int token:forced) warm_scores=runner.forward({token});
    // Warm precisely the same greedy workload as the timed run, not one decode.
    if(forced.empty()) for(int t=0;t<count;++t) {
      int token=sample(warm_sampler,warm_scores,warm_history);
      warm_history.push_back(token);
      if(std::find(stop.begin(),stop.end(),token)!=stop.end() || t==count-1) break;
      warm_scores=runner.forward({token});
    }
    runner.reset();
  }
  runner.trace=trace_directory;
  auto begin=now(); auto scores=runner.forward(input); auto prefill_end=now();
  std::vector<std::vector<float>> checkpoints;
  if(check_reset) checkpoints.push_back(scores);
  std::vector<int> generated;
  std::vector<int> argmax;
  auto pick=[&] {
    return greedy(scores); // Teacher diagnostic never consumes stochastic RNG.
  };
  argmax.push_back(pick());
  for(int token:forced) {
    scores=runner.forward({token});
    if(check_reset) checkpoints.push_back(scores);
    argmax.push_back(pick());
  }
  for(int t=0;t<count;++t) {
    int token=sample(sampler,scores,generated); generated.push_back(token);
    if(std::find(stop.begin(),stop.end(),token)!=stop.end() || t==count-1) break;
    scores=runner.forward({token});
    if(check_reset) checkpoints.push_back(scores);
  }
  auto end=now();
  // Optional same-process reset regression: exactly repeat prefill and teacher forcing.
  bool reset_ok=true;
  if(check_reset) {
    runner.trace.clear(); runner.reset(); scores=runner.forward(input);
    reset_ok=scores==checkpoints[0];
    std::vector<int> replay=forced;
    if(!generated.empty()) replay.assign(generated.begin(),generated.end()-1);
    for(size_t i=0;i<replay.size();++i) {
      scores=runner.forward({replay[i]}); reset_ok &= scores==checkpoints[i+1];
    }
    if(!reset_ok) throw std::runtime_error("reset exact-logit regression");
  }
  std::ofstream f(argv[4]);
#ifdef LFM3_NATIVE_GEMM
  // lfm::gemm, eager attention with FP32 scores, FP32 logits (G4); not the cuBLAS eager-BF16 engine.
  f<<"{\"backend\":\"native-cuda-experimental\",\"attention\":\"eager-fp32-scores\",\"gemm\":\"lfm-gemm\",\"logits\":\"fp32\",\"input_ids\":"; print_ids(f,input);
#else
  f<<"{\"backend\":\"native-cuda-experimental\",\"attention\":\"eager-bf16\",\"input_ids\":"; print_ids(f,input);
#endif
  f<<",\"generated_ids\":"; print_ids(f,generated); f<<",\"teacher_argmax\":"; print_ids(f,argmax);
  f<<",\"prefill_seconds\":"<<elapsed(begin,prefill_end)<<",\"decode_seconds\":"<<elapsed(prefill_end,end)
   <<",\"decode_forward_calls\":"<<(forced.empty()?std::max(0,int(generated.size())-1):int(forced.size()))
   <<",\"reset_checked\":"<<(check_reset?"true":"false")
   <<",\"reset_exact_logits_ok\":"<<(check_reset?(reset_ok?"true":"false"):"null")
   <<",\"warmup_runs\":"<<warmups<<",\"tracing\":"<<(trace_directory.empty()?"false":"true")
   <<",\"do_sample\":"<<(sampling.do_sample?"true":"false")<<",\"seed\":"<<sampling.seed
   <<",\"live_token_count\":"<<live;
  if(want_text) f<<",\"text\":"<<lfm::lfm3_text::json_string(lfm::lfm3_text::generated_text(*text_frontend,generated,stop))
                 <<",\"text_frontend\":\"native\"";
  f<<"}\n";
  if(!f) throw std::runtime_error("output write failed");
  return 0;
} catch(const std::exception& e) { std::cerr<<"lfm3: "<<e.what()<<"\n"; return 1; }
#endif
