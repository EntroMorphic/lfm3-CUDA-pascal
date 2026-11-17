// Single-threaded resident protocol prototype; explicit two-lane decode groups.
// Reuse the identical runtime in one translation unit; no second kernel fork.
#define LFM3_RUNTIME_ONLY
#include "lfm3_main.cu"
#include "lfm3_decode_batch.cuh"
#include <lfm/text_sessions.hpp>
#include <lfm/text_cancellation.hpp>
#include <lfm/text_batch_cancellation.hpp>
#include <lfm/lfm3_text_request.hpp>
#ifdef LFM3_NATIVE_GEMM  // speech (#2) was validated only with the native GEMM; other builds refuse op "speech"
#include "lfm3_batch.cuh"
#include "lfm3_speech.cuh"
#include "lfm/image_handoff.hpp"
#endif
#include <functional>
#include <map>
#include <iomanip>
#include <sstream>

namespace {
std::string quote(const std::string& value) {
  std::ostringstream out;out<<'"';
  for(unsigned char c:value) {
    if(c=='"' || c=='\\') out<<'\\'<<char(c);
    else if(c>=32 && c<127) out<<char(c);
    else {const char* hex="0123456789abcdef";out<<"\\u00"<<hex[c>>4]<<hex[c&15];}
  }
  out<<'"';return out.str();
}
std::string session_id(const lfm::Json& req) {
  auto id=req.at("session_id").string();
  if(id.empty() || id.size()>64) throw std::runtime_error("invalid session ID");
  for(char c:id) if(!((c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_'||c=='-'))
    throw std::runtime_error("invalid session ID");
  return id;
}
// A text generation between ops text_begin and its last text_step (validation/text-wait-v1): everything op
// "generate"'s decode loop keeps on its stack, so the loop can stop after any token and resume with the same state.
struct PendingText {
  std::vector<int> generated;
  int count=0,t=0,origin_rpc=0;
  bool want_text=false,return_logits=false,trace_sampling=false;
  size_t reused=0;
  lfm::TextSamplingOptions options;
  std::unique_ptr<lfm::TextSampler> sampler;
  std::unique_ptr<lfm::text::SessionLedger::Lease> lease;
  std::string sampling_trace;
};
struct Session {
  int capacity;
  std::unique_ptr<Runner> runner;
  std::vector<int> consumed;
  std::vector<float> last_scores;
  std::unique_ptr<PendingText> pending;  // a sliced text generation in progress: every other op is refused
};
// Native text frontend, loaded on the first request that needs it (messages / return_text) and checked
// against the verified token count.
struct TextFrontend {
  std::string path;int live=0;
  std::unique_ptr<lfm::lfm3_text::Tokenizer> tokenizer;
  const lfm::lfm3_text::Tokenizer& get() {
    if(!tokenizer) tokenizer=lfm::lfm3_text::load_checked(path,live);
    return *tokenizer;
  }
};
// Prompt ids of a generate request: input_ids as before, or natively encoded messages.
std::vector<int> request_input(const lfm::Json& req,int vocab,TextFrontend& text,int capacity,
                               const std::vector<int>& suppressed) {
  if(!req.has("messages")) return ids(req,"input_ids",vocab);
  const int count=req.at("max_new_tokens").integer();
  return lfm::lfm3_text::message_input_ids(text.get(),req,capacity,count,suppressed);
}
struct GenerationCancelled : std::exception {
  const char* what() const noexcept override {return "generation cancelled at a token boundary";}
};
// Ids of `ids` that token choice can see (those below the tokenizer's count); the full list stays in use
// for input validation, so a modality id beyond the count is still rejected as input.
std::vector<int> below(const std::vector<int>& ids,int live) {
  std::vector<int> kept;
  for(int id:ids) if(id<live) kept.push_back(id);
  return kept;
}
// top_k is validated against the output layer (unchanged contract), then k above the tokenizer's
// count, which meant "every candidate", becomes every token the tokenizer defines.
lfm::TextSamplingOptions sampling_options(const lfm::Json& request,int vocab,int live) {
  lfm::TextSamplingOptions o;
  if(request.has("do_sample")) o.do_sample=request.at("do_sample").boolean();
  if(request.has("temperature")) o.temperature=request.at("temperature").number();
  if(request.has("top_k")) o.top_k=request.at("top_k").integer();
  if(request.has("top_p")) o.top_p=request.at("top_p").number();
  if(request.has("min_p")) o.min_p=request.at("min_p").number();
  if(request.has("repetition_penalty")) o.repetition_penalty=request.at("repetition_penalty").number();
  if(request.has("seed")) o.seed=request.at("seed").integer();
  if(o.top_k>vocab) throw std::runtime_error("top_k exceeds vocabulary size");
  lfm::TextSampler validate(o);
  if(o.top_k>live) o.top_k=live;
  return o;
}
void schema(const lfm::Json& request,const std::string& op) {
  const std::set<std::string> common={"op","session_id","rpc_id"};
  const std::set<std::string> generate={"input_ids","max_new_tokens","do_sample","temperature","top_k","top_p","min_p","repetition_penalty","seed","return_logits","trace_sampling","messages","return_text"};
  const auto* object=std::get_if<lfm::Json::Object>(&request.value);
  if(!object) throw std::runtime_error("expected object");
  for(const auto& [key,value]:*object) {
    if(common.count(key)) continue;
    if(op=="create" && key=="capacity") continue;
    if((op=="generate" || op=="text_begin") && generate.count(key)) continue;
    if(op=="text_step" && (key=="tokens" || key=="cancel")) continue;
    if(op=="generate_batch" && (key=="op" || key=="rpc_id" || key=="lanes")) continue;
    if((op=="speech" || op=="speech_admit") && (key=="chunks" || key=="mode" || key=="max_new_tokens" || key=="seed" ||
                                               key=="chunked" || key=="audio" || key=="images" || key=="stream")) continue;
    if(op=="speech_admit" && key=="ticket") continue;
    if(op=="speech_step" && key=="cancel") continue;
    throw std::runtime_error("unsupported request field: "+key);
  }
}
// Metadata is collected for BOTH lanes before any lease, weight load or reset.
struct BatchLane {
  std::string id;
  Session* session=nullptr;
  std::vector<int> input,generated;
  int count=0,paired=0;
  bool return_logits=false,trace=false,done=false,cancelled=false,want_text=false;
  size_t reused=0;
  lfm::TextSamplingOptions options;
  std::unique_ptr<lfm::TextSampler> sampler;
  std::unique_ptr<lfm::text::SessionLedger::Lease> lease;
};
void generate_batch(const lfm::Json& request,int rpc_id,const std::string& directory,
                    const lfm::ModelConfig& cfg,const std::vector<int>& suppressed,int live,
                    const std::vector<int>& stops,const std::string& trace_root,
                    std::shared_ptr<lfm::Safetensors>& weights,
                    std::map<std::string,Session>& sessions,lfm::text::SessionLedger& ledger,
                    lfm::text::CancellationReader& cancellation,TextFrontend& text,std::ostream& out) {
  const auto choice_suppressed=below(suppressed,live);
  // Outer schema is intentionally narrower than the single-operation common set.
  for(const auto& [key,value]:std::get<lfm::Json::Object>(request.value))
    if(key!="op" && key!="rpc_id" && key!="lanes") throw std::runtime_error("unsupported batch field: "+key);
  const auto& requests=request.at("lanes").array();
  if(requests.size()!=2) throw std::runtime_error("generate_batch requires exactly two lanes");
  std::array<BatchLane,2> lanes;
  for(size_t i=0;i<2;++i) {
    const auto& req=requests[i];auto& lane=lanes[i];
    schema(req,"generate");
    if(req.has("op") || req.has("rpc_id")) throw std::runtime_error("lane cannot carry op or rpc_id");
    lane.id=session_id(req);
    auto found=sessions.find(lane.id);
    if(found==sessions.end()) throw std::runtime_error("unknown session");
    lane.session=&found->second;
    if(lane.session->pending) throw std::runtime_error("session has a text generation in progress");
    lane.input=request_input(req,cfg.shape.vocabulary,text,lane.session->capacity,suppressed);
    lane.want_text=lfm::lfm3_text::wants_text(req);
    for(int token:lane.input) if(std::find(suppressed.begin(),suppressed.end(),token)!=suppressed.end())
      throw std::runtime_error("modality placeholder input is unsupported");
    lane.count=req.at("max_new_tokens").integer();
    if(lane.count<1) throw std::runtime_error("max_new_tokens must be positive");
    lfm::text::require_budget(lane.input.size(),lane.count,lane.session->capacity);
    lane.return_logits=req.has("return_logits") && req.at("return_logits").boolean();
    lane.trace=req.has("trace_sampling") && req.at("trace_sampling").boolean();
    if(lane.trace && (trace_root.empty() || lane.count>32))
      throw std::runtime_error("sampling trace needs startup audit root and at most32 output tokens");
    lane.options=sampling_options(req,cfg.shape.vocabulary,live);
    lane.sampler=std::make_unique<lfm::TextSampler>(lane.options);
    if(ledger.inspect(lane.id).inflight) throw std::runtime_error("session already inflight");
  }
  if(lanes[0].id==lanes[1].id) throw std::runtime_error("batch sessions must be distinct");
  const auto trace_key=std::to_string(rpc_id);
  const auto trace_directory=trace_root+"/"+trace_key;
  auto discard=[](BatchLane& lane) {
    lane.session->runner.reset();lane.session->consumed.clear();lane.session->last_scores.clear();
    if(lane.lease) lane.lease->abort();
  };
  auto poll=[&] {
    const auto cancelled=cancellation.requested_pair(rpc_id);
    for(size_t i=0;i<2;++i) if(!lanes[i].done && cancelled[i]) {
      discard(lanes[i]);lanes[i].cancelled=true;lanes[i].done=true;
    }
  };
  int paired=0;
  try {
    for(auto& lane:lanes) lane.lease=std::make_unique<lfm::text::SessionLedger::Lease>(
      ledger.begin(lane.id,lfm::text::SessionLedger::Tokens(lane.input.begin(),lane.input.end()),lane.count));
    poll();
    if((lanes[0].trace || lanes[1].trace) && !std::filesystem::create_directory(trace_directory))
      throw std::runtime_error("sampling trace request already exists");
    // Prefills and reuse continuations remain serial, never decode-batched.
    for(size_t i=0;i<2;++i) {
      auto& lane=lanes[i];if(lane.done) continue;
      auto& session=*lane.session;
      if(!weights) weights=std::make_shared<lfm::Safetensors>(directory+"/model.safetensors");
      if(!session.runner) {
        auto shape=cfg;shape.shape.context=session.capacity;
        session.runner=std::make_unique<Runner>(directory,shape,weights);
      }
      const bool reuse=!lane.lease->plan().full_prefill;
      lane.reused=lane.lease->plan().reuse_tokens;
      if(reuse && (session.consumed.size()!=lane.reused || session.runner->cursor!=int(lane.reused)))
        throw std::runtime_error("session physical cache/ledger disagreement");
      if(!reuse) {session.runner->reset();session.consumed.clear();session.last_scores.clear();}
      poll(); // initialization/reset may have been long: check BEFORE prefill.
      if(lane.done) continue;
      if(!reuse) {
        session.last_scores=session.runner->forward(lane.input);session.consumed=lane.input;
      } else for(size_t j=lane.reused;j<lane.input.size();++j) {
        poll();if(lane.done) break;
        session.last_scores=session.runner->forward({lane.input[j]});session.consumed.push_back(lane.input[j]);
      }
      poll(); // also latch cancellation for the peer while this lane prefills.
      if(lane.trace && !std::filesystem::create_directory(trace_directory+"/lane"+std::to_string(i)))
        throw std::runtime_error("sampling trace lane already exists");
    }
    while(!lanes[0].done || !lanes[1].done) {
      poll();
      std::vector<size_t> ready;
      std::vector<Runner*> owners;std::vector<int> tokens;
      for(size_t i=0;i<2;++i) {
        auto& lane=lanes[i];if(lane.done) continue;
        auto& session=*lane.session;
        if(lane.trace) {
#ifdef LFM3_NATIVE_GEMM
          write_logits_trace(trace_directory+"/lane"+std::to_string(i)+"/"+std::to_string(lane.generated.size()),session.last_scores);
#else
          std::vector<bf16> raw(session.last_scores.size());
          for(size_t j=0;j<raw.size();++j) raw[j]=__float2bfloat16_rn(session.last_scores[j]);
          std::ofstream file(trace_directory+"/lane"+std::to_string(i)+"/"+std::to_string(lane.generated.size())+".logits.bf16",std::ios::binary);
          file.write(reinterpret_cast<const char*>(raw.data()),raw.size()*sizeof(bf16));
          if(!file) throw std::runtime_error("sampling trace write failed");
#endif
        }
        const int token=lane.sampler->sample(session.last_scores.data(),size_t(live),lane.generated,choice_suppressed);
        lane.generated.push_back(token);
        if(std::find(stops.begin(),stops.end(),token)!=stops.end() || lane.generated.size()==size_t(lane.count)) {
          poll();if(lane.done) continue;
          lane.lease->commit(lfm::text::SessionLedger::Tokens(lane.generated.begin(),lane.generated.end()),
                             lfm::text::SessionLedger::Tokens(session.consumed.begin(),session.consumed.end()));
          lane.done=true; // completed lanes are never touched by poll/discard again.
        } else ready.push_back(i);
      }
      poll();
      for(auto i:ready) if(!lanes[i].done) {
        owners.push_back(lanes[i].session->runner.get());tokens.push_back(lanes[i].generated.back());
      }
      if(owners.empty()) continue;
      std::vector<std::vector<float>> scores;
      if(owners.size()==2) {scores=lfm3_decode_batch::forward(owners,tokens);++paired;}
      else scores.push_back(owners[0]->forward({tokens[0]}));
      size_t row=0;
      for(auto i:ready) if(!lanes[i].done) {
        auto& lane=lanes[i];lane.session->last_scores=std::move(scores[row++]);
        lane.session->consumed.push_back(lane.generated.back());
        if(owners.size()==2) ++lane.paired;
      }
    }
  } catch(...) {
    // No CUDA fault isolation promise: fail closed for ALL still-inflight lanes.
    for(auto& lane:lanes) if(!lane.done && lane.lease) discard(lane);
    throw;
  }
  out<<"{\"rpc_id\":"<<rpc_id<<",\"ok\":true,\"gpu_batching\":"<<(paired?"true":"false")
           <<",\"paired_decode_calls\":"<<paired<<",\"results\":[";
  for(size_t i=0;i<2;++i) {
    const auto& lane=lanes[i];const auto& session=*lane.session;
    if(i) out<<',';
    out<<"{\"rpc_id\":"<<rpc_id<<",\"ok\":true,\"session_id\":"<<quote(lane.id)<<",\"generated_ids\":";
    print_ids(out,lane.generated);
    out<<",\"reused_tokens\":"<<(lane.cancelled?0:lane.reused)<<",\"consumed_tokens\":"<<session.consumed.size()
             <<",\"do_sample\":"<<(lane.options.do_sample?"true":"false")<<",\"seed\":"<<lane.options.seed
             <<",\"cancelled\":"<<(lane.cancelled?"true":"false")<<",\"cache_valid\":"<<(lane.cancelled?"false":"true")
             <<",\"gpu_batching\":"<<(lane.paired?"true":"false")<<",\"paired_decode_calls\":"<<lane.paired
             <<",\"live_token_count\":"<<live;  // the token-choice bound this lane used (G6)
    if(lane.want_text) out<<",\"text\":"<<lfm::lfm3_text::json_string(lfm::lfm3_text::generated_text(text.get(),lane.generated,stops));
    if(lane.trace) out<<",\"sampling_trace\":"<<quote(trace_key+"/lane"+std::to_string(i));
    if(lane.return_logits && !lane.cancelled) {
      out<<",\"last_logits\":[";
      for(size_t j=0;j<session.last_scores.size();++j) {if(j) out<<',';out<<session.last_scores[j];}
      out<<']';
    }
    out<<'}';
  }
  out<<"]}\n"<<std::flush;
}
} // namespace

namespace {
// The resident protocol, one request line at a time. liquid-lfm3-worker drives it from stdin/stdout;
// liquid-lfm3-server drives it in-process from its single engine-owner thread. Only one thread may call
// handle(); cancellation arrives on the same nonblocking control pipe in both cases.
class WorkerCore {
 public:
  WorkerCore(const std::string& model_directory,const std::string& trace_root,bool trace_root_requested,int cancellation_fd)
    : directory(model_directory),sampling_trace_root(trace_root),cancellation(cancellation_fd),
      cancellation_enabled(cancellation_fd>=0) {
    // validation/image-cache-v1 C3: what the engine's image cache holds, published for the in-process front end
    image_cache_.set_observer([](const std::string& key,bool kept) {lfm::image::shared_held_images().kept(key,kept);});
    if(trace_root_requested && (sampling_trace_root.empty() || !std::filesystem::create_directory(sampling_trace_root)))
      throw std::runtime_error("sampling trace root must be a new directory");
    const auto model=read_json(directory+"/config.json");
    cfg=lfm::ModelConfig::load(directory+"/config.json",32768);
    if(!cfg.is_lfm3() || !cfg.sandwich_norm || !cfg.tied || cfg.shape.hidden!=2048 || cfg.shape.head_dim!=128 || cfg.shape.conv_width!=7)
      throw std::runtime_error("unsupported worker geometry");
    stops=cfg.eos;
    if(cfg.architecture==lfm::ModelConfig::Architecture::Lfm3Omni) {
      for(auto name:{"audio_token_id","audio_out_token_id","image_token_id","video_token_id","mixed_start_token_id","audio_start_token_id"})
        suppressed.push_back(model.at(name).integer());
      stops.push_back(model.at("text_end_token_id").integer());
    }
    for(int id:suppressed) if(id>=cfg.shape.vocabulary) throw std::runtime_error("invalid modality token");
    for(int id:stops) if(id>=cfg.shape.vocabulary) throw std::runtime_error("invalid stop token");
    // Token choice is limited to ids the tokenizer defines (verified contiguous 0..live-1): output rows
    // at or above `live` decode to nothing. `suppressed` stays whole for input validation; token choice uses
    // `choice_suppressed`, its ids below `live` (ids at or above are excluded by the bound itself).
    live=lfm::tokenizer_token_count(directory+"/tokenizer.json",cfg.shape.vocabulary);
    choice_suppressed=below(suppressed,live);
    for(int id:stops) if(id>=live) throw std::runtime_error("stop token outside the tokenizer vocabulary");
    text.path=directory+"/tokenizer.json";text.live=live;
#ifdef LFM3_NATIVE_GEMM
    // Speech needs an Omni checkpoint with its codec, whose special ids are the ones lfm/audio_modes.hpp fixes.
    if(cfg.architecture==lfm::ModelConfig::Architecture::Lfm3Omni &&
       std::filesystem::exists(directory+"/audio_tokenizer/model.safetensors")) {
      const lfm::audio::ModeTokens t{};
      std::vector<int> placeholders;
      for(auto name:{"audio_token_id","audio_out_token_id","image_token_id","video_token_id"}) placeholders.push_back(model.at(name).integer());
      std::vector<int> fixed=t.placeholders;
      std::sort(placeholders.begin(),placeholders.end());std::sort(fixed.begin(),fixed.end());
      speech_ok=model.at("audio_start_token_id").integer()==t.audio_start && model.at("audio_out_token_id").integer()==t.audio_out &&
                model.at("mixed_start_token_id").integer()==t.mixed_start && model.at("text_end_token_id").integer()==t.text_end &&
                placeholders==fixed && std::find(cfg.eos.begin(),cfg.eos.end(),t.eos[0])!=cfg.eos.end() && live>t.audio_out;
      // Audio input (#3) also needs the encoder this build implements: its contract checked against the config and
      // the safetensors header, so a checkpoint it cannot run is refused up front (an engine refusal would fail the
      // server closed).
      if(speech_ok && model.at("audio_token_id").integer()==lfm::audio::ModeTokens{}.placeholders[0]) {
        try {
          const auto table=lfm::safetensors_header::read(directory+"/model.safetensors");
          lfm3_audio_in::Encoder::check_contract(model,[&](const std::string& name) -> std::optional<lfm::safetensors_header::Entry> {
            const auto it=table.find(name);
            if(it==table.end()) return std::nullopt;
            return it->second;
          });
          audio_in_ok=true;
        } catch(const std::exception&) {audio_in_ok=false;}
      }
      // Image input (#4): likewise the vision encoder's contract, from the header alone.
      if(speech_ok && model.at("image_token_id").integer()==lfm3_speech::kImageToken) {
        try {
          const auto table=lfm::safetensors_header::read(directory+"/model.safetensors");
          lfm3_vision::Encoder::check_contract(model,[&](const std::string& name) -> std::optional<lfm::safetensors_header::Entry> {
            const auto it=table.find(name);
            if(it==table.end()) return std::nullopt;
            return it->second;
          });
          image_in_ok=true;
        } catch(const std::exception&) {image_in_ok=false;}
      }
    }
#endif
  }
  void ready(std::ostream& out) const {
    out<<"{\"ready\":true,\"protocol\":2,\"max_sessions\":2,\"aggregate_capacity\":32768,\"gpu_batching\":false,\"decode_batching\":true,\"cancellation\":"
       <<(cancellation_enabled?"true":"false")<<"}\n"<<std::flush;
  }
#ifdef LFM3_WORKER_CORE_ONLY
  // For liquid-lfm3-server's HTTP layer: the request policy and the native tokenizer, loaded once and
  // checked against the verified token count (G6).
  int vocabulary() const {return cfg.shape.vocabulary;}
  const std::vector<int>& suppressed_tokens() const {return suppressed;}
  const std::vector<int>& stop_tokens() const {return stops;}
  const lfm::lfm3_text::Tokenizer& tokenizer() {return text.get();}
  // Whether op "speech" can run (native-GEMM build, an Omni checkpoint with its codec and the expected special ids);
  // the HTTP layer refuses /speech up front otherwise, since any engine refusal fails the server closed.
  bool speech_supported() const {return speech_ok;}
  // Whether op "speech" takes audio input (speech_supported, and the audio encoder's contract holds).
  bool audio_input_supported() const {return audio_in_ok;}
  // Whether op "speech" takes images (speech_supported, and the vision encoder's contract holds).
  bool image_input_supported() const {return image_in_ok;}
  // Concurrent speech: the batch's rows (ops speech_admit/speech_step); 0 when speech is unsupported.
  int speech_rows() const {return speech_ok?speech_rows_:0;}
  void set_speech_rows(int rows) {
    if(rows<1 || rows>16) throw std::runtime_error("speech rows must be 1..16");  // validated up to 16 (speech-batch-v1)
#ifdef LFM3_NATIVE_GEMM
    if(speech_batch) throw std::runtime_error("speech rows are fixed once the batch exists");
#endif
    speech_rows_=rows;
  }
  // validation/image-cache-v1: the engine's image cache budget in MiB (0, the default here: none; the in-process
  // server sets its --image-cache-mb)
  void set_image_cache_mb(int mb) {
    if(mb<0 || mb>65536) throw std::runtime_error("image cache must be 0..65536 MiB");
#ifdef LFM3_NATIVE_GEMM
    image_cache_.set_budget(size_t(mb)<<20);  // (its observer, set at construction, publishes what it holds: C3)
#endif
  }
#endif
  // Streaming (op "speech" with "stream": true): set by the in-process server around a call; receives the call's
  // progress lines, in order, before its reply line. Unset (the standalone worker on pipes), stream requests are refused.
  std::function<void(const std::string&)> progress;
  // One protocol line (`oversized`: the line exceeded 4 MiB). Writes exactly one reply line to `out`;
  // returns false after a shutdown request.
  bool handle(const std::string& line,bool oversized,std::ostream& out) {
    // JSON readers generally parse numbers as double. Emit enough digits to
    // preserve the exact float->double value, not merely a float32 roundtrip.
    out<<std::setprecision(std::numeric_limits<double>::max_digits10);
  int rpc_id=0;
  try {
    if(oversized) throw std::runtime_error("request exceeds 4 MiB");
    const auto request=lfm::Json::parse(line);
    rpc_id=request.at("rpc_id").integer();
    if(rpc_id<1 || last_rpc_id==2147483647 || rpc_id!=last_rpc_id+1)
      throw std::runtime_error("out-of-sequence RPC ID");
    last_rpc_id=rpc_id;
    const auto op=request.at("op").string();schema(request,op);
    if(op=="shutdown") {
      out<<"{\"rpc_id\":"<<rpc_id<<",\"ok\":true,\"shutdown\":true}\n"<<std::flush;return false;
    }
     if(op=="stats") {
       size_t free_bytes=0,total_bytes=0;
       if(weights) {cuda_check(cudaDeviceSynchronize());cuda_check(cudaMemGetInfo(&free_bytes,&total_bytes));}
       out<<"{\"rpc_id\":"<<rpc_id<<",\"ok\":true,\"sessions\":"<<ledger.size()<<",\"reserved_tokens\":"<<ledger.reserved_tokens()
                <<",\"weights_loaded\":"<<(weights?"true":"false");
       // Device-wide unified-memory accounting, NOT per-worker allocations or RSS.
       if(weights) out<<",\"cuda_free_bytes\":"<<free_bytes<<",\"cuda_total_bytes\":"<<total_bytes
                           <<",\"device_allocated_bytes\":"<<(total_bytes-free_bytes);
#ifdef LFM3_NATIVE_GEMM
       out<<",\"image_cache_entries\":"<<image_cache_.size()<<",\"image_cache_bytes\":"<<image_cache_.bytes()
          <<",\"image_cache_budget_bytes\":"<<image_cache_.budget();
       // The worker's own engine buffers (LFM_ALLOCATION_METRICS=1 only): exact, unlike the device-wide numbers.
       if(lfm::allocation_metrics::enabled()) {
         const auto m=lfm::allocation_metrics::snapshot();
         out<<",\"app_live_bytes\":"<<m.live<<",\"app_outstanding_allocations\":"<<m.outstanding
            <<",\"app_metrics_valid\":"<<(m.valid?"true":"false");
       }
#endif
       out<<"}\n"<<std::flush;return true;
    }
     if(op=="generate_batch") {
       generate_batch(request,rpc_id,directory,cfg,suppressed,live,stops,sampling_trace_root,weights,sessions,ledger,cancellation,text,out);
       return true;
     }
     if(op=="speech") {speech(request,rpc_id,out);return true;}
     if(op=="speech_admit") {speech_admit(request,rpc_id,out);return true;}
     if(op=="speech_step") {speech_step(request,rpc_id,out);return true;}
     const auto id=session_id(request);
    if(op=="create") {
      int capacity=request.at("capacity").integer();lfm::text::scratch_rows(capacity);
      ledger.create(id,capacity);
      try {sessions.emplace(id,Session{capacity,{},{},{},{}});}
      catch(...) {ledger.close(id);throw;}
      out<<"{\"rpc_id\":"<<rpc_id<<",\"ok\":true,\"session_id\":"<<quote(id)<<"}\n"<<std::flush;return true;
    }
    auto found=sessions.find(id);
    if(found==sessions.end()) throw std::runtime_error("unknown session");
    auto& session=found->second;
    // a sliced generation owns its session until its last text_step (the Service's lease serializes sessions too)
    if(session.pending && op!="text_step") throw std::runtime_error("session has a text generation in progress");
    if(op=="text_step") {
      if(!session.pending) throw std::runtime_error("no text generation in progress");
      const long long k=request.at("tokens").integer();
      if(k<1) throw std::runtime_error("text_step tokens must be positive");
      text_step(id,session,rpc_id,int(std::min<long long>(k,std::numeric_limits<int>::max())),
                request.has("cancel") && request.at("cancel").boolean(),out);
      return true;
    }
    if(op=="inspect") {
      const auto snapshot=ledger.inspect(id);
      out<<"{\"rpc_id\":"<<rpc_id<<",\"ok\":true,\"capacity\":"<<snapshot.capacity<<",\"consumed_tokens\":"<<session.consumed.size()
               <<",\"committed_consumed_tokens\":"<<snapshot.consumed.size()
               <<",\"transcript_tokens\":"<<snapshot.transcript.size()<<",\"cache_valid\":"<<(snapshot.cache_valid?"true":"false")
               <<",\"inflight\":"<<(snapshot.inflight?"true":"false")<<"}\n"<<std::flush;return true;
    }
    if(op=="close") {
      ledger.close(id);sessions.erase(found);
      out<<"{\"rpc_id\":"<<rpc_id<<",\"ok\":true,\"closed\":true}\n"<<std::flush;return true;
    }
    if(op=="reset") {
      try {if(session.runner) session.runner->reset();}
      catch(...) {session.runner.reset();session.consumed.clear();session.last_scores.clear();ledger.reset(id);throw;}
      session.consumed.clear();session.last_scores.clear();
      ledger.reset(id);
      out<<"{\"rpc_id\":"<<rpc_id<<",\"ok\":true,\"reset\":true}\n"<<std::flush;return true;
    }
    if(op!="generate" && op!="text_begin") throw std::runtime_error("unknown operation");
    // op generate is text_begin and one unbounded text_step in one call: the sliced path and the whole path are one
    // code path, so a generation sliced among speech steps computes exactly what it computes alone
    if(!text_begin(request,id,session,rpc_id,out)) return true;  // cancelled during prefill: its reply is written
    if(op=="text_begin") {
      out<<"{\"rpc_id\":"<<rpc_id<<",\"ok\":true,\"session_id\":"<<quote(id)<<",\"begun\":true}\n"<<std::flush;
      return true;
    }
    text_step(id,session,rpc_id,std::numeric_limits<int>::max(),false,out);
  } catch(const std::exception& e) {
    out<<"{\"rpc_id\":"<<rpc_id<<",\"ok\":false,\"error\":"<<quote(e.what())<<"}\n"<<std::flush;
  }
    return true;
  }
 private:
  // op text_begin (and the start of op generate): everything before the decode loop, unchanged from op generate:
  // the request's checks (all before any mutation), the sampler, the ledger lease and the prefill or reuse
  // continuation. Leaves the loop's state in session.pending and returns true; or, cancelled during prefill, writes
  // op generate's cancelled reply and returns false.
  bool text_begin(const lfm::Json& request,const std::string& id,Session& session,int rpc_id,std::ostream& out) {
    auto p=std::make_unique<PendingText>();
    const auto input=request_input(request,cfg.shape.vocabulary,text,session.capacity,suppressed);
    p->want_text=lfm::lfm3_text::wants_text(request);
    for(int token:input) if(std::find(suppressed.begin(),suppressed.end(),token)!=suppressed.end())
      throw std::runtime_error("modality placeholder input is unsupported");
    p->count=request.at("max_new_tokens").integer();
    if(p->count<1) throw std::runtime_error("max_new_tokens must be positive");
    lfm::text::require_budget(input.size(),p->count,session.capacity);
    p->return_logits=request.has("return_logits") && request.at("return_logits").boolean();
    p->trace_sampling=request.has("trace_sampling") && request.at("trace_sampling").boolean();
    if(p->trace_sampling && (sampling_trace_root.empty() || p->count>32))
      throw std::runtime_error("sampling trace needs startup audit root and at most32 output tokens");
    p->options=sampling_options(request,cfg.shape.vocabulary,live);
    p->sampler=std::make_unique<lfm::TextSampler>(p->options);
    p->origin_rpc=rpc_id;
    p->lease=std::make_unique<lfm::text::SessionLedger::Lease>(
        ledger.begin(id,lfm::text::SessionLedger::Tokens(input.begin(),input.end()),p->count));
    auto check_cancellation=[&] {
      if(cancellation.requested(rpc_id)) throw GenerationCancelled{};
    };
    // Every metadata rejection above is pre-mutation. Execution errors below
    // invalidate the entire sequence cache; no partially written cache survives.
    try {
      check_cancellation();
      if(!weights) weights=std::make_shared<lfm::Safetensors>(directory+"/model.safetensors");
      if(!session.runner) {
        auto shape=cfg;shape.shape.context=session.capacity;
        session.runner=std::make_unique<Runner>(directory,shape,weights);
      }
      auto& runner=*session.runner;
      const bool reuse=!p->lease->plan().full_prefill;
      p->reused=p->lease->plan().reuse_tokens;
      if(reuse && (session.consumed.size()!=p->reused || runner.cursor!=int(p->reused)))
        throw std::runtime_error("session physical cache/ledger disagreement");
      if(!reuse) {runner.reset();session.consumed.clear();session.last_scores.clear();}
      if(!reuse) {
        session.last_scores=lfm::text::at_cancellation_boundary(check_cancellation,[&] {return runner.forward(input);});
        session.consumed=input;
      } else for(size_t i=p->reused;i<input.size();++i) {
        check_cancellation();
        session.last_scores=runner.forward({input[i]});session.consumed.push_back(input[i]);
      }
      p->sampling_trace=sampling_trace_root+"/"+std::to_string(rpc_id);
      if(p->trace_sampling && !std::filesystem::create_directory(p->sampling_trace))
        throw std::runtime_error("sampling trace request already exists");
    } catch(const GenerationCancelled&) {
      session.runner.reset();session.consumed.clear();session.last_scores.clear();p->lease->abort();
      write_cancelled(out,rpc_id,id,*p);
      return false;
    } catch(...) {
      session.runner.reset();session.consumed.clear();session.last_scores.clear();throw;
    }
    session.pending=std::move(p);
    return true;
  }
  // op text_step (and the rest of op generate): up to k iterations of op generate's decode loop, unchanged (check
  // cancellation, trace, sample, then one forward unless the token ends the generation). Unfinished: {"busy":true}.
  // Finished: op generate's reply. `cancel` (the Service's cancellation between slices) or the pipe's cancellation
  // of this call ends it as op generate's cancellation does: the cache reset and the cancelled reply.
  void text_step(const std::string& id,Session& session,int rpc_id,int k,bool cancel,std::ostream& out) {
    auto& p=*session.pending;
    auto check_cancellation=[&] {
      // the pipe first: its frame for this call is consumed even when the Service's flag already cancels
      if(cancellation.requested(rpc_id) || cancel) throw GenerationCancelled{};
    };
    try {
      auto& runner=*session.runner;
      bool finished=false;
      for(int n=0;n<k;++n) {
        check_cancellation();
        if(p.trace_sampling) {
#ifdef LFM3_NATIVE_GEMM
          write_logits_trace(p.sampling_trace+"/"+std::to_string(p.t),session.last_scores);
#else
          std::vector<bf16> trace_scores(session.last_scores.size());
          for(size_t i=0;i<trace_scores.size();++i) trace_scores[i]=__float2bfloat16_rn(session.last_scores[i]);
          std::ofstream trace_file(p.sampling_trace+"/"+std::to_string(p.t)+".logits.bf16",std::ios::binary);
          trace_file.write(reinterpret_cast<const char*>(trace_scores.data()),trace_scores.size()*sizeof(bf16));
          if(!trace_file) throw std::runtime_error("sampling trace write failed");
#endif
        }
        int token=p.sampler->sample(session.last_scores.data(),size_t(live),p.generated,choice_suppressed);
        p.generated.push_back(token);
        if(std::find(stops.begin(),stops.end(),token)!=stops.end() || p.t==p.count-1) {finished=true;break;}
        session.last_scores=runner.forward({token});session.consumed.push_back(token);
        ++p.t;
      }
      if(!finished) {
        out<<"{\"rpc_id\":"<<rpc_id<<",\"ok\":true,\"session_id\":"<<quote(id)<<",\"busy\":true}\n"<<std::flush;
        return;
      }
      check_cancellation();
      p.lease->commit(lfm::text::SessionLedger::Tokens(p.generated.begin(),p.generated.end()),
                      lfm::text::SessionLedger::Tokens(session.consumed.begin(),session.consumed.end()));
      out<<"{\"rpc_id\":"<<rpc_id<<",\"ok\":true,\"session_id\":"<<quote(id)<<",\"generated_ids\":";print_ids(out,p.generated);
      out<<",\"reused_tokens\":"<<p.reused<<",\"consumed_tokens\":"<<session.consumed.size()
                <<",\"do_sample\":"<<(p.options.do_sample?"true":"false")<<",\"seed\":"<<p.options.seed
                <<",\"live_token_count\":"<<live;  // the token-choice bound this generation used (G6)
      if(p.want_text) out<<",\"text\":"<<lfm::lfm3_text::json_string(lfm::lfm3_text::generated_text(text.get(),p.generated,stops));
      if(p.trace_sampling) out<<",\"sampling_trace\":"<<p.origin_rpc;
      if(p.return_logits) {
        out<<",\"last_logits\":[";
        for(size_t i=0;i<session.last_scores.size();++i) {if(i)out<<',';out<<session.last_scores[i];}
        out<<']';
      }
      out<<"}\n"<<std::flush;
      session.pending.reset();
    } catch(const GenerationCancelled&) {
      const auto owned=std::move(session.pending);  // the session is free whatever the reply's writing does
      session.runner.reset();session.consumed.clear();session.last_scores.clear();owned->lease->abort();
      write_cancelled(out,rpc_id,id,*owned);
    } catch(...) {
      session.runner.reset();session.consumed.clear();session.last_scores.clear();session.pending.reset();throw;
    }
  }
  void write_cancelled(std::ostream& out,int rpc_id,const std::string& id,const PendingText& p) {
    out<<"{\"rpc_id\":"<<rpc_id<<",\"ok\":true,\"session_id\":"<<quote(id)
             <<",\"cancelled\":true,\"cache_valid\":false,\"consumed_tokens\":0,\"reused_tokens\":0,\"generated_ids\":";
    print_ids(out,p.generated);
    out<<",\"do_sample\":"<<(p.options.do_sample?"true":"false")<<",\"seed\":"<<p.options.seed
             <<",\"live_token_count\":"<<live;  // G6 bound, on cancelled results too
    if(p.want_text) out<<",\"text\":"<<lfm::lfm3_text::json_string(lfm::lfm3_text::generated_text(text.get(),p.generated,stops));
    out<<"}\n"<<std::flush;
  }
  std::string directory,sampling_trace_root;
  lfm::text::CancellationReader cancellation;
  bool cancellation_enabled;
  // Speech (#2, lfm3_speech.cuh): one request at a time on its own runner (capacity kSpeechCapacity per chunk),
  // sharing the loaded weights; created on first use. Same loop as lfm3-speech (run_chunks), so the same bytes.
  static constexpr int kSpeechCapacity=4096;
  // Decode steps per request, across all its chunks (lfm3_speech::run_chunks): bounds its worst-case time under the
  // default engine timeout (validation/speech-server-v1: ~40 frames/s measured with locked clocks, ~34 assumed).
  static constexpr size_t kSpeechStepBudget=6144;
  static constexpr size_t kSpeechMaxClips=64;  // audio clips per request (lfm::http::kSpeechMaxClips, lfm3-speech)
  static constexpr size_t kSpeechMaxImages=16;  // images per request (lfm::http::kSpeechMaxImages)
  static constexpr std::uint64_t kSpeechMaxImagePixels=64ull*1024*1024;  // lfm::http::kSpeechMaxImagePixels
  bool speech_ok=false,audio_in_ok=false,image_in_ok=false;
  int speech_rows_=8;  // rows of the concurrent speech batch (set_speech_rows, before its first use)
#ifdef LFM3_NATIVE_GEMM
  lfm3_speech::ImageCache image_cache_;  // validation/image-cache-v1: engine thread only (parse_speech, the prefill)
#endif
#ifdef LFM3_NATIVE_GEMM
  std::unique_ptr<Runner> speech_runner;
  std::unique_ptr<lfm3_speech::Session> speech_session;
  std::unique_ptr<lfm3_speech::Batch> speech_batch;
  struct BatchRow {
    long long ticket=-1;  // -1: free
    std::unique_ptr<lfm3_speech::ChunkRun> run;
    bool stream=false,cancel=false;
  };
  std::vector<BatchRow> batch_rows;
#endif
  // {"op":"speech","chunks":[[ids..],..],"mode":"audio"|"interleaved"|"text","max_new_tokens":N,"seed":S,"chunked":B,
  //  "audio":[base64 16 kHz mono WAV, ..] (audio input, #3: one clip per <|audio|> span of the single prompt),
  //  "images":[base64 JPEG or PNG, ..] (image input, #4: one image per <|image|> span of the single prompt, in order),
  //  "stream":B (in-process only, through `progress`)}
  // -> {"ok":true,"cancelled":false,"truncated":B,"slots":[..],"chunks":[{"frames","slots","ended_by_eos"}],"frames",
  //     "samples","audio_seconds","prefill_seconds","generate_seconds","wav_base64"}   (cancelled: only "cancelled")
  // Streamed, the reply has no "wav_base64"; before it, in generation order, `progress` receives {"audio":"base64 of
  // 16-bit PCM"} as the codec makes samples final (all of them, in order, are the WAV's data), {"text":[slot]} for each
  // emitted slot but <|audio_out|>, and {"chunk":{"frames","slots","ended_by_eos"}} as each chunk ends.
#ifdef LFM3_NATIVE_GEMM
  // A speech request read and checked (op "speech" and op "speech_admit" alike), before any engine state changes.
  struct SpeechJob {
    std::vector<std::vector<int>> prompts;
    lfm3_speech::Options o;
    bool chunked=false,stream=false;
  };
  SpeechJob parse_speech(const lfm::Json& request) {
    if(!speech_ok) throw std::runtime_error("speech is not supported by this checkpoint");
    SpeechJob job;
    const auto mode=request.at("mode").string();
    if(mode!="audio" && mode!="interleaved" && mode!="text") throw std::runtime_error("speech mode must be audio, interleaved or text");
    const int count=request.at("max_new_tokens").integer();
    if(count<1 || count>=kSpeechCapacity) throw std::runtime_error("speech max_new_tokens out of range");
    const long long seed=request.has("seed")?request.at("seed").integer():0;
    if(seed<0) throw std::runtime_error("speech seed must be nonnegative");
    const bool chunked=job.chunked=request.has("chunked") && request.at("chunked").boolean();
    job.stream=request.has("stream") && request.at("stream").boolean();
    if(job.stream && !progress) throw std::runtime_error("speech streaming needs the in-process server");
    auto& prompts=job.prompts;
    for(const auto& c:request.at("chunks").array()) {
      prompts.emplace_back();
      for(const auto& x:c.array()) {
        const int id=x.integer();
        if(id<0 || id>=live) throw std::runtime_error("speech prompt id out of range");
        prompts.back().push_back(id);
      }
      if(prompts.back().empty() || prompts.back().size()+size_t(count)>size_t(kSpeechCapacity))
        throw std::runtime_error("speech prompt empty or over capacity");
    }
    if(prompts.empty() || (!chunked && prompts.size()!=1)) throw std::runtime_error("speech needs one prompt, or chunks");
    std::vector<std::vector<float>> clips;  // decoded and checked (spans included) before any engine state changes
    if(request.has("audio")) {
      if(!audio_in_ok) throw std::runtime_error("audio input is not supported by this checkpoint");
      if(request.at("audio").array().size()>kSpeechMaxClips) throw std::runtime_error("too many audio clips");
      if(chunked) throw std::runtime_error("chunks cannot take audio input");
      std::vector<size_t> clip_tokens;
      for(const auto& a:request.at("audio").array()) {
        clips.push_back(lfm3_speech::clip_from_base64(a.string()));
        clip_tokens.push_back(lfm::audio_features::tokens(clips.back().size()));
      }
      lfm3_speech::check_audio_spans(prompts[0],clip_tokens);
    } else {
      for(const auto& p:prompts) lfm3_speech::check_audio_spans(p,{});
    }
    std::vector<std::shared_ptr<const lfm::image::PixelValues>> images;  // decoded, preprocessed and checked (spans included) likewise
    std::vector<std::string> image_keys;  // validation/image-cache-v1: each image's cache key (empty: not cached)
    std::vector<std::shared_ptr<const lfm3_speech::ImageRows>> image_rows;  // the cache's rows (null: encode)
    if(request.has("images")) {
      if(!image_in_ok) throw std::runtime_error("image input is not supported by this checkpoint");
      if(request.at("images").array().size()>kSpeechMaxImages) throw std::runtime_error("too many images");
      if(chunked) throw std::runtime_error("chunks cannot take image input");
      std::vector<size_t> image_tokens;
      lfm::image::Limits limits;limits.max_pixels=kSpeechMaxImagePixels;
      const bool caching=image_cache_.budget()>0;
      for(const auto& a:request.at("images").array()) {
        const std::string& key=a.string();
        // validation/image-cache-v1: these exact bytes encoded before: their rows, no decode or encode
        if(auto rows=caching?image_cache_.find(key):nullptr) {
          auto stub=std::make_shared<lfm::image::PixelValues>();
          stub->grid_h=rows->grid_h;stub->grid_w=rows->grid_w;stub->tokens=rows->tokens;
          stub->row_size=lfm3_vision::Encoder::kPatchDim;
          images.push_back(std::move(stub));
          image_rows.push_back(std::move(rows));
        } else if(const auto pixels=lfm::image::shared_pixel_handoff().find(key)) {
          images.push_back(pixels);  // image-queue-v1 part 2: decoded by the in-process server's handler thread (shared)
          image_rows.push_back(nullptr);
        } else {
          const std::string bytes=lfm::base64::decode(key);
          images.push_back(std::make_shared<const lfm::image::PixelValues>(lfm::image::preprocess(lfm::image::load_rgb(reinterpret_cast<const std::uint8_t*>(bytes.data()),bytes.size(),limits))));
          image_rows.push_back(nullptr);
        }
        image_keys.push_back(caching?key:std::string());
        image_tokens.push_back(size_t(images.back()->tokens));
      }
      lfm3_speech::check_image_spans(prompts[0],image_tokens);
    } else {
      for(const auto& p:prompts) lfm3_speech::check_image_spans(p,{});
    }
    auto& o=job.o;
    o.mode=mode=="audio"?lfm::audio::Requested::Audio:mode=="text"?lfm::audio::Requested::Text:lfm::audio::Requested::Interleaved;
    o.max_new_tokens=count;o.seed=uint64_t(seed);o.audio=std::move(clips);o.images=std::move(images);
    o.image_keys=std::move(image_keys);o.image_rows=std::move(image_rows);
    return job;
  }
  // The streaming observers: progress lines, each prefixed with `tag` (e.g. "\"ticket\":7," for a batch row).
  void observe(lfm3_speech::Options& o,const std::string& tag) {
    o.on_audio=[this,tag](const float* x,size_t n) {
      std::string pcm;lfm3_speech::append_pcm16(x,n,pcm);
      progress("{"+tag+"\"audio\":\""+lfm::base64::encode(pcm)+"\"}");
    };
    o.on_slot=[this,tag](int slot) {
      if(slot!=lfm::audio::ModeTokens{}.audio_out) progress("{"+tag+"\"text\":["+std::to_string(slot)+"]}");
    };
    o.on_chunk=[this,tag](size_t,const lfm3_speech::ChunkInfo& c) {
      progress("{"+tag+"\"chunk\":{\"frames\":"+std::to_string(c.frames)+",\"slots\":"+std::to_string(c.slots)+
               ",\"ended_by_eos\":"+(c.ended_by_eos?"true":"false")+"}}");
    };
  }
  // A finished request's reply fields, from "cancelled" on (no braces); streamed, no "wav_base64".
  static std::string speech_fields(const lfm3_speech::Result& r,const std::vector<lfm3_speech::ChunkInfo>& info,
                                   bool truncated,bool stream) {
    // the WAV before any output: a throw leaves no half line
    const std::string wav=stream?std::string():lfm::base64::encode(lfm3_speech::wav_bytes(r.wav,24000));
    std::ostringstream line;line<<std::setprecision(std::numeric_limits<double>::max_digits10);
    line<<"\"cancelled\":false,\"truncated\":"<<(truncated?"true":"false")<<",\"slots\":[";
    for(size_t i=0;i<r.slots.size();++i) line<<(i?",":"")<<r.slots[i];
    line<<"],\"chunks\":[";
    for(size_t k=0;k<info.size();++k)
      line<<(k?",":"")<<"{\"frames\":"<<info[k].frames<<",\"slots\":"<<info[k].slots<<",\"ended_by_eos\":"
          <<(info[k].ended_by_eos?"true":"false")<<"}";
    line<<"],\"frames\":"<<r.frames.size()<<",\"samples\":"<<r.wav.size()<<",\"audio_seconds\":"<<r.wav.size()/24000.0
        <<",\"prefill_seconds\":"<<r.prefill_seconds<<",\"generate_seconds\":"<<r.generate_seconds;
    if(!stream) line<<",\"wav_base64\":\""<<wav<<"\"";
    return line.str();
  }
  void ensure_speech_runner() {
    if(!weights) weights=std::make_shared<lfm::Safetensors>(directory+"/model.safetensors");
    if(!speech_runner)
      speech_runner=std::make_unique<Runner>(directory,lfm::ModelConfig::load(directory+"/config.json",kSpeechCapacity),weights);
  }
#endif
  // {"op":"speech","chunks":[[ids..],..],"mode":"audio"|"interleaved"|"text","max_new_tokens":N,"seed":S,"chunked":B,
  //  "audio":[base64 16 kHz mono WAV, ..] (audio input, #3: one clip per <|audio|> span of the single prompt),
  //  "images":[base64 JPEG or PNG, ..] (image input, #4: one image per <|image|> span of the single prompt, in order),
  //  "stream":B (in-process only, through `progress`)}
  // -> {"ok":true,"cancelled":false,"truncated":B,"slots":[..],"chunks":[{"frames","slots","ended_by_eos"}],"frames",
  //     "samples","audio_seconds","prefill_seconds","generate_seconds","wav_base64"}   (cancelled: only "cancelled")
  // Streamed, the reply has no "wav_base64"; before it, in generation order, `progress` receives {"audio":"base64 of
  // 16-bit PCM"} as the codec makes samples final (all of them, in order, are the WAV's data), {"text":[slot]} for each
  // emitted slot but <|audio_out|>, and {"chunk":{"frames","slots","ended_by_eos"}} as each chunk ends.
  void speech(const lfm::Json& request,int rpc_id,std::ostream& out) {
#ifndef LFM3_NATIVE_GEMM
    (void)request;(void)rpc_id;(void)out;
    throw std::runtime_error("speech requires the native GEMM build (LIQUID_LFM3_NATIVE_GEMM=ON)");
#else
    auto job=parse_speech(request);
    ensure_speech_runner();
    if(!speech_session) {
      speech_session=std::make_unique<lfm3_speech::Session>(*speech_runner,directory);
      speech_session->set_image_cache(&image_cache_);
    }
    auto& o=job.o;
    o.cancelled=[&] {return cancellation_enabled && cancellation.requested(uint64_t(rpc_id));};
    if(job.stream) observe(o,"");
    std::vector<lfm3_speech::ChunkInfo> info;
    lfm3_speech::Result r;
    bool truncated=false;
    try {r=lfm3_speech::run_chunks(*speech_session,job.prompts,std::move(o),job.chunked,info,kSpeechStepBudget,&truncated);}
    catch(const lfm3_speech::Cancelled&) {
      out<<"{\"rpc_id\":"<<rpc_id<<",\"ok\":true,\"cancelled\":true}\n"<<std::flush;return;
    }
    const auto fields=speech_fields(r,info,truncated,job.stream);
    out<<"{\"rpc_id\":"<<rpc_id<<",\"ok\":true,"<<fields<<"}\n"<<std::flush;
#endif
  }
  // Concurrent speech (validation/speech-batch-v1), in-process only (results and events come through `progress`):
  // {"op":"speech_admit","ticket":T,...op speech's fields...} starts the request in a free row of the speech batch
  //   (speech_rows() rows) -> {"ok":true,"ticket":T,"row":R}; refused (ok false) when no row is free or T is running.
  // {"op":"speech_step","cancel":[T..]} cancels those tickets at this step, then steps every busy row once (a split
  //   request moves to its next chunk in its row) -> {"ok":true,"busy":N}. Before the reply, in order, `progress`
  //   receives a streamed ticket's events as op speech streams them, with "ticket":T first, and every request that
  //   finished: {"ticket":T,"finished":{op speech's reply fields}} or {"ticket":T,"finished":{"cancelled":true}}.
  // Same chunk loop, step budget and outputs as op speech, row by row: a request's reply does not depend on the others.
  void speech_admit(const lfm::Json& request,int rpc_id,std::ostream& out) {
#ifndef LFM3_NATIVE_GEMM
    (void)request;(void)rpc_id;(void)out;
    throw std::runtime_error("speech requires the native GEMM build (LIQUID_LFM3_NATIVE_GEMM=ON)");
#else
    if(!progress) throw std::runtime_error("concurrent speech needs the in-process server");
    const long long ticket=request.at("ticket").integer();
    if(ticket<0) throw std::runtime_error("speech ticket must be nonnegative");
    auto job=parse_speech(request);
    ensure_speech_runner();
    if(!speech_batch) {
      speech_batch=std::make_unique<lfm3_speech::Batch>(*speech_runner,directory,speech_rows_);
      speech_batch->set_image_cache(&image_cache_);
      batch_rows.clear();batch_rows.resize(size_t(speech_rows_));
    }
    int row=-1;
    for(int r=0;r<speech_rows_;++r) {
      if(batch_rows[size_t(r)].ticket==ticket) throw std::runtime_error("speech ticket already running");
      if(row<0 && batch_rows[size_t(r)].ticket<0) row=r;
    }
    if(row<0) throw std::runtime_error("no free speech row");
    auto& o=job.o;
    o.cancelled=[this,row] {return batch_rows[size_t(row)].cancel;};
    if(job.stream) observe(o,"\"ticket\":"+std::to_string(ticket)+",");
    BatchRow& b=batch_rows[size_t(row)];
    b.run=std::make_unique<lfm3_speech::ChunkRun>(job.prompts,std::move(o),job.chunked,kSpeechStepBudget);  // moved: C4
    b.ticket=ticket,b.stream=job.stream,b.cancel=false;
    try {
      const std::vector<int>* prompt=nullptr;lfm3_speech::Options chunk;
      b.run->next(prompt,chunk);
      speech_batch->admit(row,*prompt,std::move(chunk));
    } catch(...) {b=BatchRow{};throw;}
    out<<"{\"rpc_id\":"<<rpc_id<<",\"ok\":true,\"ticket\":"<<ticket<<",\"row\":"<<row<<"}\n"<<std::flush;
#endif
  }
  void speech_step(const lfm::Json& request,int rpc_id,std::ostream& out) {
#ifndef LFM3_NATIVE_GEMM
    (void)request;(void)rpc_id;(void)out;
    throw std::runtime_error("speech requires the native GEMM build (LIQUID_LFM3_NATIVE_GEMM=ON)");
#else
    if(!progress) throw std::runtime_error("concurrent speech needs the in-process server");
    if(request.has("cancel"))
      for(const auto& t:request.at("cancel").array())
        for(auto& b:batch_rows) if(b.ticket==t.integer()) b.cancel=true;
    int busy=0;
    if(speech_batch) {
      std::vector<int> finished;
      try {finished=speech_batch->step();}
      catch(...) {for(auto& b:batch_rows) b=BatchRow{};throw;}  // the batch closed every row: an engine fault
      for(int r:finished) {
        BatchRow& b=batch_rows[size_t(r)];
        const std::string tag="{\"ticket\":"+std::to_string(b.ticket)+",\"finished\":{";
        if(speech_batch->cancelled(r)) {
          speech_batch->take(r);
          b=BatchRow{};
          progress(tag+"\"cancelled\":true}}");
          continue;
        }
        b.run->done(speech_batch->take(r));
        const std::vector<int>* prompt=nullptr;lfm3_speech::Options chunk;
        if(b.run->next(prompt,chunk)) {speech_batch->admit(r,*prompt,std::move(chunk));continue;}  // its next chunk, same row
        const auto result=b.run->take();
        const auto fields=speech_fields(result,b.run->info(),b.run->truncated(),b.stream);
        b=BatchRow{};
        progress(tag+fields+"}}");
      }
      for(const auto& b:batch_rows) busy+=b.ticket>=0;
    }
    out<<"{\"rpc_id\":"<<rpc_id<<",\"ok\":true,\"busy\":"<<busy<<"}\n"<<std::flush;
#endif
  }
  lfm::ModelConfig cfg;
  std::vector<int> suppressed,stops,choice_suppressed;
  int live=0;
  TextFrontend text;
  std::shared_ptr<lfm::Safetensors> weights;
  std::map<std::string,Session> sessions;
  lfm::text::SessionLedger ledger{2,32768};
  int last_rpc_id=0;
};
}  // namespace (worker core)

#ifndef LFM3_WORKER_CORE_ONLY
int main(int argc,char** argv) try {
  if(argc<3 || argc%2!=1 || std::string(argv[1])!="--experimental")
    throw std::runtime_error("usage: liquid-lfm3-worker --experimental MODEL_DIRECTORY [--sampling-trace-root NEW_DIRECTORY] [--cancel-fd FD]");
  const std::string directory=argv[2];
  std::string sampling_trace_root;int cancellation_fd=-1;
  std::set<std::string> options_seen;
  for(int i=3;i<argc;i+=2) {
    std::string option=argv[i];
    if(!options_seen.insert(option).second) throw std::runtime_error("duplicate worker option");
    if(option=="--sampling-trace-root") sampling_trace_root=argv[i+1];
    else if(option=="--cancel-fd") {
      std::string value=argv[i+1];size_t used=0;
      cancellation_fd=std::stoi(value,&used);
      if(used!=value.size() || cancellation_fd<3) throw std::runtime_error("invalid cancellation fd");
    } else throw std::runtime_error("unsupported worker option");
  }
  WorkerCore core(directory,sampling_trace_root,options_seen.count("--sampling-trace-root")>0,cancellation_fd);
  std::cout<<std::setprecision(std::numeric_limits<double>::max_digits10);
  core.ready(std::cout);
  for(;;) {
    // Bounded framing, unlike unbounded std::getline allocation.
    std::string line;bool oversized=false;char c;
    while(std::cin.get(c) && c!='\n') {
      if(line.size()<4*1024*1024) line+=c;else oversized=true;
    }
    if(line.empty() && !std::cin && !oversized) break;
    if(!core.handle(line,oversized,std::cout)) break;
  }
  return 0;
} catch(const std::exception& e) {std::cerr<<"lfm3-worker: "<<e.what()<<"\n";return 1;}
#endif
