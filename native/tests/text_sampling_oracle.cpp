// Independent-process wiring oracle: replay materialized logits, not model code.
#include <cmath>
#include <filesystem>
#include <lfm/model_config.hpp>
#include <lfm/text_sampling.hpp>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

lfm::Json json(const std::string& path) {
  std::ifstream file(path);
  if(!file) throw std::runtime_error("cannot open "+path);
  return lfm::Json::parse(std::string(std::istreambuf_iterator<char>(file),{}));
}
std::vector<int> ids(const lfm::Json& root,const std::string& name) {
  std::vector<int> result;
  for(const auto& x:root.at(name).array()) result.push_back(x.integer());
  return result;
}
// Whole-string decimal integer: rejects "3.9", "3abc", "", sign tricks and overflow.
int strict_int(const char* text,const char* name) {
  const std::string value(text);
  if(value.empty() || value.size()>9 || value.find_first_not_of("0123456789")!=std::string::npos)
    throw std::runtime_error(std::string("invalid ")+name);
  return std::stoi(value);
}
int main(int argc,char** argv) try {
  if(argc!=5 && argc!=6) throw std::runtime_error("usage: text-sampling-oracle REQUEST.json RESULT.json TRACE_DIR VOCAB [TOKENIZER.json]");
  const auto req=json(argv[1]),out=json(argv[2]);
  const int vocab=strict_int(argv[4],"vocabulary");
  if(vocab<1 || vocab>16777216) throw std::runtime_error("invalid vocabulary");
  // Traces hold all VOCAB logits; tokens are chosen among the ids the tokenizer defines, counted by
  // the engine's own lfm::tokenizer_token_count (one implementation, so the replay cannot drift from
  // the engine). Without TOKENIZER.json every VOCAB id is live.
  const int live=argc==6?lfm::tokenizer_token_count(argv[5],vocab):vocab;
  lfm::TextSamplingOptions options;
  options.do_sample=req.at("do_sample").boolean();
  options.temperature=req.at("temperature").number();
  options.top_k=req.at("top_k").integer();
  options.top_p=req.at("top_p").number();
  options.min_p=req.at("min_p").number();
  options.repetition_penalty=req.at("repetition_penalty").number();
  options.seed=req.at("seed").integer();
  if(!options.do_sample) throw std::runtime_error("oracle fixture must be stochastic");
  if(options.top_k>vocab) throw std::runtime_error("top_k exceeds vocabulary size");
  if(options.top_k>live) options.top_k=live;  // the engines' rule: k above the token count = all tokens
  lfm::TextSampler sampler(options);
  const auto generated=ids(out,"generated_ids");
  // Same rule as the worker: suppression ids at or above `live` are already excluded by the bound.
  std::vector<int> suppressed;
  for(int id:ids(req,"suppress_tokens")) {
    if(id<0 || id>=vocab) throw std::runtime_error("suppress token outside vocabulary");
    if(id<live) suppressed.push_back(id);
  }
  if(generated.empty()) throw std::runtime_error("missing generation");
  std::vector<int> history;
  bool differs=false;
  // Exact traces are all-or-none: an engine that samples from FP32 logits writes STEP.logits.f32 for every
  // step, and each STEP.logits.bf16 must then be its round-to-nearest-even rounding.
  const bool exact_traces=std::filesystem::exists(std::string(argv[3])+"/0.logits.f32");
  for(size_t step=0;step<generated.size();++step) {
    // An engine that samples from FP32 logits (the native-GEMM build) writes STEP.logits.f32 with the exact
    // values it used; replay those. Otherwise the BF16 trace is exact.
    const std::string stem=std::string(argv[3])+"/"+std::to_string(step);
    std::vector<float> logits(vocab);
    if(std::filesystem::exists(stem+".logits.f32")!=exact_traces)
      throw std::runtime_error("FP32 traces must exist for every step or none (step "+std::to_string(step)+")");
    if(exact_traces) {
      std::ifstream file(stem+".logits.f32",std::ios::binary);
      if(!file.read(reinterpret_cast<char*>(logits.data()),std::streamsize(vocab)*4) || file.peek()!=std::char_traits<char>::eof())
        throw std::runtime_error("wrong-size FP32 trace");
      std::ifstream rounded(stem+".logits.bf16",std::ios::binary);
      std::vector<std::uint16_t> raw(vocab);
      if(!rounded.read(reinterpret_cast<char*>(raw.data()),vocab*2) || rounded.peek()!=std::char_traits<char>::eof())
        throw std::runtime_error("missing/wrong-size BF16 trace");
      for(int i=0;i<vocab;++i) {
        if(!std::isfinite(logits[i])) throw std::runtime_error("nonfinite FP32 trace");
        std::uint32_t bits; std::memcpy(&bits,&logits[i],sizeof(bits));
        const std::uint16_t nearest_even=std::uint16_t((bits+0x7FFFu+((bits>>16)&1u))>>16);
        if(raw[i]!=nearest_even) throw std::runtime_error("BF16 trace is not the rounding of the FP32 trace at step "+std::to_string(step));
      }
    } else {
      std::ifstream file(stem+".logits.bf16",std::ios::binary);
      std::vector<std::uint16_t> raw(vocab);
      if(!file.read(reinterpret_cast<char*>(raw.data()),vocab*2) || file.peek()!=std::char_traits<char>::eof())
        throw std::runtime_error("missing/wrong-size BF16 trace");
      for(int i=0;i<vocab;++i) {std::uint32_t bits=std::uint32_t(raw[i])<<16; std::memcpy(&logits[i],&bits,sizeof(float));}
    }
    int expected=sampler.sample(logits.data(),size_t(live),history,suppressed);
    if(expected!=generated[step]) throw std::runtime_error("sampler wiring mismatch at step "+std::to_string(step));
    auto greedy_options=options;greedy_options.do_sample=false;
    lfm::TextSampler greedy(greedy_options);
    differs |= expected!=greedy.sample(logits.data(),size_t(live),history,suppressed);
    history.push_back(expected);
  }
  if(!differs) throw std::runtime_error("fixture did not demonstrate non-greedy sampling");
  std::cout<<"PASS: replayed "<<generated.size()<<" sampled tokens; non-greedy choice verified; live_token_count "<<live<<"\n";
  return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<"\n"; return 1;}
