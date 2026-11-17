// Standalone, opt-in GPU experiment. Not built/integrated by default.
#define LFM3_RUNTIME_ONLY
#include "../src/lfm3_main.cu"
#include "../src/lfm3_decode_batch.cuh"
#include <cstring>
#include <iomanip>
#include <sstream>

namespace {
std::string json_string(const std::string& text) {
  std::ostringstream out;out<<'"';
  for(unsigned char c:text) {
    if(c=='"' || c=='\\') out<<'\\'<<char(c);
    else if(c>=32 && c<127) out<<char(c);
    else out<<"\\u00"<<std::hex<<std::setw(2)<<std::setfill('0')<<int(c);
  }
  out<<'"';return out.str();
}
int argmax(const std::vector<float>& values) {
  if(values.empty()) throw std::runtime_error("empty probe logits");
  return std::max_element(values.begin(),values.end())-values.begin();
}
void require(bool condition,const std::string& message) {
  if(!condition) throw std::runtime_error(message);
}
std::vector<std::uint16_t> cache_snapshot(const Runner& r) {
  std::vector<std::uint16_t> bits;
  auto append=[&](const bf16* ptr,size_t count) {
    const size_t offset=bits.size();bits.resize(offset+count);
    cuda_check(cudaMemcpy(bits.data()+offset,ptr,count*sizeof(bf16),cudaMemcpyDeviceToHost));
  };
  for(int i=0;i<r.s.layers;++i) {
    if(r.keys[i]) {append(r.keys[i]->p,r.s.kv_layer_stride());append(r.values[i]->p,r.s.kv_layer_stride());}
    if(r.states[i]) append(r.states[i]->p,size_t(r.s.hidden)*7);
  }
  return bits;
}
struct PhysicalState {
  std::vector<std::uint16_t> bits;
  int cursor=0,step=0;
  static PhysicalState capture(const Runner& r) {return {cache_snapshot(r),r.cursor,r.step};}
  void restore(Runner& r) const {
    size_t expected=0;
    for(int i=0;i<r.s.layers;++i) {
      if(r.keys[i]) expected+=2*r.s.kv_layer_stride();
      if(r.states[i]) expected+=size_t(r.s.hidden)*7;
    }
    require(bits.size()==expected && cursor>=0 && cursor<=r.s.context && step>=0,"invalid oracle state");
    size_t offset=0;
    auto upload=[&](bf16* ptr,size_t count) {
      cuda_check(cudaMemcpy(ptr,bits.data()+offset,count*sizeof(bf16),cudaMemcpyHostToDevice));offset+=count;
    };
    for(int i=0;i<r.s.layers;++i) {
      if(r.keys[i]) {upload(r.keys[i]->p,r.s.kv_layer_stride());upload(r.values[i]->p,r.s.kv_layer_stride());}
      if(r.states[i]) upload(r.states[i]->p,size_t(r.s.hidden)*7);
    }
    cuda_check(cudaDeviceSynchronize());r.cursor=cursor;r.step=step;
  }
};
// Test-only oracle: normal owns independent GPU scratch; compact swaps ONLY
// legitimate physical cache/counters into its actual owner, never activations.
class Oracle {
  Runner& actual_;
  std::unique_ptr<Runner> independent_;
  PhysicalState state_;
  std::vector<float> execute(const std::vector<int>* input) {
    if(independent_) {
      if(input) return independent_->forward(*input);
      independent_->reset();return {};
    }
    const auto saved=PhysicalState::capture(actual_);
    state_.restore(actual_);
    std::vector<float> result;
    if(input) result=actual_.forward(*input);else actual_.reset();
    auto next=PhysicalState::capture(actual_);
    saved.restore(actual_);
    require(cache_snapshot(actual_)==saved.bits && actual_.cursor==saved.cursor && actual_.step==saved.step,
            "compact oracle failed to restore actual physical state");
    state_=std::move(next);return result;
  }
public:
  Oracle(Runner& actual,const std::string& directory,const lfm::ModelConfig& cfg,
         const std::shared_ptr<lfm::Safetensors>& weights,bool compact):actual_(actual) {
    if(compact) state_=PhysicalState::capture(actual);
    else independent_=std::make_unique<Runner>(directory,cfg,weights);
  }
  std::vector<float> forward(const std::vector<int>& input) {return execute(&input);}
  void reset() {(void)execute(nullptr);}
  std::vector<std::uint16_t> cache() const {return independent_?cache_snapshot(*independent_):state_.bits;}
  int cursor() const {return independent_?independent_->cursor:state_.cursor;}
  int step() const {return independent_?independent_->step:state_.step;}
};
}
int main(int argc,char** argv) {
  if((argc<4 || argc>6) || std::string(argv[1])!="--experimental") {
    std::cerr<<"usage: decode-batch-probe --experimental MODEL_DIRECTORY OUTPUT_DIR [scratch|sliding|softmax|tiled|upper] [normal|compact]\n";return 1;
  }
  const std::filesystem::path output=argv[3];
  try {
    if(!std::filesystem::create_directory(output)) throw std::runtime_error("output directory must be new");
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
  const std::string fixture=argc>=5?argv[4]:"scratch",strategy=argc==6?argv[5]:"normal";
  std::ofstream log(output/"checks.jsonl");int comparisons=0,rejections=0,paired_calls=0,retired_checks=0,resets=0;
  int expected_comparisons=0,expected_paired=0,expected_retired=0;
  try {
    require(bool(log),"cannot open probe log");
    require(strategy=="normal" || strategy=="compact","unknown reference strategy");
    const bool compact=strategy=="compact";
    auto begin_case=[&](const std::string& name) {
      log<<"{\"case\":"<<json_string(name)<<",\"event\":\"begin\"}\n"<<std::flush;
      require(bool(log),"case progress write failed");
    };
    int capacity_a=12,capacity_b=300,length_a=3,length_b=257;
    if(fixture=="sliding") {capacity_a=270;capacity_b=280;length_a=254;length_b=256;}
    else if(fixture=="softmax") {capacity_a=2062;capacity_b=2074;length_a=2046;length_b=2048;}
    else if(fixture=="tiled") {capacity_a=4112;capacity_b=4124;length_a=4096;length_b=4098;}
    else if(fixture=="upper") {capacity_a=2;capacity_b=32766;length_a=1;length_b=32764;}
    else require(fixture=="scratch","unknown fixture");
    const bool upper=fixture=="upper";
    expected_paired=capacity_a-length_a+(upper?0:1); // extra reversed-order call before reset
    const int b_only=capacity_b-length_b-(capacity_a-length_a);
    expected_comparisons=(upper?2:7)+2*(capacity_a-length_a)+b_only;
    expected_retired=b_only+(upper?0:1);
    log<<"{\"fixture\":"<<json_string(fixture)<<",\"reference_mode\":"<<json_string(strategy)
       <<",\"gpu_runner_owners\":"<<(compact?2:4)<<",\"capacities\":["<<capacity_a<<','<<capacity_b
       <<"],\"prompt_lengths\":["<<length_a<<','<<length_b<<"],\"expected_comparisons\":"<<expected_comparisons
       <<",\"expected_paired_calls\":"<<expected_paired<<",\"expected_retired_checks\":"<<expected_retired
       <<",\"expected_rejections\":11,\"expected_reset_checks\":"<<(upper?0:2)
       <<",\"standard_six_step_sequence\":"<<(upper?"false":"true")<<"}\n"<<std::flush;
    require(bool(log),"fixture log write failed");
    const std::string directory=argv[2];
    auto ca=lfm::ModelConfig::load(directory+"/config.json",capacity_a);
    auto cb=lfm::ModelConfig::load(directory+"/config.json",capacity_b);
    require(ca.shape.context==capacity_a && cb.shape.context==capacity_b,"configured capacity changed");
    require(ca.is_lfm3() && ca.sandwich_norm && ca.tied && ca.shape.hidden==2048 &&
      ca.shape.head_dim==128 && ca.shape.conv_width==7,"unsupported probe geometry");
    auto weights=std::make_shared<lfm::Safetensors>(directory+"/model.safetensors");
    Runner a(directory,ca,weights),b(directory,cb,weights);
    Oracle ra(a,directory,ca,weights,compact),rb(b,directory,cb,weights,compact);
    auto compare=[&](const std::string& name,const std::vector<float>& got,
                     const std::vector<float>& expected,const Runner& actual,const Oracle& ref) {
      size_t mismatch=0;
      while(mismatch<got.size() && mismatch<expected.size() &&
            std::memcmp(&got[mismatch],&expected[mismatch],sizeof(float))==0) ++mismatch;
      const bool exact=got.size()==size_t(ca.shape.vocabulary) && got.size()==expected.size() && mismatch==got.size();
      const auto cache=cache_snapshot(actual),reference_cache=ref.cache();
      size_t cache_mismatch=0;
      while(cache_mismatch<cache.size() && cache_mismatch<reference_cache.size() &&
            cache[cache_mismatch]==reference_cache[cache_mismatch]) ++cache_mismatch;
      const bool exact_cache=cache==reference_cache;
      log<<"{\"case\":"<<json_string(name)<<",\"exact_logits\":"<<(exact?"true":"false")
         <<",\"logit_comparison\":\"float_bits\""
         <<",\"exact_full_cache\":"<<(exact_cache?"true":"false")<<",\"cache_bytes\":"<<cache.size()*2
         <<",\"first_cache_mismatch_word\":"<<cache_mismatch
         <<",\"vocabulary\":"<<got.size()<<",\"first_mismatch\":"<<mismatch
         <<",\"id\":"<<argmax(got)<<",\"expected_id\":"<<argmax(expected)
         <<",\"cursor\":"<<actual.cursor<<",\"expected_cursor\":"<<ref.cursor()<<"}\n"<<std::flush;
      require(bool(log),"probe log write failed");
      require(exact && exact_cache && argmax(got)==argmax(expected) && actual.cursor==ref.cursor() && actual.step==ref.step(),
              name+": exact serial parity failed (no tolerance fallback)");++comparisons;
    };
    std::vector<int> prompt_a(length_a),prompt_b(length_b);
    for(int i=0;i<length_a;++i) prompt_a[i]=1+i%17;
    for(int i=0;i<length_b;++i) prompt_b[i]=4+i%19;
    std::vector<float> sa,sb;
    auto prefill=[&](const std::string& name) {
      begin_case(name);
      sa=ra.forward(prompt_a);sb=rb.forward(prompt_b);
      compare(name+"-A",a.forward(prompt_a),sa,a,ra);
      compare(name+"-B",b.forward(prompt_b),sb,b,rb);
    };
    prefill("prefill");
    auto reject=[&](const std::string& name,const std::vector<Runner*>& lanes,const std::vector<int>& ids) {
      begin_case(name);
      const int ac=a.cursor,bc=b.cursor,as=a.step,bs=b.step;
      const auto cache_a=cache_snapshot(a),cache_b=cache_snapshot(b);
      bool rejected=false;
      try {(void)lfm3_decode_batch::forward(lanes,ids);}
      catch(const std::invalid_argument&) {rejected=true;}
      const bool metadata=a.cursor==ac && b.cursor==bc && a.step==as && b.step==bs;
      const bool caches=cache_snapshot(a)==cache_a && cache_snapshot(b)==cache_b;
      log<<"{\"case\":"<<json_string(name)<<",\"rejected\":"<<(rejected?"true":"false")
         <<",\"metadata_unchanged\":"<<(metadata?"true":"false")
         <<",\"full_caches_unchanged\":"<<(caches?"true":"false")<<"}\n"<<std::flush;
      require(rejected && metadata && caches,name+": invalid request accepted or mutated owner/cache");
      require(bool(log),"probe log write failed");++rejections;
    };
    reject("duplicate-owner",{&a,&a},{1,2});
    reject("null-first-owner",{nullptr,&b},{1,2});reject("null-second-owner",{&a,nullptr},{1,2});
    auto saved=b.weights;b.weights.reset();reject("missing-shared-weights",{&a,&b},{1,2});b.weights=saved;
    // Legal one-past alias, never dereferenced: validation must reject identity
    // before any weight access. It retains the sole checkpoint's ownership.
    b.weights=std::shared_ptr<lfm::Safetensors>(saved,saved.get()+1);
    reject("distinct-nonnull-weights",{&a,&b},{1,2});b.weights=saved;
    const float epsilon=b.s.epsilon;b.s.epsilon=epsilon+1;
    reject("geometry-mismatch",{&a,&b},{1,2});b.s.epsilon=epsilon;
    b.trace="forbidden-probe-trace";reject("trace-enabled",{&a,&b},{1,2});b.trace.clear();
    reject("invalid-token",{&a,&b},{1,ca.shape.vocabulary});
    const int cursor=b.cursor;b.cursor=0;reject("zero-cursor",{&a,&b},{1,2});b.cursor=cursor;
    bool shifted_window=false,small_softmax=false,large_softmax=false,scratch_exceeded=false;
    auto paired=[&](const std::string& name,bool reverse=false) {
      begin_case(name);
      const int ia=argmax(sa),ib=argmax(sb);
      for(int layer=0;layer<ca.shape.layers;++layer) if(ca.attention[layer]) {
        for(const auto* r:{&a,&b}) {
          const int window=r->config.layer_specs[layer].window;
          const int start=window?std::max(0,r->cursor+1-window):0,end=r->cursor+1-start;
          shifted_window|=start>0;small_softmax|=end<=2048;large_softmax|=end>2048;
          scratch_exceeded|=r==&b && end>lfm::text::scratch_rows(a.s.context)*a.s.context;
        }
      }
      if(compact) {sa=ra.forward({ia});sb=rb.forward({ib});}
      auto got=reverse?lfm3_decode_batch::forward({&b,&a},{ib,ia}):lfm3_decode_batch::forward({&a,&b},{ia,ib});
      if(!compact) {sa=ra.forward({ia});sb=rb.forward({ib});}
      require(got.size()==2,"paired output lane count mismatch");
      compare(name+"-A",got[reverse?1:0],sa,a,ra);compare(name+"-B",got[reverse?0:1],sb,b,rb);++paired_calls;
    };
    auto single=[&](const std::string& name,bool run_b=true) {
      begin_case(name);
      Runner& active=run_b?b:a;Oracle& ref=run_b?rb:ra;
      Runner& retired=run_b?a:b;Oracle& retired_ref=run_b?ra:rb;
      auto& values=run_b?sb:sa;
      const int old_cursor=retired.cursor,old_step=retired.step;
      const auto before=cache_snapshot(retired);const int token=argmax(values);
      if(compact) values=ref.forward({token});
      auto got=lfm3_decode_batch::forward({&active},{token});
      if(!compact) values=ref.forward({token});
      const auto after=cache_snapshot(retired);
      const bool untouched=before==after && after==retired_ref.cache() &&
        retired.cursor==old_cursor && retired.step==old_step &&
        retired.cursor==retired_ref.cursor() && retired.step==retired_ref.step();
      log<<"{\"case\":"<<json_string(name+"-retired")<<",\"retired_lane\":\""<<(run_b?'A':'B')
         <<"\",\"full_cache_unchanged_and_exact\":"<<(untouched?"true":"false")<<"}\n"<<std::flush;
      require(got.size()==1 && untouched,"retired lane/cache changed");++retired_checks;
      compare(name,got[0],values,active,ref);
    };
    if(!upper) {
      single("retire-B",false);paired("reverse-lanes-resume-B",true);
      for(auto owners:{std::make_pair(&a,&ra),std::make_pair(&b,&rb)}) {
        begin_case("reset-before-standard-sequence");
        owners.first->reset();owners.second->reset();
        const bool exact=cache_snapshot(*owners.first)==owners.second->cache() &&
          owners.first->cursor==0 && owners.second->cursor()==0 && owners.first->step==0 && owners.second->step()==0;
        log<<"{\"case\":\"reset-before-standard-sequence\",\"exact_full_cache\":"<<(exact?"true":"false")<<"}\n"<<std::flush;
        require(exact,"reset cache mismatch");++resets;
      }
      prefill("replay-prefill");
      for(int i=0;i<6;++i) paired("decode-"+std::to_string(i));
      single("retire-A-0");single("retire-A-1");
      for(int i=0;i<3;++i) paired("resume-"+std::to_string(i));
    } else paired("upper-only-pair");
    while(a.cursor<capacity_a) paired("fill-A-"+std::to_string(a.cursor));
    require(a.cursor==capacity_a && b.cursor<capacity_b,"unequal boundary fixture missed");
    reject("A-at-capacity",{&a,&b},{1,2});
    while(b.cursor<capacity_b) single("fill-B-"+std::to_string(b.cursor));
    require(b.cursor==capacity_b,"B boundary fixture missed");
    reject("B-at-capacity",{&b},{1});
    const bool tiled=length_a>lfm::text::scratch_rows(capacity_a) && length_b>lfm::text::scratch_rows(capacity_b);
    log<<"{\"case\":\"fixture-coverage\",\"shifted_window\":"<<(shifted_window?"true":"false")
       <<",\"small_softmax\":"<<(small_softmax?"true":"false")<<",\"large_softmax\":"<<(large_softmax?"true":"false")
       <<",\"B_exceeded_A_serial_scratch\":"<<(scratch_exceeded?"true":"false")
       <<",\"tiled_prefill\":"<<(tiled?"true":"false")<<"}\n"<<std::flush;
    require(shifted_window,"fixture did not exercise shifted sliding-window attention");
    if(fixture=="scratch") require(scratch_exceeded,"B did not exceed A's serial attention scratch");
    if(fixture=="softmax") require(small_softmax && large_softmax,"softmax dispatch boundary missed");
    if(fixture=="tiled") require(tiled,"tiled prefill boundary missed");
    require(comparisons==expected_comparisons && paired_calls==expected_paired && retired_checks==expected_retired &&
      resets==(upper?0:2) && rejections==11,"fixture coverage/count mismatch");
    std::ofstream summary(output/"summary.json");
    summary<<"{\"status\":\"passed\",\"fixture\":"<<json_string(fixture)<<",\"comparisons\":"<<comparisons
           <<",\"reference_mode\":"<<json_string(strategy)<<",\"gpu_runner_owners\":"<<(compact?2:4)
           <<",\"expected_comparisons\":"<<expected_comparisons<<",\"paired_calls\":"<<paired_calls
           <<",\"expected_paired_calls\":"<<expected_paired<<",\"retired_checks\":"<<retired_checks
           <<",\"expected_retired_checks\":"<<expected_retired<<",\"reset_checks\":"<<resets<<",\"rejections\":"<<rejections
           <<",\"dense_batch_rows\":2,\"worker_integration\":false,\"scope\":\"finite standalone exact serial decode parity only\"}\n";
    require(bool(summary),"summary write failed");return 0;
  } catch(const std::exception& e) {
    std::ofstream summary(output/"summary.json");
    summary<<"{\"status\":\"failed\",\"error\":"<<json_string(e.what())
           <<",\"reference_mode\":"<<json_string(strategy)
           <<",\"fixture\":"<<json_string(fixture)<<",\"comparisons\":"<<comparisons
           <<",\"expected_comparisons\":"<<expected_comparisons<<",\"paired_calls\":"<<paired_calls
           <<",\"expected_paired_calls\":"<<expected_paired<<",\"retired_checks\":"<<retired_checks
           <<",\"expected_retired_checks\":"<<expected_retired<<",\"reset_checks\":"<<resets
           <<",\"rejections\":"<<rejections<<"}\n";
    std::cerr<<"decode-batch-probe: "<<e.what()<<'\n';return 1;
  }
}
