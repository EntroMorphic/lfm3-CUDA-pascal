"""Host-only tagged cancellation and the real worker scheduler with fake runners.

No CUDA kernels execute here. Numerical parity/real memory validation is separate.
"""
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASE = runpy.run_path(str(Path(__file__).with_name("test_text_cancellation.py")))["SOURCE"]
CASES = r'''
    } else if (mode == "pair_targeting") {
        Pipe p; Reader r(p.fd[0]);
        auto token=lfm::text::batch_cancellation_token;
        send(p.fd[1],token(5,1)); send(p.fd[1],token(5,0));
        send(p.fd[1],token(5,0)); send(p.fd[1],5);
        assert((r.requested_pair(5)==std::array<bool,2>{true,true}));
        assert((r.requested_pair(5)==std::array<bool,2>{true,true}));
        send(p.fd[1],token(5,0)); send(p.fd[1],token(7,1));
        assert((r.requested_pair(6)==std::array<bool,2>{false,false}));
        assert((r.requested_pair(7)==std::array<bool,2>{false,false}));
        send(p.fd[1],token(8,0)); assert(!r.requested(8)); // valid foreign tag
        send(p.fd[1],9); assert(r.requested(9));
        invalid([&] { r.requested_pair(9); });
    } else if (mode == "pair_independent") {
        for(unsigned first: {0u,1u}) {
            Pipe p; Reader r(p.fd[0]);
            send(p.fd[1],lfm::text::batch_cancellation_token(1,first));
            auto bits=r.requested_pair(1);assert(bits[first] && !bits[1-first]);
            send(p.fd[1],lfm::text::batch_cancellation_token(1,1-first));
            bits=r.requested_pair(1);assert(bits[0] && bits[1]);
        }
    } else if (mode == "pair_partial") {
        for(size_t split=1;split<8;++split) {
            Pipe p; Reader r(p.fd[0]);
            auto bytes=frame(lfm::text::batch_cancellation_token(2,1));
            assert(write(p.fd[1],bytes.data(),split)==ssize_t(split));
            assert(!r.requested_pair(2)[1]);
            assert(write(p.fd[1],bytes.data()+split,8-split)==ssize_t(8-split));
            send(p.fd[1],lfm::text::batch_cancellation_token(2,0));
            auto bits=r.requested_pair(2);assert(bits[0] && bits[1]);
        }
        Pipe p; Reader r(p.fd[0]);
        auto future=frame(lfm::text::batch_cancellation_token(3,0));
        assert(write(p.fd[1],future.data(),3)==3);r.requested_pair(2);
        assert(write(p.fd[1],future.data()+3,5)==5);
        assert(!r.requested_pair(3)[0]); // future partial is not queued either
    } else if (mode == "pair_malformed") {
        for(auto id: {UINT64_C(0),UINT64_C(1)<<32,UINT64_C(3)<<32|1,
                      UINT64_C(1)<<32|UINT64_C(0x80000000),UINT64_MAX}) {
            Pipe p; Reader r(p.fd[0]);send(p.fd[1],id);
            ioerror([&] { r.requested_pair(1); },EPROTO);
            ioerror([&] { r.requested(2); },EPROTO);
        }
    } else if (mode == "pair_disabled") {
        Reader r; assert((r.requested_pair(1)==std::array<bool,2>{false,false}));
        invalid([&] { r.requested_pair(0); });
        invalid([&] { r.requested_pair(Reader::max_rpc_id+1); });
        invalid([&] { lfm::text::batch_cancellation_token(0,0); });
        invalid([&] { lfm::text::batch_cancellation_token(1,2); });
        assert(lfm::text::batch_cancellation_token(Reader::max_rpc_id,1)==UINT64_C(0x27fffffff));
    } else if (mode == "pair_bounded") {
        Pipe p; Reader r(p.fd[0]);
        for(int kind: {1,2,3}) {
            fake_reads=0;fake_bytes=0;fake_mode=kind;
            auto bits=r.requested_pair(1);assert(!bits[0] && !bits[1]);
            fake_mode=0;assert(fake_reads==int(Reader::read_attempt_budget));
        }
        send(p.fd[1],lfm::text::batch_cancellation_token(1,1));
        assert(r.requested_pair(1)[1]);
    } else if (mode == "pair_backpressure") {
        Pipe p; Reader r(p.fd[0]);auto bytes=frame(1);
        while(write(p.fd[1],bytes.data(),8)==8) {}
        assert(errno==EAGAIN || errno==EWOULDBLOCK);
        int remaining=0;ioctl(p.fd[0],FIONREAD,&remaining);
        while(remaining) {
            r.requested_pair(1);int next=0;ioctl(p.fd[0],FIONREAD,&next);
            assert(remaining-next==std::min(remaining,int(Reader::drain_budget)));remaining=next;
        }
        send(p.fd[1],lfm::text::batch_cancellation_token(1,0));
        send(p.fd[1],lfm::text::batch_cancellation_token(1,1));
        auto bits=r.requested_pair(1);assert(bits[0] && bits[1]);
'''
READER_SOURCE = BASE.replace("#include <lfm/text_cancellation.hpp>",
                             "#include <lfm/text_batch_cancellation.hpp>")
READER_SOURCE = READER_SOURCE.replace("    } else { assert(false); }", CASES + "    } else { assert(false); }")

STUBS = r'''
#include <lfm/model_config.hpp>
#include <lfm/text_sampling.hpp>
#include <lfm/text_sessions.hpp>
#include <lfm/text_batch_cancellation.hpp>
#include <lfm/text_serving_limits.hpp>
#include <lfm/lfm3_text_request.hpp>
#include <algorithm>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <functional>
namespace lfm { struct Safetensors { explicit Safetensors(const std::string&) {} }; }
using bf16=float;
float __float2bfloat16_rn(float x) {return x;}
int forwards=0,pairs=0,fail_at=0;
std::function<void()> on_init,on_forward;
struct Runner {
 int cursor=0;
 Runner(const std::string&, const lfm::ModelConfig&, std::shared_ptr<lfm::Safetensors>) {if(on_init) on_init();}
 void reset() {cursor=0;}
 std::vector<float> forward(const std::vector<int>& tokens) {
   ++forwards;if(fail_at==forwards) throw std::runtime_error("fake execution error");
   cursor+=tokens.size();if(on_forward) on_forward();
   return {0,1,2,3,4,5,6,7};
 }
};
namespace lfm3_decode_batch {
std::vector<std::vector<float>> forward(const std::vector<Runner*>& owners,const std::vector<int>& tokens) {
 assert(owners.size()==2);++pairs;
 return {owners[0]->forward({tokens[0]}),owners[1]->forward({tokens[1]})};
}
}
std::vector<int> ids(const lfm::Json& req,const std::string& key,int vocab) {
 std::vector<int> result;for(const auto& x:req.at(key).array()) {
   const int id=x.integer();if(id>=vocab) throw std::runtime_error("bad token");result.push_back(id);
 }return result;
}
void print_ids(std::ostream& out,const std::vector<int>& ids) {
 out<<'[';for(size_t i=0;i<ids.size();++i) {if(i) out<<',';out<<ids[i];}out<<']';
}
'''
HARNESS = r'''
int main(int argc,char** argv) {
 assert(argc==3);std::string mode=argv[1];
 const std::string trace_root=mode=="trace"?argv[2]:"";
 if(!trace_root.empty()) assert(std::filesystem::create_directory(trace_root));
 lfm::ModelConfig cfg;cfg.shape.vocabulary=8;
 std::map<std::string,Session> sessions;
 sessions.emplace("A",Session{16,{},{},{},{}});sessions.emplace("B",Session{16,{},{},{},{}});
 lfm::text::SessionLedger ledger(2,32);ledger.create("A",16);ledger.create("B",16);
 std::shared_ptr<lfm::Safetensors> weights;
 int fd[2];assert(pipe2(fd,O_NONBLOCK|O_CLOEXEC)==0);
 lfm::text::CancellationReader reader(fd[0]);
 auto cancel=[&](unsigned lane) {
   auto id=lfm::text::batch_cancellation_token(1,lane);
   unsigned char frame[8];for(int i=0;i<8;++i) frame[i]=id>>(8*i);
   assert(write(fd[1],frame,8)==8);
 };
 if(mode=="cancel_init") on_init=[&] {cancel(0);};
 if(mode=="both_cancel") {cancel(0);cancel(1);}
 if(mode=="cancel_prefill_peer") on_forward=[&] {if(forwards==1) cancel(1);};
 if(mode=="execution_error") fail_at=3;
 if(mode=="error_after_commit") fail_at=3;
 int inits=0;
 if(mode=="cancel_peer_init") on_init=[&] {if(++inits==2) cancel(0);};
 if(mode=="reuse" || mode=="cancel_history") {
   for(auto name: {"A","B"}) {
     auto lease=ledger.begin(name,{1},1);lease.commit({7},{1});
     auto& s=sessions.at(name);s.runner=std::make_unique<Runner>("unused",cfg,weights);
     s.runner->cursor=1;s.consumed={1};s.last_scores={0,1,2,3,4,5,6,7};
   }
   if(mode=="cancel_history") cancel(0);
 }
 std::string second="B",extra="",count="3";
 if(mode=="duplicate") second="A";
 if(mode=="bad_metadata") extra=",\"top_k\":9";
 if(mode=="bad_schema") extra=",\"rpc_id\":1";
 if(mode=="bad_capacity") count="16";
 if(mode=="trace") extra=",\"trace_sampling\":true,\"return_logits\":true";
 const std::string first_count=mode=="error_after_commit"?"1":mode=="ragged"?"2":"3";
 const std::string first_input=(mode=="reuse" || mode=="cancel_history")?"[1,7]":"[1]";
 const std::string second_input=(mode=="reuse" || mode=="cancel_history")?"[1,7]":"[2]";
 auto request=lfm::Json::parse("{\"op\":\"generate_batch\",\"rpc_id\":1,\"lanes\":["
   "{\"session_id\":\"A\",\"input_ids\":"+first_input+",\"max_new_tokens\":"+first_count+(mode=="trace"?extra:"")+"},"
   "{\"session_id\":\""+second+"\",\"input_ids\":"+second_input+",\"max_new_tokens\":"+count+extra+"}]}");
 std::ostringstream out;auto* old=std::cout.rdbuf(out.rdbuf());bool failed=false;
 // live = the fake model's whole vocabulary: it has no output rows beyond its tokens (G6 bound is a no-op here)
 TextFrontend text{"unused-tokenizer.json",cfg.shape.vocabulary,nullptr};
 try {generate_batch(request,1,"unused",cfg,{},cfg.shape.vocabulary,mode=="eos"?std::vector<int>{7}:std::vector<int>{},trace_root,weights,sessions,ledger,reader,text,std::cout);}
 catch(const std::exception&) {failed=true;}
 std::cout.rdbuf(old);
 if(mode=="duplicate" || mode=="bad_metadata" || mode=="bad_schema" || mode=="bad_capacity") {
   assert(failed && forwards==0 && !weights);
   for(auto name: {"A","B"}) {auto s=ledger.inspect(name);assert(!s.inflight && s.transcript.empty());assert(!sessions.at(name).runner);}
 } else if(mode=="execution_error") {
   assert(failed);for(auto name: {"A","B"}) {assert(!ledger.inspect(name).inflight);assert(!ledger.inspect(name).cache_valid);assert(!sessions.at(name).runner);}
 } else if(mode=="error_after_commit") {
   assert(failed && ledger.inspect("A").cache_valid && !ledger.inspect("A").inflight);
   assert(sessions.at("A").runner && sessions.at("A").consumed==std::vector<int>{1});
   assert(!ledger.inspect("B").cache_valid && !sessions.at("B").runner);
 } else {
   assert(!failed);auto result=lfm::Json::parse(out.str());const auto& lanes=result.at("results").array();
   if(mode=="normal" || mode=="trace") {
     assert(pairs==2 && result.at("paired_decode_calls").integer()==2);
     for(const auto& lane:lanes) {assert(lane.at("consumed_tokens").integer()==3);assert(lane.at("generated_ids").array().size()==3);}
     if(mode=="trace") for(size_t i=0;i<2;++i) {
       assert(lanes[i].at("sampling_trace").string()=="1/lane"+std::to_string(i));
       assert(lanes[i].at("last_logits").array().size()==8);
       for(int step=0;step<3;++step) assert(std::filesystem::exists(trace_root+"/1/lane"+std::to_string(i)+"/"+std::to_string(step)+".logits.bf16"));
     }
   } else if(mode=="both_cancel") {
     assert(!weights && forwards==0 && pairs==0);
     for(const auto& lane:lanes) {assert(lane.at("cancelled").boolean());assert(!lane.at("cache_valid").boolean());}
   } else if(mode=="reuse") {
     assert(pairs==2 && forwards==6);
     for(const auto& lane:lanes) {assert(lane.at("reused_tokens").integer()==1);assert(lane.at("consumed_tokens").integer()==4);}
   } else if(mode=="ragged") {
     assert(pairs==1 && forwards==5);
     assert(lanes[0].at("consumed_tokens").integer()==2 && lanes[1].at("consumed_tokens").integer()==3);
   } else if(mode=="eos") {assert(pairs==0 && forwards==2);for(const auto& lane:lanes) assert(lane.at("consumed_tokens").integer()==1);}
   else {
     const int cancelled=mode=="cancel_prefill_peer"?1:0;
     assert(lanes[cancelled].at("cancelled").boolean());assert(!lanes[1-cancelled].at("cancelled").boolean());
     assert(lanes[cancelled].at("consumed_tokens").integer()==0);
     assert(ledger.inspect(cancelled?"A":"B").cache_valid);
     assert(forwards==(mode=="cancel_peer_init"?4:3) && pairs==0);
     if(mode=="cancel_history") {
       const auto prior=ledger.inspect("A");assert((prior.transcript==lfm::text::SessionLedger::Tokens{1,7}));
       assert((prior.consumed==lfm::text::SessionLedger::Tokens{1}));assert(!prior.cache_valid);
     }
   }
 }
 close(fd[0]);close(fd[1]);
}
'''


@pytest.fixture(scope="module", params=[False, True], ids=["strict", "ubsan"])
def host_binaries(tmp_path_factory, request):
    if sys.platform != "linux" or not shutil.which("g++"):
        pytest.skip("Linux and g++ required")
    tmp = tmp_path_factory.mktemp("text_batch_cancellation")
    worker = (ROOT / "native/src/lfm3_worker.cu").read_text()
    body = "namespace {" + worker.split("namespace {", 1)[1].split("} // namespace", 1)[0] + "}\n"
    result = {}
    for name, source in [("reader", READER_SOURCE), ("worker", STUBS + body + HARNESS)]:
        path = tmp / f"{name}.cpp"
        path.write_text(source)
        flags = ["-fsanitize=undefined", "-fno-sanitize-recover=all"] if request.param else []
        subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pedantic",
                        *flags, "-I", str(ROOT / "native/include"), str(path),
                        *(["-Wl,--wrap=read"] if name == "reader" else []),
                        "-o", str(tmp / name)], check=True)
        result[name] = tmp / name
    return result


@pytest.mark.parametrize("case", ["pair_targeting", "pair_independent", "pair_partial",
                                  "pair_malformed", "pair_disabled", "pair_bounded", "pair_backpressure"])
def test_pair_reader(host_binaries, case):
    subprocess.run([str(host_binaries["reader"]), case], check=True, timeout=10)


@pytest.mark.parametrize("case", ["normal", "trace", "both_cancel", "eos", "reuse", "ragged", "cancel_init", "cancel_prefill_peer",
                                  "cancel_peer_init", "cancel_history", "execution_error", "error_after_commit",
                                  "duplicate", "bad_metadata", "bad_schema", "bad_capacity"])
def test_worker_host_scheduler(host_binaries, case, tmp_path):
    subprocess.run([str(host_binaries["worker"]), case, str(tmp_path / "traces")], check=True, timeout=10)
