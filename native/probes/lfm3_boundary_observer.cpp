// Read-only diagnostic using the documented Compute Sanitizer callback/memory APIs.
// No CUDA API calls, patches, device writes or initializer operations in callbacks.
/* Notice for the NVIDIA Sanitizer API dependency (not a relicensing of this file):
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE LICENSE
 * AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THESE
 * LICENSED DELIVERABLES FOR ANY PURPOSE. IT IS PROVIDED "AS IS" WITHOUT EXPRESS
 * OR IMPLIED WARRANTY OF ANY KIND. NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD
 * TO THESE LICENSED DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE LICENSE
 * AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT,
 * INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING
 * FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
 * NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE
 * USE OR PERFORMANCE OF THESE LICENSED DELIVERABLES.
 * U.S. Government End Users. These Licensed Deliverables are a "commercial item"
 * as that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting of
 * "commercial computer software" and "commercial computer software
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) and is
 * provided to the U.S. Government only as a commercial end item. Consistent with
 * 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with only those
 * rights set forth herein.
 */
#include <sanitizer.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
constexpr size_t workspace_bytes=8519680, snapshot_bytes=49152;
std::string quote(const char* text) {
  std::string out="\"";
  for(const unsigned char c:std::string(text?text:"")) {
    if(c=='"'||c=='\\') {out+='\\';out+=char(c);}
    else if(c<32) {char escaped[7];std::snprintf(escaped,sizeof(escaped),"\\u%04x",unsigned(c));out+=escaped;}
    else out+=char(c);
  }
  return out+'"';
}
struct Observer {
  std::mutex lock;
  std::string directory,mode;
  FILE* log=nullptr;
  Sanitizer_SubscriberHandle subscriber{};
  uint64_t workspace=0;
  uint64_t control_buffers[2]{};
  CUcontext context=nullptr;
  unsigned candidates=0,captures=0,launches=0;
  unsigned control_buffer_count=0;
  unsigned modules=0;
  size_t module_bytes=0;
  bool failed=false,initialized=false;
  void fail(const std::string& message) {
    failed=true;if(log) {std::fprintf(log,"{\"error\":%s}\n",quote(message.c_str()).c_str());std::fflush(log);}
  }
  void snapshot(const char* stage,const Sanitizer_LaunchData& data,uint64_t source=0,bool primary=true) {
    if(candidates!=1||!workspace||context!=data.context) {fail("workspace identity missing/ambiguous/context mismatch");return;}
    if(data.stream!=nullptr) {fail("expected default CUDA stream");return;}
    if(!source)source=workspace;
    std::vector<unsigned char> host(snapshot_bytes);
    std::fprintf(log,"{\"snapshot_begin\":%s,\"grid_id\":%llu,\"function\":%s}\n",quote(stage).c_str(),(unsigned long long)data.gridId,quote(data.functionName).c_str());std::fflush(log);
    const SanitizerResult status=sanitizerMemcpyDeviceToHost(host.data(),reinterpret_cast<void*>(source),snapshot_bytes,data.hStream);
    if(status!=SANITIZER_SUCCESS) {fail("sanitizerMemcpyDeviceToHost status="+std::to_string(int(status)));return;}
    std::ofstream f(directory+"/"+stage+".bin",std::ios::binary);f.write(reinterpret_cast<char*>(host.data()),host.size());f.close();
    if(!f) {fail("snapshot file write failed");return;}
    if(primary)++captures;
    std::fprintf(log,"{\"snapshot_complete\":%s,\"bytes\":%zu,\"workspace\":%llu,\"source_address\":%llu,\"grid_id\":%llu}\n",quote(stage).c_str(),host.size(),(unsigned long long)workspace,(unsigned long long)source,(unsigned long long)data.gridId);std::fflush(log);
  }
  ~Observer() {
    if(log) {
      std::fprintf(log,"{\"observer_complete\":true,\"initialized\":%s,\"failed\":%s,\"workspace_candidates\":%u,\"control_buffer_count\":%u,\"captures\":%u,\"launches\":%u,\"module_images\":%u,\"full_acceptance\":false}\n",initialized?"true":"false",failed?"true":"false",candidates,control_buffer_count,captures,launches,modules);
      std::fclose(log);
    }
  }
};
Observer& observer() {static Observer instance;return instance;}

void SANITIZERAPI callback(void* user,Sanitizer_CallbackDomain domain,Sanitizer_CallbackId cbid,const void* cbdata) noexcept {
  auto& o=*static_cast<Observer*>(user);
  try {
    std::lock_guard<std::mutex> guard(o.lock);
    if(domain==SANITIZER_CB_DOMAIN_RESOURCE&&cbid==SANITIZER_CBID_RESOURCE_MODULE_LOADED) {
      const auto& d=*static_cast<const Sanitizer_ResourceModuleData*>(cbdata);
      if(!d.pCubin||!d.cubinSize||d.cubinSize>128*1024*1024) {
        std::fprintf(o.log,"{\"module_unavailable_or_bounded\":%llu,\"bytes\":%zu}\n",(unsigned long long)reinterpret_cast<uintptr_t>(d.module),d.cubinSize);std::fflush(o.log);return;
      }
      const std::string_view image(d.pCubin,d.cubinSize);
      const bool wanted=image.find("nvjet_sm110")!=std::string_view::npos||image.find("splitKreduce_kernel")!=std::string_view::npos||image.find("boundary_control_")!=std::string_view::npos||image.find("consume_all")!=std::string_view::npos;
      if(wanted) {
        if(o.modules>=16||o.module_bytes+d.cubinSize>256*1024*1024) {o.fail("selected module capture budget exceeded");return;}
        const std::string file="module-"+std::to_string(o.modules++)+".cubin";
        std::ofstream f(o.directory+"/"+file,std::ios::binary);f.write(d.pCubin,d.cubinSize);f.close();
        if(!f) {o.fail("module image write failed");return;}
        o.module_bytes+=d.cubinSize;
        std::fprintf(o.log,"{\"module_image\":%s,\"module\":%llu,\"bytes\":%zu}\n",quote(file.c_str()).c_str(),(unsigned long long)reinterpret_cast<uintptr_t>(d.module),d.cubinSize);std::fflush(o.log);
      }
    } else if(domain==SANITIZER_CB_DOMAIN_RESOURCE&&cbid==SANITIZER_CBID_RESOURCE_DEVICE_MEMORY_ALLOC) {
      const auto& d=*static_cast<const Sanitizer_ResourceMemoryData*>(cbdata);
      if(d.size==workspace_bytes) {
        ++o.candidates;o.workspace=d.address;o.context=d.context;
        std::fprintf(o.log,"{\"workspace_allocation\":%llu,\"bytes\":%llu,\"candidates\":%u}\n",(unsigned long long)d.address,(unsigned long long)d.size,o.candidates);std::fflush(o.log);
        if(o.candidates!=1)o.fail("multiple matching allocations");
      } else if(o.mode=="control"&&d.size==snapshot_bytes) {
        if(o.control_buffer_count>=2)o.fail("ambiguous control buffer allocation");
        else o.control_buffers[o.control_buffer_count++]=d.address;
      }
    } else if(domain==SANITIZER_CB_DOMAIN_LAUNCH&&cbid==SANITIZER_CBID_LAUNCH_BEGIN) {
      const auto& d=*static_cast<const Sanitizer_LaunchData*>(cbdata);const std::string name=d.functionName?d.functionName:"";
      ++o.launches;
      std::fprintf(o.log,"{\"launch_begin\":%s,\"grid_id\":%llu,\"module\":%llu,\"stream\":%llu,\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"cluster\":[%u,%u,%u]}\n",quote(d.functionName).c_str(),(unsigned long long)d.gridId,(unsigned long long)reinterpret_cast<uintptr_t>(d.module),(unsigned long long)reinterpret_cast<uintptr_t>(d.stream),d.gridDim_x,d.gridDim_y,d.gridDim_z,d.blockDim_x,d.blockDim_y,d.blockDim_z,d.clusterDim_x,d.clusterDim_y,d.clusterDim_z);std::fflush(o.log);
      const bool control=o.mode=="control";
      if(control&&o.control_buffer_count!=2) {o.fail("control destination allocations missing/ambiguous");return;}
      const bool producer=control?name.find("boundary_control_fill")!=std::string::npos:name.rfind("nvjet_",0)==0;
      const bool reducer=control?name.find("boundary_control_reduce")!=std::string::npos:name.find("splitKreduce_kernel")!=std::string::npos;
      const bool consumer=control?name.find("boundary_control_consume")!=std::string::npos:name.find("consume_all")!=std::string::npos;
      if(producer) {if(o.captures!=0)o.fail("producer order/count mismatch");else o.snapshot("before_producer",d);}
      else if(reducer) {
        if(o.captures!=1)o.fail("reducer order/count mismatch");
        else {o.snapshot("before_reducer",d);if(control)o.snapshot("before_reducer_output",d,o.control_buffers[0],false);}
      }
      else if(consumer) {
        if(o.captures!=2)o.fail("consumer order/count mismatch");
        else {o.snapshot("before_consumer",d);if(control) {o.snapshot("before_consumer_output",d,o.control_buffers[0],false);o.snapshot("before_consumer_sink",d,o.control_buffers[1],false);}}
      }
      else o.fail("unexpected kernel in exact boundary fixture");
    }
  } catch(const std::exception& e) {o.fail(e.what());} catch(...) {o.fail("callback exception");}
}

int initialize() {
  auto& o=observer();const char* dir=std::getenv("LFM_BOUNDARY_OUTPUT");const char* mode=std::getenv("LFM_BOUNDARY_MODE");
  if(!dir||!mode)return 0;
  o.directory=dir;o.mode=mode;
  o.log=std::fopen((o.directory+"/observer.jsonl").c_str(),"wx");
  if(!o.log) {std::fprintf(stderr,"boundary observer: cannot create log\n");return 0;}
  if(o.mode!="control"&&o.mode!="gemm") {o.fail("invalid observer mode");return 0;}
  auto result=sanitizerSubscribe(&o.subscriber,callback,&o);
  std::fprintf(o.log,"{\"subscribe_result\":%d,\"mode\":%s,\"read_only\":true,\"boundary_synchronization\":true}\n",int(result),quote(mode).c_str());std::fflush(o.log);
  if(result!=SANITIZER_SUCCESS) {o.fail("subscription failed");return 0;}
  for(const auto domain:{SANITIZER_CB_DOMAIN_RESOURCE,SANITIZER_CB_DOMAIN_LAUNCH}) {
    result=sanitizerEnableDomain(1,o.subscriber,domain);
    if(result!=SANITIZER_SUCCESS) {o.fail("enable domain failed status="+std::to_string(int(result)));return 0;}
  }
  o.initialized=true;
  return 0;
}
int initialized=initialize();
} // namespace
