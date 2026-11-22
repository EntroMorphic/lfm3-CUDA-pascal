// liquid-lfm3-server: one executable for LFM3 text serving. Loopback HTTP (lfm/lfm3_http.hpp, a port of
// tools/lfm3_server.py), the native text frontend (lfm/lfm3_text.hpp) and the engine (WorkerCore, the
// resident protocol of liquid-lfm3-worker, driven in-process by one engine-owner thread).
//
//   liquid-lfm3-server --experimental MODEL_DIRECTORY [--token-file FILE] [--port N] [--request-timeout S]
//                      [--socket-timeout S] [--session-idle-timeout S] [--engine-timeout S] [--total-capacity N]
//                      [--max-sessions N] [--queue-size N] [--handlers N] [--speech-rows N] [--image-cache-mb N]
// --image-cache-mb (0..65536, default 512; 0: none): validation/image-cache-v1's budget for each side's image cache (the
// front end's token counts, the engine's encoder rows on the GPU), keyed by an image's full bytes
// --speech-rows (1..16, the most validated; default 1 on the GTX 1080, validation/speech-v1 rule 12: with text at its
// limits at a total capacity of 32,768, not even one row fits in 7,740 MiB, so the total capacity is lowered (below);
// the reference's default was 8): /speech requests generated at once in the engine's speech batch
// (validation/speech-batch-v1); more wait in the queue (--queue-size of them).
// --total-capacity: default 32,768 (serve-v1 rule 11), or, when the engine serves speech, kSpeechTotalCapacity less
// kSpeechCapacityPerRow for each speech row past the first, at least kSpeechMinTotalCapacity (speech-v1 rule 12: the
// speech weights, one row and audio in at its limit leave room for 29,696; a row costs about 102 MiB, 1,024 tokens of
// capacity about 52 MiB). Without --handlers, the server takes
// max(8, 2 x speech rows) handlers: at most half of them stream at once (Service::speech_stream), so every row can.
// The bearer token comes from --token-file (an owned mode-0600 file), or LFM3_SERVER_TOKEN, or else the default file
// ~/.config/liquid-lfm3/token ($XDG_CONFIG_HOME/liquid-lfm3/token), created with a random token on first start and
// reused after; tools/lfm3_token.py reads the same file for clients. --port 0
// picks a free port. Once listening, one line {"ready":true,"port":N,"total_capacity":N,"speech_rows":N} is written to
// stdout (speech_rows 0: speech not served, as when --engine-timeout is below 300 s). SIGTERM or
// SIGINT stops admission, cancels running generations at their next token boundary, and exits 0; a
// second signal exits at once (status 1). An engine call longer than --engine-timeout (default 300 s,
// the Python worker client's) fails the server closed: every request then gets 503; restart it.
#define LFM3_WORKER_CORE_ONLY
#include "lfm3_worker.cu"
#include <lfm/lfm3_http.hpp>

#include <csignal>
#include <cstdlib>
#include <malloc.h>
#include <pthread.h>

namespace {
// validation/speech-v1 rule 12: the largest multiple of 1,024 at which serve-v1 rule 11's configurations, scaled, pass
// beside one speech row on the GTX 1080, audio in at the request limit (a 98 s clip) among them: 29,696
// (validation/red-team-2: with the clip the rule run used, 87 s, 30,720 had passed; with the full clip it peaks at
// 7,759 MiB).
constexpr int kSpeechTotalCapacity=29696;
// Each further speech row (red-team review: more rows beside the one-row default could exhaust the card and fail the
// service closed), down to kSpeechMinTotalCapacity at 15 and 16 rows (measured at 1, 8 and 16 rows,
// validation/red-team-2).
constexpr int kSpeechCapacityPerRow=2048;
constexpr int kSpeechMinTotalCapacity=1024;

class CoreBackend final : public lfm::http::Backend {
 public:
  explicit CoreBackend(const std::string& directory) {
    if(::pipe2(pipe_,O_NONBLOCK|O_CLOEXEC)!=0) throw std::runtime_error("cancellation pipe failed");
    core_=std::make_unique<WorkerCore>(directory,"",false,pipe_[0]);
  }
  WorkerCore& core() {return *core_;}
  std::string call(const std::string& line) override {
    if(closed_) throw std::runtime_error("engine closed");
    std::ostringstream out;
    if(!core_->handle(line,false,out)) throw std::runtime_error("engine shut down");
    return out.str();
  }
  std::string call_streaming(const std::string& line,const Progress& progress) override {
    if(closed_) throw std::runtime_error("engine closed");
    struct Hook {
      WorkerCore& core;
      ~Hook() {core.progress=nullptr;}
    } hook{*core_};
    core_->progress=progress;
    std::ostringstream out;
    if(!core_->handle(line,false,out)) throw std::runtime_error("engine shut down");
    return out.str();
  }
  int cancellation_fd() const override {return pipe_[1];}
  // The engine may hold a faulted CUDA context: refuse further calls, and let process exit release it.
  void close() override {closed_=true;}
 private:
  int pipe_[2]={-1,-1};
  std::unique_ptr<WorkerCore> core_;
  bool closed_=false;
};

int whole(const std::string& option,const std::string& value) {
  if(value.empty() || value.size()>9) throw std::runtime_error(option+" must be an integer");
  for(char c:value) if(c<'0' || c>'9') throw std::runtime_error(option+" must be an integer");
  return std::stoi(value);
}
double real(const std::string& option,const std::string& value) {
  size_t used=0;double v=0;
  try {v=std::stod(value,&used);} catch(...) {throw std::runtime_error(option+" must be a number");}
  if(used!=value.size()) throw std::runtime_error(option+" must be a number");
  return v;
}

}  // namespace

int main(int argc,char** argv) try {
  // A fixed mmap threshold (validation/audio-assurance-v1, the soak's memory finding): glibc otherwise raises it to
  // 32 MiB after the first large free, so each unstreamed /speech reply's MB-sized WAV, base64 and JSON strings come
  // from an arena heap they fragment; over a 2 h soak RSS grew ~530 MiB without a bound. With the threshold fixed,
  // every buffer of 128 KiB or more is its own mapping, returned to the system when freed (600 unstreamed replies:
  // +4.7 MiB, flat). Allocation only: no output changes. MALLOC_MMAP_THRESHOLD_ in the environment still wins.
  if(!std::getenv("MALLOC_MMAP_THRESHOLD_") && mallopt(M_MMAP_THRESHOLD,128*1024)!=1)
    throw std::runtime_error("mallopt(M_MMAP_THRESHOLD) failed");
  if(argc<3 || argc%2!=1 || std::string(argv[1])!="--experimental")
    throw std::runtime_error("usage: liquid-lfm3-server --experimental MODEL_DIRECTORY [--token-file FILE] [--port N] ...");
  const std::string directory=argv[2];
  std::string token_path;int port=8080,speech_rows=1;
  lfm::http::Settings settings;
  std::set<std::string> seen;
  for(int i=3;i<argc;i+=2) {
    const std::string option=argv[i],value=argv[i+1];
    if(!seen.insert(option).second) throw std::runtime_error("duplicate option "+option);
    if(option=="--token-file") token_path=value;
    else if(option=="--port") port=whole(option,value);
    else if(option=="--request-timeout") settings.request_timeout=real(option,value);
    else if(option=="--socket-timeout") settings.socket_timeout=real(option,value);
    else if(option=="--session-idle-timeout") settings.session_idle_timeout=real(option,value);
    else if(option=="--engine-timeout") settings.engine_timeout=real(option,value);
    else if(option=="--total-capacity") settings.total_capacity=whole(option,value);
    else if(option=="--max-sessions") settings.max_sessions=whole(option,value);
    else if(option=="--queue-size") settings.queue_size=whole(option,value);
    else if(option=="--handlers") settings.handlers=whole(option,value);
    else if(option=="--speech-rows") speech_rows=whole(option,value);
    else if(option=="--image-cache-mb") settings.image_cache_mb=whole(option,value);
    else throw std::runtime_error("unsupported option "+option);
  }
  std::string note;
  std::string token=lfm::http::startup_token(token_path,note);
  if(!note.empty()) std::cerr<<"liquid-lfm3-server: "<<note<<"\n";
  ::unsetenv("LFM3_SERVER_TOKEN");
  if(speech_rows<1 || speech_rows>16) throw std::runtime_error("--speech-rows must be 1..16");
  if(!seen.count("--handlers")) settings.handlers=std::max(8,2*speech_rows);
  settings.validate();
  if(port<0 || port>65535) throw std::runtime_error("port must be an integer in 0..65535");

  // Signals go to one waiting thread, never to handler or engine threads.
  sigset_t signals;
  sigemptyset(&signals);sigaddset(&signals,SIGTERM);sigaddset(&signals,SIGINT);
  if(pthread_sigmask(SIG_BLOCK,&signals,nullptr)!=0) throw std::runtime_error("cannot block signals");
  std::signal(SIGPIPE,SIG_IGN);

  CoreBackend backend(directory);
  backend.core().set_speech_rows(speech_rows);
  // Speech is served when the engine supports it and the engine timeout is one /speech accepts (lfm3_http.hpp
  // kSpeechMinEngineTimeout); the default capacity and the ready line follow what is served.
  const bool speech_served=backend.core().speech_supported() && settings.engine_timeout>=lfm::http::kSpeechMinEngineTimeout;
  if(!seen.count("--handlers") && !speech_served) settings.handlers=8;  // the default above assumed speech rows
  if(!seen.count("--total-capacity") && speech_served) {
    settings.total_capacity=std::max(kSpeechMinTotalCapacity,kSpeechTotalCapacity-kSpeechCapacityPerRow*(speech_rows-1));
    settings.validate();
  }
  backend.core().set_image_cache_mb(settings.image_cache_mb);
  lfm::http::Policy policy;
  policy.vocabulary=backend.core().vocabulary();
  policy.suppress=backend.core().suppressed_tokens();
  policy.stops=backend.core().stop_tokens();
  policy.speech=backend.core().speech_supported();
  policy.audio_input=backend.core().audio_input_supported();
  policy.image_input=backend.core().image_input_supported();
  policy.speech_rows=backend.core().speech_rows();
  policy.image_handoff=true;  // the engine runs in this process (image-queue-v1 part 2)
  for(auto* ids:{&policy.suppress,&policy.stops}) {
    std::sort(ids->begin(),ids->end());ids->erase(std::unique(ids->begin(),ids->end()),ids->end());
  }
  for(int id:policy.stops)
    if(std::binary_search(policy.suppress.begin(),policy.suppress.end(),id)) throw std::runtime_error("conflicting text generation policy");
  const auto& tokenizer=backend.core().tokenizer();  // loaded and checked before the first request

  auto service=std::make_unique<lfm::http::Service>(backend,tokenizer,policy,settings);
  auto server=std::make_unique<lfm::http::Server>(*service,token,port);
  token.assign(token.size(),'\0');
  // First SIGTERM/SIGINT: orderly shutdown. A second one: exit now.
  std::thread waiter([&] {
    int received=0;
    sigwait(&signals,&received);
    server->stop();
    sigwait(&signals,&received);
    std::_Exit(1);
  });
  std::cout<<"{\"ready\":true,\"port\":"<<server->port()<<",\"total_capacity\":"<<settings.total_capacity
           <<",\"speech_rows\":"<<(speech_served?policy.speech_rows:0)<<"}\n"<<std::flush;
  server->serve();
  // Admission is closed: fail queued work, cancel running generations at their next token boundary,
  // wait for the engine owner, then give handlers one socket timeout to deliver their replies.
  service->close();
  server->drain(lfm::http::seconds(settings.socket_timeout));
  const int status=service->cleanup_failed()?1:0;
  std::cout<<std::flush;
  // Handler threads may still be writing replies to slow peers; they own no engine state. Exit without
  // running destructors under them.
  std::_Exit(status);
} catch(const std::exception& e) {std::cerr<<"lfm3-server: "<<e.what()<<"\n";return 1;}
