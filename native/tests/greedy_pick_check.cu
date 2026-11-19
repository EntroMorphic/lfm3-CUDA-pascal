// The greedy choice on the card (lfm/greedy_pick.cuh) against the host's greedy pick (lfm3_main.cu's `greedy`:
// ascending ids, strict '>', blocked ids and ids at or above `live` skipped) and its refusal of non-finite logits,
// on the card: random logits with many ties (including -0 against +0), blocked ids from none to all but one, `live`
// below the vocabulary, and a NaN or an infinity planted anywhere in the output layer.
//   greedy_pick_check        exit 0 if every case agrees
#include <lfm/greedy_pick.cuh>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
namespace {
void check(cudaError_t e) { if (e != cudaSuccess) { std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); std::exit(2); } }
unsigned long long state = 0x2545F4914F6CDD1Dull;
unsigned next() { state = state * 6364136223846793005ull + 1442695040888963407ull; return unsigned(state >> 33); }
}
int main() {
  using namespace lfm::greedy;
  const int vocabulary = 128000;
  float* dl; unsigned char* db; GreedyPart* dp; int* dout;
  check(cudaMalloc(&dl, vocabulary * 4)); check(cudaMalloc(&db, vocabulary)); check(cudaMalloc(&dp, kGreedyBlocks * sizeof(GreedyPart)));
  check(cudaMalloc(&dout, 8));
  int cases = 0, failures = 0;
  for (int trial = 0; trial < 400; ++trial) {
    const int live = trial % 3 == 0 ? vocabulary : 125033 - int(next() % 1000);
    std::vector<float> l(vocabulary); std::vector<unsigned char> b(live, 0);
    const int levels = 1 + next() % (trial % 4 == 0 ? 3 : 2000);  // few levels: many ties
    for (auto& v : l) { const int k = next() % levels; v = k == 0 ? (next() & 1 ? -0.f : 0.f) : float(k) * 0.25f - 100.f; }
    const int density = trial % 5;  // 0: none blocked ... 4: all but a few
    for (int i = 0; i < live; ++i) b[i] = density == 4 ? (next() % 1000 != 0) : (next() % 4 < unsigned(density));
    if (trial == 7) for (int i = 0; i < live; ++i) b[i] = 1;  // every id blocked: id -1
    int planted = -1;
    if (trial % 7 == 3) { planted = next() % vocabulary; l[planted] = trial % 2 ? NAN : (trial % 3 ? INFINITY : -INFINITY); }
    int want = -1; bool bad = false;  // the host's rule
    for (int i = 0; i < vocabulary; ++i) bad |= !std::isfinite(l[i]);
    for (int t = 0; t < live; ++t) if (!b[t] && (want < 0 || l[t] > l[want])) want = t;
    check(cudaMemcpy(dl, l.data(), vocabulary * 4, cudaMemcpyHostToDevice)); check(cudaMemcpy(db, b.data(), live, cudaMemcpyHostToDevice));
    greedy_pick<float><<<kGreedyBlocks, kGreedyThreads>>>(dl, vocabulary, db, live, dp);
    greedy_finish<<<1, kGreedyThreads>>>(dp, kGreedyBlocks, dout);
    check(cudaGetLastError());
    int got[2]; check(cudaMemcpy(got, dout, 8, cudaMemcpyDeviceToHost));
    ++cases;
    const bool ok = bool(got[1]) == bad && (bad || got[0] == want);  // with a non-finite logit the step is refused whatever the id
    if (!ok) { ++failures; std::printf("FAIL trial %d: card id %d flag %d, host id %d flag %d (planted %d)\n", trial, got[0], got[1], want, int(bad), planted); }
  }
  std::printf("greedy_pick_check: %d cases, %d failed: %s\n", cases, failures, failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
