// validation/image-cache-v1 rule 3 (host): lfm::LruBytes and lfm::LeasedMap under their contracts. Exits nonzero with a
// message on the first violation. Built by tests/test_lru_bytes.py (plain, ASan+UBSan; TSan for the threaded part).
// GCC's mismatched-new-delete warning cannot see that the operator new below is malloc's (the warning fires where the
// containers' code is inlined, so it is off for the whole file)
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
#include <lfm/leased_map.hpp>
#include <lfm/lru_bytes.hpp>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <random>
#include <string>
#include <thread>
#include <vector>

#define CHECK(c)                                                    \
  do {                                                              \
    if (!(c)) {                                                     \
      std::fprintf(stderr, "FAILED %s:%d %s\n", __FILE__, __LINE__, #c); \
      std::exit(1);                                                 \
    }                                                               \
  } while (0)

// Every byte operator new hands out (validation/image-cache-v1 red-team: what holding a key really costs). The keys are
// whole images' base64 text, up to 16 MiB each: a copy of one is the size of the entry itself. Counted in the plain and
// ASan+UBSan builds; TSan's runtime defines operator new itself (it cannot be replaced there), so that build checks
// the rest, its threads.
#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define LFM_NO_COUNTING 1
#endif
#endif
#ifndef LFM_NO_COUNTING
static std::atomic<size_t> g_allocated{0};
void* operator new(size_t n) {
  g_allocated += n;
  if (void* p = std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
#define ALLOCATED() g_allocated.load()
#define CHECK_ALLOCATED(c) CHECK(c)
#else
#define ALLOCATED() size_t(0)
#define CHECK_ALLOCATED(c) ((void)0)
#endif

int main() {
  // red-team (image-cache-v1 rule 6): a cost that would overflow size_t is not kept (it wrapped to a small cost and
  // was kept under any budget)
  {
    lfm::LruBytes<int> lru(1000);
    lru.put("k", std::make_shared<const int>(1), SIZE_MAX);
    CHECK(lru.size() == 0 && lru.bytes() == 0);
    lru.put("k", std::make_shared<const int>(1), SIZE_MAX - 100);
    CHECK(lru.size() == 0 && lru.bytes() == 0);
    lru.put(std::string(10, 'k'), std::make_shared<const int>(1), 990);  // exactly the budget: kept
    CHECK(lru.size() == 1 && lru.bytes() == 1000);
  }
  // red-team (image-cache-v1 rule 6): an entry holds its key once, as it is charged once. Before, the order list kept a
  // second copy: the front end's token cache, all keys, could hold twice its budget.
  {
    const std::string key(1 << 20, 'k');
    lfm::LruBytes<int> lru(4 << 20);
    const auto value = std::make_shared<const int>(1);
    const size_t before = ALLOCATED();
    lru.put(key, value, sizeof(int));
    [[maybe_unused]] const size_t taken = ALLOCATED() - before;
    CHECK_ALLOCATED(taken >= key.size() && taken < key.size() + 4096);
    CHECK(lru.find(key) && lru.bytes() == key.size() + sizeof(int));
    // eviction still finds and frees the least recent entry by its key
    lru.put(std::string(3 << 20, 'z'), value, sizeof(int));
    CHECK(!lru.find(key) && lru.size() == 1);
  }
  // red-team (image-queue-v1 rule 5): a lease does not copy its key. Before, each lease held its own copy of the
  // image's base64 text: a request's images took their size again per lease, for the request's whole life.
  {
    const std::string key(1 << 20, 'k');
    lfm::LeasedMap<int> m;
    [[maybe_unused]] size_t before = ALLOCATED();
    auto first = m.put(key, std::make_shared<const int>(1));
    CHECK_ALLOCATED(ALLOCATED() - before < key.size() + 4096);  // the map's own copy only
    before = ALLOCATED();
    auto second = m.put(key, std::make_shared<const int>(2));
    CHECK_ALLOCATED(ALLOCATED() - before < 4096);
    CHECK(m.size() == 1 && *m.find(key) == 1);
    {
      auto moved = std::move(first);
      lfm::LeasedMap<int>::Lease assigned;
      assigned = std::move(second);
      CHECK(m.size() == 1);
    }
    CHECK(m.size() == 0 && !m.find(key));
  }
  // 200 distinct keys of random sizes under a budget: the total never exceeds it; the most recent entry is kept
  {
    std::mt19937 rng(1117);
    const size_t budget = 64 << 10;
    lfm::LruBytes<int> lru(budget);
    std::vector<std::string> keys;
    for (int i = 0; i < 200; ++i) {
      std::string k(1 + rng() % 4000, 'a' + i % 26);
      k += std::to_string(i);
      const size_t value_bytes = rng() % 8000;
      lru.put(k, std::make_shared<const int>(i), value_bytes);
      CHECK(lru.bytes() <= budget);
      if (k.size() + value_bytes <= budget) CHECK(lru.find(k) && *lru.find(k) == i);
      keys.push_back(k);
    }
    CHECK(lru.size() > 1);
  }
  // least recently used goes first; a find makes an entry recent
  {
    lfm::LruBytes<int> lru(30);
    lru.put("a", std::make_shared<const int>(1), 9);  // 10 each
    lru.put("b", std::make_shared<const int>(2), 9);
    lru.put("c", std::make_shared<const int>(3), 9);
    CHECK(lru.find("a"));                              // a recent: b is the oldest
    lru.put("d", std::make_shared<const int>(4), 9);
    CHECK(!lru.find("b") && lru.find("a") && lru.find("c") && lru.find("d") && lru.bytes() == 30);
    // the same key again keeps the first value
    lru.put("a", std::make_shared<const int>(99), 9);
    CHECK(*lru.find("a") == 1 && lru.bytes() == 30);
    // an entry larger than the budget is not kept, and evicts nothing
    lru.put("huge", std::make_shared<const int>(5), 100);
    CHECK(!lru.find("huge") && lru.size() == 3);
    // keys differing in their last byte are different entries
    lru.set_budget(1000);
    lru.put(std::string(100, 'x') + "1", std::make_shared<const int>(7), 0);
    lru.put(std::string(100, 'x') + "2", std::make_shared<const int>(8), 0);
    CHECK(*lru.find(std::string(100, 'x') + "1") == 7 && *lru.find(std::string(100, 'x') + "2") == 8);
    // an evicted value lives on for whoever holds it
    auto held = lru.find("c");
    lru.set_budget(0);
    CHECK(lru.size() == 0 && lru.bytes() == 0 && held && *held == 3);
  }
  // budget 0 keeps nothing
  {
    lfm::LruBytes<int> lru(0);
    lru.put("a", std::make_shared<const int>(1), 0);
    CHECK(lru.size() == 0);
  }
  // LeasedMap: an entry lives while a lease lives; the same key shares one entry; threads put and find at once
  {
    lfm::LeasedMap<int> m;
    {
      auto l1 = m.put("k", std::make_shared<const int>(1));
      {
        auto l2 = m.put("k", std::make_shared<const int>(2));
        CHECK(*m.find("k") == 1 && m.size() == 1);
      }
      CHECK(m.find("k"));
      auto moved = std::move(l1);
      CHECK(m.find("k"));
    }
    CHECK(!m.find("k") && m.size() == 0);
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t)
      threads.emplace_back([&m, t] {
        for (int i = 0; i < 2000; ++i) {
          const std::string key = "key" + std::to_string((t + i) % 5);
          auto lease = m.put(key, std::make_shared<const int>(i));
          CHECK(m.find(key));
        }
      });
    for (auto& t : threads) t.join();
    CHECK(m.size() == 0);
  }
  // LockedLruBytes from several threads: the budget holds
  {
    lfm::LockedLruBytes<int> lru;
    lru.set_budget(10000);
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t)
      threads.emplace_back([&lru, t] {
        for (int i = 0; i < 2000; ++i) {
          lru.put("k" + std::to_string(t * 100000 + i), std::make_shared<const int>(i), size_t(i % 300));
          lru.find("k" + std::to_string(t * 100000 + i / 2));
          CHECK(lru.bytes() <= 10000);
        }
      });
    for (auto& t : threads) t.join();
  }
  std::puts("lru_bytes_check: PASS");
}
