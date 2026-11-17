"""Host-only session/admission contract; no CUDA or resident worker required."""
from pathlib import Path
import shutil
import subprocess

import pytest


SOURCE = r'''
#include <lfm/text_sessions.hpp>
#include <cassert>
#include <functional>
#include <limits>
#include <memory>
#include <type_traits>
using Ledger = lfm::text::SessionLedger;
using Tokens = Ledger::Tokens;
void rejected(const std::function<void()>& fn) {
    bool caught = false;
    try { fn(); } catch (const std::exception&) { caught = true; }
    assert(caught);
}
void unchanged(const Ledger::Snapshot& a, const Ledger::Snapshot& b) {
    assert(a.capacity == b.capacity && a.transcript == b.transcript);
    assert(a.consumed == b.consumed && a.cache_valid == b.cache_valid);
    assert(a.inflight == b.inflight);
}
int main(int argc, char** argv) {
    assert(argc == 2);
    std::string mode = argv[1];
    static_assert(!std::is_copy_constructible<Ledger::Lease>::value, "unique leases");
    if (mode == "admission") {
        rejected([] { Ledger l(0, 1); });
        rejected([] { Ledger l(1, 0); });
        Ledger l(2, 10);
        for (const auto& id : {std::string(), std::string(65, 'a'), std::string("a/b"),
                              std::string(".."), std::string("a b"), std::string("a\n"),
                              std::string("a\0b", 3), std::string("\xff")})
            rejected([&] { l.create(id, 1); });
        rejected([&] { l.create("A", 0); });
        rejected([&] { l.create("A", 32769); });
        l.create("A", 6);
        rejected([&] { l.create("A", 1); });
        rejected([&] { l.create("B", 5); });
        assert(l.size() == 1 && l.reserved_tokens() == 6);
        l.create("B", 4);
        rejected([&] { l.create("C", 1); });
        l.reset("A");
        assert(l.reserved_tokens() == 10);
        l.close("A");
        l.create("safe_0-ABC", 6);
        assert(l.size() == 2 && l.reserved_tokens() == 10);
        rejected([&] { l.inspect("missing"); });
        rejected([&] { l.close("A"); });
    } else if (mode == "isolation") {
        Ledger l(2, 32); l.create("A", 16); l.create("B", 16);
        auto a = l.begin("A", {1, 2}, 3);
        auto b = l.begin("B", {8}, 2);
        assert(a.plan().full_prefill && b.plan().full_prefill);
        rejected([&] { l.begin("A", {1}, 1); });
        rejected([&] { l.reset("A"); });
        rejected([&] { l.close("A"); });
        b.commit({9, 10}, {8, 9});
        a.commit({3, 4, 5}, {1, 2, 3, 4});
        assert(l.inspect("A").transcript == Tokens({1, 2, 3, 4, 5}));
        assert(l.inspect("B").consumed == Tokens({8, 9}));
        const auto saved_b = l.inspect("B");
        auto before = l.inspect("A");
        rejected([&] { a.commit({3}, {1, 2}); });
        unchanged(before, l.inspect("A"));
        auto next = l.begin("A", {1, 2, 3, 4, 5, 6}, 2);
        assert(!next.plan().full_prefill && next.plan().reuse_tokens == 4);
        rejected([&] { a.commit({7}, {1, 2}); }); // old ticket cannot publish
        a.abort(); // old lease cannot cancel new turn
        assert(l.inspect("A").inflight);
        next.commit({7}, {1, 2, 3, 4, 5, 6}); // final prediction/EOS not forwarded
        unchanged(saved_b, l.inspect("B"));
    } else if (mode == "prefix") {
        Ledger l(1, 32); l.create("A", 32);
        auto seed = [&] { auto t = l.begin("A", {1, 2}, 3); t.commit({3, 4, 5}, {1, 2, 3, 4}); };
        for (const Tokens& prompt : {Tokens{1, 2}, Tokens{1, 2, 3, 4},
                                    Tokens{1, 2, 9, 4, 5}, Tokens{8, 2, 3, 4, 5}}) {
            seed();
            auto t = l.begin("A", prompt, 1);
            assert(t.plan().full_prefill && t.plan().reuse_tokens == 0);
        }
        seed();
        auto t = l.begin("A", {1, 2, 3, 4, 99}, 1);
        assert(t.plan().reuse_tokens == 4); // unconsumed old prediction need not match
        t.abort();
        auto saved = l.inspect("A");
        assert(!saved.cache_valid && saved.transcript == Tokens({1, 2, 3, 4, 5}));
        auto restore = l.begin("A", saved.transcript, 1);
        assert(restore.plan().full_prefill);
        restore.commit({6}, saved.transcript);
        l.reset("A");
        auto empty = l.inspect("A");
        assert(empty.transcript.empty() && empty.consumed.empty() && !empty.cache_valid);
    } else if (mode == "invalid_commit") {
        Ledger l(1, 16); l.create("A", 16);
        { auto t = l.begin("A", {1}, 1); t.commit({2}, {1}); }
        auto t = l.begin("A", {1, 2}, 2);
        const auto before = l.inspect("A");
        rejected([&] { t.commit({3, 4, 5}, {1, 2, 3, 4}); });
        rejected([&] { t.commit({}, {1, 2}); });
        rejected([&] { t.commit({-1}, {1, 2}); });
        rejected([&] { t.commit({3, 4}, {1, 2, 3, 4}); }); // final token consumed
        rejected([&] { t.commit({3, 4}, {1, 2}); });
        rejected([&] { t.commit({3, 4}, {1, 9, 3}); });
        unchanged(before, l.inspect("A"));
        t.abort(); t.abort();
        rejected([&] { t.commit({3}, {1, 2}); });
        auto after = l.inspect("A");
        assert(after.transcript == before.transcript && after.consumed == before.consumed);
        assert(!after.inflight && !after.cache_valid);
        auto bad = l.inspect("A");
        rejected([&] { l.begin("A", {}, 1); });
        rejected([&] { l.begin("A", {-1}, 1); });
        rejected([&] { l.begin("A", {1}, std::numeric_limits<std::size_t>::max()); });
        unchanged(bad, l.inspect("A"));
    } else if (mode == "lifetime") {
        Ledger l(2, 16); l.create("A", 8); l.create("B", 8);
        { auto abandoned = l.begin("A", {1}, 1); }
        assert(!l.inspect("A").inflight && !l.inspect("A").cache_valid);
        auto a = l.begin("A", {1}, 1);
        auto b = l.begin("B", {2}, 1);
        b = std::move(a); // cancels B, transfers A
        assert(!l.inspect("B").inflight && l.inspect("A").inflight);
        rejected([&] { a.commit({3}, {1}); });
        b.commit({3}, {1});
        l.close("A"); l.create("A", 8);
        rejected([&] { b.commit({3}, {1}); });
        assert(l.inspect("A").transcript.empty());
        std::unique_ptr<Ledger::Lease> survivor;
        { Ledger temporary(1, 8); temporary.create("A", 8);
          survivor = std::make_unique<Ledger::Lease>(temporary.begin("A", {1}, 1)); }
        survivor->commit({2}, {1}); // shared internal ownership: no dangling ledger
    } else if (mode == "boundary") {
        Ledger l(2, 65536); l.create("A", 32768); l.create("B", 32768);
        Tokens prompt(32767, 7);
        auto t = l.begin("A", prompt, 1);
        t.commit({0}, prompt); // EOS remains in transcript, absent from consumed
        assert(l.inspect("A").transcript.size() == 32768);
        auto before = l.inspect("A");
        rejected([&] { l.begin("A", before.transcript, 1); });
        rejected([&] { l.begin("A", prompt, 2); });
        unchanged(before, l.inspect("A"));
        auto zero = l.begin("A", before.transcript, 0);
        assert(zero.plan().reuse_tokens == 32767);
        zero.commit({}, before.transcript); // no prediction: entire prompt consumed
        assert(l.inspect("A").consumed.size() == 32768);
        auto replay = l.begin("A", before.transcript, 0);
        assert(replay.plan().full_prefill); // no stored logits assumption
        replay.abort();
        rejected([&] { l.begin("B", Tokens(32769, 0), 0); });
        assert(l.reserved_tokens() == 65536);
    } else { assert(false); }
}
'''


@pytest.fixture(scope="module")
def session_exe(tmp_path_factory):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ required")
    root = Path(__file__).resolve().parents[1]
    tmp = tmp_path_factory.mktemp("text_sessions")
    source = tmp / "sessions.cpp"
    source.write_text(SOURCE)
    exe = tmp / "sessions"
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "-pedantic", "-I", str(root / "native/include"), str(source),
                    "-o", str(exe)], check=True)
    return exe


@pytest.mark.parametrize("case", ["admission", "isolation", "prefix", "invalid_commit",
                                  "lifetime", "boundary"])
def test_text_sessions(session_exe, case):
    subprocess.run([str(session_exe), case], check=True)
