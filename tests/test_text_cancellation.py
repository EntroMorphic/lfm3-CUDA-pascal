"""Linux host-only cancellation pipe contract; no worker/CUDA integration."""
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


SOURCE = r'''
#include <lfm/text_cancellation.hpp>
#include <lfm/text_sessions.hpp>
#include <cassert>
#include <functional>
#include <string>
#include <sys/ioctl.h>
#include <type_traits>
using Reader = lfm::text::CancellationReader;
// A deterministic endless source proves the poll bound even on hosts whose
// pipe capacity is <=8192. Real-pipe cases still use the actual read syscall.
int fake_reads=0, fake_mode=0, fake_bytes=0;
extern "C" ssize_t __real_read(int, void*, size_t);
extern "C" ssize_t __wrap_read(int fd, void* data, size_t size) {
    if (!fake_mode) return __real_read(fd, data, size);
    ++fake_reads;
    assert(fake_reads <= static_cast<int>(Reader::read_attempt_budget)+1);
    if (fake_mode == 2) { errno=EINTR; return -1; }
    auto* out=static_cast<unsigned char*>(data);
    const auto count=fake_mode == 3 ? std::size_t(1) : size;
    for (std::size_t i=0;i<count;++i) out[i]=((fake_bytes++)%8 == 0) ? 1 : 0;
    return static_cast<ssize_t>(count);
}
struct Pipe {
    int fd[2];
    Pipe() { assert(pipe2(fd, O_NONBLOCK | O_CLOEXEC) == 0); }
    ~Pipe() { for (int f : fd) if (f >= 0) close(f); }
    void end_writer() { assert(close(fd[1]) == 0); fd[1] = -1; }
};
std::array<unsigned char, 8> frame(std::uint64_t id) {
    std::array<unsigned char, 8> out{};
    for (unsigned i=0;i<8;++i) out[i] = static_cast<unsigned char>(id >> (8*i));
    return out;
}
void send(int fd, std::uint64_t id) {
    const auto bytes = frame(id);
    assert(write(fd, bytes.data(), bytes.size()) == 8);
}
void invalid(const std::function<void()>& fn) {
    bool caught = false;
    try { fn(); } catch (const std::invalid_argument&) { caught = true; }
    assert(caught);
}
void ioerror(const std::function<void()>& fn, int code) {
    bool caught = false;
    try { fn(); } catch (const std::system_error& e) {
        assert(e.code().value() == code); caught = true;
    }
    assert(caught);
}
int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string mode = argv[1];
    static_assert(!std::is_copy_constructible<Reader>::value, "single poll owner");
    if (mode == "validation") {
        Reader off;
        assert(!off.requested(1) && !off.requested(1) && !off.requested(2));
        invalid([&] { off.requested(0); });
        invalid([&] { off.requested(Reader::max_rpc_id+1); });
        invalid([&] { off.requested(1); });
        for (int fd : {-2, 0, 1, 2}) invalid([&] { Reader bad(fd); });
        Pipe p;
        invalid([&] { Reader bad(p.fd[1]); });
        assert(fcntl(p.fd[0], F_SETFL, 0) == 0);
        invalid([&] { Reader bad(p.fd[0]); });
        assert(fcntl(p.fd[0], F_SETFL, O_NONBLOCK) == 0);
        int dead = dup(p.fd[0]); assert(dead >= 3); assert(close(dead) == 0);
        ioerror([&] { Reader bad(dead); }, EBADF);
        int regular = open("/dev/null", O_RDONLY | O_NONBLOCK); assert(regular >= 3);
        invalid([&] { Reader bad(regular); }); assert(close(regular) == 0);
        { Reader borrowed(p.fd[0]); assert(!borrowed.requested(1)); }
        assert(fcntl(p.fd[0], F_GETFL) & O_NONBLOCK); // reader does not own fd
    } else if (mode == "targeting") {
        Pipe p; Reader r(p.fd[0]);
        assert(!r.requested(10));
        for (auto id : {1, 9, 11, 200, 9}) send(p.fd[1], id);
        assert(!r.requested(10));
        assert(!r.requested(11)); // drained future frame was NOT saved
        send(p.fd[1], 11); send(p.fd[1], 11);
        assert(r.requested(11) && r.requested(11)); // latch survives empty pipe
        send(p.fd[1], 11); // late cancellation after completion
        assert(!r.requested(12) && !r.requested(12));
        send(p.fd[1], Reader::max_rpc_id);
        assert(r.requested(Reader::max_rpc_id));
    } else if (mode == "partial") {
        for (std::size_t split=1;split<8;++split) {
            Pipe p; Reader r(p.fd[0]); const auto bytes=frame(0x01020304);
            assert(write(p.fd[1], bytes.data(), split) == static_cast<ssize_t>(split));
            assert(!r.requested(0x01020304) && !r.requested(0x01020304));
            assert(write(p.fd[1], bytes.data()+split, 8-split) == static_cast<ssize_t>(8-split));
            assert(r.requested(0x01020304));
        }
        Pipe p; Reader r(p.fd[0]); const auto stale=frame(7);
        assert(write(p.fd[1], stale.data(), 3) == 3);
        assert(!r.requested(7));
        assert(write(p.fd[1], stale.data()+3, 5) == 5);
        send(p.fd[1], 8);
        assert(r.requested(8)); // partial old frame cannot misalign next target
    } else if (mode == "malformed") {
        for (auto id : {std::uint64_t(0), Reader::max_rpc_id+1, UINT64_MAX}) {
            Pipe p; Reader r(p.fd[0]); send(p.fd[1], 5); send(p.fd[1], id);
            ioerror([&] { r.requested(5); }, EPROTO); // even after matching frame
            send(p.fd[1], 6);
            ioerror([&] { r.requested(6); }, EPROTO); // sticky fail closed
        }
    } else if (mode == "eof") {
        for (int partial : {0, 3, 8}) {
            Pipe p; Reader r(p.fd[0]); const auto bytes=frame(1);
            if (partial) assert(write(p.fd[1], bytes.data(), partial) == partial);
            p.end_writer();
            ioerror([&] { r.requested(1); }, EPIPE);
            ioerror([&] { r.requested(2); }, EPIPE);
        }
    } else if (mode == "changed_fd") {
        Pipe p; Reader r(p.fd[0]); assert(!r.requested(1));
        assert(fcntl(p.fd[0], F_SETFL, 0) == 0);
        ioerror([&] { r.requested(1); }, EINVAL);
        assert(fcntl(p.fd[0], F_SETFL, O_NONBLOCK) == 0);
        ioerror([&] { r.requested(1); }, EINVAL);
        Pipe q; Reader closed(q.fd[0]); assert(close(q.fd[0]) == 0); q.fd[0]=-1;
        ioerror([&] { closed.requested(1); }, EBADF);
    } else if (mode == "bounded") {
        for (int kind : {1, 2, 3}) {
            Pipe p; Reader r(p.fd[0]);
            fake_reads=0; fake_bytes=0; fake_mode=kind;
            assert(!r.requested(2));
            fake_mode=0;
            assert(fake_reads == static_cast<int>(Reader::read_attempt_budget));
            assert(fake_bytes == (kind == 1 ? static_cast<int>(Reader::drain_budget) :
                                  kind == 2 ? 0 : static_cast<int>(Reader::read_attempt_budget)));
            send(p.fd[1], 2); assert(r.requested(2));
        }
    } else if (mode == "backpressure") {
        Pipe p; Reader r(p.fd[0]);
        assert(fpathconf(p.fd[1], _PC_PIPE_BUF) >= 8);
        int count=0;
        for (;;) {
            const auto bytes=frame(1);
            const auto n=write(p.fd[1], bytes.data(), bytes.size());
            if (n == -1) { assert(errno == EAGAIN || errno == EWOULDBLOCK); break; }
            assert(n == 8); ++count;
        }
        assert(count > 0);
        int before=0; assert(ioctl(p.fd[0], FIONREAD, &before) == 0);
        assert(before == count*8);
        assert(!r.requested(2));
        int after=0; assert(ioctl(p.fd[0], FIONREAD, &after) == 0);
        const int drained=std::min(before, static_cast<int>(Reader::drain_budget));
        assert(before-after == drained); // exact fixed upper bound, not timing
        while (after) {
            assert(!r.requested(2));
            int next=0; assert(ioctl(p.fd[0], FIONREAD, &next) == 0);
            assert(after-next == std::min(after, static_cast<int>(Reader::drain_budget)));
            after=next;
        }
        send(p.fd[1], 2); assert(r.requested(2)); // writer progresses after drain
    } else if (mode == "owner_contract") {
        Pipe p; Reader r(p.fd[0]);
        lfm::text::SessionLedger l(2, 32); l.create("A",16); l.create("B",16);
        { auto a=l.begin("A",{1},1); a.commit({2},{1}); }
        { auto b=l.begin("B",{8},1); b.commit({9},{8}); }
        const auto old=l.inspect("A"), other=l.inspect("B");
        send(p.fd[1], 1); // queued before execution
        auto a=l.begin("A",{1,2},1);
        int forwards=0;
        if (!r.requested(1)) ++forwards;
        else a.abort(); // simulated owner action; header never touches ledger
        assert(forwards == 0);
        assert(!l.inspect("A").cache_valid && !l.inspect("A").inflight);
        assert(l.inspect("A").transcript == old.transcript);
        assert(l.inspect("A").consumed == old.consumed);
        assert(l.inspect("B").cache_valid == other.cache_valid);
        assert(l.inspect("B").transcript == other.transcript);
        assert(l.inspect("B").consumed == other.consumed);
        send(p.fd[1], 1); assert(!r.requested(2));
        auto b=l.begin("B",{8,9},1); b.commit({10},{8,9});
        assert(l.inspect("B").cache_valid);
        const auto completed=l.inspect("B");
        send(p.fd[1], 2); // arrives after B's successful commit
        assert(!r.requested(3));
        assert(l.inspect("B").cache_valid);
        assert(l.inspect("B").transcript == completed.transcript);
        assert(l.inspect("B").consumed == completed.consumed);
    } else { assert(false); }
}
'''


@pytest.fixture(scope="module", params=[False, True], ids=["strict", "ubsan"])
def cancellation_exe(tmp_path_factory, request):
    if sys.platform != "linux":
        pytest.skip("Linux pipe contract")
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ required")
    root = Path(__file__).resolve().parents[1]
    tmp = tmp_path_factory.mktemp("text_cancellation")
    source = tmp / "cancellation.cpp"
    source.write_text(SOURCE)
    exe = tmp / "cancellation"
    flags = ["-fsanitize=undefined", "-fno-sanitize-recover=all"] if request.param else []
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pedantic",
                    *flags, "-I", str(root / "native/include"), str(source),
                    "-Wl,--wrap=read", "-o", str(exe)], check=True)
    return exe


@pytest.mark.parametrize("case", ["validation", "targeting", "partial", "malformed", "eof",
                                  "changed_fd", "bounded", "backpressure", "owner_contract"])
def test_text_cancellation(cancellation_exe, case):
    subprocess.run([str(cancellation_exe), case], check=True, timeout=10)
