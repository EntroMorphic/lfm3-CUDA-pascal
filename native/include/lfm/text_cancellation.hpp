#pragma once

// Linux host-only control-pipe foundation, NOT worker cancellation integration.
// Borrowed fd; one externally serialized owner must perform all polling.
#include <array>
#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <stdexcept>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>
#include <utility>

namespace lfm::text {

// Poll at the last host boundary after potentially long initialization/reset,
// not merely before it. This does not interrupt the operation once launched.
template<class Check, class Operation>
decltype(auto) at_cancellation_boundary(Check&& check, Operation&& operation) {
    std::forward<Check>(check)();
    return std::forward<Operation>(operation)();
}

class CancellationReader {
public:
    static constexpr std::size_t frame_bytes = 8;
    static constexpr std::size_t drain_budget = 8192;
    // Also bound interrupted/partial read attempts, not just successful bytes.
    static constexpr std::size_t read_attempt_budget = drain_budget / frame_bytes;
    static constexpr std::uint64_t max_rpc_id = INT_MAX;

private:
    int fd_;
    int failure_ = 0;
    std::array<unsigned char, frame_bytes> partial_{};
    std::size_t used_ = 0;
    std::uint64_t current_ = 0;
    std::array<bool, 2> latched_{};
    bool pair_ = false;
    std::uint64_t partial_generation_ = 0;

    static void validate_fd(int fd) {
        if (fd < 3) throw std::invalid_argument("cancellation fd must be >=3 or -1");
        const int flags = ::fcntl(fd, F_GETFL);
        if (flags == -1) throw std::system_error(errno, std::generic_category(), "cancellation fd");
        if ((flags & O_ACCMODE) == O_WRONLY || !(flags & O_NONBLOCK))
            throw std::invalid_argument("cancellation fd must be readable and nonblocking");
        struct stat info{};
        if (::fstat(fd, &info) == -1)
            throw std::system_error(errno, std::generic_category(), "cancellation fstat");
        if (!S_ISFIFO(info.st_mode)) throw std::invalid_argument("cancellation fd must be a pipe");
    }
    [[noreturn]] void fail(int error) {
        failure_ = error;
        throw std::system_error(error, std::generic_category(), "cancellation pipe failed");
    }

public:
    explicit CancellationReader(int fd = -1) : fd_(fd) {
        if (fd_ != -1) validate_fd(fd_);
    }
    CancellationReader(const CancellationReader&) = delete;
    CancellationReader& operator=(const CancellationReader&) = delete;
    CancellationReader(CancellationReader&&) = delete;
    CancellationReader& operator=(CancellationReader&&) = delete;
    // Does not close the borrowed descriptor or change its flags.

    // Poll IDs are positive protocol-2 IDs and must never decrease or be reused
    // for a different RPC. Complete foreign frames are discarded, not queued.
    // A matching frame stays latched through repeated polls of the same RPC.
    std::array<bool, 2> poll(std::uint64_t rpc_id, bool pair) {
        if (failure_) fail(failure_);
        if (rpc_id == 0 || rpc_id > max_rpc_id || rpc_id < current_)
            throw std::invalid_argument("invalid or decreasing cancellation RPC ID");
        if (rpc_id != current_) {
            current_ = rpc_id;
            latched_ = {};
            pair_ = pair;
        }
        if (pair_ != pair) throw std::invalid_argument("cannot change cancellation mode within RPC");
        if (fd_ == -1) return latched_;
        // Detect closure/changed flags before a read could unexpectedly block.
        try { validate_fd(fd_); }
        catch (const std::system_error& e) { fail(e.code().value()); }
        catch (const std::invalid_argument&) { fail(EINVAL); }
        std::size_t bytes = 0;
        for (std::size_t attempts = 0;
             attempts < read_attempt_budget && bytes < drain_budget; ++attempts) {
            if (!used_) partial_generation_ = current_;
            const auto n = ::read(fd_, partial_.data() + used_, frame_bytes - used_);
            if (n == -1) {
                const int error = errno;
                if (error == EAGAIN || error == EWOULDBLOCK) break;
                if (error == EINTR) continue;
                fail(error);
            }
            // Clean EOF is also a lost control channel, never "not cancelled".
            if (n == 0) fail(EPIPE);
            used_ += static_cast<std::size_t>(n);
            bytes += static_cast<std::size_t>(n);
            if (used_ == frame_bytes) {
                std::uint64_t target = 0;
                for (std::size_t i = 0; i < frame_bytes; ++i)
                    target |= std::uint64_t(partial_[i]) << (8 * i);
                used_ = 0;
                const auto tag = target >> 32;
                const auto outer = target & 0xffffffffULL;
                if (tag > 2 || outer == 0 || outer > max_rpc_id) fail(EPROTO);
                if (partial_generation_ == current_ && outer == rpc_id) {
                    if (pair && tag) latched_[tag - 1] = true;
                    if (!pair && !tag) latched_[0] = true;
                }
            }
        }
        return latched_;
    }
    bool requested(std::uint64_t rpc_id) { return poll(rpc_id, false)[0]; }
    // One drain and one partial-frame buffer for both lane targets. Foreign
    // untagged/tagged frames are valid but never saved for a future generation.
    std::array<bool, 2> requested_pair(std::uint64_t rpc_id) { return poll(rpc_id, true); }
};

} // namespace lfm::text
