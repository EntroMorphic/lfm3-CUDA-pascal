#pragma once

// Host-only bookkeeping, not a resident KV implementation. All operations,
// including lease destruction, require external serialization (no mutexes).
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lfm { namespace text {

class SessionLedger {
public:
    using Tokens = std::vector<std::int32_t>;
    static constexpr std::size_t context_limit = 32768;
    static constexpr std::size_t id_limit = 64;

    struct Snapshot {
        std::size_t capacity = 0;
        Tokens transcript; // Full prompt + predictions, including final EOS if any.
        Tokens consumed;   // Exactly the prefix actually forwarded through the model.
        bool cache_valid = false;
        bool inflight = false;
    };
    struct Plan {
        Tokens prompt;
        std::size_t requested_output = 0;
        std::size_t reuse_tokens = 0;
        // Forward prompt[reuse_tokens:] before sampling. If true, discard/reset
        // old device state first; the ledger never performs that device action.
        bool full_prefill = true;
    };

private:
    struct Entry {
        Snapshot value;
        std::uint64_t ticket = 0;
    };
    struct State {
        std::size_t max_sessions, token_budget, reserved = 0;
        std::uint64_t next_ticket = 1;
        std::map<std::string, Entry> entries;
        State(std::size_t n, std::size_t b) : max_sessions(n), token_budget(b) {}
    };
    std::shared_ptr<State> state_;

    static void check_id(const std::string& id) {
        if (id.empty() || id.size() > id_limit)
            throw std::invalid_argument("session ID length must be 1..64");
        for (unsigned char c : id)
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == '_' || c == '-'))
                throw std::invalid_argument("session ID must be safe ASCII [A-Za-z0-9_-]");
    }
    static void check_tokens(const Tokens& tokens) {
        for (auto token : tokens)
            if (token < 0) throw std::invalid_argument("negative token ID");
    }
    Entry& find(const std::string& id) const {
        check_id(id);
        auto it = state_->entries.find(id);
        if (it == state_->entries.end()) throw std::invalid_argument("unknown session");
        return it->second;
    }

public:
    class Lease {
        friend class SessionLedger;
        std::shared_ptr<State> state_;
        std::string id_;
        std::uint64_t ticket_ = 0;
        Plan plan_;
        Lease(std::shared_ptr<State> state, std::string id, std::uint64_t ticket, Plan plan)
            : state_(std::move(state)), id_(std::move(id)), ticket_(ticket), plan_(std::move(plan)) {}
        Entry& active_entry() const {
            if (!state_ || ticket_ == 0) throw std::logic_error("inactive transaction");
            auto it = state_->entries.find(id_);
            if (it == state_->entries.end() || !it->second.value.inflight ||
                it->second.ticket != ticket_)
                throw std::logic_error("stale transaction");
            return it->second;
        }
    public:
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept
            : state_(std::move(other.state_)), id_(std::move(other.id_)),
              ticket_(other.ticket_), plan_(std::move(other.plan_)) { other.ticket_ = 0; }
        Lease& operator=(Lease&& other) noexcept {
            if (this != &other) {
                abort();
                state_ = std::move(other.state_);
                id_ = std::move(other.id_);
                ticket_ = other.ticket_;
                plan_ = std::move(other.plan_);
                other.ticket_ = 0;
            }
            return *this;
        }
        ~Lease() { abort(); }
        const Plan& plan() const noexcept { return plan_; }
        bool active() const noexcept { return state_ && ticket_ != 0; }

        // Caller attests device work succeeded, and supplies the exact forwarded
        // token history. A final predicted token (including EOS) is NOT consumed.
        // Invalid commit leaves the lease active and committed history unchanged.
        void commit(const Tokens& generated, const Tokens& consumed) {
            Entry& entry = active_entry();
            if (generated.size() > plan_.requested_output ||
                (plan_.requested_output != 0 && generated.empty()))
                throw std::invalid_argument("invalid generated token count");
            check_tokens(generated);
            Tokens transcript = plan_.prompt;
            transcript.insert(transcript.end(), generated.begin(), generated.end());
            const std::size_t expected = transcript.size() - (generated.empty() ? 0 : 1);
            if (consumed.size() != expected ||
                !std::equal(consumed.begin(), consumed.end(), transcript.begin()))
                throw std::invalid_argument("consumed history must exclude final prediction");
            // Allocate everything before publishing any change (strong guarantee).
            Tokens forwarded = consumed;
            entry.value.transcript.swap(transcript);
            entry.value.consumed.swap(forwarded);
            entry.value.cache_valid = true;
            entry.value.inflight = false;
            entry.ticket = 0;
            ticket_ = 0;
        }
        // Idempotent cancellation; keep the last committed transcript/consumed
        // history, but never trust potentially partially modified device state.
        void abort() noexcept {
            if (state_ && ticket_) {
                auto it = state_->entries.find(id_);
                if (it != state_->entries.end() && it->second.ticket == ticket_) {
                    it->second.value.cache_valid = false;
                    it->second.value.inflight = false;
                    it->second.ticket = 0;
                }
                ticket_ = 0;
            }
        }
    };

    SessionLedger(std::size_t max_sessions, std::size_t token_budget)
        : state_(std::make_shared<State>(max_sessions, token_budget)) {
        if (!max_sessions || !token_budget) throw std::invalid_argument("zero admission limit");
    }
    SessionLedger(const SessionLedger&) = delete;
    SessionLedger& operator=(const SessionLedger&) = delete;
    SessionLedger(SessionLedger&&) = delete;
    SessionLedger& operator=(SessionLedger&&) = delete;

    void create(const std::string& id, std::size_t capacity) {
        check_id(id);
        if (!capacity || capacity > context_limit) throw std::invalid_argument("invalid capacity");
        if (state_->entries.count(id)) throw std::invalid_argument("duplicate session");
        if (state_->entries.size() >= state_->max_sessions || capacity > state_->token_budget - state_->reserved)
            throw std::runtime_error("session admission quota exceeded");
        Entry entry;
        entry.value.capacity = capacity;
        state_->entries.emplace(id, std::move(entry));
        state_->reserved += capacity;
    }
    Snapshot inspect(const std::string& id) const { return find(id).value; }
    std::size_t reserved_tokens() const noexcept { return state_->reserved; }
    std::size_t size() const noexcept { return state_->entries.size(); }

    Lease begin(const std::string& id, const Tokens& prompt, std::size_t requested_output) {
        Entry& entry = find(id);
        if (entry.value.inflight) throw std::logic_error("session already inflight");
        if (prompt.empty() || prompt.size() > entry.value.capacity ||
            requested_output > entry.value.capacity - prompt.size())
            throw std::invalid_argument("prompt plus requested output exceeds capacity");
        check_tokens(prompt);
        Plan plan{prompt, requested_output, 0, true};
        const auto& old = entry.value.consumed;
        // No rollback/truncation assumption and no implicit saved-logits cache.
        // At least one prompt token must be forwarded to produce next logits.
        if (entry.value.cache_valid && !old.empty() && old.size() < prompt.size() &&
            std::equal(old.begin(), old.end(), prompt.begin())) {
            plan.reuse_tokens = old.size();
            plan.full_prefill = false;
        }
        if (state_->next_ticket == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("transaction ticket space exhausted");
        // Construct the potentially allocating lease before marking inflight.
        Lease lease(state_, id, state_->next_ticket, std::move(plan));
        entry.ticket = state_->next_ticket++;
        entry.value.inflight = true;
        return lease;
    }
    void reset(const std::string& id) {
        Entry& entry = find(id);
        if (entry.value.inflight) throw std::logic_error("cannot reset inflight session");
        entry.value.transcript.clear();
        entry.value.consumed.clear();
        entry.value.cache_valid = false;
    }
    void close(const std::string& id) {
        Entry& entry = find(id);
        if (entry.value.inflight) throw std::logic_error("cannot close inflight session");
        state_->reserved -= entry.value.capacity;
        state_->entries.erase(id);
    }
};

}} // namespace lfm::text
