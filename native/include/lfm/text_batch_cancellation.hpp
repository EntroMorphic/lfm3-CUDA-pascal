#pragma once
#include <lfm/text_cancellation.hpp>

namespace lfm::text {
// Wire tokens are not RPC IDs: only the low 31 bits carry the outer RPC.
inline std::uint64_t batch_cancellation_token(std::uint64_t rpc_id, unsigned lane) {
    if (!rpc_id || rpc_id > CancellationReader::max_rpc_id || lane > 1)
        throw std::invalid_argument("invalid batch cancellation target");
    return (std::uint64_t(lane + 1) << 32) | rpc_id;
}
} // namespace lfm::text
