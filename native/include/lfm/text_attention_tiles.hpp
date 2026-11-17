#pragma once
#include <lfm/text_serving_limits.hpp>

namespace lfm::text {
struct AttentionKeyRange {
  int begin;
  int end; // Exclusive absolute cache position, not a key count.
  int span() const { return end-begin; }
};

// Host-only policy shared by the native runner's prefill and decode calls.
inline AttentionKeyRange attention_key_range(int capacity,int cursor,int total_rows,
                                            int first,int rows,int window) {
  const int tile=scratch_rows(capacity);
  if(cursor<0 || cursor>=capacity || total_rows<1 || total_rows>capacity-cursor ||
     first<0 || first>=total_rows || rows<1 || rows>total_rows-first || rows>tile ||
     window<0 || (cursor>0 && total_rows!=1))
    throw std::runtime_error("invalid attention tile");
  const bool enabled=total_rows>4096 && total_rows>tile;
  if(enabled) {
    // Every query is in [cursor+first, cursor+first+rows). Future keys
    // beyond this tile are causal-masked for every query. For a sliding
    // layer, keys before the earliest query's window are also all masked.
    const int query_start=cursor+first;
    return {window?std::max(0,query_start-window+1):0,query_start+rows};
  }
  // Keep the validated eager shapes and dynamic one-token cache unchanged.
  return {(total_rows==1 && window>0)?std::max(0,cursor+rows-window):0,
          cursor+total_rows};
}
} // namespace lfm::text
