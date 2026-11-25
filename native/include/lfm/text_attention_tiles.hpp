#pragma once
#include <lfm/text_serving_limits.hpp>
#include <algorithm>
#include <stdexcept>

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
// A continuation's tile (continuation-v1): rows [query_start, query_start+rows) of a prompt of `prompt_rows` positions
// whose first positions are already cached, extended in one call (Runner::extend). It reads the keys the same rows
// read when the whole prompt is one prefill: those of the prefill's tile that holds them (tiles start at multiples of
// scratch_rows(capacity) from position 0), so every row's scores, softmax and products have the prefill's spans.
inline AttentionKeyRange prefill_key_range(int capacity,int prompt_rows,int query_start,int rows,int window) {
  const int tile=scratch_rows(capacity);
  if(prompt_rows<1 || prompt_rows>capacity || query_start<0 || rows<1 || rows>prompt_rows-query_start ||
     window<0 || query_start/tile!=(query_start+rows-1)/tile)
    throw std::runtime_error("invalid continuation tile");
  const int tile_start=query_start/tile*tile;
  if(prompt_rows>4096 && prompt_rows>tile)
    return {window?std::max(0,tile_start-window+1):0,std::min(tile_start+tile,prompt_rows)};
  return {0,prompt_rows};
}
} // namespace lfm::text
