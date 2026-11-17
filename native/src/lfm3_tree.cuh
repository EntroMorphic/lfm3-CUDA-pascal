#pragma once
// Tree-verify decode (journal/lfm3-single-stream_synth.md): every pass fills its rows with each request's pending
// token plus a tree of guessed continuations, verifies them all in one weight read, and commits the longest path the
// model itself produces. Include after lfm3_main.cu and lfm3_batch.cuh (LFM3_RUNTIME_ONLY, LFM3_NATIVE_GEMM).
//
// Exactness by construction. Row r (slot s, depth d, ancestors path[r][0..d]) is the token at position q = c_s + d,
// where c_s is the slot's committed length. Its attention reads keys [begin, q] in order: positions < c_s from the
// slot's cache, then its ancestors' keys from the pass scratch, in the same 256-key splits relative to `begin` as
// lfm3_batch's attention; its conv window is the last 6 committed or ancestor inputs plus its own, in the same order
// and arithmetic as convolution7; everything else is row-wise. So every row's logits are bitwise what plain batched
// decode computes at that position after that prefix. Tokens are picked per row exactly as plain decode picks them
// (greedy, or sample_rows with the request's seed and step = generated + d), so following the tree while the pick
// has a matching child yields token for token the plain decode output, greedy or sampled, whatever the guesses.
#include <algorithm>
#include <map>

namespace lfm3_tree {

using lfm3_batch::kSplit;
constexpr int kMaxDepth = 64;

// K/V of every row into the pass scratch [row][kv_head][dim].
__global__ void scratch_kv(const bf16* k, const bf16* v, bf16* ks, bf16* vs, size_t n) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) ks[i] = k[i], vs[i] = v[i];
}

// The key/value of logical position j for row r: the slot cache below the committed length, else an ancestor's row.
__device__ __forceinline__ const bf16* kv_at(const bf16* cache, const bf16* scratch, int slot, int kvh, int kv_heads,
                                             int capacity, int committed, const int* path, int j, int dim) {
  if (j < committed) return cache + ((size_t(slot) * kv_heads + kvh) * capacity + j) * dim;
  return scratch + (size_t(path[j - committed]) * kv_heads + kvh) * dim;
}

// lfm3_batch::attention_partial with keys resolved per row (cache, then ancestors). Same arithmetic and order.
template <int GROUP>
__global__ void __launch_bounds__(256) tree_attention_partial(const bf16* q, const bf16* kc, const bf16* vc,
                                                              const bf16* ks, const bf16* vs, const int* slot_of,
                                                              const int* depth_of, const int* committed_of,
                                                              const int* paths, int window, int capacity, int kv_heads,
                                                              int max_splits, float* part_o, float* part_ml) {
  constexpr int D = 128;
  const int split = blockIdx.x, kvh = blockIdx.y, r = blockIdx.z;
  const int slot = slot_of[r], c = committed_of[slot], pos = c + depth_of[r];
  const int begin = window ? max(0, pos - window + 1) : 0, end = pos + 1;
  const int k0 = begin + split * kSplit;
  if (k0 >= end) return;
  const int n = min(kSplit, end - k0);
  const int* path = paths + size_t(r) * (kMaxDepth + 1);
  const int heads = kv_heads * GROUP, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  __shared__ float qs[GROUP][D];
  __shared__ float e[GROUP][kSplit];
  __shared__ float stats[GROUP][2];
  for (int i = threadIdx.x; i < GROUP * D; i += 256)
    qs[i / D][i % D] = __bfloat162float(q[(size_t(r) * heads + kvh * GROUP + i / D) * D + i % D]);
  __syncthreads();
  const float scale = rsqrtf(float(D));
  for (int j = warp; j < n; j += 8) {
    const bf16* key = kv_at(kc, ks, slot, kvh, kv_heads, capacity, c, path, k0 + j, D);
    const uint2 raw = *reinterpret_cast<const uint2*>(key + lane * 4);
    const __nv_bfloat162* k2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
    const float2 ka = __bfloat1622float2(k2[0]), kb = __bfloat1622float2(k2[1]);
#pragma unroll
    for (int g = 0; g < GROUP; ++g) {
      const float* qq = qs[g] + lane * 4;
      float s = __fmul_rn(qq[0], ka.x);
      s = __fmaf_rn(qq[1], ka.y, s);
      s = __fmaf_rn(qq[2], kb.x, s);
      s = __fmaf_rn(qq[3], kb.y, s);
#pragma unroll
      for (int o = 16; o; o >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, o));
      if (lane == 0) e[g][j] = __fmul_rn(s, scale);
    }
  }
  __syncthreads();
  if (warp < GROUP) {
    const int g = warp;
    float m = -INFINITY;
    for (int j = lane; j < n; j += 32) m = fmaxf(m, e[g][j]);
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float l = 0.f;
    for (int j = lane; j < n; j += 32) {
      const float x = expf(__fsub_rn(e[g][j], m));
      e[g][j] = x;
      l = __fadd_rn(l, x);
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) l = __fadd_rn(l, __shfl_xor_sync(0xffffffffu, l, o));
    if (lane == 0) stats[g][0] = m, stats[g][1] = l;
  }
  __syncthreads();
  for (int g = threadIdx.x / 64; g < GROUP; g += 4) {
    const int d = (threadIdx.x % 64) * 2;
    float o0 = 0.f, o1 = 0.f;
    for (int j = 0; j < n; ++j) {
      const bf16* value = kv_at(vc, vs, slot, kvh, kv_heads, capacity, c, path, k0 + j, D);
      const float2 vv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(value + d));
      const float p = e[g][j];
      o0 = __fmaf_rn(p, vv.x, o0);
      o1 = __fmaf_rn(p, vv.y, o1);
    }
    const size_t idx = (size_t(r) * heads + kvh * GROUP + g) * max_splits + split;
    part_o[idx * D + d] = o0;
    part_o[idx * D + d + 1] = o1;
    if (d == 0) part_ml[idx * 2] = stats[g][0], part_ml[idx * 2 + 1] = stats[g][1];
  }
}

__global__ void tree_attention_combine(const float* part_o, const float* part_ml, const int* slot_of,
                                       const int* depth_of, const int* committed_of, int window, int heads,
                                       int max_splits, bf16* out) {
  constexpr int D = 128;
  const int h = blockIdx.x, r = blockIdx.y, d = threadIdx.x;
  const int pos = committed_of[slot_of[r]] + depth_of[r], begin = window ? max(0, pos - window + 1) : 0;
  const int splits = (pos + 1 - begin + kSplit - 1) / kSplit;
  const size_t base = (size_t(r) * heads + h) * max_splits;
  float m = -INFINITY;
  for (int s = 0; s < splits; ++s) m = fmaxf(m, part_ml[(base + s) * 2]);
  float l = 0.f, acc = 0.f;
  for (int s = 0; s < splits; ++s) {
    const float w = expf(__fsub_rn(part_ml[(base + s) * 2], m));
    l = __fmaf_rn(part_ml[(base + s) * 2 + 1], w, l);
    acc = __fmaf_rn(part_o[(base + s) * D + d], w, acc);
  }
  out[(size_t(r) * heads + h) * D + d] = __float2bfloat16_rn(__fdiv_rn(acc, l));
}

// convolution7 for tree rows: the window is the last 6 inputs along (committed state, ancestors) plus the row's own;
// the committed state is not modified (tree_commit_conv shifts the accepted inputs in).
__global__ void tree_conv7(const bf16* projected, const bf16* weights, const bf16* state, bf16* output, int width,
                           int rows, const int* slot_of, const int* depth_of, const int* paths) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(rows) * width) return;
  const int r = i / width, d = i % width, depth = depth_of[r];
  const int* path = paths + size_t(r) * (kMaxDepth + 1);
  const bf16* st = state + (size_t(slot_of[r]) * width + d) * 7;
  auto input = [&](int row) {
    const size_t base = size_t(row) * 3 * width + d;
    return __hmul(projected[base], projected[base + 2 * width]);
  };
  // the sequence: st[0..6] (oldest..newest), then path[0..depth] (the last is this row)
  const int len = 7 + depth + 1;
  float products[8];
  for (int k = 0; k < 7; ++k) {
    const int idx = len - 7 + k;
    const bf16 value = idx < 7 ? st[idx] : input(path[idx - 7]);
    products[k] = __fmul_rn(__bfloat162float(value), __bfloat162float(weights[d * 7 + k]));
  }
  products[7] = 0.0f;
  const float even = __fadd_rn(__fadd_rn(products[0], products[4]), __fadd_rn(products[2], products[6]));
  const float odd = __fadd_rn(__fadd_rn(products[1], products[5]), __fadd_rn(products[3], products[7]));
  const float sum = __fadd_rn(even, odd);
  output[i] = __hmul(projected[size_t(r) * 3 * width + width + d], __float2bfloat16_rn(sum));
}

// Commit: for each slot, its accepted rows' K/V into the cache at their positions, and their conv inputs shifted
// into the state. accept[slot] lists `count[slot]` rows (root first), at positions committed..committed+count-1.
__global__ void tree_commit_kv(const bf16* ks, const bf16* vs, bf16* kc, bf16* vc, const int* accept_rows,
                               const int* accept_slot, const int* accept_pos, int accepted, int kv_heads, int dim,
                               int capacity) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t per = size_t(kv_heads) * dim;
  if (i >= size_t(accepted) * per) return;
  const int a = i / per, e = i % per, h = e / dim, d = e % dim;
  const size_t dst = ((size_t(accept_slot[a]) * kv_heads + h) * capacity + accept_pos[a]) * dim + d;
  kc[dst] = ks[size_t(accept_rows[a]) * per + e];
  vc[dst] = vs[size_t(accept_rows[a]) * per + e];
}
__global__ void tree_commit_conv(const bf16* projected, bf16* state, int width, const int* slots, const int* first,
                                 const int* count, const int* accept_rows, int nslots) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(nslots) * width) return;
  const int k = i / width, d = i % width, s = slots[k], n = count[k];
  bf16* st = state + (size_t(s) * width + d) * 7;
  bf16 seq[7 + kMaxDepth + 1];
  for (int j = 0; j < 7; ++j) seq[j] = st[j];
  for (int j = 0; j < n; ++j) {
    const size_t base = size_t(accept_rows[first[k] + j]) * 3 * width + d;
    seq[7 + j] = __hmul(projected[base], projected[base + 2 * width]);
  }
  for (int j = 0; j < 7; ++j) st[j] = seq[n + j];
}

// One row of a pass: its slot, its token, its parent row (-1 for the root) and its depth.
struct Node { int slot, token, parent, depth; };

class TreeDecoder {
 public:
  Runner& base;
  const lfm::RuntimeShape s;
  const int slots, max_rows;
  Buffer<bf16> hidden, normalized, projected, operated, updated, q, k, v, gates, attended, gated, a, b, act;
  Buffer<float> logits, part_o, part_ml;
  Buffer<int> tokens, positions, slot_dev, depth_dev, committed_dev, paths_dev, picked, acc_rows, acc_slot, acc_pos,
      cs_slots, cs_first, cs_count;
  Buffer<unsigned char> active_dev;
  Buffer<uint64_t> seeds_dev, steps_dev;
  std::vector<std::unique_ptr<Buffer<bf16>>> keys, values, states;
  std::vector<int> committed;
  int max_splits;

  TreeDecoder(Runner& runner, int nslots, int rows)
      : base(runner), s(runner.s), slots(nslots), max_rows(rows),
        hidden(size_t(rows) * s.hidden), normalized(size_t(rows) * s.hidden), projected(size_t(rows) * 3 * s.hidden),
        operated(size_t(rows) * s.hidden), updated(size_t(rows) * s.hidden), q(size_t(rows) * s.hidden),
        k(size_t(rows) * s.kv_width()), v(size_t(rows) * s.kv_width()), gates(size_t(rows) * s.query_heads),
        attended(size_t(rows) * s.hidden), gated(size_t(rows) * s.hidden), a(size_t(rows) * s.mlp),
        b(size_t(rows) * s.mlp), act(size_t(rows) * s.mlp), logits(size_t(rows) * s.vocabulary),
        part_o(size_t(rows) * s.query_heads * ((s.context + kSplit - 1) / kSplit) * 128),
        part_ml(size_t(rows) * s.query_heads * ((s.context + kSplit - 1) / kSplit) * 2),
        tokens(rows), positions(rows), slot_dev(rows), depth_dev(rows), committed_dev(nslots),
        paths_dev(size_t(rows) * (kMaxDepth + 1)), picked(rows), acc_rows(rows), acc_slot(rows), acc_pos(rows),
        cs_slots(nslots), cs_first(nslots), cs_count(nslots), active_dev(rows), seeds_dev(rows), steps_dev(rows),
        committed(nslots, 0), max_splits((s.context + kSplit - 1) / kSplit) {
    if (nslots < 1 || rows < nslots || rows > 256) throw std::invalid_argument("tree decoder: 1 <= slots <= rows <= 256");
    if (s.head_dim != 128 || s.query_heads / s.kv_heads != 4) throw std::invalid_argument("tree decoder: head_dim 128, group 4");
    for (int i = 0; i < s.layers; ++i) {
      if (base.config.attention[i]) {
        keys.emplace_back(new Buffer<bf16>(size_t(nslots) * s.kv_layer_stride()));
        values.emplace_back(new Buffer<bf16>(size_t(nslots) * s.kv_layer_stride()));
        states.emplace_back(nullptr);
      } else {
        keys.emplace_back(nullptr), values.emplace_back(nullptr);
        states.emplace_back(new Buffer<bf16>(size_t(nslots) * s.hidden * 7));
      }
    }
    cuda_check(cudaMemset(active_dev.p, 1, rows));
    lfm3_batch::pack_model(base);
  }

  // Slot `slot` takes the base runner's state after its prefill (KV, conv states, committed length). The prefill's
  // last-position logits are the caller's: the first token is picked from them (as plain decode does).
  void load(int slot) {
    if (slot < 0 || slot >= slots || base.cursor < 1) throw std::invalid_argument("tree load: invalid slot");
    for (int i = 0; i < s.layers; ++i) {
      if (keys[i]) {
        cuda_check(cudaMemcpy(keys[i]->p + size_t(slot) * s.kv_layer_stride(), base.keys[i]->p,
                              s.kv_layer_stride() * sizeof(bf16), cudaMemcpyDeviceToDevice));
        cuda_check(cudaMemcpy(values[i]->p + size_t(slot) * s.kv_layer_stride(), base.values[i]->p,
                              s.kv_layer_stride() * sizeof(bf16), cudaMemcpyDeviceToDevice));
      } else {
        cuda_check(cudaMemcpy(states[i]->p + size_t(slot) * s.hidden * 7, base.states[i]->p,
                              size_t(s.hidden) * 7 * sizeof(bf16), cudaMemcpyDeviceToDevice));
      }
    }
    committed[slot] = base.cursor;
  }

  // One verification pass over `nodes` (each slot's root first, parents before children). Writes every row's logits.
  void pass(const std::vector<Node>& nodes) {
    const int R = int(nodes.size()), H = s.hidden;
    if (R < 1 || R > max_rows) throw std::invalid_argument("tree pass: row count");
    std::vector<int> tok(R), pos(R), slot(R), dep(R), paths(size_t(R) * (kMaxDepth + 1), -1);
    int longest = 0;
    for (int r = 0; r < R; ++r) {
      const Node& n = nodes[r];
      if (n.depth > kMaxDepth || (n.parent < 0) != (n.depth == 0) || (n.parent >= r) ||
          (n.parent >= 0 && (nodes[n.parent].slot != n.slot || nodes[n.parent].depth != n.depth - 1)))
        throw std::invalid_argument("tree pass: malformed tree");
      tok[r] = n.token, slot[r] = n.slot, dep[r] = n.depth, pos[r] = committed[n.slot] + n.depth;
      if (pos[r] >= s.context) throw std::runtime_error("tree pass: capacity");
      longest = std::max(longest, pos[r] + 1);
      int* p = paths.data() + size_t(r) * (kMaxDepth + 1);
      for (int x = r, d = n.depth; d >= 0; x = nodes[x].parent, --d) p[d] = x;
    }
    cuda_check(cudaMemcpy(tokens.p, tok.data(), R * 4, cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(positions.p, pos.data(), R * 4, cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(slot_dev.p, slot.data(), R * 4, cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(depth_dev.p, dep.data(), R * 4, cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(committed_dev.p, committed.data(), slots * 4, cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(paths_dev.p, paths.data(), paths.size() * 4, cudaMemcpyHostToDevice));
    const int t_hidden = norm_threads(1, H), t_q = norm_threads(s.query_heads, s.head_dim),
              t_k = norm_threads(s.kv_heads, s.head_dim);
    const int full_splits = std::max(1, (longest + kSplit - 1) / kSplit);
    auto w = [&](const std::string& n) { return base.w(n); };
    using lfm3_batch::linear;
    embed<<<grid(size_t(R) * H), 256>>>(w("embed_tokens.weight"), tokens.p, hidden.p, R, H);
    for (int layer = 0; layer < s.layers; ++layer) {
      const std::string p = "layers." + std::to_string(layer) + ".";
      const auto spec = base.config.layer_specs[layer];
      norm<<<R, t_hidden>>>(hidden.p, w(p + "operator_norm.weight"), normalized.p, H, s.epsilon);
      if (!base.config.attention[layer]) {
        linear(normalized.p, w(p + "conv.in_proj.weight"), projected.p, R, H, 3 * H);
        // the pass's conv inputs (projected) are kept per layer: the commit shifts the accepted ones into the state
        tree_conv7<<<grid(size_t(R) * H), 256>>>(projected.p, w(p + "conv.conv.weight"), states[layer]->p, operated.p,
                                                 H, R, slot_dev.p, depth_dev.p, paths_dev.p);
        save_conv_inputs(layer, R);
        linear(operated.p, w(p + "conv.out_proj.weight"), updated.p, R, H, H);
      } else {
        const std::string at = p + "self_attn.";
        linear(normalized.p, w(at + "q_proj.weight"), q.p, R, H, H);
        linear(normalized.p, w(at + "k_proj.weight"), k.p, R, H, s.kv_width());
        linear(normalized.p, w(at + "v_proj.weight"), v.p, R, H, s.kv_width());
        norm<<<R * s.query_heads, t_q>>>(q.p, w(at + "q_norm.weight"), q.p, s.head_dim, s.epsilon);
        norm<<<R * s.kv_heads, t_k>>>(k.p, w(at + "k_norm.weight"), k.p, s.head_dim, s.epsilon);
        rotary_from_frequencies<<<grid(size_t(R) * H / 2), 256>>>(q.p, positions.p, base.frequencies[layer]->p, R,
                                                                  s.query_heads, s.head_dim, spec.rotary_pairs);
        rotary_from_frequencies<<<grid(size_t(R) * s.kv_width() / 2), 256>>>(k.p, positions.p, base.frequencies[layer]->p,
                                                                             R, s.kv_heads, s.head_dim, spec.rotary_pairs);
        save_kv(layer, R);
        const int splits = spec.window ? (spec.window + kSplit - 1) / kSplit : full_splits;
        tree_attention_partial<4><<<dim3(splits, s.kv_heads, R), 256>>>(
            q.p, keys[layer]->p, values[layer]->p, layer_ks(layer), layer_vs(layer), slot_dev.p, depth_dev.p,
            committed_dev.p, paths_dev.p, spec.window, s.context, s.kv_heads, max_splits, part_o.p, part_ml.p);
        tree_attention_combine<<<dim3(s.query_heads, R), 128>>>(part_o.p, part_ml.p, slot_dev.p, depth_dev.p,
                                                                committed_dev.p, spec.window, s.query_heads, max_splits,
                                                                attended.p);
        linear(normalized.p, w(at + "gate_proj.weight"), gates.p, R, H, s.query_heads);
        lfm::lfm3::head_gate<<<grid(size_t(R) * H), 256>>>(attended.p, gates.p, gated.p, size_t(R) * H, s.head_dim);
        linear(gated.p, w(at + "o_proj.weight"), updated.p, R, H, H);
      }
      norm<<<R, t_hidden>>>(updated.p, w(p + "post_operator_norm.weight"), hidden.p, H, s.epsilon, hidden.p);
      norm<<<R, t_hidden>>>(hidden.p, w(p + "ffn_norm.weight"), normalized.p, H, s.epsilon);
      linear(normalized.p, w(p + "feed_forward.w1.weight"), a.p, R, H, s.mlp);
      linear(normalized.p, w(p + "feed_forward.w3.weight"), b.p, R, H, s.mlp);
      swiglu<<<grid(size_t(R) * s.mlp), 256>>>(a.p, b.p, act.p, size_t(R) * s.mlp);
      linear(act.p, w(p + "feed_forward.w2.weight"), updated.p, R, s.mlp, H);
      norm<<<R, t_hidden>>>(updated.p, w(p + "post_ffn_norm.weight"), hidden.p, H, s.epsilon, hidden.p);
    }
    norm<<<R, t_hidden>>>(hidden.p, w("embedding_norm.weight"), normalized.p, H, s.epsilon);
    linear(normalized.p, w("embed_tokens.weight"), nullptr, R, H, s.vocabulary, logits.p);
    cuda_check(cudaGetLastError());
    rows_in_pass = R;
  }

  // Picks per row: greedy, or sample_rows with each row's (seed, step).
  std::vector<int> pick(int R, double temperature, const std::vector<uint64_t>& seed, const std::vector<uint64_t>& step,
                        const unsigned char* suppress_dev = nullptr) {
    if (temperature > 0) {
      cuda_check(cudaMemcpy(seeds_dev.p, seed.data(), R * 8, cudaMemcpyHostToDevice));
      cuda_check(cudaMemcpy(steps_dev.p, step.data(), R * 8, cudaMemcpyHostToDevice));
      lfm3_batch::sample_rows<<<R, 1024>>>(logits.p, s.vocabulary, suppress_dev, active_dev.p, temperature, seeds_dev.p,
                                           steps_dev.p, picked.p);
    } else {
      lfm3_batch::greedy_rows<<<R, 1024>>>(logits.p, s.vocabulary, suppress_dev, active_dev.p, picked.p);
    }
    cuda_check(cudaGetLastError());
    std::vector<int> out(R);
    cuda_check(cudaMemcpy(out.data(), picked.p, R * 4, cudaMemcpyDeviceToHost));
    return out;
  }

  // Commit the accepted rows of each slot (root first, consecutive depths): their K/V and conv inputs, from the last
  // pass. accepted: per slot, the row list.
  void commit(const std::vector<std::vector<int>>& accepted) {
    std::vector<int> rows, slot_of, pos, cslots, first, count;
    for (int sl = 0; sl < slots; ++sl) {
      const auto& a = accepted[sl];
      if (a.empty()) continue;
      cslots.push_back(sl), first.push_back(int(rows.size())), count.push_back(int(a.size()));
      for (size_t j = 0; j < a.size(); ++j) rows.push_back(a[j]), slot_of.push_back(sl), pos.push_back(committed[sl] + int(j));
    }
    if (rows.empty()) return;
    const int n = int(rows.size()), ns = int(cslots.size());
    cuda_check(cudaMemcpy(acc_rows.p, rows.data(), n * 4, cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(acc_slot.p, slot_of.data(), n * 4, cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(acc_pos.p, pos.data(), n * 4, cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(cs_slots.p, cslots.data(), ns * 4, cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(cs_first.p, first.data(), ns * 4, cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(cs_count.p, count.data(), ns * 4, cudaMemcpyHostToDevice));
    for (int i = 0; i < s.layers; ++i) {
      if (keys[i]) {
        tree_commit_kv<<<grid(size_t(n) * s.kv_width()), 256>>>(layer_ks(i), layer_vs(i), keys[i]->p, values[i]->p,
                                                                acc_rows.p, acc_slot.p, acc_pos.p, n, s.kv_heads,
                                                                s.head_dim, s.context);
      } else {
        tree_commit_conv<<<grid(size_t(ns) * s.hidden), 256>>>(layer_inputs(i), states[i]->p, s.hidden, cs_slots.p,
                                                               cs_first.p, cs_count.p, acc_rows.p, ns);
      }
    }
    cuda_check(cudaGetLastError());
    for (int k2 = 0; k2 < ns; ++k2) committed[cslots[k2]] += count[k2];
  }

 private:
  int rows_in_pass = 0;
  // Per layer: the pass's K/V (attention layers) and projected rows (conv layers), kept until the commit.
  std::vector<std::unique_ptr<Buffer<bf16>>> pass_ks, pass_vs, pass_proj;
  void ensure_layer_buffers() {
    if (!pass_ks.empty()) return;
    for (int i = 0; i < s.layers; ++i) {
      if (base.config.attention[i]) {
        pass_ks.emplace_back(new Buffer<bf16>(size_t(max_rows) * s.kv_width()));
        pass_vs.emplace_back(new Buffer<bf16>(size_t(max_rows) * s.kv_width()));
        pass_proj.emplace_back(nullptr);
      } else {
        pass_ks.emplace_back(nullptr), pass_vs.emplace_back(nullptr);
        pass_proj.emplace_back(new Buffer<bf16>(size_t(max_rows) * 3 * s.hidden));
      }
    }
  }
  bf16* layer_ks(int i) { ensure_layer_buffers(); return pass_ks[i]->p; }
  bf16* layer_vs(int i) { ensure_layer_buffers(); return pass_vs[i]->p; }
  bf16* layer_inputs(int i) { ensure_layer_buffers(); return pass_proj[i]->p; }
  void save_kv(int layer, int R) {
    scratch_kv<<<grid(size_t(R) * s.kv_width()), 256>>>(k.p, v.p, layer_ks(layer), layer_vs(layer), size_t(R) * s.kv_width());
  }
  void save_conv_inputs(int layer, int R) {
    cuda_check(cudaMemcpyAsync(layer_inputs(layer), projected.p, size_t(R) * 3 * s.hidden * sizeof(bf16),
                               cudaMemcpyDeviceToDevice));
  }
};

// Free drafts: continuations that followed earlier occurrences of the context's longest matching suffix (n <= 8),
// most recent first, merged into a token tree of at most `budget` nodes (prompt lookup; tools/lfm3_acceptance.py).
// Returns nodes as (token, parent index in the returned list or -1 for a child of the root, depth >= 1).
struct Draft { int token, parent, depth; };
inline std::vector<Draft> ngram_tree(const std::vector<int>& ctx, int budget, int max_depth) {
  std::vector<Draft> out;
  if (budget <= 0 || ctx.size() < 2) return out;
  std::vector<std::vector<int>> conts;
  const int L = int(ctx.size());
  for (int n = std::min(8, L - 1); n >= 1 && conts.empty(); --n) {
    for (int j = L - n - 1; j >= 0; --j) {
      bool same = true;
      for (int t = 0; t < n && same; ++t) same = ctx[j + t] == ctx[L - n + t];
      if (!same) continue;
      std::vector<int> c(ctx.begin() + j + n, ctx.begin() + std::min(L, j + n + max_depth));
      if (!c.empty()) conts.push_back(std::move(c));
    }
  }
  // trie: children keyed by (parent, token)
  std::map<std::pair<int, int>, int> child;
  for (const auto& c : conts) {
    int parent = -1;
    for (size_t d = 0; d < c.size(); ++d) {
      auto it = child.find({parent, c[d]});
      if (it == child.end()) {
        if (int(out.size()) >= budget) break;
        out.push_back({c[d], parent, int(d) + 1});
        it = child.emplace(std::make_pair(parent, c[d]), int(out.size()) - 1).first;
      }
      parent = it->second;
    }
  }
  return out;
}

}  // namespace lfm3_tree
