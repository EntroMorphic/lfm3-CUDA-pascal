# Memory per token, and how far context can go in 8 GB

Status (2026-11-22): [memory-v2](../validation/memory-v2/) also shares a layer's convolution and attention buffers
(of the hidden-width buffers and the convolution projection below, 30 KiB a token become 16).

Status (2026-11-18): the two changes under "Changes that keep the arithmetic" **are built and measured**
([memory-v1](../validation/memory-v1/), [long-decode-v1](../validation/long-decode-v1/)); the compact window
buffers are on by default above 4,096 tokens of capacity. The first table and "At 32,768 tokens" describe the
engine **before** those changes, as written on 2026-11-17. "LFM3-1.7B at 262,144 tokens" is still calculation only,
and is on the parking list in [PLAN.md](../PLAN.md).

## What the engine reserved before the changes

`Runner` (`native/src/lfm3_main.cu`) allocates every buffer once, for the requested capacity. Per token of
capacity, for the LFM3 backbone (hidden 2,048, MLP 7,680, 16 query heads, 4 key-value heads × 128, 11 attention
layers: 6 full, 5 with a 256-token window; 13 convolution layers):

| What | Bytes per token | KiB |
|---|---:|---:|
| MLP activations (`a`, `b`, `act`: 3 × 7,680, BF16) | 46,080 | 45.0 |
| Hidden-width work buffers (7 × 2,048, BF16) | 28,672 | 28.0 |
| Key/value cache (11 layers × keys and values × 512, BF16) | 22,528 | 22.0 |
| Convolution projection (`projected`: 3 × 2,048, BF16) | 12,288 | 12.0 |
| Per-step keys and values, gates, positions, tokens | 2,088 | 2.0 |
| Decode attention workspace (split-K P·V) | 256 | 0.25 |
| **Linear part** | **111,912** | **109.3** |
| Attention scores (FP32) and probabilities (BF16), capacity above 4,096: 128 query rows × 16 heads | 12,288 | 12.0 |
| **Above 4,096 tokens** | **124,200** | **121.3** |

At or below 4,096 tokens the score and probability buffers are a whole-prompt tile instead: quadratic in capacity,
with the heads per call chosen so the scores stay within 256 MiB (probabilities half that).

Only the key/value cache is state that persists between steps. The other 87 KiB per token is prefill scratch.

Check against the card (scorer, whole Omni-2B checkpoint resident at 4,134 MiB, 108 MiB CUDA context, and the
scorer's 256-row logit tile taken as 125 MiB):

| Capacity | Computed peak | Measured peak (`nvidia-smi`, 0.2 s samples) |
|---|---:|---:|
| 256 | 4,400 MiB | 4,407 MiB |
| 4,096 | 5,188 MiB | 5,195 MiB |

At 32,768 tokens the per-token buffers come to 3,881 MiB against about 3,737 MiB free: about 144 MiB short.

## Changes that keep the arithmetic

Both are meant to leave every output bit unchanged. The check for each: the same ids on the fixtures and
byte-identical NLL files at 256, 257 and 4,096 tokens as before the change.

### 1. Keys and values: 22 → 14 KiB per token

The 5 windowed layers keep every token's keys and values, but in decode they read only the last 256
(`lfm::text::attention_key_range`). They need a fixed buffer each, not one that grows with context.

| Part | Now | Needed |
|---|---:|---:|
| 6 full layers | 12 KiB per token | 12 KiB per token (every key is read at every step) |
| 5 windowed layers | 10 KiB per token | fixed: 2 × 256 tokens each, about 5 MiB in all |
| Prefill scratch for the windowed layers | — | 2 KiB per token, one buffer shared by the five |

- Prefill computes a whole prompt's keys one layer at a time and attention reads them in per-head layout, hence the
  one shared full-length buffer.
- Key order must be kept: a ring buffer that wraps would change the order keys enter the P·V sum, and so the
  result. A linear buffer of twice the window, compacted when full, keeps it.
- The full layers cannot shrink without storing keys and values at lower precision, which changes outputs and
  would have to pass the perplexity bar.

### 2. MLP activations: 45 KiB per token → about 11 MiB fixed

The MLP is row-independent (`w1` and `w3`, SwiGLU, `w2`), so a layer can run a block of rows at a time through three
small buffers (256 rows × 7,680 × 3, BF16 = 11.25 MiB).

- `lfm::gemm::run` picks its kernel by row count: 8 rows or fewer take the decode kernels, which sum in a different
  order from the prefill kernel. A last block of 1–8 rows would therefore change bits. The kernel must be chosen
  from the prompt's row count, not the block's.

### Effect at 32,768 tokens

| Change | Saves |
|---|---:|
| Keys and values | about 256 MiB |
| MLP tiling | about 1,430 MiB |
| Shortfall today | about 144 MiB |

### Further, less cleanly

The other prefill scratch (about 40 KiB per token) can also be tiled, with more care: a convolution row needs its
six predecessors, and attention needs a whole layer's keys. Loading only the language model's weights for text-only
use would free about 837 MiB of the 4,134.

## LFM3-1.7B at 262,144 tokens

LFM3-1.7B (a checkpoint in the owner's workspace, not part of this repository) is the same backbone as Omni-2B's language model, with
`max_position_embeddings` 262,144 (Omni-2B's is 128,000, and its card documents 32K). Its weights are 3,297 MiB.

| | MiB |
|---|---:|
| Free on the GTX 1080 with a CUDA context | 7,996 |
| Weights | −3,297 |
| Keys and values, 6 full layers: 12 KiB × 262,144 | −3,072 |
| Windowed layers (fixed) | −5 |
| **Left for everything else** | **about 1,620** |

It fits only with all three of:

1. **Change 1.** At 22 KiB per token the cache alone is 5,632 MiB.
2. **Row-tiled prefill scratch** (change 2 and the rest). At 121 KiB per token, 262,144 tokens need about 31 GiB.
3. **Chunked prefill.** A one-pass prefill keeps the hidden state of every row (4 KiB per token: 1,024 MiB) and the
   shared windowed-layer scratch (2 KiB per token: 512 MiB), leaving under 100 MiB. Prefilling the prompt in chunks
   through the cache avoids both. The engine cannot do this today: `Runner::forward` accepts an empty-cache prefill
   or a single-token decode. Hugging Face has such a path (`tools/ppl_reference.py --chunk-rows`), which is what a
   chunked engine would be compared with.

Not known:

- **Speed.** Attention cost grows with the square of the prompt and the card has no tensor cores. No measurement.
- **Quality.** Which Thor records exist for LFM3-1.7B at long context has not been checked.
- **Decode.** Each token reads all 3,072 MiB of keys and values; the time per token at that length is unmeasured.
