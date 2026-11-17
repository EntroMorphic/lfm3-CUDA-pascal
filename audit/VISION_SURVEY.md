# LFM3-Omni-2B vision path: survey for a native C/CUDA port

Read-only survey, 2026-11-13. Nothing was run on a GPU. The only computation was host-side Python that copies the
`smart_resize` arithmetic to get the example numbers below, plus a parse of the safetensors JSON header.

Paths: `M/` = `models/LFM3-Omni-2B/`, `N/` = `liquid-CUDA/native/`.

**One caveat up front.** `get_vision_position_ids`, `get_vision_attention_seqlens` and `get_max_seqlen` are imported
from `transformers.vision_utils` (M/modeling_lfm3_vl.py:48). The image and video processor base classes
(`TorchvisionBackend`, `BaseVideoProcessor`, `ProcessorMixin` expansion) also come from transformers. The checkpoint
targets transformers 5.19.0.dev0 (config.json `transformers_version`), and the README asks for `>=5.20.2`
(M/README.md:219). No 5.x transformers is installed on this machine; only 4.57.3 and 4.57.6 are. So **those helpers
are not checked here**. The behaviour below is inferred from how they are called, and each inferred point is
labelled. Before any kernel work, dump their outputs from a real HF 5.20 install (stage 0 in §9).

---

## 0. Summary in numbers

| | value | source |
|---|---|---|
| patch | 16×16 px, 3 ch, temporal 2 → patch vector 1536 | config.json vision_config; M/modeling_lfm3_vl.py:472 |
| resize factor | 32 px (patch 16 × merge 2) | M/image_processing_lfm3_vl.py:225 |
| image pixel budget | min 65,536 (256²) and max 2,097,152 px → **64 to 2,048 LM tokens per image** | M/processor_config.json:44-46 |
| encoder | 14 layers, width 1024: 8 gated depthwise-conv layers, 6 bidirectional GQA attention layers (16 q / 8 kv heads, head_dim 64), SwiGLU 2560, RMSNorm eps 1e-5, 2D axial RoPE θ=1e6 | config.json vision_config |
| attention scope | one image, or one temporal-pair "frame" of a video; no cross-frame attention | M/modeling_lfm3_vl.py:195-305 |
| projector | LayerNorm(1024, eps 1e-6) → concatenate each 2×2 block (4096) → Linear 4096→4096 → exact GELU → Linear 4096→2048 | M/modeling_lfm3_vl.py:515-537 |
| tokens per image | `(H'/16)·(W'/16)/4 = H'·W'/1024` | M/processing_lfm3_omni.py:754-756 |
| prompt form, image | `<\|vision_start\|>` + `<\|image\|>`×N + `<\|vision_end\|>` (ids 125009, 125030, 125010) | M/processing_lfm3_omni.py:491-492, 754-756 |
| weights | vision tower 164,206,336 params; projector 25,174,016; **total 189,380,352 params = 378,760,704 B, all BF16** | safetensors header |
| LM positions | ordinary 1-D positions (no M-RoPE); image tokens take consecutive positions | M/modeling_lfm3.py:676 |

---

## 1. Image preprocessing

### 1.1 Settings that apply (processor_config.json overrides the class defaults)
- `do_convert_rgb=True`, `do_resize=True`, `resample=3` (BICUBIC), `do_rescale=True`, `rescale_factor=1/255`
  (0.00392156862745098), `do_normalize=True`, `image_mean=image_std=[0.5,0.5,0.5]`, `patch_size=16`,
  `merge_size=2`, `temporal_patch_size=2`, `size={shortest_edge: 65536, longest_edge: 2097152}`
  (M/processor_config.json:24-52).
- The class default `longest_edge` is 16,777,216 (M/image_processing_lfm3_vl.py:83), but the config sets
  **2,097,152**. The README's "up to 2,048 tokens per image" (M/README.md:462) agrees with the config:
  2,097,152 / 32² = 2048.
- The names are misleading: `shortest_edge` and `longest_edge` are **pixel counts** (min_pixels and max_pixels), not
  edge lengths (M/image_processing_lfm3_vl.py:97-106, 151-157).
- No tiling, no thumbnail, no crop. The tokenizer still has `<|img_thumbnail|>` (125008), `<|image_split|>` (125011)
  and `<|img_row_R_col_C|>` tokens from the LFM2-VL tiling scheme, but **no code on this path emits them**. The
  whole image is resized once to a multiple of 32 at roughly native resolution.

### 1.2 smart_resize (M/image_processing_lfm3_vl.py:50-76), with factor = 32 (line 225)
```
def smart_resize(H, W, factor=32, min_px=65536, max_px=2097152):
    if max(H,W)/min(H,W) > 200: error                       # float division
    h = round(H/factor)*factor                               # Python round(): ROUND HALF TO EVEN on a float64
    w = round(W/factor)*factor
    if h*w > max_px:
        beta = sqrt(H*W / max_px)                            # float64
        h = max(factor, floor(H/beta/factor)*factor)         # note: H/beta, then /factor (two float64 divides)
        w = max(factor, floor(W/beta/factor)*factor)
    elif h*w < min_px:
        beta = sqrt(min_px / (H*W))
        h = ceil(H*beta/factor)*factor
        w = ceil(W*beta/factor)*factor
    return h, w
```
Points that matter for a bit-exact port:
- `round` is **banker's rounding**. 1040 px → 1040/32 = 32.5 → 32 → **1024**, where round-half-up would give 1056.
  In C use `nearbyint` under the default rounding mode, not `round`.
- The `beta` branches run in IEEE float64 in this exact order of operations. `floor` and `ceil` sit right on
  boundaries. For example 224×224 → 224·sqrt(65536/50176)/32 evaluates to exactly 8.0 → 256. Change the evaluation
  order and that can become 8.000…1 → 288. Copy the expression order exactly, in `double`.
- The down branch does not re-check the result against `max_px`, and the up branch does not re-check `min_px`.
  Neither do the token counts.

Worked examples (host Python of the formula above; H×W given as height×width):

| input H×W | resized H'×W' | branch | patch grid | patches | LM tokens |
|---|---|---|---|---|---|
| 512×512 | 512×512 | none | 32×32 | 1024 | 256 |
| 1080×1920 | **1088×1920** (upscaled 8 rows) | none | 68×120 | 8160 | 2040 |
| 1920×1080 | 1920×1088 | none | 120×68 | 8160 | 2040 |
| 2160×3840 | 1056×1920 | down | 66×120 | 7920 | 1980 |
| 3000×4000 | 1248×1664 | down | 78×104 | 8112 | 2028 |
| 480×640 | 480×640 | none | 30×40 | 1200 | 300 |
| 1040×1040 | 1024×1024 (banker's) | none | 64×64 | 4096 | 1024 |
| 224×224 | 256×256 | up | 16×16 | 256 | 64 |
| 100×10000 | 96×9984 | none | 6×624 | 3744 | 936 |

### 1.3 Pixel pipeline
```
img  = decode -> RGB uint8 [3][H][W]                          # do_convert_rgb
img  = resize(img, (H', W'), BICUBIC, antialias)              # torchvision.v2 functional on a uint8 tensor -> uint8
x    = (float32(img) - 127.5) / 127.5                         # fused rescale+normalize, float32  [see note]
# patchify (M/image_processing_lfm3_vl.py:164-197), gh=H'/16, gw=W'/16:
for bh in 0..gh/2-1, bw in 0..gw/2-1, mh in 0..1, mw in 0..1:          # row order = merge-block major
    r = ((bh*(gw/2) + bw)*2 + mh)*2 + mw
    h = 2*bh+mh ; w = 2*bw+mw
    for c in 0..2, t in 0..1, py in 0..15, px in 0..15:                 # column order (C, T, py, px)
        pixel_values[r][((c*2 + t)*16 + py)*16 + px] = x[c][16h+py][16w+px]   # t duplicated for an image
image_grid_thw = [1, gh, gw]
```
- **Row order is merge-block major, not raster.** Each run of 4 consecutive rows is one 2×2 block. Everything
  downstream depends on this: the conv layer un-permutes it (§3.3), and the projector concatenates runs of 4 rows
  (§4).
- For images the frame is **duplicated** along T, with `expand` (M/image_processing_lfm3_vl.py:188-196).
  Optimisation: on the native side, an image could use a patchifier with the two temporal halves pre-summed,
  `W'[:, c, py, px] = W[:, c, 0, py, px] + W[:, c, 1, py, px]`, giving K=768. The sum of two BF16 values is
  almost always exact in FP32. This changes the rounding order relative to HF's K=1536 GEMM, so it belongs to the
  "at least as accurate" class of differences, not bit-exact.
- **Not verifiable locally (5.x `TorchvisionBackend`):**
  - Resize: in 4.57.6 the fast processor calls `F.resize(uint8 tensor, size, interpolation, antialias=True)`
    (miniconda3 …/image_processing_utils_fast.py:310-360). 5.x is presumably the same. That is torchvision's
    **antialiased bicubic on uint8**: Keys a=-0.5, support scaled by the downscale factor, fixed-point uint8
    arithmetic with round and clamp. It is the hardest piece to match bit-for-bit. Pillow's `Image.resize(BICUBIC)`
    is a similar but not identical algorithm.
  - Normalize: in 4.57 rescale and normalize are fused as `(x - 0.5·255) / (0.5·255)` in float32 (same file,
    :434-476), i.e. `(u8 - 127.5) / 127.5`. 5.x is unverified.
  - RGB conversion: in 4.57 this is PIL `image.convert("RGB")`, which drops alpha without compositing
    (…/image_transforms.py:781-798). 5.x is unverified.
- The model then casts `pixel_values` to BF16 before the patchifier (M/modeling_lfm3_vl.py:498). In the HF
  reference the normalized pixels are rounded to BF16 (8 mantissa bits). Per-pixel steps are 1/127.5, so above
  |x| = 0.5 the BF16 grid (spacing 2⁻⁸ … 2⁻⁷) is coarser than the uint8 steps, and HF loses input precision there.

---

## 2. Video preprocessing (M/video_processing_lfm3_vl.py; settings from M/video_preprocessor_config.json)

Settings: `fps=2.0`, `min_frames=4`, `max_frames=384` (class default 768, M/video_processing_lfm3_vl.py:138; config
:117), `size={shortest_edge: 4096, longest_edge: 25,165,824}` pixels **summed over all frames**,
`cap_pixels_per_frame=False`, `max_video_tokens=768` (used only when the cap is on), BICUBIC, mean and std 0.5.

### 2.1 Frame sampling (`sample_frames`, M/video_processing_lfm3_vl.py:152-182)
```
n = int(total_frames / clip_fps * 2.0)                 # truncation
n = min(max(n, 4), 384, total_frames)
idx = numpy.linspace(0, total_frames-1, n).round()     # numpy round = half-to-even
```
- If metadata has no fps, the shared sampler assumes 24 (:168-173). The **Omni processor refuses** decoded frames
  without fps (M/processing_lfm3_omni.py:731-744).
- `fps` is rounded to 6 decimals before timestamps are computed (M/processing_lfm3_omni.py:706).

### 2.2 Resize: one size for the whole clip (`smart_resize`, :71-104)
- Same as for images, except: tiny frames (< 32 px) are first scaled up with `int()` truncation (:82-85).
  `t_bar = round(n/2)*2` (banker's) enters only the threshold test. Both the test and `beta` use frame counts
  (`beta = sqrt(n·H·W / 25,165,824)`, using the **unpadded** `n`).
- With `cap_pixels_per_frame=True` (off in this config) the budget becomes
  `max(min(768·32², 25,165,824 // n), int(4096·1.05))·n` (:200-204).
- Uncapped (the configured default), short clips keep near-native resolution. Examples:

| clip | frames | per-frame size | patches/frame | tokens/group | groups | video tokens |
|---|---|---|---|---|---|---|
| 2 s 1080p30 | 4 | 1088×1920 | 8160 | 2040 | 2 | 4080 |
| 2 s 2160p30 | 4 | 1856×3328 | **24,128** | 6032 | 2 | 12,064 |
| 10 s 1080p30 | 20 | 832×1472 | 4784 | 1196 | 10 | 11,960 |
| 60 s 720p30 | 120 | 320×608 | 760 | 190 | 60 | 11,400 |
| 300 s 1080p30 | 384 | 192×320 | 240 | 60 | 192 | 11,520 |

The total settles around 25.1M px / 1024 / 2 ≈ 12.3k LM tokens per long video. A short 4K clip, however, gives
**24k-patch attention sequences** in the encoder, three times the largest image.

### 2.3 Patchify (:222-260)
- Pad frames to even by repeating the last frame (:233-236). `grid_t = n_padded/2`.
- View `(gt, T=2, C, gh/2, 2, 16, gw/2, 2, 16)`, permute to `(gt, gh/2, gw/2, mh, mw, C, T, py, px)`. That gives the
  same row and column layout as §1.3, with T now indexing the 2 real frames of each group and rows offset by
  `g·gh·gw`.
- **Temporal merging happens only in the patchifier**: the two frames of a group share one 1536-wide patch
  vector. After that, each group is an independent "frame" in the encoder (§3). There are no temporal position
  embeddings; timing reaches the LM only through the timestamp text (§5).

### 2.4 Timestamps (Omni processor, M/processing_lfm3_omni.py:759-766)
`ts[g] = (idx[2g] + idx[2g+1]) / (2·fps)`, with the index list padded with its last element. It is printed as
`f"<{ts:.1f} seconds>"` (:726). Python's `.1f` is correctly rounded from the exact binary value; glibc `%.1f` is
too, so the two should agree. Check the boundary cases anyway. The VL processor has a slightly different formula
(M/processing_lfm3_vl.py:217-228), but the Omni processor's version is the one used.

### 2.5 Decoding
HF decodes video with torchcodec, and fps comes from the container header (comment at M/processing_lfm3_omni.py:
704-705). A native port needs its own decoder (FFmpeg/NVDEC). Exact frame-index semantics (pts vs. decode order,
VFR clips) are a fidelity risk; see §9.

---

## 3. Vision encoder (`Lfm3VlVisionModel`, M/modeling_lfm3_vl.py:460-512)

### 3.1 Top level
```
x   = patchifier(pixel_values.bf16)        # Linear 1536 -> 1024, no bias                    (:473, :498)
x   = embedding_norm(x)                    # RMSNorm(1024, eps 1e-5)                        (:474, :499)
cos, sin = rotary_pos_emb(x, position_ids) # 2D axial, computed once                       (:500)
for layer in layers[0..13]: x = layer(x, frame_shapes, cos/sin)
return x                                   # NO final norm (the projector's LayerNorm acts as one)
```
`layer_types` = `[conv, conv, attn, conv, attn, conv, attn, conv, attn, conv, attn, conv, attn, conv]`
(config.json). Conv layers are 0,1,3,5,7,9,11,13 (8); attention layers are 2,4,6,8,10,12 (6).

### 3.2 Every layer: pre-norm residual (:354-386)
```
h = x + Operator(RMSNorm_op(x))            # operator_norm, eps 1e-5
y = h + W2( silu(W1·n) * (W3·n) ),  n = RMSNorm_ffn(h)       # SwiGLU 1024 -> 2560 -> 1024, no biases (:75-83)
```
RMSNorm (:54-69): `w * bf16(x * rsqrt(mean(x²) + eps))`, with statistics in FP32. The weight multiplies directly
(it is not 1 + w). The input is rounded back to the input dtype **before** the weight multiply.

### 3.3 Conv layer: gated depthwise 3×3 over the 2-D patch grid (`Lfm3VlVisionConv2d`, :308-351)
```
B, C, X = split(in_proj(n), 3)              # in_proj 1024 -> 3072, no bias; chunk order B | C | X
g = B * X                                   # [L][1024], merge-block order
G = to_raster(g, gh, gw)                    # (gh/2, gw/2, 2, 2, D) -> (D, gh, gw)
Y[d][y][x] = sum_{a,b in 0..2} K[d][0][a][b] * G[d][y+a-1][x+b-1]    # zero padding at frame edges, no bias
out = out_proj( C * from_raster(Y) )        # out_proj 1024 -> 1024, no bias
```
- PyTorch Conv2d is cross-correlation, so the kernel is not flipped. Padding is 1 and `groups=1024`.
- The convolution runs on the **full-resolution patch grid** (gh×gw), per frame. Padding does not leak across
  frames, images, or a video's temporal groups (frames are batched along N or looped, :330-350).
- Unlike the audio/LFM2 short conv, this is **non-causal 2-D**. Nothing in `N/` can be reused for it except the
  pattern of an elementwise kernel.

### 3.4 Attention layer (`Lfm3VlVisionAttention`, :195-305)
- `q = q_proj(n)` gives 16 heads × 64. `k, v = k_proj(n), v_proj(n)` give 8 heads × 64 (q 1024×1024; k, v 512×1024;
  out 1024×1024; no biases).
- **QK-norm**: RMSNorm over each 64-wide head (`q_layernorm`, `k_layernorm`, weights [64], eps 1e-5) **before**
  RoPE (:225-232).
- GQA: query head h uses kv head `h // 2` (`repeat_kv`, :158-167).
- Scale `64^-0.5 = 0.125` (:205). Bidirectional (`is_causal=False`), no mask. Softmax over all L = gh·gw patches of
  one frame.
- Attention runs separately for each image and each temporal group. In SDPA/eager mode equal-shaped frames are
  batched and mixed shapes are looped (:259-301); flash-attention mode uses `cu_seqlens` (:242-258). The maths is
  the same either way.

### 3.5 2-D axial RoPE (`Lfm3VlVisionRotaryEmbedding`, :86-155)
```
spatial_dim = 1024/16/2 = 32
inv_freq[j] = 1 / (1e6 ** ((2j)/32)),  j = 0..15          # float32 (:110-111)
fh = pos_h * inv_freq ; fw = pos_w * inv_freq              # float32, autocast disabled (:117-122)
cos_vec (64) = [cos fh(16) | cos fw(16) | cos fh(16) | cos fw(16)]   (:128-134), same for sin
q' = q*cos + rotate_half(q)*sin                            # in FP32, then cast back to BF16 (:144-155)
```
In terms of pairs: element d is paired with d+32. For d in 0..15 the angle is `pos_h · inv_freq[d]`; for d in 16..31
it is `pos_w · inv_freq[d−16]`. This is **not** the same as N/'s `fp32::rotary`, which uses `θ^(−2d/dim)` with one
1-D position (N/include/lfm/fp32_ops.cuh:229-243). A new kernel is needed, but it is small.

**Ambiguity:**
- The comment at :116 says `position_ids` is `(2, N)`. But `recomposition_frequencies` indexes `freq[:, 0]` and
  `freq[:, 1]` (:132), which only works if the shape is `(N, 2)`, giving `freqs` of `(N, 2, 16)`. The code wins:
  treat it as `(N, 2)`, columns (h, w).
- `get_vision_position_ids(grid_thw, spatial_merge_size)` (:494) is not available locally. It is *inferred* to
  give integer (row, col) patch coordinates **in merge-block order**, matching the `pixel_values` row order (§1.3),
  restarting at 0 for each frame and each temporal group. No interpolation, centring or scaling is expected, and
  there is no learned position table (no such tensor exists in the checkpoint).
- This must be confirmed by dumping `position_ids` from HF. A wrong order here (raster vs. block) would give
  plausible-looking but degraded outputs.

### 3.6 Shapes for reference inputs (L = patches per frame)

| stage | 512×512 (L=1024) | 1080×1920 → 1088×1920 (L=8160) |
|---|---|---|
| pixel_values | [1024, 1536] | [8160, 1536] |
| patchifier, embedding_norm | [1024, 1024] | [8160, 1024] |
| conv in_proj | [1024, 3072] → B, C, X [1024,1024] each | [8160, 3072] |
| conv raster grid | [1024 ch, 32, 32] | [1024 ch, 68, 120] |
| attn q / k / v | [1024,16,64] / [1024,8,64] / [1024,8,64] | [8160,16,64] / [8160,8,64] ×2 |
| attn score matrix | 16 × 1024 × 1024 | 16 × 8160 × 8160 (266 MB per head in FP32 if materialised) |
| MLP hidden | [1024, 2560] | [8160, 2560] |
| tower output | [1024, 1024] | [8160, 1024] |
| projector in (after concatenating 2×2 blocks) | [256, 4096] | [2040, 4096] |
| projector out | [256, 2048] | [2040, 2048] |
| prompt span | 258 tokens | 2042 tokens |

Rough forward cost (2·MACs): 512² ≈ 0.37 TFLOP; 1088×1920 ≈ **4.4 TFLOP**, of which about 1.9 is attention
QKᵀ/PV, 1.8 MLP and 0.55 conv-layer projections. The audio encoder ran at roughly 125 rows per 10 s clip. A large
image has 8160 rows through 164M parameters (~2.7 TFLOP in linears alone), so a CUDA-core FP32 GEMM will not do;
see §7 and §9.

---

## 4. Projector (`Lfm3OmniMultiModalProjector`, M/modeling_lfm3_omni.py:349-371; identical to M/modeling_lfm3_vl.py:515-537)
```
n  = LayerNorm(x)              # per vision token, 1024, weight+bias, eps = projector_layer_norm_eps = 1e-6
m  = n.reshape(-1, 4096)       # 4 consecutive rows = one 2x2 block, concatenated [ (0,0) | (0,1) | (1,0) | (1,1) ]
h  = gelu(linear_1(m))         # 4096 -> 4096 + bias; ACT2FN["gelu"] = exact erf GELU
o  = linear_2(h)               # 4096 -> 2048 + bias   (2048 = text_config.hidden_size)
```
- LayerNorm runs **before** the 2×2 merge, on each 1024-wide token.
- The "pixel-unshuffle" is just this reshape. It works only because rows are already in merge-block order. With
  mixed image sizes it is still correct, since every image has gh and gw even and contributes a multiple of 4 rows.
- `downsample_factor` (2) must equal `spatial_merge_size` (M/configuration_lfm3_omni.py:291-292).
- The eps is **1e-6**. The audio projector's LayerNorm uses the PyTorch default 1e-5 (M/modeling_lfm3_omni.py:503),
  and the native audio code hard-codes 1e-5 (N/include/lfm/audio_encoder.cuh:296). Do not copy that constant.
- Output rows per image = gh·gw/4, in block order (row-major over the (gh/2)×(gw/2) block grid). That is the order
  they replace the `<|image|>` placeholders.

---

## 5. Prompt integration

### 5.1 Template
`chat_template.jinja` renders every image part as `<|vision_start|><|image|><|vision_end|>` and every video part as
`<|vision_start|><|video|><|vision_end|>` (M/chat_template.jinja:22-25, 39-42). Audio is wrapped the same way
(:26-27, 43-44), which is the native engine's existing convention (N/include/lfm/lfm3_text.hpp:557-559, 682).

### 5.2 Processor expansion (Omni processor)
- First it strips the template's span pair from each placeholder (`_strip_spans`, M/processing_lfm3_omni.py:494-501).
  Then it adds exactly one pair while expanding (`_wrap`, :491-492; `vision_markers=True`, M/processor_config.json:93).
- The span tokens configured are `<|vision_start|>` / `<|vision_end|>` (M/processor_config.json:92-94), overriding
  the constructor default `<|media_start|>` (M/processing_lfm3_omni.py:154). Ids: 125009 and 125010 (tokenizer.json
  added_tokens; also M/configuration_lfm3_vl.py:144-145).
- **Image**, id 125030 (:754-756):
  `<|vision_start|>` + `<|image|>` × (gh·gw/4) + `<|vision_end|>`. With H',W' from §1.2 the count is
  **N = H'·W'/1024**. This is the same splice as native `expand_audio`, except the span tokens are already in the
  template.
- **Video**, id 125029 (`_video_expansion`, :693-729): spans are nested.
  ```
  <|vision_start|>
    for g in 0..grid_t-1:
      "<{ts[g]:.1f} seconds>" <|vision_start|> <|video|> × (gh·gw/4) <|vision_end|> [<|audio|> × k_g if soundtrack]
  <|vision_end|>
  ```
  The timestamp is ordinary text, tokenized by BPE as part of the expanded string. The tokenizer runs on the fully
  expanded string (:682). A native port can splice token ids, but must tokenize each `"<x.y seconds>"` in context
  exactly as BPE would within the string; check that against HF.
- With `use_audio_in_video=True`, the soundtrack's encoder tokens are split per group at
  `round(start_g / 0.08 s)` (hop 160 × stride 8 / 16 kHz) and made monotone (:709-724, 727-728). No audio markers
  surround them inside the video span.
- No `<|img_thumbnail|>`, `<|image_split|>` or row/column tokens are used.

### 5.3 Scatter into the LM (M/modeling_lfm3_omni.py:682-693)
`inputs_embeds = embed(input_ids)`. Then image features (BF16, cast with `.to(inputs_embeds)`) are
`masked_scatter`ed into the positions where `input_ids == 125030`, in order. Video does the same at 125029, then
audio at 125031 (:695-707). The counts are checked (`get_placeholder_mask`, :612-621). The LM uses ordinary 1-D
`arange` positions (M/modeling_lfm3.py:676) and causal masks (M/modeling_lfm3_vl.py:389-418). There is **no M-RoPE
and no bidirectional attention among image tokens** in the LM. The comment "Qwen3-VL convention" (:679) refers only
to placeholders.

---

## 6. Weights (safetensors header; header 84,864 B, `__metadata__ = {format: pt}`; every vision/projector tensor BF16)

Vision tower, `model.vision_tower.` (132 tensors):

| tensor | shape | count |
|---|---|---|
| patchifier.weight | [1024, 1536] | 1 |
| embedding_norm.weight | [1024] | 1 |
| layers.{i}.operator_norm.weight, ffn_norm.weight | [1024] | 14 × 2 |
| layers.{i}.feed_forward.w1 / w3 .weight | [2560, 1024] | 14 × 2 |
| layers.{i}.feed_forward.w2.weight | [1024, 2560] | 14 |
| conv layers i∈{0,1,3,5,7,9,11,13}: conv.in_proj.weight | [3072, 1024] | 8 |
| conv.out_proj.weight | [1024, 1024] | 8 |
| conv.spatial_conv.weight | [1024, 1, 3, 3] | 8 |
| attn layers i∈{2,4,6,8,10,12}: self_attn.q_proj.weight | [1024, 1024] | 6 |
| self_attn.k_proj.weight, v_proj.weight | [512, 1024] | 6 × 2 |
| self_attn.out_proj.weight | [1024, 1024] | 6 |
| self_attn.q_layernorm.weight, k_layernorm.weight | [64] | 6 × 2 |

Per conv layer: 12,069,888 params. Per attention layer: 11,012,224. Total: 8·12,069,888 + 6·11,012,224 + 1,572,864
+ 1,024 = **164,206,336 params, 328,412,672 B**.

Projector, `model.multi_modal_projector.` (6 tensors): `layer_norm.weight` [1024], `layer_norm.bias` [1024],
`linear_1.weight` [4096, 4096], `linear_1.bias` [4096], `linear_2.weight` [2048, 4096], `linear_2.bias` [2048].
That is **25,174,016 params, 50,348,032 B**.

Vision + projector: **189,380,352 params, 378,760,704 B (~361 MiB)**. For comparison, from the same header: LM
1,728,801,536; audio encoder 109,505,344; audio projector 5,772,800; depth decoder 83,898,560; codebook embedding
49,569,792.

No tensor for a final vision norm, positional table, or patch bias exists. The conv weight name is
`conv.spatial_conv.weight` [1024,1,3,3]; audio's is `conv.conv.weight` [768,1,3].

---

## 7. Numerics: what a faithful port must match or decide

HF reference precision (README loads `dtype=torch.bfloat16` on CUDA, M/README.md:230-233; `vision_config.dtype`
is bfloat16):
1. **Input rounding.** `pixel_values` (FP32 in [-1,1]) is cast to BF16 before the patchifier (:498).
2. **RMSNorm.** FP32 statistics, rounded to BF16, then multiplied by the BF16 weight in BF16 (:64-69). eps is 1e-5
   for embedding_norm, operator/ffn norms and q/k norms.
3. **RoPE.** inv_freq, angles and cos/sin in FP32. q and k are upcast to FP32 for the rotation, then rounded to
   BF16 (:144-155).
4. **Attention.** Scale 0.125. Eager softmax is in FP32 (:187). The default SDPA backend (flash/mem-efficient) keeps
   FP32 accumulators with BF16 inputs and output.
5. **Conv gating.** `B*X` and `C*conv` are BF16 products. The depthwise conv is cuDNN BF16, probably with FP32
   accumulation; not verifiable here.
6. **Projector LayerNorm.** eps **1e-6** with bias. GELU is **exact erf**. The final output is cast to the LM
   embedding dtype, BF16 (M/modeling_lfm3_omni.py:685).
7. **Resize.** Antialiased bicubic on uint8 with rounding and clamp (torchvision). Banker's rounding in
   smart_resize. Float64 `beta` expressions.
8. **No interpolation of position embeddings.** RoPE is evaluated directly at integer positions, so any resolution
   works.

The native audio-in precedent kept FP32 activations over stored BF16 weights and rounded only the output. Round 1,
at HF's BF16 rounding points, failed the accuracy-ratio rule narrowly; FP32 passed (liquid-CUDA/validation/
audio-in-v1/README.md:46-60). For vision the same rule ("native relative-L2 error to HF-FP32 ≤ HF-BF16's") is the
natural gate. **But compute is about 35× larger**, so `lfm::fp32::linear` (a CUDA-core FMA GEMM, no cuBLAS,
N/include/lfm/fp32_ops.cuh:1-8) is likely too slow for a 2048-token image at batch scale. A candidate middle ground:
- BF16 tensor-core GEMMs with FP32 accumulation, and FP32 outputs kept between ops (no BF16 rounding of
  activations);
- FP32 norms, RoPE, softmax and conv.

Whether that beats HF-BF16 error has to be measured; it is not guaranteed. Measure before choosing.

---

## 8. Comparison with the native audio-in integration

**Reusable as-is or nearly so:**
- *Embedding splice*: `Runner::forward(prompt, injected, placeholder)` replaces placeholder rows in order at prefill
  (N/src/lfm3_main.cu:538-559, `inject_rows`). This is exactly HF's `masked_scatter`. It currently takes **one
  placeholder id**. A prompt with an image, a video and audio needs either several (id, rows) pairs, or one combined
  row buffer plus a per-row source index. Note that HF scatters per modality, so the concatenation order is
  images, then videos, then audio, each in placeholder order.
- *Placeholder expansion*: `expand_audio` (N/include/lfm/lfm3_text.hpp:776-797) is the right shape for images (one
  id → N copies inside the template's existing span). Video needs a richer expansion: nested spans, timestamp text
  token ids, and optional audio interleave.
- *Contract checking*: the `Encoder::check_contract` pattern of config plus safetensors header, checked at startup
  (N/include/lfm/audio_encoder.cuh:163-226), and the 501-style "unsupported" fallback.
- *Session plumbing*: encode before prefill into a row buffer sized from the token counts
  (N/src/lfm3_speech.cuh:189, 202-218, `encoder_`, `audio_rows_`), release scratch after each item
  (audio_encoder.cuh:308-313), and per-stage `Dumps` for HF parity.
- *Kernels*: `fp32::rms_norm` (with a stride, usable for the per-head QK-norm), `swiglu`, `layer_norm` (audio
  kernel; parameterise the eps), exact GELU `activate`, `to_bf16`, `add`.
- *Refusals*: today image and video placeholders are refused in speech prompts (N/src/lfm3_speech.cuh:166-177).
  That becomes the gate that vision support lifts.

**New work:**
1. Image decode (JPEG/PNG/WebP → RGB uint8). No decoder exists in `N/`.
2. A bit-exact antialiased bicubic uint8 resize matching torchvision, plus smart_resize with banker's rounding.
3. Block-order patchify (trivial) and a K=1536 patch GEMM.
4. A 2-D depthwise 3×3 conv with raster/block re-indexing. The permutation can be folded into the conv's indexing:
   read the neighbour's block-order row directly.
5. **Long-sequence bidirectional GQA attention**: head_dim 64, 16/8 heads, L up to ~8k for images and ~24k for
   short 4K videos, non-causal, segmented per frame. The audio `window_attention` (32-key window, one warp per
   query) and the text attention (causal, head_dim 128) do not fit. This is a flash-style tiled kernel.
6. 2-D axial RoPE.
7. A 2×2 concatenate inside the projector, which is free given the row order.
8. Video: decoding, frame sampling, per-group timestamp text, nested spans, soundtrack interleave (reusing the
   audio encoder's per-clip tokens split at the 0.08 s boundaries).
9. Throughput design. Speed is P0 and is judged at batch 32: one large image is about 4.4 TFLOP before the LM sees
   it, and 2040 prompt tokens of prefill. Encoder batching across requests, and its scheduling against decode,
   become real questions that audio (small) never raised.

---

## 9. Open questions, risks, and validation order

**Open questions and risks**
1. *(blocking)* The `transformers.vision_utils` helpers (position ids, seqlens) and the 5.x image/video backends are
   not available locally. Dump `position_ids`, `cu_seqlens` and `pixel_values` from an HF 5.20.2 install for a few
   shapes (square, odd block counts such as 68×120, a video with odd frame count). Do this before writing kernels.
2. Bit-exact resize. Torchvision's uint8 AA-bicubic has its own fixed-point path, and different input decoders
   (PIL vs. libjpeg-turbo settings, EXIF orientation) already change pixels. Decide whether the parity target is
   "same uint8 resized image" (needs a torchvision-exact port) or "same `pixel_values` given HF's resized image".
   The audio work had an equivalent split (features rule vs. encoder rule).
3. EXIF orientation and alpha handling in HF 5.x loaders (`load_image`) are unverified here.
4. Precision strategy (§7): FP32 activations as in audio vs. tensor-core BF16 with FP32 accumulation. Both need to
   meet an accuracy-ratio gate against HF-FP32.
5. Attention memory and time at L = 8160 to 24,128. Attention must not materialise scores. Also consider capping
   video per-frame pixels; HF itself warns the cap will become the default in v5.22
   (M/video_processing_lfm3_vl.py:280-289). If the reference moves, the native default must follow, and the token
   counts change.
6. Prompt capacity: a single 1080p image is 2042 tokens and a long video about 12k. Check the per-request context
   limits and the KV memory at batch 32.
7. Video decode fidelity: frame-index semantics, VFR clips, and container-header fps vs. the averaged fps (the
   processor rounds to 6 decimals).
8. Timestamp text tokenization in context, and `.1f` rounding equality between C and Python.
9. Cross-check the Omni vs. VL timestamp formula (§2.4). Omni is used; the VL variant can differ in the last ulp.

**Suggested validation stages** (mirroring audio-in-v1: rules fixed before each run; timing on a quiet Thor)
0. *Host layout, no GPU.* For a corpus of (H, W) values, including the banker's-rounding and ceil-edge cases above,
   with images and videos: compare native smart_resize, grid_thw, token counts, expanded prompt ids (spans,
   timestamps) and sampled frame indices to the HF processor. Require exact equality. Also capture HF
   `position_ids` and `cu_seqlens`.
1. *Pixels.* Native resized uint8 vs. HF (target exact, or report max abs diff), then `pixel_values` [P,1536]
   exact given the same uint8 image.
2. *Encoder stages.* Dump patchifier+embedding_norm, each of the 14 layer outputs, and the tower output. Compare
   relative L2 to HF-FP32 (TF32 off) and require a ratio ≤ HF-BF16's, per image. Isolate conv layers (0,1) and the
   first attention layer (2) first, since the layout and RoPE bugs show up there.
3. *Projected features.* [N, 2048] BF16 vs. HF, with the same ratio rule.
4. *Tokens.* Prefill with injected rows and greedy decode vs. HF on an image-QA/caption set: token-identical rate
   plus a benchmark-style accuracy delta with a paired-bootstrap bound, as audio did with WER.
5. *Coherency.* Read the generated answers and captions end to end, as with every engine change.
6. *Video.* Repeat stages 0 to 5 with videos, then with `use_audio_in_video`.
7. *Edge and refusal cases.* 32×32, 100×10000 (aspect 100), aspect > 200 (refused), odd frame counts, mixed image
   sizes in one request, and a placeholder/media count mismatch.
8. *Throughput.* Encode seconds per image size, and aggregate tok/s at batch 32 with image prompts.
