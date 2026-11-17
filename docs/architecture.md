# Architecture

LFM3-Omni-2B is a hybrid language model (gated short convolutions and grouped-query attention) with an audio encoder,
a vision encoder, and speech output through a depth decoder and a neural codec. The engine implements each piece
natively and composes them the way the reference's `generate` does.

```
                        ┌──────────────── liquid-lfm3-server (loopback HTTP, bearer token) ────────────────┐
 request ─▶ JSON parse ─▶ chat template + tokenizer ─▶ media front ends ─▶ engine-owner thread ─▶ reply/stream
                        │   lfm3_text.hpp              audio: WAV → log-mel      WorkerCore           │
                        │                              image: JPEG/PNG → pixels  (text sessions,      │
                        │                                                         speech batch rows)  │
                        └──────────────────────────────────────────────────────────────────────────────┘
 engine:  audio encoder ─┐
          vision encoder ┼─▶ rows injected at their placeholders ─▶ LFM3 backbone ─▶ text tokens
                         ┘                                             │
                                                                       └─▶ depth decoder ─▶ frames ─▶ codec ─▶ 24 kHz audio
```

## Components

| Component | Code | Notes |
|---|---|---|
| Tokenizer and chat template | `native/include/lfm/lfm3_text.hpp`, `lfm3_text_classes.hpp` | Byte-level BPE and the checkpoint's template, bit-exact against HF's |
| Text backbone | `native/src/lfm3_main.cu` (`Runner`), `lfm3_ops.cuh`, `gemm.cuh` | BF16 weights, native GEMM (no cuBLAS), FP32 logits; prefill and single-token decode |
| Batched decode | `native/src/lfm3_batch.cuh`, `lfm3_decode_batch.cuh` | Many rows per step, each row computed exactly as alone |
| Tree-verify decode | `native/src/lfm3_tree.cuh` | Exact single-request speculation |
| Sampling | `native/include/lfm/text_sampling.hpp`, `audio_sampling.hpp`, `batch_sampler.cuh` | Seeded, reproducible |
| Speech control | `native/include/lfm/audio_modes.hpp`, `native/src/lfm3_speech.cuh` | The reference's mode machine (text, audio, interleaved) and the speech batch |
| Depth decoder | `native/include/lfm/depth.cuh` | Hidden state → 12 codebook entries per frame |
| Codec | `native/include/lfm/codec.cuh` | Frames → 24 kHz waveform, streamed |
| Audio input | `native/include/lfm/audio_features.hpp`, `audio_encoder.cuh` | 16 kHz WAV → log-mel → encoder → projected rows |
| Image input | `native/src/image_decode.cpp`, `image_preprocess.cpp`, `native/include/lfm/vision_encoder.cuh` | JPEG/PNG → RGB (Pillow's rules) → resize (torchvision's) → patches → encoder → projected rows |
| Speech text | `native/include/lfm/speech_text.hpp`, `speech_normalize.hpp` | Sentence splitting for long-form TTS; numbers and abbreviations as words |
| HTTP | `native/include/lfm/lfm3_http.hpp`, `native/src/lfm3_server.cu` | Strict parsing, admission, deadlines, cancellation, streaming |
| Worker | `native/src/lfm3_worker.cu` | The engine core shared by the server and the out-of-process worker |

## Multimodal prompts

Media enter the prompt as placeholder tokens: each audio clip as `<|vision_start|>`, N × `<|audio|>`,
`<|vision_end|>`, and each image likewise with `<|image|>` (N = grid / 4). The encoders' projected rows replace the
placeholders' embeddings in prompt order before the backbone's prefill (`Runner::forward`), as HF's
`masked_scatter` does.

## Precision policy

Weights are BF16. Where the reference's numerics are integer or fixed-point (tokenization, JPEG and PNG decoding,
uint8 resizing) the engine is bit-identical. Elsewhere activations are FP32 over BF16 weights, with exact
transcendental functions where it matters, so the engine sits at least as close to an FP32 reference as HF's own BF16
GPU path — measured per round in `validation/`.
