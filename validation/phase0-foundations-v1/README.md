# phase0-foundations-v1

Phase 0 of [PLAN.md](../../PLAN.md): the ground the port stands on. No engine code runs in this round.

## What it will be

The repository imported from the Thor engine, a verified way to get code and the checkpoint onto the GTX 1080
machine and run there, the host test suite running on the workstation, and a measured memory budget.

## Rules

1. **Sync.** `tools/pascal/remote.sh` copies the tree and compares one SHA-256 over every file on both ends; a
   difference fails before anything runs. A command's exit status comes back unchanged.
2. **Checkpoint.** Every file of LFM3-Omni-2B on the target matches the SHA-256 recorded when it was downloaded.
3. **Host tests.** The imported host suite gives the same result here as in the reference tree on this workstation.
4. **Memory.** The free device memory with every weight resident is measured on the card, not computed.

## Target

| | |
|---|---|
| GPU | GeForce GTX 1080 (GP104, `sm_61`), 8,104 MiB |
| Host | Ubuntu 24.04, kernel 6.8.0-149, NVIDIA kernel module 580.178.04 |
| Container | LXD `lfm3-pascal`, Ubuntu 24.04, the GPU as an LXD `gpu` device; NVIDIA userspace 580.178.04 (held) |
| Toolchain | CUDA 12.9 (`nvcc` 12.9.86), GCC 13.3, CMake 3.28 |
| Flags | `-O3 --fmad=false -std=c++20 -arch=sm_61` (the reference's, retargeted) |

CUDA 12.9 warns that targets before `sm_75` will be removed in a future release; this project stays on 12.9.

## Results

### 1. Sync — PASS

First run: tree `5ce49a02…503c8b` on both ends, 2.9 s for sync, hash and a command; exit status 7 returned as 7.

### 2. Checkpoint — PASS

29 files, 4,499,839,711 bytes; `sha256sum -c` against `model-checksums.sha256` passes on the workstation and inside
the container.

### 3. Host tests

(pending)

### 4. Memory — measured

`native/probes/pascal_mem_budget.cu` (binary `b1291dc6…18cf8c`), which uploads both files with the engine's own
loader (`lfm::Safetensors`, unmodified) and checks the small tensors' device bytes against the files:

| | MiB |
|---|---:|
| Device total | 8,103.9 |
| Free with a CUDA context and nothing else | 7,995.8 |
| `model.safetensors` resident (665 tensors) | −4,134.0 |
| `audio_tokenizer/model.safetensors` resident (203 tensors) | −142.0 |
| **Free with every weight resident** | **3,719.8** |
| Largest single allocation then granted | 3,718.0 |

Upload of both files: 4.1 s wall (page cache warm). After release the device returns to 7,995.8 MiB.

What the remaining 3.7 GiB must hold, from the checkpoint's configuration (computed, to be measured in phase 1):
the backbone has 12 attention layers (6 full, 6 with a 256-token window) of 4 key-value heads × 128, so keys and
values cost 2,048 bytes per token per layer in BF16 — 384 MiB for the 6 full layers at 32,768 tokens, 768 MiB if
the windowed layers also keep every token.

Also learned: a BF16-input, FP32-accumulate kernel built with these flags gives sums bit-identical to the same
order on the host CPU (4,096 rows × 2,048), so `__nv_bfloat16` conversions are exact on `sm_61`.
