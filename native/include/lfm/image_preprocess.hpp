#pragma once
// The vision front end after decoding (validation/images-v1 rule 1): what HF's Lfm3VlImageProcessor (transformers
// 5.20.2, the checkpoint's image_processing_lfm3_vl.py and processor_config.json; torchvision 0.25, torch 2.10, CPU)
// makes of an RGB image, bit for bit:
//   1. smart_resize: both sides to multiples of factor = patch * merge (32), the pixel count kept within
//      [min_pixels, max_pixels] = [65,536, 2,097,152], the aspect ratio as close as possible; an aspect ratio over 200
//      is refused. Python's arithmetic reproduced: round() ties to even, the true divisions correctly rounded doubles.
//   2. Resize: torchvision's resize on the uint8 tensor (BICUBIC, antialias=True) runs torch's native uint8 kernel
//      (_upsample_bicubic2d_aa), on x86 (AVX2) and on aarch64 alike: Keys cubic (a = -0.5) weights over each output
//      pixel's support, computed in double, quantized to int16 at the precision that keeps the largest below 2^15;
//      the horizontal pass first, rounded and clamped to uint8, then the vertical; a dimension that does not change
//      is not resampled. Ported from aten/src/ATen/native/cpu/UpSampleKernel.cpp (v2.10.0).
//   3. Rescale and normalize, fused as HF fuses them: float32 (x - 127.5) / 127.5.
//   4. Patchify: rows in (h/32, w/32, 2, 2) order, each 3 x 2 x 16 x 16 values (channel, the frame duplicated for the
//      temporal patch of 2, row, column); grid (1, h/16, w/16); tokens h/32 * w/32.
#include <lfm/image_decode.hpp>

#include <cstdint>
#include <vector>

namespace lfm::image {

struct PreprocessConfig {
  int patch = 16;
  int merge = 2;
  int temporal = 2;
  std::uint64_t min_pixels = 65536;     // processor_config.json size.shortest_edge
  std::uint64_t max_pixels = 2097152;   // size.longest_edge
};

struct Size {
  int height = 0, width = 0;
};

// HF's smart_resize. Throws DecodeError("aspect_ratio", ...) above 200, as HF raises ValueError.
Size smart_resize(int height, int width, int factor, std::uint64_t min_pixels, std::uint64_t max_pixels);

// torch's uint8 antialiased bicubic resize (align_corners=False, scales from the sizes).
Rgb resize_bicubic_aa(const Rgb& in, int out_width, int out_height);

struct PixelValues {
  int grid_t = 1, grid_h = 0, grid_w = 0;  // image_grid_thw
  int row_size = 0;                        // 3 * temporal * patch * patch
  int tokens = 0;                          // grid_t * grid_h * grid_w / merge^2: the language model's image tokens
  std::vector<float> values;               // grid_h * grid_w rows of row_size
};

PixelValues preprocess(const Rgb& image, const PreprocessConfig& config = {});

// preprocess(image, config).tokens without the resize: smart_resize alone (refusing as it does). For the server's
// front end, which needs only the count (validation/image-queue-v1).
int tokens(const Rgb& image, const PreprocessConfig& config = {});

}  // namespace lfm::image
