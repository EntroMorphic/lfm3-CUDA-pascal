#pragma once
// The vision encoder's interface with no encoder behind it (validation/speech-v1): the Pascal build's speech and worker
// before phase 5 (images). lfm/vision_encoder.cuh's kernels use the tensor cores (mma.sync, ldmatrix), which sm_61
// does not have. Everything that would encode refuses: the worker's contract check throws, so it reports image input
// unsupported and refuses requests with images before any state changes (its own message), and the speech engine's
// constructor throws should an image reach it. LFM3_VISION=1 builds the real encoder (lfm/vision_encoder.cuh).
#include <lfm/safetensors.hpp>
#include <lfm/safetensors_header.hpp>

#include <cuda_bf16.h>

#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace lfm3_vision {

inline constexpr const char* kAbsent = "image input is not built in this engine (LFM3_VISION=0)";

class Encoder {
 public:
  static constexpr int kPatchDim = 1536;  // lfm/vision_encoder.cuh's; the speech engine's image check reads it
  struct Precision {};
  struct Dumps {};
  static void check_contract(const lfm::Json&,
                             const std::function<std::optional<lfm::safetensors_header::Entry>(const std::string&)>&) {
    throw std::runtime_error(kAbsent);
  }
  Encoder(std::shared_ptr<lfm::Safetensors>, const lfm::Json&, Precision = {}) { throw std::runtime_error(kAbsent); }
  int encode(const float*, int, int, __nv_bfloat16*, Dumps* = nullptr) { throw std::runtime_error(kAbsent); }
};

}  // namespace lfm3_vision
