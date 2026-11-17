#pragma once

namespace lfm {

// Number of independent blocks used by the greedy LM-head argmax first stage.
// Autoresearch candidates may change only this integer. The evaluator accepts
// values in [64, 1024] and requires exact control-output preservation.
inline constexpr int kLmHeadBlocks = 320;

}  // namespace lfm
