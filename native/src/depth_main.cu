// liquid-depth: the native depth decoder on recorded passes (#2 step 3, A1/B1). Inputs, as lfm3_depth_anchor.py
// writes them: HIDDEN.bf16 [frames][2048] raw BF16; CODES.i32 and MASKED.u8 [frames][3][12]. Output: FP32 logits
// [frames][3][12][2017]. Each frame starts once (begin_frame) and runs its 3 passes, all 12 heads scored.
//   liquid-depth MODEL_DIR HIDDEN.bf16 CODES.i32 MASKED.u8 OUT.f32
#include <lfm/depth.cuh>

#include <chrono>
#include <fstream>
#include <iostream>

namespace {
template <class T>
std::vector<T> read(const char* path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error(std::string("cannot read ") + path);
  const auto bytes = size_t(in.tellg());
  if (bytes % sizeof(T)) throw std::runtime_error(std::string("ragged file ") + path);
  std::vector<T> v(bytes / sizeof(T));
  in.seekg(0);
  in.read(reinterpret_cast<char*>(v.data()), std::streamsize(bytes));
  return v;
}
}  // namespace

int main(int argc, char** argv) try {
  using namespace lfm::depth;
  if (argc != 6) {
    std::cerr << "usage: liquid-depth MODEL_DIR HIDDEN.bf16 CODES.i32 MASKED.u8 OUT.f32\n";
    return 2;
  }
  const auto hidden = read<uint16_t>(argv[2]);
  const auto codes = read<int32_t>(argv[3]);
  const auto masked = read<uint8_t>(argv[4]);
  const size_t frames = hidden.size() / kModelHidden;
  if (!frames || hidden.size() != frames * kModelHidden || codes.size() != frames * 3 * kCodebooks ||
      masked.size() != codes.size())
    throw std::runtime_error("inputs disagree on the number of frames");
  auto weights = std::make_shared<lfm::Safetensors>(std::string(argv[1]) + "/model.safetensors");
  Decoder decoder(weights);
  lfm::fp32::DeviceArray<__nv_bfloat16> hidden_device(kModelHidden);
  lfm::fp32::DeviceArray<float> logits_device(size_t(kCodebooks) * kCodes);
  std::vector<float> logits(frames * 3 * kCodebooks * kCodes);
  const bool all[kCodebooks] = {true, true, true, true, true, true, true, true, true, true, true, true};
  lfm::fp32::check(cudaDeviceSynchronize(), "setup");
  const auto start = std::chrono::steady_clock::now();
  for (size_t f = 0; f < frames; ++f) {
    lfm::fp32::check(cudaMemcpy(hidden_device.p, hidden.data() + f * kModelHidden, kModelHidden * 2, cudaMemcpyHostToDevice),
                     "hidden");
    decoder.begin_frame(hidden_device.p);
    for (int p = 0; p < 3; ++p) {
      bool m[kCodebooks];
      for (int c = 0; c < kCodebooks; ++c) m[c] = masked[(f * 3 + p) * kCodebooks + c] != 0;
      decoder.pass(codes.data() + (f * 3 + p) * kCodebooks, m, all, logits_device.p);
      lfm::fp32::check(cudaMemcpy(logits.data() + (f * 3 + p) * kCodebooks * kCodes, logits_device.p,
                                  logits_device.bytes(), cudaMemcpyDeviceToHost), "logits");
    }
  }
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  std::ofstream out(argv[5], std::ios::binary);
  out.write(reinterpret_cast<const char*>(logits.data()), std::streamsize(logits.size() * sizeof(float)));
  if (!out) throw std::runtime_error(std::string("cannot write ") + argv[5]);
  std::cout << "{\"frames\": " << frames << ", \"ms_per_frame\": " << ms / frames
            << ", \"buffer_bytes\": " << decoder.buffer_bytes() << "}\n";
  return 0;
} catch (const std::exception& e) {
  std::cerr << "liquid-depth: " << e.what() << "\n";
  return 1;
}
