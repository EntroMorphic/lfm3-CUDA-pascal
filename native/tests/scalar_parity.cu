// Standalone source-driven probe. Build with the same flags/libraries as build.sh.
// No production source is copied or changed: even BF16 coefficients use the
// production kernel, launched once for each position.
#define main native_main_for_test
#include "../src/main.cu"
#undef main

#include <filesystem>

namespace {

constexpr int kFiniteBf16Count = 65280;

__global__ void scalar_rope_intermediates(
    float* frequencies, float* angles, float* cosines, float* sines) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kContextLength * 32) return;
  const int position = index / 32;
  const int dimension = index % 32;
  const float frequency = rope_frequency(dimension);
  const float angle = static_cast<float>(position) * frequency;
  if (position == 0) frequencies[dimension] = frequency;
  angles[index] = angle;
  cosines[index] = cosf(angle);
  sines[index] = sinf(angle);
}

template <typename T>
void write_scalar_dump(const std::filesystem::path& path, const DeviceBuffer<T>& buffer) {
  std::vector<T> host(buffer.size);
  check_cuda(cudaMemcpy(host.data(), buffer.data, buffer.size * sizeof(T),
                        cudaMemcpyDeviceToHost), "copy scalar dump");
  std::ofstream stream(path, std::ios::binary);
  stream.write(reinterpret_cast<const char*>(host.data()), host.size() * sizeof(T));
  if (!stream) throw std::runtime_error("cannot write " + path.string());
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("usage: scalar-parity OUTPUT_DIRECTORY");
    const std::filesystem::path output(argv[1]);
    if (!std::filesystem::create_directory(output)) {
      throw std::runtime_error("output directory must not already exist");
    }
    std::vector<uint16_t> encodings;
    for (uint32_t bits = 0; bits < 65536; ++bits) {
      if ((bits & 0x7f80) != 0x7f80) encodings.push_back(bits);
    }
    require(encodings.size() == kFiniteBf16Count, "finite BF16 enumeration");
    DeviceBuffer<__nv_bfloat16> gate(kFiniteBf16Count), value(kFiniteBf16Count);
    DeviceBuffer<__nv_bfloat16> activated(kFiniteBf16Count);
    check_cuda(cudaMemcpy(gate.data, encodings.data(), encodings.size() * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy finite BF16 gates");
    const std::vector<uint16_t> ones(kFiniteBf16Count, 0x3f80);
    check_cuda(cudaMemcpy(value.data, ones.data(), ones.size() * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy unit values");
    swiglu<<<(kFiniteBf16Count + 255) / 256, 256>>>(
        gate.data, value.data, activated.data, kFiniteBf16Count);
    check_cuda(cudaGetLastError(), "swiglu launch");

    DeviceBuffer<float> frequencies(32), angles(kContextLength * 32);
    DeviceBuffer<float> cosines(kContextLength * 32), sines(kContextLength * 32);
    DeviceBuffer<__nv_bfloat16> coefficients(kContextLength * 64);
    scalar_rope_intermediates<<<(kContextLength * 32 + 255) / 256, 256>>>(
        frequencies.data, angles.data, cosines.data, sines.data);
    check_cuda(cudaGetLastError(), "RoPE intermediate launch");
    for (int position = 0; position < kContextLength; ++position) {
      rope_coefficients<<<1, 32>>>(coefficients.data + position * 64, position);
    }
    check_cuda(cudaGetLastError(), "production rope_coefficients launches");
    check_cuda(cudaDeviceSynchronize(), "scalar probe synchronize");
    write_scalar_dump(output / "gate.bf16", gate);
    write_scalar_dump(output / "swiglu.bf16", activated);
    write_scalar_dump(output / "rope_frequency.f32", frequencies);
    write_scalar_dump(output / "rope_angle.f32", angles);
    write_scalar_dump(output / "rope_cos.f32", cosines);
    write_scalar_dump(output / "rope_sin.f32", sines);
    write_scalar_dump(output / "rope_coefficients.bf16", coefficients);
    std::ofstream manifest(output / "manifest.json");
    manifest << "{\"finite_bf16_count\":65280,\"positions\":32768,"
                "\"rotary_dimension\":64,\"byte_order\":\"little\","
                "\"coefficient_layout\":\"position,cos[32]+sin[32]\","
                "\"coefficient_source\":\"production rope_coefficients\"}\n";
    if (!manifest) throw std::runtime_error("cannot write manifest");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
