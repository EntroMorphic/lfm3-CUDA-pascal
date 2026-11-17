#include <lfm/safetensors.hpp>
#include <lfm/checkpoint_path.hpp>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <stdexcept>

int main(int argc, char** argv) {
  try {
    if (argc < 2) return 2;
    lfm::Safetensors model(argv[1]);
    std::printf("%zu %zu %zu\n", model.tensor_count(), model.payload_bytes(), model.device_bytes());
    if (argc == 3 && std::string(argv[2]) == "--verify-small") {
      model.verify_small_tensors();
      std::puts("small_tensor_verification: pass");
      return 0;
    }
    if (argc == 4 && std::string(argv[2]) == "--corrupt-small") {
      const auto& tensor = model.at(argv[3]);
      if (tensor.bytes() > 64 * 1024) throw std::runtime_error("negative control requires a small tensor");
      const auto status = cudaMemset(const_cast<__nv_bfloat16*>(tensor.device_data), 0, tensor.bytes());
      if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
      model.verify_small_tensors();
      throw std::runtime_error("corruption negative control was not rejected");
    }
    if (argc == 3 && std::string(argv[2]) == "--verify-all") {
      const auto checkpoint = lfm::checkpoint_path(argv[1]);
      std::vector<std::pair<std::string, std::filesystem::path>> weights;
      if (lfm::is_checkpoint_index(checkpoint)) {
        std::ifstream index_file(checkpoint);
        const std::string text((std::istreambuf_iterator<char>(index_file)), {});
        const auto index = lfm::Json::parse(text);
        for (const auto& [name, file] : std::get<lfm::Json::Object>(index.at("weight_map").value))
          weights.emplace_back(name, lfm::checkpoint_shard(checkpoint.parent_path(), file.string()));
      } else {
        std::ifstream input(checkpoint, std::ios::binary);
        uint64_t header_length = 0;
        input.read(reinterpret_cast<char*>(&header_length), 8);
        if (!input || header_length > std::filesystem::file_size(checkpoint) - 8)
          throw std::runtime_error("invalid source header length");
        std::string text(header_length, '\0');
        input.read(text.data(), text.size());
        if (!input) throw std::runtime_error("cannot read source header");
        const auto header = lfm::Json::parse(text);
        for (const auto& [name, value] : std::get<lfm::Json::Object>(header.value))
          if (name != "__metadata__") weights.emplace_back(name, checkpoint);
      }
      std::vector<unsigned char> source(8 * 1024 * 1024), device(source.size());
      size_t verified = 0;
      for (const auto& [name, file] : weights) {
        std::ifstream input(file, std::ios::binary);
        uint64_t header_length = 0;
        input.read(reinterpret_cast<char*>(&header_length), 8);
        const auto& tensor = model.at(name);
        input.seekg(8 + header_length + tensor.begin);
        for (size_t offset = 0; offset < tensor.bytes(); offset += source.size()) {
          const size_t count = std::min(source.size(), tensor.bytes() - offset);
          input.read(reinterpret_cast<char*>(source.data()), count);
          if (!input) throw std::runtime_error("cannot read source tensor: " + name);
          const auto status = cudaMemcpy(device.data(), reinterpret_cast<const unsigned char*>(tensor.device_data) + offset,
                                          count, cudaMemcpyDeviceToHost);
          if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
          if (std::memcmp(source.data(), device.data(), count)) throw std::runtime_error("GPU upload mismatch: " + name);
          verified += count;
        }
      }
      if (weights.size() != model.tensor_count() || verified != model.payload_bytes())
        throw std::runtime_error("source/device tensor coverage differs");
      std::printf("all_tensor_uploads_byte_identical: %zu tensors, %zu bytes\n", weights.size(), verified);
      return 0;
    }
    for (int i = 2; i < argc; ++i) {
      const auto& tensor = model.at(argv[i]);
      std::vector<unsigned char> bytes(tensor.bytes());
      auto status = cudaMemcpy(bytes.data(), tensor.device_data, bytes.size(), cudaMemcpyDeviceToHost);
      if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
      std::printf("%s %s ", argv[i], tensor.dtype.c_str());
      for (auto b : bytes) std::printf("%02x", b);
      std::puts("");
    }
  } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
