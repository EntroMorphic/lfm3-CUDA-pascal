#pragma once

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfm {

struct TensorView {
  std::string dtype;
  std::vector<int64_t> shape;
  size_t begin = 0;
  size_t end = 0;
  const __nv_bfloat16* device_data = nullptr;

  size_t elements() const;
  size_t bytes() const { return end - begin; }
};

class Safetensors {
 public:
  explicit Safetensors(const std::string& path);
  ~Safetensors();

  Safetensors(const Safetensors&) = delete;
  Safetensors& operator=(const Safetensors&) = delete;

  const TensorView& at(const std::string& name) const;
  bool has(const std::string& name) const { return tensors_.count(name) != 0; }
  size_t tensor_count() const { return tensors_.size(); }
  size_t payload_bytes() const { return payload_bytes_; }
  size_t device_bytes() const { return device_bytes_; }
  void debug_snapshot_tail(const std::string& stage) const;
  void verify_small_tensors() const;

 private:
  Safetensors(const std::string& path, bool upload);
  void load_index(const std::string& path);
  int file_descriptor_ = -1;
  void* mapping_ = nullptr;
  size_t file_size_ = 0;
  size_t payload_offset_ = 0;
  void* device_payload_ = nullptr;
  size_t payload_bytes_ = 0;
  size_t device_bytes_ = 0;
  std::unordered_map<std::string, TensorView> tensors_;
  std::vector<std::unique_ptr<Safetensors>> source_shards_;
};

}  // namespace lfm
