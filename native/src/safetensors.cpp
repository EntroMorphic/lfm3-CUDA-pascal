#include <lfm/safetensors.hpp>
#include <lfm/checkpoint_path.hpp>

#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>

namespace lfm {
namespace {
void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

size_t json_size(const Json& j) {
  const double value = j.number();
  // All offsets must be exactly representable by the shared JSON parser.
  if (!std::isfinite(value) || value < 0 || std::floor(value) != value ||
      value > 9007199254740991.0 || value > static_cast<double>(std::numeric_limits<size_t>::max()))
    throw std::runtime_error("Safetensors integer out of range");
  return static_cast<size_t>(value);
}

const Json::Object& object(const Json& j) {
  const auto* result = std::get_if<Json::Object>(&j.value);
  if (!result) throw std::runtime_error("expected Safetensors JSON object");
  return *result;
}

size_t add_size(size_t a, size_t b) {
  if (b > std::numeric_limits<size_t>::max() - a) throw std::runtime_error("checkpoint allocation overflow");
  return a + b;
}

struct UploadStream {
  cudaStream_t stream = nullptr;
  UploadStream() { check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create checkpoint upload stream"); }
  ~UploadStream() {
    if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); }
  }
};
}  // namespace

size_t TensorView::elements() const {
  size_t count = 1;
  for (const int64_t dimension : shape) {
    if (dimension <= 0 || count > std::numeric_limits<size_t>::max() / static_cast<size_t>(dimension))
      throw std::runtime_error("invalid tensor shape");
    count *= static_cast<size_t>(dimension);
  }
  return count;
}

Safetensors::Safetensors(const std::string& path) : Safetensors(checkpoint_path(path).string(), true) {}

Safetensors::Safetensors(const std::string& path, bool upload) {
  try {
    if (is_checkpoint_index(path)) {
      if (!upload) throw std::runtime_error("nested checkpoint indexes are not supported");
      load_index(path);
      return;
    }
    file_descriptor_ = open(path.c_str(), O_RDONLY);
    if (file_descriptor_ < 0) throw std::runtime_error("cannot open Safetensors: " + path);
    struct stat status {};
    if (fstat(file_descriptor_, &status) != 0 || status.st_size < 8) throw std::runtime_error("invalid Safetensors file size");
    file_size_ = static_cast<size_t>(status.st_size);
    mapping_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, file_descriptor_, 0);
    if (mapping_ == MAP_FAILED) { mapping_ = nullptr; throw std::runtime_error("cannot memory-map Safetensors"); }
    uint64_t header_length = 0;
    std::memcpy(&header_length, mapping_, sizeof(header_length));
    if (header_length > file_size_ - 8 || header_length > 64 * 1024 * 1024)
      throw std::runtime_error("invalid Safetensors header size");
    payload_offset_ = 8 + static_cast<size_t>(header_length);
    payload_bytes_ = file_size_ - payload_offset_;
    const auto header = Json::parse(std::string(static_cast<const char*>(mapping_) + 8, header_length));
    for (const auto& [name, value] : object(header)) {
      if (name == "__metadata__") continue;
      TensorView tensor;
      tensor.dtype = value.at("dtype").string();
      for (const auto& dimension : value.at("shape").array()) tensor.shape.push_back(static_cast<int64_t>(json_size(dimension)));
      const auto& offsets = value.at("data_offsets").array();
      if (offsets.size() != 2) throw std::runtime_error("invalid tensor offsets: " + name);
      tensor.begin = json_size(offsets[0]); tensor.end = json_size(offsets[1]);
      if (tensor.begin > tensor.end || tensor.end > payload_bytes_) throw std::runtime_error("invalid tensor offsets: " + name);
      const size_t item_bytes = tensor.dtype == "BF16" ? 2 : tensor.dtype == "F32" ? 4 : 0;
      if (!item_bytes || tensor.elements() > std::numeric_limits<size_t>::max() / item_bytes ||
          tensor.bytes() != tensor.elements() * item_bytes || tensor.begin % item_bytes != 0)
        throw std::runtime_error("unsupported tensor encoding: " + name);
      tensors_.emplace(name, std::move(tensor));
    }
    std::vector<std::pair<size_t, size_t>> ranges;
    for (const auto& [name, tensor] : tensors_) ranges.emplace_back(tensor.begin, tensor.end);
    std::sort(ranges.begin(), ranges.end());
    size_t covered = 0;
    for (const auto& [begin, end] : ranges) {
      if (begin != covered) throw std::runtime_error("Safetensors payload has gaps or overlaps");
      covered = end;
    }
    if (!payload_bytes_ || covered != payload_bytes_) throw std::runtime_error("Safetensors payload is not fully described");
    if (upload) {
      // Preserve the original single-file layout and weight alignment exactly.
      check_cuda(cudaMalloc(&device_payload_, payload_bytes_), "cudaMalloc model payload");
      device_bytes_ = payload_bytes_;
      check_cuda(cudaMemcpy(device_payload_, static_cast<const std::byte*>(mapping_) + payload_offset_,
                            payload_bytes_, cudaMemcpyHostToDevice), "copy model payload");
      for (auto& [name, tensor] : tensors_)
        tensor.device_data = reinterpret_cast<const __nv_bfloat16*>(static_cast<const std::byte*>(device_payload_) + tensor.begin);
    }
  } catch (...) {
    if (device_payload_) cudaFree(device_payload_);
    if (mapping_) munmap(mapping_, file_size_);
    if (file_descriptor_ >= 0) close(file_descriptor_);
    throw;
  }
}

void Safetensors::load_index(const std::string& path) {
  const auto index_path = std::filesystem::path(path);
  const auto index_bytes = std::filesystem::file_size(index_path);
  if (!index_bytes || index_bytes > 16 * 1024 * 1024) throw std::runtime_error("invalid checkpoint index size");
  std::ifstream file(index_path);
  const std::string text((std::istreambuf_iterator<char>(file)), {});
  const auto index = Json::parse(text);
  const auto& weights = object(index.at("weight_map"));
  if (weights.empty()) throw std::runtime_error("empty checkpoint weight map");
  std::set<std::string> filenames;
  for (const auto& [name, shard] : weights) filenames.insert(shard.string());
  std::unordered_map<std::string, const Safetensors*> owners;
  for (const auto& filename : filenames) {
    const auto resolved = checkpoint_shard(index_path.parent_path(), filename);
    auto shard = std::unique_ptr<Safetensors>(new Safetensors(resolved.string(), false));
    payload_bytes_ = add_size(payload_bytes_, shard->payload_bytes_);
    for (const auto& [name, tensor] : shard->tensors_) {
      const auto entry = weights.find(name);
      if (entry == weights.end() || entry->second.string() != filename)
        throw std::runtime_error("checkpoint index/shard tensor mismatch: " + name);
      if (!tensors_.emplace(name, tensor).second) throw std::runtime_error("duplicate tensor across checkpoint shards: " + name);
      owners.emplace(name, shard.get());
    }
    source_shards_.push_back(std::move(shard));
  }
  if (tensors_.size() != weights.size()) throw std::runtime_error("checkpoint index references missing tensors");
  if (index.has("metadata") && index.at("metadata").has("total_size") &&
      json_size(index.at("metadata").at("total_size")) != payload_bytes_)
    throw std::runtime_error("checkpoint index total_size mismatch");

  // One final GPU allocation. Gate/up pairs land adjacently on initial upload,
  // even when split across shards. No temporary source-payload allocation or
  // additional expert gate/up copy resides on the GPU; disk files are unchanged.
  std::vector<std::string> order;
  std::set<std::string> placed;
  std::unordered_map<std::string, size_t> destinations;
  size_t device_bytes = 0;
  for (const auto& [name, unused] : weights) {
    if (placed.count(name)) continue;
    device_bytes = add_size(device_bytes, 255) & ~size_t(255);
    auto place = [&](const std::string& tensor_name) {
      order.push_back(tensor_name); placed.insert(tensor_name);
      destinations.emplace(tensor_name, device_bytes);
      device_bytes = add_size(device_bytes, tensors_.at(tensor_name).bytes());
    };
    place(name);
    const auto& tensor = tensors_.at(name);
    if (name.find(".experts.") != std::string::npos && name.ends_with(".w1.weight") &&
        tensor.dtype == "BF16" && tensor.shape.size() == 2) {
      const auto up = name.substr(0, name.size() - 9) + "w3.weight";
      const auto found = tensors_.find(up);
      if (found != tensors_.end() && !placed.count(up) && found->second.dtype == tensor.dtype && found->second.shape == tensor.shape)
        place(up);
    }
  }
  check_cuda(cudaMalloc(&device_payload_, device_bytes), "cudaMalloc indexed model payload");
  device_bytes_ = device_bytes;
  UploadStream upload;
  for (const auto& name : order) {
    auto& tensor = tensors_.at(name);
    const auto* owner = owners.at(name);
    auto* destination = static_cast<std::byte*>(device_payload_) + destinations.at(name);
    check_cuda(cudaMemcpyAsync(destination, static_cast<const std::byte*>(owner->mapping_) + owner->payload_offset_ + tensor.begin,
                               tensor.bytes(), cudaMemcpyHostToDevice, upload.stream), "upload indexed tensor");
    tensor.device_data = reinterpret_cast<const __nv_bfloat16*>(destination);
  }
  // Keep every mapped source alive until all device transfers are complete.
  // Do not infer completion from host-side return/staging of pageable copies:
  // releasing the last mappings early exposed zeroed tail norm tensors on Thor.
  check_cuda(cudaStreamSynchronize(upload.stream), "complete checkpoint upload stream");
  debug_snapshot_tail("uploaded-mapped");
  // Retain read-only file mappings with the model, as for single-file loading.
  // Their clean file-backed pages are reclaimable; this is not a second GPU
  // allocation or a materialized CPU weight copy.
  debug_snapshot_tail("sources-retained");
}

void Safetensors::debug_snapshot_tail(const std::string& stage) const {
  const char* prefix = std::getenv("LFM_TRACE_WEIGHT_TAIL");
  if (!prefix || !device_payload_) return;
  const char* filter = std::getenv("LFM_TRACE_WEIGHT_TAIL_STAGE");
  if (filter && stage != filter && !(std::string(filter) == "layers" && (stage == "state-reset" || stage.starts_with("layer-")))) return;
  const size_t count = std::min<size_t>(8192, device_bytes_);
  std::vector<unsigned char> bytes(count);
  check_cuda(cudaMemcpy(bytes.data(), static_cast<const std::byte*>(device_payload_) + device_bytes_ - count,
                        count, cudaMemcpyDeviceToHost), "snapshot weight tail");
  std::ofstream file(std::string(prefix) + "." + stage + ".bin", std::ios::binary);
  file.write(reinterpret_cast<const char*>(bytes.data()), count);
  if (!file) throw std::runtime_error("cannot write weight-tail diagnostic");
}

void Safetensors::verify_small_tensors() const {
  // Cheap fail-closed check for norm vectors, biases and small kernels before
  // model execution. Full matrix readback remains an explicit validation tool.
  std::vector<unsigned char> observed;
  for (const auto& [name, tensor] : tensors_) {
    if (tensor.bytes() > 64 * 1024) continue;
    const Safetensors* owner = mapping_ ? this : nullptr;
    if (!owner) for (const auto& shard : source_shards_) {
      if (shard->has(name)) { owner = shard.get(); break; }
    }
    if (!owner) throw std::runtime_error("missing retained source for device verification: " + name);
    observed.resize(tensor.bytes());
    check_cuda(cudaMemcpy(observed.data(), tensor.device_data, observed.size(), cudaMemcpyDeviceToHost),
               "verify small checkpoint tensor");
    const auto& source = owner->at(name);
    const auto* expected = static_cast<const std::byte*>(owner->mapping_) + owner->payload_offset_ + source.begin;
    if (std::memcmp(observed.data(), expected, observed.size()))
      throw std::runtime_error("checkpoint device bytes differ from source: " + name);
  }
}

Safetensors::~Safetensors() {
  if (device_payload_) cudaFree(device_payload_);
  if (mapping_) munmap(mapping_, file_size_);
  if (file_descriptor_ >= 0) close(file_descriptor_);
}

const TensorView& Safetensors::at(const std::string& name) const {
  const auto iterator = tensors_.find(name);
  if (iterator == tensors_.end()) throw std::out_of_range("missing tensor: " + name);
  return iterator->second;
}
}  // namespace lfm
