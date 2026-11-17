#pragma once
#include <lfm/model_config.hpp>
#include <filesystem>

namespace lfm {
inline bool is_checkpoint_index(const std::filesystem::path& path) {
  return path.filename().string().ends_with(".safetensors.index.json");
}

inline std::filesystem::path checkpoint_path(std::filesystem::path path) {
  if (std::filesystem::is_directory(path)) {
    const auto single = path / "model.safetensors";
    const auto index = path / "model.safetensors.index.json";
    if (std::filesystem::exists(single) && std::filesystem::exists(index))
      throw std::runtime_error("ambiguous checkpoint directory; select the file or index explicitly");
    path = std::filesystem::exists(single) ? single : index;
  }
  if (!std::filesystem::is_regular_file(path) ||
      (!is_checkpoint_index(path) && path.extension() != ".safetensors"))
    throw std::runtime_error("expected a Safetensors file, index or checkpoint directory: " + path.string());
  return std::filesystem::canonical(path);
}

inline std::filesystem::path checkpoint_shard(const std::filesystem::path& directory, const std::string& name) {
  const std::filesystem::path relative(name);
  if (relative.empty() || name.find('\0') != std::string::npos || name.find('\\') != std::string::npos ||
      relative.is_absolute() || relative.filename() != relative ||
      relative.extension() != ".safetensors")
    throw std::runtime_error("unsafe checkpoint shard filename: " + name);
  const auto root = std::filesystem::canonical(directory);
  const auto shard = checkpoint_path(root / relative);
  if (shard.parent_path() != root) throw std::runtime_error("checkpoint shard escapes its directory: " + name);
  return shard;
}
}  // namespace lfm
