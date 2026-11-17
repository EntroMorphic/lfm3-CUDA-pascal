#pragma once
// A safetensors file's tensor table (dtype and shape by name), read from its header alone: no payload is read or
// uploaded. For capability checks made before the weights load (audio input, #3). Host-only.
#include <lfm/model_config.hpp>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace lfm::safetensors_header {

struct Entry {
  std::string dtype;
  std::vector<int64_t> shape;
};

inline std::map<std::string, Entry> read(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  uint64_t length = 0;
  f.read(reinterpret_cast<char*>(&length), 8);
  if (!f || length == 0 || length > 64 * 1024 * 1024) throw std::runtime_error("invalid Safetensors header size");
  std::string text(length, '\0');
  f.read(text.data(), std::streamsize(length));
  if (!f) throw std::runtime_error("truncated Safetensors header");
  const auto header = Json::parse(text);
  const auto* object = std::get_if<Json::Object>(&header.value);
  if (!object) throw std::runtime_error("invalid Safetensors header");
  std::map<std::string, Entry> out;
  for (const auto& [name, value] : *object) {
    if (name == "__metadata__") continue;
    Entry e;
    e.dtype = value.at("dtype").string();
    for (const auto& d : value.at("shape").array()) e.shape.push_back(d.integer());
    out.emplace(name, std::move(e));
  }
  return out;
}

}  // namespace lfm::safetensors_header
