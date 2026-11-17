// lfm3-vision-check: the native vision encoder and projector (lfm/vision_encoder.cuh) on HF's own inputs, every stage
// written for comparison with HF's FP32 and BF16 dumps (tools/dump_lfm3_vision_reference.py,
// tools/compare_lfm3_vision_stages.py; validation/images-v1 rule 2).
// Usage: lfm3-vision-check MODEL_DIR CASE_DIR OUT_DIR [--linear fp32|x1|x2] [--attention fp32|x1|x2] [--repeat R]
//   --linear, --attention: the encoder's precision modes (lfm3_vision::Precision; validation/images-speed-v1);
//   --repeat R: each image's timed encode runs R times after one untimed encode, stages.json then holding the median.
//   CASE_DIR: pixel_values.npy (float32 [rows][1536]) and image_grid_thw.npy (int64 [images][3]), as HF dumped them.
//   OUT_DIR/<stage>.f32: each stage, the images' rows concatenated in order (HF's layout): patchifier, embedding_norm,
//   rotary.0 (cos) and rotary.1 (sin), layer00..layer13, proj.layer_norm, proj.linear_1, proj.gelu, proj.linear_2,
//   image_features (the BF16 output, as float); OUT_DIR/stages.json: each stage's shape, and each image's encode time.
#include <lfm/model_config.hpp>
#include <lfm/safetensors.hpp>
#include <lfm/vision_encoder.cuh>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

// A C-order little-endian .npy of float32 ('<f4') or int64 ('<i8'): its shape and raw bytes.
struct Npy {
  std::string descr;
  std::vector<int64_t> shape;
  std::vector<char> data;
};

Npy read_npy(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  char magic[8];
  f.read(magic, 8);
  if (std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error(path + ": not .npy");
  size_t header_len = 0;
  if (magic[6] == 1) {
    unsigned char b[2];
    f.read(reinterpret_cast<char*>(b), 2);
    header_len = b[0] | (b[1] << 8);
  } else {
    unsigned char b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    header_len = b[0] | (b[1] << 8) | (b[2] << 16) | (size_t(b[3]) << 24);
  }
  std::string header(header_len, ' ');
  f.read(header.data(), std::streamsize(header_len));
  Npy n;
  const auto d = header.find("'descr': '");
  n.descr = header.substr(d + 10, header.find('\'', d + 10) - d - 10);
  if (header.find("'fortran_order': False") == std::string::npos) throw std::runtime_error(path + ": Fortran order");
  const auto s = header.find('(', header.find("'shape'")), e = header.find(')', s);
  std::stringstream ss(header.substr(s + 1, e - s - 1));
  std::string item;
  size_t count = 1;
  while (std::getline(ss, item, ','))
    if (item.find_first_not_of(' ') != std::string::npos) n.shape.push_back(std::stoll(item)), count *= n.shape.back();
  const size_t width = n.descr == "<f4" ? 4 : n.descr == "<i8" ? 8 : 0;
  if (!width) throw std::runtime_error(path + ": dtype " + n.descr);
  n.data.resize(count * width);
  f.read(n.data.data(), std::streamsize(n.data.size()));
  if (!f) throw std::runtime_error(path + ": short");
  return n;
}

void append(std::map<std::string, std::vector<float>>& out, const std::string& name, const std::vector<float>& v) {
  auto& o = out[name];
  o.insert(o.end(), v.begin(), v.end());
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4 || argc % 2) {
    std::fprintf(stderr, "usage: lfm3-vision-check MODEL_DIR CASE_DIR OUT_DIR [--linear M] [--attention M] [--repeat R]\n");
    return 2;
  }
  try {
    const std::string model = argv[1], dir = argv[2], out_dir = argv[3];
    std::ifstream cf(model + "/config.json");
    std::stringstream cs;
    cs << cf.rdbuf();
    const lfm::Json root = lfm::Json::parse(cs.str());
    auto weights = std::make_shared<lfm::Safetensors>(model);
    lfm3_vision::Precision precision;
    int repeat = 0;
    for (int i = 4; i + 1 < argc; i += 2) {
      const std::string o = argv[i], v = argv[i + 1];
      if (o == "--linear") precision.linear = lfm3_vision::Precision::parse(v);
      else if (o == "--attention") precision.attention = lfm3_vision::Precision::parse(v);
      else if (o == "--repeat") {
        size_t used = 0;
        repeat = std::stoi(v, &used);
        if (used != v.size() || repeat < 0 || repeat > 1000) throw std::runtime_error("--repeat: an integer 0-1000");
      } else throw std::runtime_error("unknown option " + o);
    }
    lfm3_vision::Encoder encoder(weights, root, precision);
    const Npy pv = read_npy(dir + "/pixel_values.npy"), grid = read_npy(dir + "/image_grid_thw.npy");
    if (pv.descr != "<f4" || pv.shape.size() != 2 || pv.shape[1] != 1536) throw std::runtime_error("pixel_values shape");
    if (grid.descr != "<i8" || grid.shape.size() != 2 || grid.shape[1] != 3) throw std::runtime_error("grid shape");
    const auto* g = reinterpret_cast<const int64_t*>(grid.data.data());
    const auto* pixels = reinterpret_cast<const float*>(pv.data.data());
    std::map<std::string, std::vector<float>> stages;
    std::vector<double> seconds;
    size_t row = 0;
    lfm::fp32::DeviceArray<__nv_bfloat16> features(1);
    for (int64_t i = 0; i < grid.shape[0]; ++i) {
      const int t = int(g[i * 3]), h = int(g[i * 3 + 1]), w = int(g[i * 3 + 2]);
      if (t != 1) throw std::runtime_error("images only (t = 1)");
      const size_t n = size_t(h) * w;
      if (row + n > size_t(pv.shape[0])) throw std::runtime_error("grid larger than pixel_values");
      if (features.n < n / 4 * 2048) features = lfm::fp32::DeviceArray<__nv_bfloat16>(n / 4 * 2048);
      // timed without dumps first (the encode alone), then again with them
      std::vector<double> times;
      if (repeat > 0) encoder.encode(pixels + row * 1536, h, w, features.p);  // untimed, then the timed runs
      for (int r = 0; r < std::max(1, repeat); ++r) {
        cudaDeviceSynchronize();
        const auto t0 = std::chrono::steady_clock::now();
        encoder.encode(pixels + row * 1536, h, w, features.p);
        times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
      }
      std::sort(times.begin(), times.end());
      seconds.push_back(times[times.size() / 2]);
      lfm3_vision::Dumps d;
      encoder.encode(pixels + row * 1536, h, w, features.p, &d);
      append(stages, "patchifier", d.patchifier);
      append(stages, "embedding_norm", d.embedding_norm);
      append(stages, "rotary.0", d.cos);
      append(stages, "rotary.1", d.sin);
      for (size_t l = 0; l < d.layers.size(); ++l) {
        char name[32];
        std::snprintf(name, sizeof name, "layer%02zu", l);
        append(stages, name, d.layers[l]);
      }
      append(stages, "proj.layer_norm", d.proj_layer_norm);
      append(stages, "proj.linear_1", d.proj_linear_1);
      append(stages, "proj.gelu", d.proj_gelu);
      append(stages, "proj.linear_2", d.proj_linear_2);
      append(stages, "image_features", d.features);
      row += n;
    }
    if (row != size_t(pv.shape[0])) throw std::runtime_error("pixel_values rows left over");
    std::filesystem::create_directories(out_dir);
    std::ofstream meta(out_dir + "/stages.json");
    meta << "{\"rows\": " << row << ", \"seconds\": [";
    for (size_t i = 0; i < seconds.size(); ++i) meta << (i ? ", " : "") << seconds[i];
    meta << "], \"stages\": {";
    bool first = true;
    for (const auto& [name, v] : stages) {
      std::ofstream o(out_dir + "/" + name + ".f32", std::ios::binary);
      o.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * sizeof(float)));
      if (!o) throw std::runtime_error("write " + name);
      meta << (first ? "" : ", ") << "\"" << name << "\": " << v.size();
      first = false;
    }
    meta << "}}\n";
    std::printf("%zu rows, %zu images, encode seconds:", row, seconds.size());
    for (double s : seconds) std::printf(" %.3f", s);
    std::printf("\n");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "lfm3-vision-check: %s\n", e.what());
    return 1;
  }
}
