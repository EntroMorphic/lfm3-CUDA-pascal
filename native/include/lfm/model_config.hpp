#pragma once

// Checkpoint metadata is host-only. Kernel launch parameters use the POD shape,
// never model names or a second set of architecture constants.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace lfm {

inline size_t checked_product(std::initializer_list<size_t> factors) {
  size_t result = 1;
  for (const size_t n : factors) {
    if (n && result > std::numeric_limits<size_t>::max() / n)
      throw std::runtime_error("allocation size overflow");
    result *= n;
  }
  return result;
}

class Json {
 public:
  using Object = std::map<std::string, Json>;
  using Array = std::vector<Json>;
  std::variant<std::nullptr_t, bool, double, std::string, Array, Object> value;
  // True for a number written without a fraction or exponent (Python json's int); numbers stay doubles.
  bool integral = false;
  const Json& at(const std::string& key) const {
    const auto* o = std::get_if<Object>(&value);
    if (!o || !o->count(key)) throw std::runtime_error("missing configuration field: " + key);
    return o->at(key);
  }
  bool has(const std::string& key) const {
    const auto* o = std::get_if<Object>(&value);
    return o && o->count(key);
  }
  double number() const {
    if (const auto* n = std::get_if<double>(&value)) return *n;
    throw std::runtime_error("configuration field must be a number");
  }
  int integer() const {
    const double n = number();
    if (std::floor(n) != n || n < 0 || n > std::numeric_limits<int>::max())
      throw std::runtime_error("configuration integer out of range");
    return static_cast<int>(n);
  }
  std::string string() const {
    if (const auto* s = std::get_if<std::string>(&value)) return *s;
    throw std::runtime_error("configuration field must be a string");
  }
  bool boolean() const {
    if (const auto* b = std::get_if<bool>(&value)) return *b;
    throw std::runtime_error("configuration field must be boolean");
  }
  const Array& array() const {
    if (const auto* a = std::get_if<Array>(&value)) return *a;
    throw std::runtime_error("configuration field must be an array");
  }
  static Json parse(const std::string& text) {
    size_t p = 0;
    const Json result = parse_value(text, p, 0);
    space(text, p);
    if (p != text.size()) throw std::runtime_error("trailing configuration JSON");
    return result;
  }
 private:
  static void space(const std::string& s, size_t& p) {
    while (p < s.size() && (s[p] == ' ' || s[p] == '\n' || s[p] == '\r' || s[p] == '\t')) ++p;
  }
  static char take(const std::string& s, size_t& p) {
    if (p == s.size()) throw std::runtime_error("truncated configuration JSON");
    return s[p++];
  }
  static unsigned hex4(const std::string& s, size_t& p) {
    unsigned n = 0;
    for (int i = 0; i < 4; ++i) {
      const char h = take(s, p);
      const int d = h >= '0' && h <= '9' ? h - '0' :
                    h >= 'a' && h <= 'f' ? h - 'a' + 10 :
                    h >= 'A' && h <= 'F' ? h - 'A' + 10 : -1;
      if (d < 0) throw std::runtime_error("invalid JSON Unicode escape");
      n = n * 16 + d;
    }
    return n;
  }
  static void utf8(std::string& out, unsigned n) {
    if (n < 128) out += static_cast<char>(n);
    else if (n < 2048) { out += static_cast<char>(0xc0 | (n >> 6)); out += static_cast<char>(0x80 | (n & 63)); }
    else if (n < 65536) {
      out += static_cast<char>(0xe0 | (n >> 12)); out += static_cast<char>(0x80 | ((n >> 6) & 63)); out += static_cast<char>(0x80 | (n & 63));
    } else {
      out += static_cast<char>(0xf0 | (n >> 18)); out += static_cast<char>(0x80 | ((n >> 12) & 63));
      out += static_cast<char>(0x80 | ((n >> 6) & 63)); out += static_cast<char>(0x80 | (n & 63));
    }
  }
  static std::string quoted(const std::string& s, size_t& p) {
    if (take(s, p) != '"') throw std::runtime_error("expected JSON string");
    std::string out;
    for (;;) {
      // the plain run up to the next quote, backslash or control character, appended at once (the same bytes as one
      // at a time: a 13 MB base64 image part parsed ~6x faster)
      size_t run = p;
      while (run < s.size()) {
        const unsigned char u = static_cast<unsigned char>(s[run]);
        if (u == '"' || u == '\\' || u < 32) break;
        ++run;
      }
      out.append(s, p, run - p);
      p = run;
      char c = take(s, p);
      if (c == '"') return out;
      if (static_cast<unsigned char>(c) < 32) throw std::runtime_error("control character in JSON string");
      if (c == '\\') {
        c = take(s, p);
        switch (c) {
          case '"': case '\\': case '/': break;
          case 'b': c = '\b'; break; case 'f': c = '\f'; break;
          case 'n': c = '\n'; break; case 'r': c = '\r'; break; case 't': c = '\t'; break;
          case 'u': {
            unsigned n = hex4(s, p);
            if (n >= 0xd800 && n <= 0xdbff) {
              if (take(s, p) != '\\' || take(s, p) != 'u') throw std::runtime_error("missing JSON low surrogate");
              const unsigned low = hex4(s, p);
              if (low < 0xdc00 || low > 0xdfff) throw std::runtime_error("invalid JSON low surrogate");
              n = 0x10000 + ((n - 0xd800) << 10) + low - 0xdc00;
            } else if (n >= 0xdc00 && n <= 0xdfff) {
              throw std::runtime_error("unpaired JSON low surrogate");
            }
            utf8(out, n);
            continue;
          }
          default: throw std::runtime_error("invalid JSON escape");
        }
      }
      out += c;
    }
  }
  static Json parse_value(const std::string& s, size_t& p, int depth) {
    if (depth > 64) throw std::runtime_error("configuration JSON nesting too deep");
    space(s, p);
    if (p == s.size()) throw std::runtime_error("missing configuration JSON value");
    const char c = s[p];
    if (c == '"') return Json{quoted(s, p)};
    if (c == '{' || c == '[') {
      ++p;
      Object object;
      Array array;
      space(s, p);
      const char end = c == '{' ? '}' : ']';
      if (p < s.size() && s[p] == end) { ++p; return c == '{' ? Json{std::move(object)} : Json{std::move(array)}; }
      for (;;) {
        space(s, p);
        if (c == '{') {
          const auto key = quoted(s, p);
          space(s, p);
          if (take(s, p) != ':') throw std::runtime_error("expected JSON colon");
          if (!object.emplace(key, parse_value(s, p, depth + 1)).second)
            throw std::runtime_error("duplicate configuration key: " + key);
        } else array.push_back(parse_value(s, p, depth + 1));
        space(s, p);
        const char separator = take(s, p);
        if (separator == end) return c == '{' ? Json{std::move(object)} : Json{std::move(array)};  // moved: a copy
        // here copied every nested value once per level
        if (separator != ',') throw std::runtime_error("invalid JSON separator");
      }
    }
    for (const auto& literal : {std::string("true"), std::string("false"), std::string("null")}) {
      if (s.compare(p, literal.size(), literal) == 0) {
        p += literal.size();
        if (literal == "null") return Json{nullptr};
        return Json{literal == "true"};
      }
    }
    const size_t begin = p;
    if (s[p] == '-') ++p;
    if (p == s.size() || s[p] < '0' || s[p] > '9') throw std::runtime_error("invalid JSON number");
    if (s[p] == '0') ++p;
    else while (p < s.size() && s[p] >= '0' && s[p] <= '9') ++p;
    bool integral = true;
    if (p < s.size() && s[p] == '.') {
      integral = false;
      const size_t digits = ++p;
      while (p < s.size() && s[p] >= '0' && s[p] <= '9') ++p;
      if (p == digits) throw std::runtime_error("invalid JSON fraction");
    }
    if (p < s.size() && (s[p] == 'e' || s[p] == 'E')) {
      integral = false;
      ++p;
      if (p < s.size() && (s[p] == '+' || s[p] == '-')) ++p;
      const size_t digits = p;
      while (p < s.size() && s[p] >= '0' && s[p] <= '9') ++p;
      if (p == digits) throw std::runtime_error("invalid JSON exponent");
    }
    double n = 0;
    try {
      n = std::stod(s.substr(begin, p - begin));
    } catch (const std::out_of_range&) {  // overflow, or underflow to a subnormal/zero (ERANGE)
      throw std::runtime_error("JSON number outside the double range");
    }
    if (!std::isfinite(n)) throw std::runtime_error("nonfinite JSON number");
    Json number{n};
    number.integral = integral;
    return number;
  }
};

struct RuntimeShape {
  int hidden = 0, mlp = 0, vocabulary = 0, layers = 0;
  int query_heads = 0, kv_heads = 0, head_dim = 0, conv_width = 0;
  int attention_layers = 0, context = 0;
  float epsilon = 0, rope_theta = 0;
  int kv_width() const { return kv_heads * head_dim; }
  int qkv_width() const { return hidden + 2 * kv_width(); }
  size_t kv_layer_stride() const { return static_cast<size_t>(kv_heads) * context * head_dim; }
  size_t kv_batch_stride() const { return static_cast<size_t>(attention_layers) * kv_layer_stride(); }
};

// Number of token ids the tokenizer defines (base BPE vocabulary plus added tokens). The output
// layer may have more rows than tokens: LFM3-Omni-2B scores 128000 ids but its tokenizer defines
// 125033, so ids 125033..127999 decode to nothing. Generation must only choose ids below the
// returned count, which is safe as a prefix bound only because the ids are verified to be exactly
// 0..count-1 (no gaps). Anything else is rejected rather than assumed.
inline int tokenizer_token_count(const std::string& path, int model_vocabulary) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open tokenizer: " + path);
  stream.seekg(0, std::ios::end);
  const auto bytes = stream.tellg();
  if (bytes <= 0 || bytes > 256 * 1024 * 1024) throw std::runtime_error("invalid tokenizer size");
  stream.seekg(0);
  const std::string text((std::istreambuf_iterator<char>(stream)), {});
  const Json j = Json::parse(text);
  if (model_vocabulary < 1) throw std::runtime_error("invalid model vocabulary");
  // seen: 0 = unused, 1 = base vocabulary, 2 = added token only.
  std::vector<unsigned char> seen(static_cast<size_t>(model_vocabulary), 0);
  int max_id = -1;
  const auto check = [&](int id) {
    if (id < 0 || id >= model_vocabulary) throw std::runtime_error("tokenizer id outside the model vocabulary");
    max_id = std::max(max_id, id);
    return static_cast<size_t>(id);
  };
  const auto* vocab = std::get_if<Json::Object>(&j.at("model").at("vocab").value);
  if (!vocab || vocab->empty()) throw std::runtime_error("tokenizer model.vocab must be a nonempty object");
  for (const auto& entry : *vocab) {
    const size_t id = check(entry.second.integer());
    // Two base tokens on one id is a corrupt vocabulary (decoding would be ambiguous).
    if (seen[id]) throw std::runtime_error("tokenizer base vocabulary repeats an id");
    seen[id] = 1;
  }
  // An added token may reuse a base id (it is the same id).
  for (const auto& token : j.at("added_tokens").array()) {
    const size_t id = check(token.at("id").integer());
    if (!seen[id]) seen[id] = 2;
  }
  const int count = max_id + 1;
  for (int id = 0; id < count; ++id)
    if (!seen[static_cast<size_t>(id)]) throw std::runtime_error("tokenizer ids are not contiguous");
  return count;
}

struct ModelConfig {
  enum class Architecture { Lfm2, Lfm3, Lfm3Omni };
  enum class LayerKind { Convolution, FullAttention, SlidingAttention };
  struct LayerSpec {
    LayerKind kind = LayerKind::Convolution;
    float rope_theta = 0;
    int rotary_pairs = 0;  // Frequency denominator remains the full head dimension.
    int window = 0;        // 0 means unrestricted causal attention.
  };
  Architecture architecture = Architecture::Lfm2;
  std::vector<LayerSpec> layer_specs;
  bool sandwich_norm = false;
  bool attention_gate = false;
  std::string tensor_prefix = "model.";
  std::string attention_output_name = "out_proj.weight";
  std::string query_norm_name = "q_layernorm.weight";
  std::string key_norm_name = "k_layernorm.weight";

  bool is_lfm3() const { return architecture != Architecture::Lfm2; }
  void require_execution_support() const {
    if (is_lfm3())
      throw std::runtime_error("LFM3 metadata is supported, but native LFM3 execution is not implemented yet: operator forward-pass integration and end-to-end parity validation are still required");
  }
  RuntimeShape shape;
  std::vector<bool> attention;
  std::vector<int> eos;
  bool tied = true;
  bool moe = false, norm_topk_prob = false, use_expert_bias = false;
  int num_experts = 0, experts_per_token = 0, expert_width = 0, dense_layers = 0;
  float routed_scaling = 1.0f;
  static ModelConfig load(const std::string& path, int context_limit = 0) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open model config: " + path);
    stream.seekg(0, std::ios::end);
    const auto bytes = stream.tellg();
    if (bytes < 0 || bytes > 4 * 1024 * 1024) throw std::runtime_error("invalid model config size");
    stream.seekg(0);
    const std::string text((std::istreambuf_iterator<char>(stream)), {});
    if (text.size() > 4 * 1024 * 1024) throw std::runtime_error("model config too large");
    const Json j = Json::parse(text);
    const std::string type = j.at("model_type").string();
    if (type == "lfm3" || type == "lfm3_omni") return load_lfm3(j, context_limit);
    if (type != "lfm2" && type != "lfm2_moe") throw std::runtime_error("unsupported model architecture: " + type + " (implemented: LFM2 text)");
    ModelConfig c;
    c.moe = type == "lfm2_moe";
    auto& s = c.shape;
    s.hidden = j.at("hidden_size").integer();
    s.vocabulary = j.at("vocab_size").integer();
    s.layers = j.at("num_hidden_layers").integer();
    s.query_heads = j.at("num_attention_heads").integer();
    s.kv_heads = j.at("num_key_value_heads").integer();
    s.conv_width = j.at("conv_L_cache").integer();
    s.epsilon = static_cast<float>(j.at("norm_eps").number());
    const Json& rope = j.has("rope_parameters") && !std::holds_alternative<std::nullptr_t>(j.at("rope_parameters").value) ? j.at("rope_parameters") : j;
    if (rope.has("rope_type") && rope.at("rope_type").string() != "default")
      throw std::runtime_error("unsupported RoPE scaling type");
    if (j.has("rope_scaling") && !std::holds_alternative<std::nullptr_t>(j.at("rope_scaling").value))
      throw std::runtime_error("unsupported rope_scaling");
    s.rope_theta = static_cast<float>(rope.at("rope_theta").number());
    const int maximum = j.at("max_position_embeddings").integer();
    if (context_limit == 0) context_limit = std::min(32768, maximum);
    if (context_limit <= 0 || context_limit > maximum) throw std::runtime_error("context capacity exceeds model positional limit");
    s.context = context_limit;
    if (s.hidden <= 0 || s.vocabulary <= 0 || s.layers <= 0 || s.layers > 4096 ||
        s.query_heads <= 0 || s.kv_heads <= 0 || s.hidden % s.query_heads || s.query_heads % s.kv_heads ||
        s.conv_width <= 0 || s.epsilon <= 0 || !std::isfinite(s.epsilon) ||
        s.rope_theta <= 0 || !std::isfinite(s.rope_theta))
      throw std::runtime_error("invalid model dimensions or normalization/RoPE parameters");
    s.head_dim = s.hidden / s.query_heads;
    if (s.head_dim % 2 || s.head_dim > 1024 || s.hidden > 1048576 || s.vocabulary > 16777216)
      throw std::runtime_error("unsupported or oversized model dimensions");
    if (j.has("head_dim") && j.at("head_dim").integer() != s.head_dim)
      throw std::runtime_error("head_dim conflicts with hidden size and head count");
    if (j.has("conv_dim") && j.at("conv_dim").integer() != s.hidden)
      throw std::runtime_error("convolution width differs from hidden size; operator not implemented");
    if (j.has("partial_rotary_factor") && j.at("partial_rotary_factor").number() != 1.0)
      throw std::runtime_error("partial rotary embeddings are not implemented");
    // HF LFM2 exposes the nominal FF dimension before its SwiGLU adjustment.
    double ff = j.has("block_ff_dim") ? j.at("block_ff_dim").integer() : j.at("intermediate_size").integer();
    if (j.has("block_use_swiglu") && !j.at("block_use_swiglu").boolean())
      throw std::runtime_error("only SwiGLU feed-forward blocks are implemented");
    if (!c.moe && (!j.has("block_auto_adjust_ff_dim") || j.at("block_auto_adjust_ff_dim").boolean())) {
      ff = std::floor(2.0 * ff / 3.0);
      if (!j.has("block_ffn_dim_multiplier") || !std::holds_alternative<std::nullptr_t>(j.at("block_ffn_dim_multiplier").value)) {
        if (j.has("block_ffn_dim_multiplier")) ff = std::floor(ff * j.at("block_ffn_dim_multiplier").number());
        const int multiple = j.at("block_multiple_of").integer();
        if (!multiple) throw std::runtime_error("invalid FF alignment");
        ff = std::ceil(ff / multiple) * multiple;
      }
    }
    if (ff <= 0 || ff > 1048576 || !std::isfinite(ff)) throw std::runtime_error("invalid feed-forward width");
    s.mlp = static_cast<int>(ff);
    if (c.moe) {
      c.num_experts = j.at("num_experts").integer();
      c.experts_per_token = j.at("num_experts_per_tok").integer();
      c.expert_width = j.at("moe_intermediate_size").integer();
      c.dense_layers = j.at("num_dense_layers").integer();
      c.norm_topk_prob = j.at("norm_topk_prob").boolean();
      c.use_expert_bias = j.at("use_expert_bias").boolean();
      c.routed_scaling = static_cast<float>(j.at("routed_scaling_factor").number());
      if (c.num_experts < 1 || c.num_experts > 4096 || c.experts_per_token < 1 ||
          c.experts_per_token > c.num_experts || c.expert_width < 1 || c.expert_width > 1048576 ||
          c.dense_layers > s.layers || c.routed_scaling < 0 || !std::isfinite(c.routed_scaling))
        throw std::runtime_error("invalid MoE routing/expert dimensions");
    }
    if (j.has("conv_bias") && j.at("conv_bias").boolean()) throw std::runtime_error("convolution bias is not implemented");
    if (j.has("use_pos_enc") && !j.at("use_pos_enc").boolean()) throw std::runtime_error("models without positional encoding are not implemented");
    c.tied = j.has("tie_word_embeddings") ? j.at("tie_word_embeddings").boolean() : j.at("tie_embedding").boolean();
    if (j.has("tie_word_embeddings") && j.has("tie_embedding") && c.tied != j.at("tie_embedding").boolean())
      throw std::runtime_error("conflicting embedding tying configuration");
    if (j.has("layer_types") && !std::holds_alternative<std::nullptr_t>(j.at("layer_types").value)) {
      const auto& kinds = j.at("layer_types").array();
      if (kinds.size() != static_cast<size_t>(s.layers)) throw std::runtime_error("layer_types length mismatch");
      for (const auto& kind : kinds) {
        const std::string name = kind.string();
        if (name != "conv" && name != "full_attention") throw std::runtime_error("unsupported layer operator: " + name);
        c.attention.push_back(name == "full_attention");
      }
    } else {
      const bool indexed = j.has("full_attn_idxs") && !std::holds_alternative<std::nullptr_t>(j.at("full_attn_idxs").value);
      c.attention.assign(s.layers, !indexed);
      if (indexed) for (const auto& i : j.at("full_attn_idxs").array()) {
        const int layer = i.integer();
        if (layer >= s.layers) throw std::runtime_error("attention layer index out of range");
        c.attention[layer] = true;
      }
    }
    s.attention_layers = std::count(c.attention.begin(), c.attention.end(), true);
    const auto& eos = j.at("eos_token_id");
    if (std::holds_alternative<Json::Array>(eos.value)) {
      for (const auto& id : eos.array()) c.eos.push_back(id.integer());
    } else c.eos.push_back(eos.integer());
    if (c.eos.empty()) throw std::runtime_error("empty EOS token list");
    for (int id : c.eos) if (id >= s.vocabulary) throw std::runtime_error("EOS token outside vocabulary");
    if (s.qkv_width() < s.hidden || s.kv_batch_stride() > static_cast<size_t>(std::numeric_limits<int>::max()))
      throw std::runtime_error("model cache stride exceeds kernel index range; reduce --context-length");
    checked_product({2, static_cast<size_t>(s.hidden), static_cast<size_t>(s.layers), static_cast<size_t>(s.conv_width)});
    return c;
  }

  static ModelConfig load_lfm3(const Json& root, int context_limit) {
    ModelConfig c;
    const bool omni = root.at("model_type").string() == "lfm3_omni";
    const Json& j = omni ? root.at("text_config") : root;
    if (j.at("model_type").string() != "lfm3")
      throw std::runtime_error("LFM3-Omni requires an lfm3 text_config");
    c.architecture = omni ? Architecture::Lfm3Omni : Architecture::Lfm3;
    c.tensor_prefix = omni ? "model.language_model." : "model.";
    c.attention_output_name = "o_proj.weight";
    c.query_norm_name = "q_norm.weight";
    c.key_norm_name = "k_norm.weight";
    c.attention_gate = true;
    c.sandwich_norm = j.at("use_sandwich_norm").boolean();
    auto& s = c.shape;
    s.hidden = j.at("hidden_size").integer();
    s.mlp = j.at("intermediate_size").integer(); // Already the actual SwiGLU width.
    s.vocabulary = j.at("vocab_size").integer();
    s.layers = j.at("num_hidden_layers").integer();
    s.query_heads = j.at("num_attention_heads").integer();
    s.kv_heads = j.at("num_key_value_heads").integer();
    s.conv_width = j.at("conv_L_cache").integer();
    s.epsilon = static_cast<float>(j.at("norm_eps").number());
    if (s.hidden <= 0 || s.hidden > 1048576 || s.mlp <= 0 || s.mlp > 1048576 ||
        s.vocabulary <= 0 || s.vocabulary > 16777216 || s.layers <= 0 || s.layers > 4096 ||
        s.query_heads <= 0 || s.kv_heads <= 0 || s.hidden % s.query_heads ||
        s.query_heads % s.kv_heads || s.conv_width <= 0 || s.conv_width > 1024 ||
        !std::isfinite(s.epsilon) || s.epsilon <= 0)
      throw std::runtime_error("invalid LFM3 dimensions or normalization parameters");
    s.head_dim = j.has("head_dim") ? j.at("head_dim").integer() : s.hidden / s.query_heads;
    if (s.head_dim <= 0 || s.head_dim > 1024 || s.head_dim % 2 || s.head_dim != s.hidden / s.query_heads)
      throw std::runtime_error("unsupported LFM3 head dimension");
    if (j.has("conv_bias") && j.at("conv_bias").boolean())
      throw std::runtime_error("LFM3 convolution bias is not implemented");
    if (j.has("rope_scaling") && !std::holds_alternative<std::nullptr_t>(j.at("rope_scaling").value))
      throw std::runtime_error("LFM3 rope_scaling is not implemented");
    const int maximum = j.at("max_position_embeddings").integer();
    // This Omni checkpoint is documented for 32K despite the backbone's larger limit.
    const int supported_maximum = omni ? std::min(32768, maximum) : maximum;
    if (context_limit == 0) context_limit = std::min(32768, supported_maximum);
    if (context_limit <= 0 || context_limit > supported_maximum)
      throw std::runtime_error("context capacity exceeds LFM3 model limit");
    s.context = context_limit;
    c.tied = j.at("tie_word_embeddings").boolean();
    if (omni && root.has("tie_word_embeddings") && root.at("tie_word_embeddings").boolean() != c.tied)
      throw std::runtime_error("conflicting Omni/text embedding tying configuration");
    const auto& kinds = j.at("layer_types").array();
    if (kinds.size() != static_cast<size_t>(s.layers))
      throw std::runtime_error("LFM3 layer_types length mismatch");
    for (const auto& kind : kinds) {
      const std::string name = kind.string();
      LayerSpec spec;
      if (name == "conv") {
        c.attention.push_back(false);
        c.layer_specs.push_back(spec);
        continue;
      }
      if (name != "full_attention" && name != "sliding_attention")
        throw std::runtime_error("unsupported LFM3 layer operator: " + name);
      spec.kind = name == "full_attention" ? LayerKind::FullAttention : LayerKind::SlidingAttention;
      const Json& rope = j.at("rope_parameters").at(name);
      if (rope.at("rope_type").string() != "default")
        throw std::runtime_error("unsupported LFM3 RoPE type");
      spec.rope_theta = static_cast<float>(rope.at("rope_theta").number());
      const double factor = rope.has("partial_rotary_factor") ? rope.at("partial_rotary_factor").number() : 1.0;
      if (!std::isfinite(spec.rope_theta) || spec.rope_theta <= 0 || factor <= 0 || factor > 1)
        throw std::runtime_error("invalid LFM3 RoPE parameters");
      spec.rotary_pairs = static_cast<int>(factor * s.head_dim) / 2;
      if (spec.kind == LayerKind::SlidingAttention) {
        spec.window = j.at("sliding_window").integer();
        if (spec.window <= 0) throw std::runtime_error("invalid LFM3 sliding window");
      }
      if (s.rope_theta == 0) s.rope_theta = spec.rope_theta;
      c.attention.push_back(true);
      c.layer_specs.push_back(spec);
      ++s.attention_layers;
    }
    const Json& eos = j.at("eos_token_id");
    if (std::holds_alternative<Json::Array>(eos.value)) {
      for (const auto& id : eos.array()) c.eos.push_back(id.integer());
    } else c.eos.push_back(eos.integer());
    if (c.eos.empty()) throw std::runtime_error("empty LFM3 EOS token list");
    for (int id : c.eos) if (id >= s.vocabulary) throw std::runtime_error("LFM3 EOS outside vocabulary");
    if (s.kv_batch_stride() > static_cast<size_t>(std::numeric_limits<int>::max()))
      throw std::runtime_error("LFM3 cache stride exceeds kernel index range");
    checked_product({static_cast<size_t>(s.hidden), static_cast<size_t>(s.layers), static_cast<size_t>(s.conv_width)});
    return c;
  }
};

// Set once, before any allocations or launches, and copied into kernel args.
inline RuntimeShape runtime_shape;
}  // namespace lfm
