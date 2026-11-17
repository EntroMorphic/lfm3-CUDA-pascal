#include <lfm/model_config.hpp>
#include <iostream>

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 3) return 2;
    const auto c = lfm::ModelConfig::load(argv[1]);
    if (argc == 3) c.require_execution_support();
    const auto& s = c.shape;
    std::cout << "{\"hidden\":" << s.hidden << ",\"mlp\":" << s.mlp
              << ",\"head_dim\":" << s.head_dim << ",\"layers\":" << s.layers
              << ",\"conv_width\":" << s.conv_width << ",\"context\":" << s.context
              << ",\"prefix\":\"" << c.tensor_prefix << "\",\"gate\":" << c.attention_gate
              << ",\"sandwich_norm\":" << c.sandwich_norm << ",\"schedule\":[";
    for (size_t i = 0; i < c.layer_specs.size(); ++i) {
      if (i) std::cout << ',';
      const auto& l = c.layer_specs[i];
      std::cout << "{\"kind\":" << static_cast<int>(l.kind)
                << ",\"theta\":" << l.rope_theta << ",\"rotary_pairs\":" << l.rotary_pairs
                << ",\"window\":" << l.window << '}';
    }
    std::cout << "]}\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
