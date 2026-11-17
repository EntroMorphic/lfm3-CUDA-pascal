#include <lfm/model_config.hpp>
#include <iostream>

int main(int argc, char** argv) {
  try {
    if (argc != 2) return 2;
    const auto c = lfm::ModelConfig::load(argv[1]);
    const auto& s = c.shape;
    std::cout << s.hidden << ' ' << s.mlp << ' ' << s.query_heads << ' '
              << s.kv_heads << ' ' << s.layers << ' ' << s.attention_layers << '\n';
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
