// Host probe for lfm::tokenizer_token_count. Usage: tokenizer-count-probe TOKENIZER.json MODEL_VOCAB
// Prints the count, or "error: <message>" with exit status 2.
#include <lfm/model_config.hpp>

#include <iostream>

int main(int argc, char** argv) {
  if (argc != 3) return 1;
  try {
    std::cout << lfm::tokenizer_token_count(argv[1], std::stoi(argv[2])) << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cout << "error: " << e.what() << "\n";
    return 2;
  }
}
