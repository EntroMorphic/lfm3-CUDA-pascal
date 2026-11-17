// Standalone captured-tensor replay; build/run ONLY in the offline Thor container.
// nvcc -O3 -std=c++20 -arch=sm_110 --default-stream=per-thread \
//   -I native/include native/tests/parity_attention_replay.cu -lcublas -o parity-attention-replay
// Usage: parity-attention-replay native.bf16.attention POSITION OUTPUT_PREFIX [split|post] [ITERATIONS]
// Trace layout is the existing 8704-BF16-elements/token attention diagnostic.
// OUTPUT_PREFIX.{bf16,scores.fp32,probabilities.fp32,output.fp32} are raw arrays.
// Prints local native/FP64-rounded comparisons; neither is a math-SDPA certificate.
// ITERATIONS>0 also reports CUDA-event attention-only latency (not full decode).
// Additional modes, requiring no captured trace:
//   --synthetic TOKENS OUTPUT_PREFIX [ITERATIONS] [INPUT_OFFSET] [OUTPUT_OFFSET]
//   --softmax INPUT_FP32 TOKENS OUTPUT_PREFIX [ITERATIONS] [INPUT_OFFSET] [OUTPUT_OFFSET]
// Both write [16,TOKENS] .scores.fp32 and .probabilities.fp32 for exact Torch
// softmax comparison. Float offsets 0..3 exercise base/row alignment; output
// defaults to the input offset. No certificate
// is inferred from CPU FP64 error or probability-sum checks.
//   --check-dispatch   CPU-only metadata and ILP index-coverage checks.
// Example Torch check after --synthetic N PREFIX 0 INPUT_OFFSET OUTPUT_OFFSET:
//   a = np.fromfile(PREFIX + '.scores.fp32', dtype=np.float32).reshape(16, N)
//   xs = torch.empty(16*N + 4, device='cuda', dtype=torch.float32)
//   ys = torch.empty_like(xs)
//   x = xs[INPUT_OFFSET:INPUT_OFFSET+16*N].view(16,N)
//   y = ys[OUTPUT_OFFSET:OUTPUT_OFFSET+16*N].view(16,N)
//   x.copy_(torch.from_numpy(a))
//   torch.softmax(x, -1, out=y)
//   b = np.fromfile(PREFIX + '.probabilities.fp32', dtype=np.float32).reshape(16,N)
//   print(np.count_nonzero(y.cpu().numpy().view(np.uint32) != b.view(np.uint32)))
// Check Torch version/backend explicitly; run this only in the offline container.

#include <lfm/parity_attention.cuh>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace {
template <typename T>
class Buffer {
 public:
  T* data = nullptr;
  explicit Buffer(size_t count) {
    lfm::parity::detail::cuda_check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)), "test malloc");
  }
  ~Buffer() { cudaFree(data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};

template <typename T>
void write_device(const std::string& path, const T* device, size_t count) {
  std::vector<T> host(count);
  lfm::parity::detail::cuda_check(cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost), "test copy result");
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(host.data()), count * sizeof(T));
  if (!out) throw std::runtime_error("write failed: " + path);
}

void print_plan(const lfm::parity::SoftmaxPlan& plan, int shared_limit) {
  std::cout << "softmax_source=" << lfm::parity::kSoftmaxSourceVersion
            << " kernel=" << lfm::parity::softmax_kernel_name(plan.kernel)
            << " tokens=" << plan.tokens << " threads=" << plan.threads
            << " registers=" << plan.registers_per_thread
            << " vector_width=" << plan.vector_width
            << " dynamic_shared_bytes=" << plan.dynamic_shared_bytes
            << " default_shared_limit=" << shared_limit
            << " bitwise_certified=" << lfm::parity::kBitwiseParityCertified << '\n';
}

int check_dispatch() {
  using lfm::parity::SoftmaxKernel;
  auto expect = [](bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
  };
  for (int n : {0, 32769}) {
    bool rejected = false;
    try { (void)lfm::parity::softmax_plan(n, 49152); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "invalid length was accepted");
  }
  expect(lfm::parity::softmax_plan(2048, 49152).kernel == SoftmaxKernel::PersistentWarp, "2048 dispatch");
  expect(lfm::parity::softmax_plan(2049, 49152).kernel == SoftmaxKernel::Register, "2049 dispatch");
  expect(lfm::parity::softmax_plan(9216, 49152).registers_per_thread == 9, "9216 dispatch");
  expect(lfm::parity::softmax_plan(9217, 49152).kernel == SoftmaxKernel::GlobalIlp, "9217 dispatch");
  expect(lfm::parity::softmax_plan(9220, 49152).kernel == SoftmaxKernel::SharedVector, "9220 dispatch");
  expect(lfm::parity::softmax_plan(9220, 49152, false).kernel == SoftmaxKernel::GlobalIlp, "unaligned dispatch");
  for (int limit : {49152, 65536, 101376}) {
    const int threshold = (limit - 128) / 4;
    expect(lfm::parity::softmax_plan(threshold - 4, limit).kernel == SoftmaxKernel::SharedVector, "below shared threshold");
    expect(lfm::parity::softmax_plan(threshold, limit).kernel == SoftmaxKernel::GlobalIlp, "strict shared threshold");
  }
  expect(lfm::parity::softmax_plan(32768, 49152).kernel == SoftmaxKernel::GlobalIlp, "32768 dispatch");
  // Independently enumerate upstream's ilpReduce index schedule, checking
  // every element is consumed exactly once at irregular/alignment boundaries.
  for (int n : {9217, 9218, 9219, 9220, 10239, 10240, 12252, 12253,
                12255, 12256, 12257, 16348, 16352, 32765, 32766, 32767, 32768}) {
    for (int shift = 0; shift < 4; ++shift) {
      std::vector<int> visits(n, 0);
      int size = n, origin = 0;
      if (shift) {
        for (int thread = shift; thread < 1024; ++thread) ++visits[thread - shift];
        size += shift - 1024;
        origin = 1024 - shift;
      }
      const int tail = size % 4096;
      for (int thread = 0; thread < 1024; ++thread) {
        for (int offset = thread; offset * 4 < size - tail; offset += 1024)
          for (int j = 0; j < 4; ++j) ++visits.at(origin + offset * 4 + j);
        for (int offset = size - tail + thread; offset < size; offset += 1024)
          ++visits.at(origin + offset);
      }
      for (int count : visits) expect(count == 1, "ILP index coverage error");
    }
  }
  std::cout << "CPU dispatch and ILP boundary coverage passed; no CUDA calls or numerical parity test\n";
  return 0;
}

int softmax_test(int argc, char** argv, bool synthetic) {
  const int base = synthetic ? 2 : 3;
  if (argc < base + 2 || argc > base + 5)
    throw std::invalid_argument("usage: --synthetic TOKENS PREFIX [ITERATIONS] [INPUT_OFFSET] [OUTPUT_OFFSET] OR --softmax INPUT TOKENS PREFIX [ITERATIONS] [INPUT_OFFSET] [OUTPUT_OFFSET]");
  const int tokens = std::stoi(argv[base]);
  if (tokens < 1 || tokens > 32768) throw std::invalid_argument("softmax tokens out of range");
  const std::string prefix = argv[base + 1];
  const int iterations = argc > base + 2 ? std::stoi(argv[base + 2]) : 0;
  const int offset = argc > base + 3 ? std::stoi(argv[base + 3]) : 0;
  const int output_offset = argc > base + 4 ? std::stoi(argv[base + 4]) : offset;
  if (iterations < 0 || offset < 0 || offset > 3 || output_offset < 0 || output_offset > 3)
    throw std::invalid_argument("invalid iterations/base offset");
  const size_t count = static_cast<size_t>(16) * tokens;
  std::vector<float> host(count);
  if (synthetic) {
    for (int head = 0; head < 16; ++head) {
      for (int t = 0; t < tokens; ++t) {
        uint32_t hash = static_cast<uint32_t>(t) * 0x9e3779b9U + head * 0x85ebca6bU;
        hash ^= hash >> 16;
        float value = static_cast<float>(static_cast<int>(hash & 32767) - 16384) / 1024.0f;
        switch (head % 8) {
          case 0: value = 0.0f; break;
          case 1: value = 2.0f; break;
          case 2: value = static_cast<float>((t % 513) - 256) / 32.0f; break;
          case 3: value += 1000.0f; break;
          case 4: value -= 1000.0f; break;
          case 5: value = t == tokens / 2 ? 80.0f : -80.0f; break;
          case 6: value *= 0.0009765625f; break;
          default: break;
        }
        host[static_cast<size_t>(head) * tokens + t] = value;
      }
    }
  } else {
    std::ifstream in(argv[2], std::ios::binary);
    in.read(reinterpret_cast<char*>(host.data()), count * sizeof(float));
    if (!in) throw std::runtime_error("softmax input missing or too short");
    if (in.peek() != std::ifstream::traits_type::eof()) throw std::runtime_error("softmax input must contain exactly 16*TOKENS floats");
    for (float value : host) if (!std::isfinite(value)) throw std::invalid_argument("nonfinite softmax input unsupported");
  }
  Buffer<float> input(count + 4), output(count + 4);
  lfm::parity::detail::cuda_check(cudaMemcpy(input.data + offset, host.data(), count * sizeof(float), cudaMemcpyHostToDevice), "softmax test input");
  const int shared_limit = lfm::parity::detail::default_shared_memory_per_block();
  auto run = [&] {
    return lfm::parity::softmax_fp32(input.data + offset, output.data + output_offset, tokens,
                                    cudaStreamPerThread, shared_limit);
  };
  const auto plan = run();
  lfm::parity::detail::cuda_check(cudaStreamSynchronize(cudaStreamPerThread), "softmax test synchronize");
  write_device(prefix + ".scores.fp32", input.data + offset, count);
  write_device(prefix + ".probabilities.fp32", output.data + output_offset, count);
  print_plan(plan, shared_limit);
  std::cout << "input_offset_floats=" << offset << " output_offset_floats=" << output_offset
            << " pattern=" << (synthetic ? "synthetic" : "file") << '\n';
  if (iterations) {
    for (int i = 0; i < 10; ++i) run();
    cudaEvent_t start, stop;
    lfm::parity::detail::cuda_check(cudaEventCreate(&start), "softmax event create");
    lfm::parity::detail::cuda_check(cudaEventCreate(&stop), "softmax event create");
    lfm::parity::detail::cuda_check(cudaEventRecord(start, cudaStreamPerThread), "softmax event start");
    for (int i = 0; i < iterations; ++i) run();
    lfm::parity::detail::cuda_check(cudaEventRecord(stop, cudaStreamPerThread), "softmax event stop");
    lfm::parity::detail::cuda_check(cudaEventSynchronize(stop), "softmax event synchronize");
    float milliseconds = 0;
    lfm::parity::detail::cuda_check(cudaEventElapsedTime(&milliseconds, start, stop), "softmax event elapsed");
    std::cout << "softmax_only_us=" << milliseconds * 1000 / iterations << " iterations=" << iterations << '\n';
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--check-dispatch") return check_dispatch();
    if (argc > 1 && std::string(argv[1]) == "--synthetic") return softmax_test(argc, argv, true);
    if (argc > 1 && std::string(argv[1]) == "--softmax") return softmax_test(argc, argv, false);
    if (argc < 4 || argc > 6) throw std::invalid_argument("usage: TRACE POSITION OUTPUT_PREFIX [split|post] [ITERATIONS]");
    const int position = std::stoi(argv[2]);
    if (position < 0 || position >= 32768) throw std::invalid_argument("position out of range");
    const int tokens = position + 1;
    const std::string mode = argc > 4 ? argv[4] : "split";
    if (mode != "split" && mode != "post") throw std::invalid_argument("scaling must be split or post");
    const int iterations = argc > 5 ? std::stoi(argv[5]) : 0;
    if (iterations < 0) throw std::invalid_argument("negative iterations");
    const auto scaling = mode == "split" ? lfm::parity::AttentionScaling::MathSdpaSplit
                                         : lfm::parity::AttentionScaling::PostQk;
    constexpr size_t row_width = 8704;
    std::ifstream in(argv[1], std::ios::binary);
    std::vector<__nv_bfloat16> trace(static_cast<size_t>(tokens) * row_width);
    in.read(reinterpret_cast<char*>(trace.data()), trace.size() * sizeof(trace[0]));
    if (!in) throw std::runtime_error("trace missing or too short");
    std::vector<__nv_bfloat16> keys(static_cast<size_t>(8) * tokens * 64), values(keys.size());
    for (int t = 0; t < tokens; ++t) {
      for (int head = 0; head < 8; ++head) {
        for (int d = 0; d < 64; ++d) {
          const size_t dst = (static_cast<size_t>(head) * tokens + t) * 64 + d;
          keys[dst] = trace[static_cast<size_t>(t) * row_width + 5632 + head * 64 + d];
          values[dst] = trace[static_cast<size_t>(t) * row_width + 6144 + head * 64 + d];
        }
      }
    }
    const auto* query = trace.data() + static_cast<size_t>(position) * row_width + 4608;
    const auto* native = trace.data() + static_cast<size_t>(position) * row_width + 6656;
    Buffer<__nv_bfloat16> dq(1024), dk(keys.size()), dv(values.size()), output(1024);
    auto copy = [](auto* dst, const auto* src, size_t count) {
      lfm::parity::detail::cuda_check(cudaMemcpy(dst, src, count * sizeof(*src), cudaMemcpyHostToDevice), "test copy input");
    };
    copy(dq.data, query, 1024);
    copy(dk.data, keys.data(), keys.size());
    copy(dv.data, values.data(), values.size());
    lfm::parity::MathAttention attention(tokens);
    auto run = [&] { attention.run(dq.data, dk.data, dv.data, output.data, tokens, tokens, scaling); };
    run();
    lfm::parity::detail::cuda_check(cudaStreamSynchronize(attention.stream()), "test synchronize");
    std::vector<__nv_bfloat16> actual(1024);
    lfm::parity::detail::cuda_check(cudaMemcpy(actual.data(), output.data, actual.size() * sizeof(actual[0]), cudaMemcpyDeviceToHost), "test actual");
    int native_differences = 0, oracle_differences = 0;
    double max_oracle_error = 0;
    for (int h = 0; h < 16; ++h) {
      std::vector<double> scores(tokens);
      for (int t = 0; t < tokens; ++t) {
        double sum = 0;
        for (int d = 0; d < 64; ++d)
          sum += static_cast<double>(__bfloat162float(query[h * 64 + d])) *
                 __bfloat162float(keys[(static_cast<size_t>(h / 2) * tokens + t) * 64 + d]);
        scores[t] = sum * 0.125;
      }
      const double maximum = *std::max_element(scores.begin(), scores.end());
      double denominator = 0;
      for (double& score : scores) { score = std::exp(score - maximum); denominator += score; }
      for (int d = 0; d < 64; ++d) {
        double numerator = 0;
        for (int t = 0; t < tokens; ++t)
          numerator += scores[t] * __bfloat162float(values[(static_cast<size_t>(h / 2) * tokens + t) * 64 + d]);
        // Double->float->BF16 double rounding is possible at exact BF16 midpoints;
        // report as a diagnostic, not an exact double-to-BF16 oracle certificate.
        const auto oracle = __float2bfloat16_rn(static_cast<float>(numerator / denominator));
        const int i = h * 64 + d;
        native_differences += std::memcmp(&actual[i], &native[i], sizeof(actual[i])) != 0;
        oracle_differences += std::memcmp(&actual[i], &oracle, sizeof(oracle)) != 0;
        max_oracle_error = std::max(max_oracle_error, std::abs(static_cast<double>(__bfloat162float(actual[i])) - numerator / denominator));
      }
    }
    const std::string prefix = argv[3];
    write_device(prefix + ".bf16", output.data, 1024);
    write_device(prefix + ".scores.fp32", attention.scores(), 16 * tokens);
    write_device(prefix + ".probabilities.fp32", attention.probabilities(), 16 * tokens);
    write_device(prefix + ".output.fp32", attention.output_fp32(), 1024);
    std::cout << "position=" << position << " scaling=" << mode
              << " native_differences=" << native_differences
              << " fp64_via_fp32_bf16_differences=" << oracle_differences
              << " max_unrounded_fp64_error=" << max_oracle_error
              << " scratch_bytes=" << attention.scratch_bytes(tokens) << '\n';
    print_plan(attention.plan(tokens), attention.default_shared_memory_bytes());
    if (iterations) {
      for (int i = 0; i < 10; ++i) run();
      cudaEvent_t start, stop;
      lfm::parity::detail::cuda_check(cudaEventCreate(&start), "event create");
      lfm::parity::detail::cuda_check(cudaEventCreate(&stop), "event create");
      lfm::parity::detail::cuda_check(cudaEventRecord(start, attention.stream()), "event start");
      for (int i = 0; i < iterations; ++i) run();
      lfm::parity::detail::cuda_check(cudaEventRecord(stop, attention.stream()), "event stop");
      lfm::parity::detail::cuda_check(cudaEventSynchronize(stop), "event synchronize");
      float milliseconds = 0;
      lfm::parity::detail::cuda_check(cudaEventElapsedTime(&milliseconds, start, stop), "event elapsed");
      std::cout << "attention_only_us=" << milliseconds * 1000 / iterations << " iterations=" << iterations << '\n';
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
