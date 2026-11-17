cmake_minimum_required(VERSION 3.24)
set(CMAKE_SUPPRESS_REGENERATION ON)
project(liquid_native LANGUAGES CXX CUDA)
option(LIQUID_PROFILE_NVTX "Enable diagnostic host-side NVTX component ranges" OFF)
option(LIQUID_LFM3_OPS_TESTS "Build the isolated LFM3 operator test bridge" OFF)
# G4: the LFM3 engine's GEMMs from lfm::gemm (native/include/lfm/gemm.cuh) instead of cuBLAS; the
# engine targets then link no cuBLAS. Held to journal §9's perplexity bar, not bitwise identity.
# G4: the engine runs on lfm::gemm by default (validation/g4-gates-v2/). OFF builds the cuBLAS engine, kept for A/B.
option(LIQUID_LFM3_NATIVE_GEMM "LFM3 engine without cuBLAS (lfm::gemm)" ON)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
function(lfm3_engine_gemm target)
  if(LIQUID_LFM3_NATIVE_GEMM)
    target_compile_definitions(${target} PRIVATE LFM3_NATIVE_GEMM)
  else()
    target_link_libraries(${target} PRIVATE CUDA::cublas)
  endif()
endfunction()

# Image input (#4, validation/images-v1): the vendored libjpeg-turbo 3.2.0, built here from its release tarball
# (hash-checked, no network), and the image front end, compiled at -O2 as its bit-exactness against HF was validated.
include(ExternalProject)
set(LFM_LJT_PREFIX ${CMAKE_BINARY_DIR}/libjpeg-turbo)
ExternalProject_Add(libjpeg_turbo_build
  URL ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libjpeg-turbo/libjpeg-turbo-3.2.0.tar.gz
  URL_HASH SHA256=6f30092cef9fb839779646608f4ee14ae3cbac989c47fa05e841b0841f09878e
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  INSTALL_DIR ${LFM_LJT_PREFIX}
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DCMAKE_INSTALL_LIBDIR=lib
             -DENABLE_SHARED=OFF -DENABLE_STATIC=ON -DWITH_TURBOJPEG=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  BUILD_BYPRODUCTS ${LFM_LJT_PREFIX}/lib/libjpeg.a)
file(MAKE_DIRECTORY ${LFM_LJT_PREFIX}/include)
add_library(lfm_jpeg STATIC IMPORTED)
set_target_properties(lfm_jpeg PROPERTIES IMPORTED_LOCATION ${LFM_LJT_PREFIX}/lib/libjpeg.a
                      INTERFACE_INCLUDE_DIRECTORIES ${LFM_LJT_PREFIX}/include)
add_dependencies(lfm_jpeg libjpeg_turbo_build)
add_library(lfm_image STATIC src/image_decode.cpp src/image_preprocess.cpp)
target_include_directories(lfm_image PUBLIC include)
target_compile_features(lfm_image PUBLIC cxx_std_20)
target_compile_options(lfm_image PRIVATE -O2)
target_link_libraries(lfm_image PUBLIC lfm_jpeg)

add_executable(liquid-lfm3 src/lfm3_main.cu src/safetensors.cpp)
target_include_directories(liquid-lfm3 PRIVATE include)
target_compile_features(liquid-lfm3 PRIVATE cxx_std_20 cuda_std_20)
target_compile_options(liquid-lfm3 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false>)
target_link_libraries(liquid-lfm3 PRIVATE CUDA::cudart)
lfm3_engine_gemm(liquid-lfm3)
set_target_properties(liquid-lfm3 PROPERTIES CUDA_ARCHITECTURES 110)

add_executable(liquid-lfm3-worker src/lfm3_worker.cu src/safetensors.cpp)
target_include_directories(liquid-lfm3-worker PRIVATE include)
target_compile_features(liquid-lfm3-worker PRIVATE cxx_std_20 cuda_std_20)
target_compile_options(liquid-lfm3-worker PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false>)
target_link_libraries(liquid-lfm3-worker PRIVATE CUDA::cudart lfm_image)
lfm3_engine_gemm(liquid-lfm3-worker)
set_target_properties(liquid-lfm3-worker PROPERTIES CUDA_ARCHITECTURES 110)

# One executable for text serving: loopback HTTP, native text frontend and the worker engine in-process.
add_executable(liquid-lfm3-server src/lfm3_server.cu src/safetensors.cpp)
target_include_directories(liquid-lfm3-server PRIVATE include)
target_compile_features(liquid-lfm3-server PRIVATE cxx_std_20 cuda_std_20)
target_compile_options(liquid-lfm3-server PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false>)
target_link_libraries(liquid-lfm3-server PRIVATE CUDA::cudart Threads::Threads lfm_image)
lfm3_engine_gemm(liquid-lfm3-server)
set_target_properties(liquid-lfm3-server PROPERTIES CUDA_ARCHITECTURES 110)

# #2 step 2 (A2): the native codec decoder (FP32 activations; no cuBLAS, no cuFFT).
add_executable(liquid-codec src/codec_main.cu src/safetensors.cpp)
target_include_directories(liquid-codec PRIVATE include)
target_compile_features(liquid-codec PRIVATE cxx_std_20 cuda_std_20)
target_compile_options(liquid-codec PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false>)
target_link_libraries(liquid-codec PRIVATE CUDA::cudart)
set_target_properties(liquid-codec PROPERTIES CUDA_ARCHITECTURES 110)

# #2 step 3 (A1/B1): the native depth decoder on recorded passes.
add_executable(liquid-depth src/depth_main.cu src/safetensors.cpp)
target_include_directories(liquid-depth PRIVATE include)
target_compile_features(liquid-depth PRIVATE cxx_std_20 cuda_std_20)
target_compile_options(liquid-depth PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false>)
target_link_libraries(liquid-depth PRIVATE CUDA::cudart)
set_target_properties(liquid-depth PROPERTIES CUDA_ARCHITECTURES 110)

add_executable(liquid-native src/main.cu src/safetensors.cpp src/tokenizer.cpp)
target_include_directories(liquid-native PRIVATE include)
if(LIQUID_PROFILE_NVTX)
  target_compile_definitions(liquid-native PRIVATE LFM_PROFILE_NVTX)
endif()
target_compile_features(liquid-native PRIVATE cxx_std_20)
target_compile_options(liquid-native PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --default-stream=per-thread>
)
target_link_libraries(liquid-native PRIVATE CUDA::cudart CUDA::cublas CUDA::cublasLt pcre2-8)
set_target_properties(liquid-native PROPERTIES
    CUDA_ARCHITECTURES 110
    CUDA_SEPARABLE_COMPILATION ON
)

if(LIQUID_LFM3_OPS_TESTS)
  # validation/text-wait-v1: sliced text generation against whole, alone and beside speech rows (the worker in-process,
  # built as liquid-lfm3-server builds it).
  add_executable(lfm3-text-slice-check EXCLUDE_FROM_ALL tests/lfm3_text_slice_check.cu src/safetensors.cpp)
  target_include_directories(lfm3-text-slice-check PRIVATE include)
  target_compile_features(lfm3-text-slice-check PRIVATE cxx_std_20 cuda_std_20)
  target_compile_options(lfm3-text-slice-check PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false>)
  target_link_libraries(lfm3-text-slice-check PRIVATE CUDA::cudart Threads::Threads lfm_image)
  lfm3_engine_gemm(lfm3-text-slice-check)
  set_target_properties(lfm3-text-slice-check PROPERTIES CUDA_ARCHITECTURES 110)
  add_executable(lfm3-decode-batch-probe EXCLUDE_FROM_ALL probes/lfm3_decode_batch_probe.cu src/safetensors.cpp)
  target_include_directories(lfm3-decode-batch-probe PRIVATE include)
  target_compile_features(lfm3-decode-batch-probe PRIVATE cxx_std_20 cuda_std_20)
  target_compile_options(lfm3-decode-batch-probe PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false>)
  target_link_libraries(lfm3-decode-batch-probe PRIVATE CUDA::cudart)
  lfm3_engine_gemm(lfm3-decode-batch-probe)
  set_target_properties(lfm3-decode-batch-probe PROPERTIES CUDA_ARCHITECTURES 110)
  # Probes the cuBLAS engine's BF16-score softmax kernels directly, so it always builds that engine (G4: the default
  # engine is lfm::gemm, with FP32 scores; tools/check_lfm3_forward.py --native-tiers covers it).
  add_executable(lfm3-serving-ops-probe probes/lfm3_serving_ops_probe.cu src/safetensors.cpp)
  target_include_directories(lfm3-serving-ops-probe PRIVATE include)
  target_compile_features(lfm3-serving-ops-probe PRIVATE cxx_std_20 cuda_std_20)
  target_compile_options(lfm3-serving-ops-probe PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false>)
  target_link_libraries(lfm3-serving-ops-probe PRIVATE CUDA::cudart CUDA::cublas)
  set_target_properties(lfm3-serving-ops-probe PROPERTIES CUDA_ARCHITECTURES 110)
  add_executable(text-sampling-oracle tests/text_sampling_oracle.cpp)
  target_include_directories(text-sampling-oracle PRIVATE include)
  target_compile_features(text-sampling-oracle PRIVATE cxx_std_17)
  add_executable(lfm3-projection-probe probes/lfm3_projection_probe.cu)
  target_link_libraries(lfm3-projection-probe PRIVATE CUDA::cudart CUDA::cublas)
  set_target_properties(lfm3-projection-probe PROPERTIES CUDA_ARCHITECTURES 110)
  add_executable(lfm3-projection-modes-probe EXCLUDE_FROM_ALL probes/lfm3_projection_modes_probe.cu)
  target_compile_features(lfm3-projection-modes-probe PRIVATE cuda_std_17)
  target_link_libraries(lfm3-projection-modes-probe PRIVATE CUDA::cudart CUDA::cublas)
  set_target_properties(lfm3-projection-modes-probe PROPERTIES CUDA_ARCHITECTURES 110)
  add_executable(lfm3-projection-bedrock-probe EXCLUDE_FROM_ALL probes/lfm3_projection_bedrock_probe.cu)
  target_compile_features(lfm3-projection-bedrock-probe PRIVATE cuda_std_17)
  target_link_libraries(lfm3-projection-bedrock-probe PRIVATE CUDA::cudart CUDA::cublas)
  set_target_properties(lfm3-projection-bedrock-probe PROPERTIES CUDA_ARCHITECTURES 110)
  add_executable(lfm3-lt-policy-probe EXCLUDE_FROM_ALL probes/lfm3_lt_policy_probe.cu)
  target_compile_features(lfm3-lt-policy-probe PRIVATE cuda_std_17)
  target_link_libraries(lfm3-lt-policy-probe PRIVATE CUDA::cudart CUDA::cublasLt)
  set_target_properties(lfm3-lt-policy-probe PROPERTIES CUDA_ARCHITECTURES 110)
  add_executable(lfm3-wmma-policy-probe EXCLUDE_FROM_ALL probes/lfm3_wmma_policy_probe.cu)
  target_compile_features(lfm3-wmma-policy-probe PRIVATE cuda_std_17)
  target_compile_options(lfm3-wmma-policy-probe PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false>)
  target_link_libraries(lfm3-wmma-policy-probe PRIVATE CUDA::cudart)
  set_target_properties(lfm3-wmma-policy-probe PROPERTIES CUDA_ARCHITECTURES 110)
  add_executable(lfm3-boundary-control EXCLUDE_FROM_ALL probes/lfm3_boundary_control.cu)
  target_compile_features(lfm3-boundary-control PRIVATE cuda_std_17)
  target_link_libraries(lfm3-boundary-control PRIVATE CUDA::cudart)
  set_target_properties(lfm3-boundary-control PROPERTIES CUDA_ARCHITECTURES 110)
  add_executable(lfm3-tma-store-probe EXCLUDE_FROM_ALL probes/lfm3_tma_store_probe.cu)
  target_compile_features(lfm3-tma-store-probe PRIVATE cuda_std_17)
  target_link_libraries(lfm3-tma-store-probe PRIVATE CUDA::cudart CUDA::cuda_driver)
  set_target_properties(lfm3-tma-store-probe PROPERTIES CUDA_ARCHITECTURES 110)
  set(LIQUID_SANITIZER_SDK "" CACHE PATH "Optional standalone Sanitizer SDK for diagnostic observer")
  if(LIQUID_SANITIZER_SDK)
    add_library(lfm3-boundary-observer SHARED EXCLUDE_FROM_ALL probes/lfm3_boundary_observer.cpp)
    target_compile_features(lfm3-boundary-observer PRIVATE cxx_std_17)
    target_include_directories(lfm3-boundary-observer PRIVATE ${CUDAToolkit_INCLUDE_DIRS} "${LIQUID_SANITIZER_SDK}/include")
    target_link_libraries(lfm3-boundary-observer PRIVATE "${LIQUID_SANITIZER_SDK}/libsanitizer-public.so")
    set_target_properties(lfm3-boundary-observer PROPERTIES BUILD_RPATH "${LIQUID_SANITIZER_SDK}")
  endif()
  add_library(lfm3_ops SHARED tests/lfm3_ops_bridge.cu)
  target_include_directories(lfm3_ops PRIVATE include)
  target_compile_features(lfm3_ops PRIVATE cuda_std_20)
  target_compile_options(lfm3_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
  set_target_properties(lfm3_ops PROPERTIES CUDA_ARCHITECTURES 110 CUDA_RUNTIME_LIBRARY Static)
endif()
