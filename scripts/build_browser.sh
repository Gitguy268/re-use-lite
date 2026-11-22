#!/bin/bash
# Browser (WASM32 + WebGPU) build of patched audio.cpp with only the `reuse` family.
# Flags follow audio.cpp docs/build/webgpu.md (Release instead of Debug).
set -euo pipefail
TOOLS="${TOOLS:-$HOME/re-use-lite-tools}"
source "$TOOLS/ext/emsdk/emsdk_env.sh" >/dev/null
cd "$TOOLS/ext/audio.cpp"
emcmake cmake -S . -B build/browser \
  -DCMAKE_BUILD_TYPE=Release \
  -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=reuse \
  -DAUDIOCPP_DEPLOYMENT_BUILD=ON \
  -DENGINE_ENABLE_WEBGPU=ON -DGGML_WEBGPU_JSPI=ON \
  -DENGINE_ENABLE_CUDA=OFF -DENGINE_ENABLE_HIP=OFF \
  -DENGINE_ENABLE_VULKAN=OFF -DENGINE_ENABLE_METAL=OFF \
  -DENGINE_ENABLE_OPENMP=OFF -DGGML_OPENMP=OFF \
  -DENGINE_ENABLE_NATIVE_CPU=OFF -DENGINE_ENABLE_LLAMAFILE=OFF \
  -DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=OFF \
  -DAUDIOCPP_BUILD_SERVER_FRONTENDS=OFF -DAUDIOCPP_BUILD_C_API=OFF \
  -DENGINE_BUILD_TESTS=OFF \
  '-DCMAKE_CXX_FLAGS_RELEASE=-O3' '-DCMAKE_C_FLAGS_RELEASE=-O3' \
  '-DCMAKE_EXE_LINKER_FLAGS=-pthread -sPTHREAD_POOL_SIZE=navigator.hardwareConcurrency' \
  '-DCMAKE_CXX_FLAGS=-fwasm-exceptions -msimd128 -pthread' \
  '-DCMAKE_C_FLAGS=-msimd128 -pthread'
cmake --build build/browser --target audiocpp_cli -j "${JOBS:-6}"
ls -la build/browser/bin
