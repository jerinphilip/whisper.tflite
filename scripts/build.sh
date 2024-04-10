#!/bin/bash

CMAKE_ARGS=(
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  # -DWITH_ASAN=ON
  # -DCMAKE_BUILD_TYPE=Debug
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
  -DCMAKE_C_COMPILER_LAUNCHER=ccache
)

cmake -B build -S $PWD "${CMAKE_ARGS[@]}"
cmake --build build --target all --parallel 31
