#!/bin/bash

CMAKE_ARGS=(
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  # -DWITH_ASAN=ON
  # -DCMAKE_BUILD_TYPE=Debug
  -DCMAKE_BUILD_TYPE=Release
)

cmake -B build -S $PWD "${CMAKE_ARGS[@]}"
cmake --build build --target all --parallel 31
