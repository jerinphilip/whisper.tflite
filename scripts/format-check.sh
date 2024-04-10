#!/bin/bash

set -eo pipefail
set -x

function whisper-tflitecheck-clang-format {
  # clang-format
  python3 run-clang-format.py --style file -r app whisper.tflite
}

function whisper-tflitecheck-clang-tidy {
  # clang-tidy
  mkdir -p build
  ARGS=(
    -DCMAKE_EXPORT_COMPILE_COMMANDS=on
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
  )

  cmake -B build -S . "${ARGS[@]}"
  set +e
  FILES=$(find app whisper.tflite -type f)
  run-clang-tidy -export-fixes build/clang-tidy.whisper.tflite.yml -fix -format -p build -header-filter="$PWD/whisper.tflite" ${FILES[@]}
  CHECK_STATUS=$?
  git diff
  set -e
  return $CHECK_STATUS

}

function whisper-tflitecheck-python {
  python3 -m black --diff --check scripts/
  python3 -m isort --profile black --diff --check scripts/
}

function whisper-tflitecheck-sh {
  shfmt -i 2 -ci -bn -sr -d scripts/
}

function whisper-tflitecheck-cmake {
  set +e
  CMAKE_FILES=$(find -name "CMakeLists.txt" -not -path "./3rd-party/*" -not -path "build")
  cmake-format ${CMAKE_FILES[@]} --check
  CHECK_STATUS=$?
  set -e
  cmake-format ${CMAKE_FILES[@]} --in-place
  git diff
  return $CHECK_STATUS
}

function whisper-tflitecheck-iwyu {
  iwyu-tool -p build whisper.tflite/* > build/iwyu.out
}

whisper-tflitecheck-clang-format
whisper-tflitecheck-python
whisper-tflitecheck-sh
# whisper-tflitecheck-cmake
whisper-tflitecheck-clang-tidy
