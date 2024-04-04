# # Check if the build is Android

set(SOURCES TFLiteEngine.cpp whisper.cpp wav_util.cpp)
set(EXT_LIBS
    tensorflow-lite
    "/home/jerin/.cache/bazel/_bazel_jerin/a618ac3dc748926a1e43bc0dce8515ed/execroot/org_tensorflow/bazel-out/k8-opt/bin/tensorflow/lite/delegates/flex/libtensorflowlite_flex.so"
)
set(TFLITE_INCLUDE_DIRS
    "${CMAKE_SOURCE_DIR}/deps/tensorflow/"
    "${CMAKE_SOURCE_DIR}/deps/tensorflow/tensorflow/lite"
    "${CMAKE_SOURCE_DIR}/deps/flatbuffers/include"
    "${CMAKE_CURRENT_SOURCE_DIR}")

message(STATUS "TFLITE_INCLUDE_DIRS: ${TFLITE_INCLUDE_DIRS}")

if(ANDROID)
  add_library(whisper-tflite SHARED ${SOURCES})
  target_link_libraries(whisper-tflite PUBLIC ${EXT_LIBS})
  target_include_directories(whisper-tflite PUBLIC ${TFLITE_INCLUDE_DIRS})
endif()

if(NOT ANDROID)
  add_library(whisper-tflite SHARED ${SOURCES})
  target_link_libraries(whisper-tflite PUBLIC ${EXT_LIBS})
  target_include_directories(whisper-tflite PUBLIC ${TFLITE_INCLUDE_DIRS})
endif()
