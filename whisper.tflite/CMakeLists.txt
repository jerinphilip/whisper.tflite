# # Check if the build is Android

set(SOURCES TFLiteEngine.cpp whisper.cpp wav_util.cpp)
set(EXT_LIBS
    tensorflow-lite
    # The following is not pleasant, but gets to build. tensorflowlite_flex can
    # only be built via bazel, then added the `.so` manually picked up and
    # linked to the rest of the libs.
    # TODO(@jerinphilip): Find a cleaner integration
    "/home/jerin/.cache/bazel/_bazel_jerin/a618ac3dc748926a1e43bc0dce8515ed/execroot/org_tensorflow/bazel-out/k8-opt/bin/tensorflow/lite/delegates/flex/libtensorflowlite_flex.so"
)
set(WHISPER_TFLITE_INCLUDE_DIRS
    "${CMAKE_SOURCE_DIR}/deps/tensorflow/"
    "${CMAKE_SOURCE_DIR}/deps/tensorflow/tensorflow/lite"
    "${CMAKE_SOURCE_DIR}/deps/flatbuffers/include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/deps")

message(STATUS "TFLITE_INCLUDE_DIRS: ${TFLITE_INCLUDE_DIRS}")

if(ANDROID)
  add_library(whisper-tflite SHARED ${SOURCES})
  target_link_libraries(whisper-tflite PUBLIC ${EXT_LIBS})
  target_include_directories(whisper-tflite
                             PUBLIC ${WHISPER_TFLITE_INCLUDE_DIRS})
  target_compile_options(whisper-tflite
                         PUBLIC ${WHISPER_TFLITE_COMPILE_OPTIONS})
  target_link_options(whisper-tflite PRIVATE ${WHISPER_TFLITE_LINK_OPTIONS})
endif()

if(NOT ANDROID)
  add_library(whisper-tflite SHARED ${SOURCES})
  target_link_libraries(whisper-tflite PUBLIC ${EXT_LIBS})
  target_include_directories(whisper-tflite
                             PUBLIC ${WHISPER_TFLITE_INCLUDE_DIRS})
  target_compile_options(whisper-tflite
                         PRIVATE ${WHISPER_TFLITE_COMPILE_OPTIONS})
  target_link_options(whisper-tflite PRIVATE ${WHISPER_TFLITE_LINK_OPTIONS})
endif()
