# # Check if the build is Android

set(TFLITE_INCLUDE_DIRS
    "${CMAKE_SOURCE_DIR}/deps/tensorflow/"
    "${CMAKE_SOURCE_DIR}/deps/tensorflow/tensorflow/lite"
    "${CMAKE_SOURCE_DIR}/deps/flatbuffers/include")

message(STATUS "TFLITE_INCLUDE_DIRS: ${TFLITE_INCLUDE_DIRS}")

if(ANDROID)
  add_library(audioEngine SHARED TFLiteEngine.cpp TFLiteEngineJNI.cpp)
  target_link_libraries(audioEngine PUBLIC tensorflow-lite)
  target_include_directories(audioEngine PUBLIC ${TFLITE_INCLUDE_DIRS})
endif()

if(NOT ANDROID)
  add_library(audioEngine SHARED TFLiteEngine.cpp)
  add_executable(my_audio_app main.cpp)
  target_link_libraries(my_audio_app PUBLIC audioEngine tensorflow-lite)
  target_include_directories(audioEngine PUBLIC ${TFLITE_INCLUDE_DIRS})
endif()
