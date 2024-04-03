# # Check if the build is Android

find_package(JNI REQUIRED)

set(SOURCES TFLiteEngine.cpp TFLiteEngineJNI.cpp whisper.cpp)
set(EXT_LIBS tensorflow-lite JNI::JNI)
set(TFLITE_INCLUDE_DIRS
    "${CMAKE_SOURCE_DIR}/deps/tensorflow/"
    "${CMAKE_SOURCE_DIR}/deps/tensorflow/tensorflow/lite"
    "${CMAKE_SOURCE_DIR}/deps/flatbuffers/include")

message(STATUS "TFLITE_INCLUDE_DIRS: ${TFLITE_INCLUDE_DIRS}")

if(ANDROID)
  add_library(audioEngine SHARED ${SOURCES})
  target_link_libraries(audioEngine PUBLIC ${EXT_LIBS})
  target_include_directories(audioEngine PUBLIC ${TFLITE_INCLUDE_DIRS})
endif()

if(NOT ANDROID)
  add_library(audioEngine SHARED ${SOURCES})
  target_link_libraries(audioEngine PUBLIC ${EXT_LIBS})
  target_include_directories(audioEngine PUBLIC ${TFLITE_INCLUDE_DIRS})

  add_executable(my_audio_app main.cpp)
  target_link_libraries(my_audio_app PUBLIC audioEngine)
endif()
