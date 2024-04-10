# Define your project and its version
cmake_minimum_required(VERSION 3.0)
project(TFLiteEngine VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)

if(WITH_ASAN)
  list(APPEND WHISPER_TFLITE_COMPILE_OPTIONS -fsanitize=address
       -fsanitize=undefined)
  list(APPEND WHISPER_TFLITE_LINK_OPTIONS -fsanitize=address
       -fsanitize=undefined)
endif(WITH_ASAN)

add_subdirectory(deps)
add_subdirectory(whisper.tflite)
add_subdirectory(app)
add_subdirectory(bindings/java)
