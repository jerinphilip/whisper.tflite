if(NOT ANDROID)
  add_executable(my_audio_app main.cpp)
  target_link_libraries(my_audio_app PUBLIC audioEngine)
endif(NOT ANDROID)

set(TENSORFLOW_SOURCE_DIR
    ""
    CACHE PATH "Directory that contains the TensorFlow project")

if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
                         "${CMAKE_SOURCE_DIR}/deps/tensorflow" ABSOLUTE)
endif()

set(CMAKE_CXX_STANDARD 17)
add_executable(minimal minimal.cc)
target_link_libraries(minimal tensorflow-lite audioEngine)
target_include_directories(
  minimal PUBLIC "${CMAKE_SOURCE_DIR}/deps" ${TFLITE_INCLUDE_DIRS}
                 "${CMAKE_SOURCE_DIR}/minimal" "${CMAKE_SOURCE_DIR}/")
