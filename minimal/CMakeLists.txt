cmake_minimum_required(VERSION 3.16)
project(minimal C CXX)

set(TENSORFLOW_SOURCE_DIR
    ""
    CACHE PATH "Directory that contains the TensorFlow project")

if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
                         "${CMAKE_SOURCE_DIR}/deps/tensorflow" ABSOLUTE)
endif()

set(CMAKE_CXX_STANDARD 17)
add_executable(minimal minimal.cc)
target_link_libraries(minimal tensorflow-lite)
