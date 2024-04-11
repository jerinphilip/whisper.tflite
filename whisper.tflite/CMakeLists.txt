# # Check if the build is Android

set(WHISPER_TFLITE_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
                                "${CMAKE_SOURCE_DIR}/deps")

set(SOURCES whisper.cpp wav_util.cpp mmap_file.cpp)

set(TENSORFLOWLITE_LIBS tensorflowlite tensorflowlite_flex)

add_library(whisper-tflite SHARED ${SOURCES})
target_link_libraries(whisper-tflite PUBLIC tflite)
target_include_directories(whisper-tflite PUBLIC ${WHISPER_TFLITE_INCLUDE_DIRS})
target_compile_options(whisper-tflite PRIVATE ${WHISPER_TFLITE_COMPILE_OPTIONS})
target_link_options(whisper-tflite PRIVATE ${WHISPER_TFLITE_LINK_OPTIONS})
