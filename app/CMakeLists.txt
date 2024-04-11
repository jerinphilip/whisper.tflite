set(TENSORFLOW_SOURCE_DIR
    ""
    CACHE PATH "Directory that contains the TensorFlow project")

if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
                         "${CMAKE_SOURCE_DIR}/deps/tensorflow" ABSOLUTE)
endif()

set(APPS minimal encdec vad silent_detection)
foreach(APP ${APPS})
  add_executable(${APP} ${APP}.cpp)
  target_link_libraries(${APP} whisper-tflite)
  target_include_directories(
    ${APP} PUBLIC ${WHISPER_TFLITE_INCLUDE_DIRS} "${CMAKE_SOURCE_DIR}/${APP}"
                  "${CMAKE_SOURCE_DIR}/")
  target_compile_options(${APP} PRIVATE ${WHISPER_TFLITE_COMPILE_OPTIONS})
  target_link_options(${APP} PRIVATE ${WHISPER_TFLITE_LINK_OPTIONS})
endforeach()
