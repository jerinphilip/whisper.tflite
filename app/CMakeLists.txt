if(NOT ANDROID)
  add_executable(my_audio_app main.cpp)
  target_link_libraries(my_audio_app PUBLIC whisper-tflite)
endif(NOT ANDROID)

set(TENSORFLOW_SOURCE_DIR
    ""
    CACHE PATH "Directory that contains the TensorFlow project")

if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
                         "${CMAKE_SOURCE_DIR}/deps/tensorflow" ABSOLUTE)
endif()

set(APPS minimal encdec)
foreach(APP ${APPS})
  add_executable(${APP} ${APP}.cpp)
  target_link_libraries(${APP} tensorflow-lite whisper-tflite)
  target_include_directories(
    ${APP} PUBLIC "${CMAKE_SOURCE_DIR}/deps" ${TFLITE_INCLUDE_DIRS}
                  "${CMAKE_SOURCE_DIR}/${APP}" "${CMAKE_SOURCE_DIR}/")
endforeach()
