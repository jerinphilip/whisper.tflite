set(TENSORFLOWLITE_SUFFIX "x86_64")
set(TENSORFLOWLITE_LIBS tensorflowlite tensorflowlite_flex)
set(TENSORFLOWLITE_INCLUDE_DIRECTORIES
    "${CMAKE_CURRENT_SOURCE_DIR}/tensorflow/"
    "${CMAKE_CURRENT_SOURCE_DIR}/tensorflow/tensorflow/lite"
    "${CMAKE_CURRENT_SOURCE_DIR}/flatbuffers/include")

if(ANDROID)
  set(TENSORFLOWLITE_SUFFIX "android/${ANDROID_ABI}")
endif(ANDROID)

foreach(TFLIB ${TENSORFLOWLITE_LIBS})
  add_library(${TFLIB} SHARED IMPORTED)
  set_target_properties(
    ${TFLIB}
    PROPERTIES
      IMPORTED_LOCATION
      "${CMAKE_CURRENT_SOURCE_DIR}/prebuilt/${TENSORFLOWLITE_SUFFIX}/lib${TFLIB}.so"
  )
endforeach()

add_library(tflite INTERFACE)
target_link_libraries(tflite INTERFACE ${TENSORFLOWLITE_LIBS})
target_include_directories(tflite
                           INTERFACE ${TENSORFLOWLITE_INCLUDE_DIRECTORIES})
