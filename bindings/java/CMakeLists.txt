find_package(JNI REQUIRED)

add_library(whisper-tflite-jni SHARED whisper.tflite.cpp)
target_link_libraries(whisper-tflite-jni PUBLIC whisper-tflite JNI::JNI)
