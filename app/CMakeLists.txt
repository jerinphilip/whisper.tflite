if(NOT ANDROID)
  add_executable(my_audio_app main.cpp)
  target_link_libraries(my_audio_app PUBLIC audioEngine)
endif(NOT ANDROID)
