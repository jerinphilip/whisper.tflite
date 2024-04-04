#include <iostream>
#include <string>

#include "TFLiteEngine.h"

int main() {
  whisper::TFLiteEngine engine;
  bool is_multilingual = false;

  // Load the TFLite model and vocabulary
  const char* model_path = "../../assets/whisper-tiny-en.tflite";
  const char* vocab_path = nullptr;
  if (is_multilingual) {
    model_path = "../../assets/whisper-tiny.tflite";
  }

  int result = engine.loadModel(model_path, nullptr, is_multilingual);
  if (result != 0) {
    std::cerr << "Error loading the TFLite model or vocabulary." << '\n';
    return 1;
  }

  // Transcribe an audio file
  const char* audio_file_path = "../../assets/jfk.wav";
  // audioFilePath = "../resources/MicInput.wav";
  audio_file_path = "../english_test_3_bili.wav";
  std::string transcription = engine.transcribeFile(audio_file_path);
  if (!transcription.empty()) {
    std::cout << "Transcription: " << transcription << '\n';
  } else {
    std::cerr << "Error transcribing the audio file." << '\n';
    return 2;
  }

  return 0;
}
