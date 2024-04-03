#ifndef _TFLITEENGINE_H_
#define _TFLITEENGINE_H_

#include <string>
#include <vector>

#include "whisper.h"

class TFLiteEngine {
 public:
  TFLiteEngine() = default;
  ~TFLiteEngine() = default;

  int loadModel(const char* modelPath, const bool isMultilingual);
  void freeModel();

  std::string transcribeBuffer(std::vector<float> samples);
  std::string transcribeFile(const char* waveFile);

 private:
  // Convert a token to a string
  const char* decode(int token) { return vocab_.id_to_token.at(token).c_str(); }

  // Add any private members or helper functions as needed
  whisper_tflite whisper_;
  whisper_vocab vocab_;
  whisper_filters filters_;
  whisper_mel mel_;
};

#endif  // _TFLITEENGINE_H_
