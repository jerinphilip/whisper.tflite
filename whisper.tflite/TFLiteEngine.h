#ifndef _TFLITEENGINE_H_
#define _TFLITEENGINE_H_

#include <string>
#include <vector>

#include "whisper.h"

class TFLiteEngine {
 public:
  TFLiteEngine() = default;
  ~TFLiteEngine() = default;

  // NOLINTBEGIN(readability-identifier-naming)
  int loadModel(const char* modelPath, const char* vocabPath,
                bool isMultilingual);
  void freeModel();

  std::string transcribeBuffer(std::vector<float> samples);
  std::string transcribeFile(const char* waveFile);
  // NOLINTEND(readability-identifier-naming)

 private:
  // Convert a token to a string
  const char* decode(int token) { return vocab_.id_to_token.at(token).c_str(); }

  // Add any private members or helper functions as needed
  WhisperTFLite whisper_;
  WhisperVocab vocab_;
  WhisperFilters filters_;
  WhisperMel mel_;

  std::unique_ptr<char[]> vocab_holder_;
};

#endif  // _TFLITEENGINE_H_
