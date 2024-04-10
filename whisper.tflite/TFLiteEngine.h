#ifndef _TFLITEENGINE_H_
#define _TFLITEENGINE_H_

#include <string>
#include <vector>

#include "whisper.h"

namespace whisper {

class TFLiteEngine {
 public:
  TFLiteEngine() = default;
  ~TFLiteEngine() = default;

  // NOLINTBEGIN(readability-identifier-naming)
  int create(const char* modelPath, const char* vocabPath, bool isMultilingual);
  void destroy();

  std::string transcribe(std::vector<float> samples);
  std::string transcribe(const char* waveFile);
  // NOLINTEND(readability-identifier-naming)

 private:
  // Convert a token to a string
  const char* decode(int token) { return vocab_.id_to_token.at(token).c_str(); }

  // Add any private members or helper functions as needed
  TFLite whisper_;
  Vocab vocab_;
  Filters filters_;
  Mel mel_;

  MmapFile vocab_file_;
};
}  // namespace whisper

#endif  // _TFLITEENGINE_H_
