#pragma once

#include <optional>
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
  void destroy() const;

  std::string transcribe(std::vector<float> samples);
  std::string transcribe(const char* waveFile);
  // NOLINTEND(readability-identifier-naming)

 private:
  // Add any private members or helper functions as needed
  std::unique_ptr<Atom> whisper_;
  Vocab vocab_;
  Filters filters_;
  Mel mel_;

  MmapFile vocab_file_;
};
}  // namespace whisper
