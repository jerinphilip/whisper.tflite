/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdio>
#include <string>

#include "whisper.tflite/whisper.h"

int main(int argc, char* argv[]) {
  if (argc != 4) {
    fprintf(stderr,
            "Usage: minimal <tflite model> <vocab file> <pcm_file name>\n");
    return 1;
  }

  // NOLINTNEXTLINE
  using namespace whisper;

  const char* model_prefix = argv[1];
  const char* vocab_path = argv[2];
  const char* pcmfilename = argv[3];
  bool multilingual = false;
  Monolith monolith(model_prefix, vocab_path, multilingual);

  std::string text = monolith.transcribe(pcmfilename);

  // Remove extra spaces between words
  text = remove_extra_spaces(text);

  printf("\n%s\n", text.c_str());
  printf("\n");

  return 0;
}
