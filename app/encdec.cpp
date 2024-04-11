
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

#include <cstdlib>
#include <iostream>
#include <string>

#include "CLI11/CLI11.hpp"
#include "whisper.tflite/whisper.h"

struct Options {
  std::string model_prefix;
  std::string decoder;
  std::string vocab;
  std::string input;

  template <class App>
  void setup(App& app) {
    app.add_option("--model-prefix", model_prefix, "Model prefix")->required();
    app.add_option("--vocab", vocab, "Path to vocabulary")->required();
    app.add_option("--input", input, "Path to prefix other filenames to")
        ->required();
  }
};

int main(int argc, char* argv[]) {
  CLI::App app{"slimt"};
  Options options;
  options.setup(app);

  try {
    app.parse(argc, argv);
    using namespace whisper;  // NOLINT
    bool multilingual = true;
    EncDec encdec(options.model_prefix, options.vocab, multilingual);
    std::string text = encdec.transcribe(options.input.c_str());
    std::cout << text << "\n";
  } catch (const CLI::ParseError& e) {
    exit(app.exit(e));
  }

  return 0;
}
