
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

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "CLI11/CLI11.hpp"
#define DR_WAV_IMPLEMENTATION
#include "dr_libs/dr_wav.h"
#include "whisper.tflite/wav_util.h"
#include "whisper.tflite/whisper.h"

struct Options {
  std::string encoder;
  std::string decoder;
  std::string vocab;
  std::string input;

  template <class App>
  void setup(App& app) {
    app.add_option("--encoder", encoder, "Path encoder")->required();
    app.add_option("--decoder", decoder, "Path to decoder")->required();
    app.add_option("--vocab", vocab, "Path to vocabulary")->required();
    app.add_option("--input", input, "Path to prefix other filenames to")
        ->required();
  }
};

int run(const Options& options) {
  // NOLINTNEXTLINE
  using namespace whisper;

  uint64_t vocab_size;
  FILE* vocab_fp = fopen(options.vocab.c_str(), "rb");
  fread(&vocab_size, sizeof(uint64_t), 1, vocab_fp);
  auto vocab_holder = std::make_unique<char[]>(vocab_size);
  fread(vocab_holder.get(), vocab_size, 1, vocab_fp);
  fclose(vocab_fp);

  // Create a pointer to the start of the unsigned char array
  char* ptr = vocab_holder.get();
  bool multilingual = true;
  Reader reader(ptr, multilingual);
  Vocab vocab;
  Filters filters;
  reader.read(filters, vocab);

  // Generate input_features for Audio file
  const char* pcmfilename = options.input.c_str();
  std::vector<float> pcmf32 = wav_read(pcmfilename);

  // Hack if the audio file size is less than 30ms, append with 0's
  pcmf32.resize((kSampleRate * kChunkSize), 0);

  Mel mel;
  if (!log_mel_spectrogram(pcmf32.data(), pcmf32.size(), kSampleRate, kNFFT,
                           kHopLength, kNMEL, 1, filters, mel)) {
    fprintf(stderr, "%s: failed to compute mel spectrogram\n", __func__);
    return -1;
  }

  printf("Mel{ n_len: %d, n_mel = %d}\n", mel.n_len, mel.n_mel);

  // Allocate tensor buffers.
  //
  Encoder encoder(options.encoder);
  auto encoder_out = encoder.forward(mel);

  Decoder decoder(options.decoder, vocab);
  std::vector<int64_t> generated = decoder.forward(encoder_out);
  bool omit_special_tokens = false;
  std::string surface = decode(vocab, generated, omit_special_tokens);
  fprintf(stderr, "\n");
  fprintf(stderr, "surface: [%s]\n", surface.c_str());

  return 0;
}

int main(int argc, char* argv[]) {
  CLI::App app{"slimt"};
  Options options;
  options.setup(app);

  try {
    app.parse(argc, argv);
    return run(options);
  } catch (const CLI::ParseError& e) {
    exit(app.exit(e));
  }

  return 0;
}
