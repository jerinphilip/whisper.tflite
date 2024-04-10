#include "TFLiteEngine.h"

#include <bits/types/struct_timeval.h>
#include <sys/time.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/interpreter_builder.h"
#include "core/model_builder.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/interpreter.h"
#include "wav_util.h"
#include "whisper.h"

#define TIME_DIFF_MS(start, end)                  \
  (((((end).tv_sec - (start).tv_sec) * 1000000) + \
    ((end).tv_usec - (start).tv_usec)) /          \
   1000)

namespace whisper {
int TFLiteEngine::create(const char *modelPath, const char *vocabPath,
                         const bool isMultilingual) {
  std::cout << "Entering " << __func__ << "()" << '\n';

  timeval start_time{};
  timeval end_time{};
  if (whisper_ == nullptr) {
    gettimeofday(&start_time, nullptr);
    std::cout << "Initializing TFLite..." << '\n';

    /////////////// Load filters and vocab data ///////////////
    FILE *vocab_fp = fopen(vocabPath, "rb");
    if (vocab_fp == nullptr) {
      fprintf(stderr, "Unable to open vocabulary file: %s", vocabPath);
      return -1;
    }

    int64_t vocab_size;
    vocab_file_ = std::move(MmapFile(vocabPath));
    const char *ptr = static_cast<const char *>(vocab_file_.data());
    std::memcpy(&vocab_size, ptr, sizeof(uint64_t));

    ptr = ptr + sizeof(uint64_t);
    Reader reader(ptr, isMultilingual);
    reader.read(filters_, vocab_);

    whisper_ = std::make_unique<Atom>(modelPath);
    gettimeofday(&end_time, nullptr);
    std::cout << "Time taken for TFLite initialization: "
              << TIME_DIFF_MS(start_time, end_time) << " ms" << '\n';
  }

  std::cout << "Exiting " << __func__ << "()" << '\n';
  return 0;
}

std::string TFLiteEngine::transcribe(std::vector<float> samples) {
  timeval start_time{};
  timeval end_time{};
  gettimeofday(&start_time, nullptr);

  // Hack if the audio file size is less than 30ms append with 0's
  samples.resize((kSampleRate * kChunkSize), 0);
  const auto processor_count = std::thread::hardware_concurrency();

  if (!log_mel_spectrogram(samples.data(), samples.size(), kSampleRate, kNFFT,
                           kHopLength, kNMEL, processor_count, filters_,
                           mel_)) {
    std::cerr << "Failed to compute mel_ spectrogram" << '\n';
    return "";
  }

  gettimeofday(&end_time, nullptr);
  std::cout << "Time taken for Spectrogram: "
            << TIME_DIFF_MS(start_time, end_time) << " ms" << '\n';

  auto *interpreter = whisper_->interpreter();
  auto *input = interpreter->typed_input_tensor<float>(0);
  memcpy(input, mel_.data.data(), mel_.n_mel * mel_.n_len * sizeof(float));
  gettimeofday(&start_time, nullptr);

  // Run inference
  interpreter->SetNumThreads(processor_count);
  if (interpreter->Invoke() != kTfLiteOk) {
    return "";
  }

  gettimeofday(&end_time, nullptr);
  std::cout << "Time taken for Interpreter: "
            << TIME_DIFF_MS(start_time, end_time) << " ms" << '\n';

  int output = interpreter->outputs()[0];
  TfLiteTensor *output_tensor = interpreter->tensor(output);
  TfLiteIntArray *output_dims = output_tensor->dims;
  // assume output dims to be something like (1, 1, ... ,size)
  auto output_size = output_dims->data[output_dims->size - 1];
  int *output_int = interpreter->typed_output_tensor<int>(0);
  bool omit_special_tokens = false;
  std::string text =
      decode(vocab_, output_int, output_int + output_size, omit_special_tokens);

  return text;
}

std::string TFLiteEngine::transcribe(const char *waveFile) {
  std::vector<float> pcmf32 = wav_read_legacy(waveFile);
  pcmf32.resize((kSampleRate * kChunkSize), 0);
  std::string text = transcribe(pcmf32);
  return text;
}

}  // namespace whisper
