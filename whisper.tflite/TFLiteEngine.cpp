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
  if (!whisper_.is_whisper_tflite_initialized) {
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
    // add additional vocab ids
    int n_vocab_expected = kVocabEnSize;
    transform_vocab_multilingual(vocab_);

    /////////////// Load tflite model buffer ///////////////

    // Open the TFLite model file for reading
    std::ifstream model_file(modelPath, std::ios::binary | std::ios::ate);
    if (!model_file.is_open()) {
      std::cerr << "Unable to open model file: " << modelPath << '\n';
      return -1;
    }

    // Get the size of the model file
    std::streamsize size = model_file.tellg();
    model_file.seekg(0, std::ios::beg);

    // Allocate memory for the model buffer
    char *buffer = new char[size];

    // Read the model data into the buffer
    if (model_file.read(buffer, size)) {
      model_file.close();
    } else {
      std::cerr << "Error reading model data from file." << '\n';
    }

    whisper_.size = size;
    whisper_.buffer = buffer;

    whisper_.model = tflite::FlatBufferModel::BuildFromBuffer(whisper_.buffer,
                                                              whisper_.size);
    TFLITE_MINIMAL_CHECK(whisper_.model != nullptr);

    // Build the interpreter with the InterpreterBuilder.
    tflite::InterpreterBuilder builder(*(whisper_.model), whisper_.resolver);

    builder(&(whisper_.interpreter));
    TFLITE_MINIMAL_CHECK(whisper_.interpreter != nullptr);

    // Allocate tensor buffers.
    TFLITE_MINIMAL_CHECK(whisper_.interpreter->AllocateTensors() == kTfLiteOk);

    whisper_.input = whisper_.interpreter->typed_input_tensor<float>(0);
    whisper_.is_whisper_tflite_initialized = true;

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

  memcpy(whisper_.input, mel_.data.data(),
         mel_.n_mel * mel_.n_len * sizeof(float));
  gettimeofday(&start_time, nullptr);

  // Run inference
  whisper_.interpreter->SetNumThreads(processor_count);
  if (whisper_.interpreter->Invoke() != kTfLiteOk) {
    return "";
  }

  gettimeofday(&end_time, nullptr);
  std::cout << "Time taken for Interpreter: "
            << TIME_DIFF_MS(start_time, end_time) << " ms" << '\n';

  int output = whisper_.interpreter->outputs()[0];
  TfLiteTensor *output_tensor = whisper_.interpreter->tensor(output);
  TfLiteIntArray *output_dims = output_tensor->dims;
  // assume output dims to be something like (1, 1, ... ,size)
  auto output_size = output_dims->data[output_dims->size - 1];
  int *output_int = whisper_.interpreter->typed_output_tensor<int>(0);
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

void TFLiteEngine::destroy() const {
  std::cout << "Entering " << __func__ << "()" << '\n';

  if (whisper_.buffer) {
    std::cout << __func__ << ": free buffer " << whisper_.buffer << " memory"
              << '\n';
    delete[] whisper_.buffer;
  }

  std::cout << "Exiting " << __func__ << "()" << '\n';
}
}  // namespace whisper
