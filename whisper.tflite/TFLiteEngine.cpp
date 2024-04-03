#include "TFLiteEngine.h"

#include <bits/types/struct_timeval.h>
#include <sys/time.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "core/interpreter_builder.h"
#include "core/model_builder.h"
#include "tensorflow/lite/core/interpreter.h"
#include "wav_util.h"
#include "whisper.h"

#define TIME_DIFF_MS(start, end)                  \
  (((((end).tv_sec - (start).tv_sec) * 1000000) + \
    ((end).tv_usec - (start).tv_usec)) /          \
   1000)
#define TFLITE_MINIMAL_CHECK(x)                            \
  if (!(x)) {                                              \
    fprintf(stderr, "Error at %s:%d", __FILE__, __LINE__); \
    exit(1);                                               \
  }

int TFLiteEngine::loadModel(const char *modelPath, const char *vocabPath,
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

    uint64_t vocab_size = 0;
    fread(&vocab_size, sizeof(uint64_t), 1, vocab_fp);
    vocab_holder_ = std::make_unique<char[]>(vocab_size);
    fread(vocab_holder_.get(), vocab_size, 1, vocab_fp);

    const char *vocab_data =
        reinterpret_cast<const char *>(vocab_holder_.get());

    // Read the magic number
    int magic = 0;
    std::memcpy(&magic, vocab_data, sizeof(magic));
    vocab_data += sizeof(magic);

    // Check the magic number
    constexpr int kVocabMagic = 0x57535052;
    if (magic != kVocabMagic) {  // 'WSPR'
      std::cerr << "Invalid vocab data (bad magic)" << '\n';
      return -1;
    }

    // Load mel_ filters_
    std::memcpy(&filters_.n_mel, vocab_data, sizeof(filters_.n_mel));
    vocab_data += sizeof(filters_.n_mel);

    std::memcpy(&filters_.n_fft, vocab_data, sizeof(filters_.n_fft));
    vocab_data += sizeof(filters_.n_fft);

    std::cout << "n_mel:" << filters_.n_mel << " n_fft:" << filters_.n_fft
              << '\n';

    filters_.data.resize(filters_.n_mel * filters_.n_fft);
    std::memcpy(filters_.data.data(), vocab_data,
                filters_.data.size() * sizeof(float));
    vocab_data += filters_.data.size() * sizeof(float);

    // Load vocab
    int n_vocab = 0;
    std::memcpy(&n_vocab, vocab_data, sizeof(n_vocab));
    vocab_data += sizeof(n_vocab);

    std::cout << "n_vocab:" << n_vocab << '\n';

    for (int i = 0; i < n_vocab; i++) {
      int len = 0;
      std::memcpy(&len, vocab_data, sizeof(len));
      vocab_data += sizeof(len);

      std::string word(vocab_data, len);
      vocab_data += len;

      vocab_.id_to_token[i] = word;
    }

    // add additional vocab ids
    int n_vocab_additional = 51864;
    if (isMultilingual) {
      n_vocab_additional = 51865;
      vocab_.token_eot++;
      vocab_.token_sot++;
      vocab_.token_prev++;
      vocab_.token_solm++;
      vocab_.token_not++;
      vocab_.token_beg++;
    }

    for (int i = n_vocab; i < n_vocab_additional; i++) {
      std::string word;
      if (i > vocab_.token_beg) {
        word = "[_TT_" + std::to_string(i - vocab_.token_beg) + "]";
      } else if (i == vocab_.token_eot) {
        word = "[_EOT_]";
      } else if (i == vocab_.token_sot) {
        word = "[_SOT_]";
      } else if (i == vocab_.token_prev) {
        word = "[_PREV_]";
      } else if (i == vocab_.token_not) {
        word = "[_NOT_]";
      } else if (i == vocab_.token_beg) {
        word = "[_BEG_]";
      } else {
        word = "[_extra_token_" + std::to_string(i) + "]";
      }
      vocab_.id_to_token[i] = word;
      // printf("%s: vocab_[%d] = '%s'", __func__, i, word.c_str());
    }

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

std::string TFLiteEngine::transcribeBuffer(std::vector<float> samples) {
  timeval start_time{};
  timeval end_time{};
  gettimeofday(&start_time, nullptr);

  // Hack if the audio file size is less than 30ms append with 0's
  samples.resize((kWhisperSampleRate * kWhisperChunkSize), 0);
  const auto processor_count = std::thread::hardware_concurrency();

  if (!log_mel_spectrogram(samples.data(), samples.size(), kWhisperSampleRate,
                           kWhisperNFFT, kWhisperHopLength, kWhisperNMEL,
                           processor_count, filters_, mel_)) {
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
  std::string text;

  for (int i = 0; i < output_size; i++) {
    if (output_int[i] == vocab_.token_eot) {
      break;
    }

    if (output_int[i] < vocab_.token_eot) {
      text += decode(output_int[i]);
    }
  }

  return text;
}

std::string TFLiteEngine::transcribeFile(const char *waveFile) {
  std::vector<float> pcmf32 = readWAVFile(waveFile);
  pcmf32.resize((kWhisperSampleRate * kWhisperChunkSize), 0);
  std::string text = transcribeBuffer(pcmf32);
  return text;
}

void TFLiteEngine::freeModel() {
  std::cout << "Entering " << __func__ << "()" << '\n';

  if (whisper_.buffer) {
    std::cout << __func__ << ": free buffer " << whisper_.buffer << " memory"
              << '\n';
    delete[] whisper_.buffer;
  }

  std::cout << "Exiting " << __func__ << "()" << '\n';
}
