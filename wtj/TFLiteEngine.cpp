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
#include "filters_vocab_en.h"
#include "filters_vocab_multilingual.h"
#include "input_features.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/interpreter.h"
#include "wav_util.h"
#include "whisper.h"

enum { INFERENCE_ON_AUDIO_FILE = 1 };
#define TIME_DIFF_MS(start, end)                  \
  (((((end).tv_sec - (start).tv_sec) * 1000000) + \
    ((end).tv_usec - (start).tv_usec)) /          \
   1000)
#define TFLITE_MINIMAL_CHECK(x)                            \
  if (!(x)) {                                              \
    fprintf(stderr, "Error at %s:%d", __FILE__, __LINE__); \
    exit(1);                                               \
  }

int TFLiteEngine::loadModel(const char *modelPath, const bool isMultilingual) {
  std::cout << "Entering " << __func__ << "()" << '\n';

  timeval start_time{};
  timeval end_time{};
  if (!g_whisper_tflite.is_whisper_tflite_initialized) {
    gettimeofday(&start_time, nullptr);
    std::cout << "Initializing TFLite..." << '\n';

    /////////////// Load filters and vocab data ///////////////

    const char *vocab_data = nullptr;
    if (isMultilingual)
      vocab_data = reinterpret_cast<const char *>(filters_vocab_multilingual);
    else
      vocab_data = reinterpret_cast<const char *>(filters_vocab_en);

    // Read the magic number
    int magic = 0;
    std::memcpy(&magic, vocab_data, sizeof(magic));
    vocab_data += sizeof(magic);

    // Check the magic number
    if (magic != 0x57535052) {  // 'WSPR'
      std::cerr << "Invalid vocab data (bad magic)" << '\n';
      return -1;
    }

    // Load mel filters
    std::memcpy(&filters.n_mel, vocab_data, sizeof(filters.n_mel));
    vocab_data += sizeof(filters.n_mel);

    std::memcpy(&filters.n_fft, vocab_data, sizeof(filters.n_fft));
    vocab_data += sizeof(filters.n_fft);

    std::cout << "n_mel:" << filters.n_mel << " n_fft:" << filters.n_fft
              << '\n';

    filters.data.resize(filters.n_mel * filters.n_fft);
    std::memcpy(filters.data.data(), vocab_data,
                filters.data.size() * sizeof(float));
    vocab_data += filters.data.size() * sizeof(float);

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

      g_vocab.id_to_token[i] = word;
    }

    // add additional vocab ids
    int n_vocab_additional = 51864;
    if (isMultilingual) {
      n_vocab_additional = 51865;
      g_vocab.token_eot++;
      g_vocab.token_sot++;
      g_vocab.token_prev++;
      g_vocab.token_solm++;
      g_vocab.token_not++;
      g_vocab.token_beg++;
    }

    for (int i = n_vocab; i < n_vocab_additional; i++) {
      std::string word;
      if (i > g_vocab.token_beg) {
        word = "[_TT_" + std::to_string(i - g_vocab.token_beg) + "]";
      } else if (i == g_vocab.token_eot) {
        word = "[_EOT_]";
      } else if (i == g_vocab.token_sot) {
        word = "[_SOT_]";
      } else if (i == g_vocab.token_prev) {
        word = "[_PREV_]";
      } else if (i == g_vocab.token_not) {
        word = "[_NOT_]";
      } else if (i == g_vocab.token_beg) {
        word = "[_BEG_]";
      } else {
        word = "[_extra_token_" + std::to_string(i) + "]";
      }
      g_vocab.id_to_token[i] = word;
      // printf("%s: g_vocab[%d] = '%s'", __func__, i, word.c_str());
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

    g_whisper_tflite.size = size;
    g_whisper_tflite.buffer = buffer;

    g_whisper_tflite.model = tflite::FlatBufferModel::BuildFromBuffer(
        g_whisper_tflite.buffer, g_whisper_tflite.size);
    TFLITE_MINIMAL_CHECK(g_whisper_tflite.model != nullptr);

    // Build the interpreter with the InterpreterBuilder.
    tflite::InterpreterBuilder builder(*(g_whisper_tflite.model),
                                       g_whisper_tflite.resolver);

    builder(&(g_whisper_tflite.interpreter));
    TFLITE_MINIMAL_CHECK(g_whisper_tflite.interpreter != nullptr);

    // Allocate tensor buffers.
    TFLITE_MINIMAL_CHECK(g_whisper_tflite.interpreter->AllocateTensors() ==
                         kTfLiteOk);

    g_whisper_tflite.input =
        g_whisper_tflite.interpreter->typed_input_tensor<float>(0);
    g_whisper_tflite.is_whisper_tflite_initialized = true;

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
  samples.resize((WHISPER_SAMPLE_RATE * WHISPER_CHUNK_SIZE), 0);
  const auto processor_count = std::thread::hardware_concurrency();

  if (!log_mel_spectrogram(samples.data(), samples.size(), WHISPER_SAMPLE_RATE,
                           WHISPER_N_FFT, WHISPER_HOP_LENGTH, WHISPER_N_MEL,
                           processor_count, filters, mel)) {
    std::cerr << "Failed to compute mel spectrogram" << '\n';
    return "";
  }

  gettimeofday(&end_time, nullptr);
  std::cout << "Time taken for Spectrogram: "
            << TIME_DIFF_MS(start_time, end_time) << " ms" << '\n';

  if (INFERENCE_ON_AUDIO_FILE) {
    memcpy(g_whisper_tflite.input, mel.data.data(),
           mel.n_mel * mel.n_len * sizeof(float));
  } else {
    memcpy(g_whisper_tflite.input, _content_input_features_bin,
           WHISPER_N_MEL * WHISPER_MEL_LEN *
               sizeof(float));  // to load pre-generated input_features
  }                             // end of audio file processing

  gettimeofday(&start_time, nullptr);

  // Run inference
  g_whisper_tflite.interpreter->SetNumThreads(processor_count);
  if (g_whisper_tflite.interpreter->Invoke() != kTfLiteOk) {
    return "";
  }

  gettimeofday(&end_time, nullptr);
  std::cout << "Time taken for Interpreter: "
            << TIME_DIFF_MS(start_time, end_time) << " ms" << '\n';

  int output = g_whisper_tflite.interpreter->outputs()[0];
  TfLiteTensor *output_tensor = g_whisper_tflite.interpreter->tensor(output);
  TfLiteIntArray *output_dims = output_tensor->dims;
  // assume output dims to be something like (1, 1, ... ,size)
  auto output_size = output_dims->data[output_dims->size - 1];

  int *output_int = g_whisper_tflite.interpreter->typed_output_tensor<int>(0);
  std::string text;

  for (int i = 0; i < output_size; i++) {
    if (output_int[i] == g_vocab.token_eot) {
      break;
    }

    if (output_int[i] < g_vocab.token_eot) {
      text += whisper_token_to_str(output_int[i]);
    }
  }

  return text;
}

std::string TFLiteEngine::transcribeFile(const char *waveFile) {
  std::vector<float> pcmf32 = readWAVFile(waveFile);
  pcmf32.resize((WHISPER_SAMPLE_RATE * WHISPER_CHUNK_SIZE), 0);
  std::string text = transcribeBuffer(pcmf32);
  return text;
}

void TFLiteEngine::freeModel() {
  std::cout << "Entering " << __func__ << "()" << '\n';

  if (g_whisper_tflite.buffer) {
    std::cout << __func__ << ": free buffer " << g_whisper_tflite.buffer
              << " memory" << '\n';
    delete[] g_whisper_tflite.buffer;
  }

  std::cout << "Exiting " << __func__ << "()" << '\n';
}
