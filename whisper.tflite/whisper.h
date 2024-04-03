#ifndef _WHISPER_H_
#define _WHISPER_H_

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

// Define constants
#define WHISPER_SAMPLE_RATE 16000
#define WHISPER_N_FFT 400
#define WHISPER_N_MEL 80
#define WHISPER_HOP_LENGTH 160
#define WHISPER_CHUNK_SIZE 30
#define WHISPER_MEL_LEN 3000

// Constants
static constexpr int kNumGoldenGeneratedIDs = 21;
static constexpr int kGoldenGeneratedIDs[kNumGoldenGeneratedIDs] = {
    50257, 50362, 1770, 13, 2264, 346, 353, 318,  262, 46329, 286,
    262,   3504,  6097, 11, 290,  356, 389, 9675, 284, 7062};
static constexpr int kWhisperSampleRate = 16000;
static constexpr int kWhisperNFFT = 400;
static constexpr int kWhisperNMEL = 80;
static constexpr int kWhisperHopLength = 160;
static constexpr int kWhisperChunkSize = 30;
static constexpr int kWhisperMelLen = 3000;

struct WhisperVocab {
  std::map<int, std::string> id_to_token;

  // NOLINTBEGIN(readability-magic-numbers)
  // clang-format off
  int n_vocab_additional  = 51864;

  int token_eot           = 50256;
  int token_sot           = 50257;
  int token_prev          = 50360;
  int token_solm          = 50361;   // ??
  int token_not           = 50362;   // no timestamps
  int token_beg           = 50363;

  int token_translwordate = 50358;
  int token_transcribe    = 50359;
  // clang-format on
  // NOLINTEND(readability-magic-numbers)
};

struct WhisperTFLite {
  char* buffer = nullptr;
  int64_t size = 0;
  std::unique_ptr<tflite::FlatBufferModel> model;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  float* input;

  bool is_whisper_tflite_initialized = false;
};

struct WhisperFilters {
  int n_mel;
  int n_fft;

  std::vector<float> data;
};

struct WhisperMel {
  int n_len;
  int n_mel;
  std::vector<float> data;
};

// Print a vector of float values
void print(const std::vector<float>& a);

// Naive Discrete Fourier Transform
void dft(const std::vector<float>& in, std::vector<float>& out);

// Cooley-Tukey FFT
void fft(const std::vector<float>& in, std::vector<float>& out);

// Forward declarations
const char* whisper_token_to_str(int token);

bool log_mel_spectrogram(const float* samples, int n_samples, int sample_rate,
                         int fft_size, int fft_step, int n_mel, int n_threads,
                         WhisperFilters& filters, WhisperMel& mel);

#endif  // _WHISPER_H_
