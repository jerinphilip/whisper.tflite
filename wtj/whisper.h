#ifndef _WHISPER_H_
#define _WHISPER_H_

#include <cmath>
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

// Forward declarations
struct whisper_vocab;
struct whisper_filters;
struct whisper_mel;
const char* whisper_token_to_str(int token);
bool log_mel_spectrogram(const float* samples, const int n_samples,
                         const int sample_rate, const int fft_size,
                         const int fft_step, const int n_mel,
                         const int n_threads, const whisper_filters& filters,
                         whisper_mel& mel);

// whisper_vocab structure
struct whisper_vocab {
  std::map<int, std::string> id_to_token;

  int n_vocab_additional = 51864;

  int token_eot = 50256;
  int token_sot = 50257;
  int token_prev = 50360;
  int token_solm = 50361;  // ??
  int token_not = 50362;   // no timestamps
  int token_beg = 50363;

  static const int token_translwordate = 50358;
  static const int token_transcribe = 50359;
};

// whisper_tflite structure
struct whisper_tflite {
  char* buffer = nullptr;
  long size = 0;
  std::unique_ptr<tflite::FlatBufferModel> model;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  float* input;

  bool is_whisper_tflite_initialized = false;
};

// whisper_filters structure
struct whisper_filters {
  int n_mel;
  int n_fft;

  std::vector<float> data;
};

// whisper_mel structure
struct whisper_mel {
  int n_len;
  int n_mel;

  std::vector<float> data;
};

// Global whisper_vocab instance
extern whisper_vocab g_vocab;
extern whisper_tflite g_whisper_tflite;
extern whisper_filters filters;
extern whisper_mel mel;

// Print a vector of float values
void print(const std::vector<float>& a);

// Convert a token to a string
const char* whisper_token_to_str(int token);

// Naive Discrete Fourier Transform
void dft(const std::vector<float>& in, std::vector<float>& out);

// Cooley-Tukey FFT
void fft(const std::vector<float>& in, std::vector<float>& out);

// Log mel spectrogram computation
bool log_mel_spectrogram(const float* samples, const int n_samples,
                         const int sample_rate, const int fft_size,
                         const int fft_step, const int n_mel,
                         const int n_threads, const whisper_filters& filters,
                         whisper_mel& mel);

#endif  // _WHISPER_H_
