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
#include <bits/types/struct_timeval.h>
#include <sys/time.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#define DR_WAV_IMPLEMENTATION
#include "dr_libs/dr_wav.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/core/model_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "whisper.tflite/whisper.h"

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

std::string remove_extra_spaces(const std::string& input) {
  std::string result;
  result.reserve(input.length());
  bool space = false;

  for (char c : input) {
    if (c == ' ') {
      if (!space) {
        result += c;
      }
      space = true;
    } else {
      result += c;
      space = false;
    }
  }

  return result;
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    fprintf(stderr,
            "Usage: minimal <tflite model> <vocab file> <pcm_file name>\n");
    return 1;
  }

  // NOLINTNEXTLINE
  using namespace whisper;

  const char* filename = argv[1];
  Mel mel;  // Use the correct struct from whisper.h
  struct timeval start_time;
  struct timeval end_time;
  Vocab vocab;

  uint64_t vocab_size;
  FILE* vocab_fp = fopen(argv[2], "rb");
  fread(&vocab_size, sizeof(uint64_t), 1, vocab_fp);
  auto vocab_holder = std::make_unique<char[]>(vocab_size);
  fread(vocab_holder.get(), vocab_size, 1, vocab_fp);
  fclose(vocab_fp);

  // Create a pointer to the start of the unsigned char array
  char* ptr = vocab_holder.get();
  // Read the magic number
  uint32_t magic = 0;
  memcpy(&magic, ptr, sizeof(magic));
  // tflt
  constexpr uint32_t kTFLTExpectedMagic = 0x74666C74;
  if (magic != kTFLTExpectedMagic) {
    printf("Invalid vocab file (bad magic)\n");
    return 0;
  }
  ptr += sizeof(magic);  // Move the pointer to the next position

  Filters filters;  // Use the correct struct from whisper.h
  // Load mel filters
  memcpy(&filters.n_mel, ptr, sizeof(filters.n_mel));
  ptr += sizeof(filters.n_mel);

  memcpy(&filters.n_fft, ptr, sizeof(filters.n_fft));
  ptr += sizeof(filters.n_fft);

  // Allocate memory for the vector and copy data
  filters.data.resize(filters.n_mel * filters.n_fft);
  memcpy(filters.data.data(), ptr,
         filters.n_mel * filters.n_fft * sizeof(float));
  ptr += filters.n_mel * filters.n_fft * sizeof(float);

  // Load vocab
  int32_t n_vocab = 0;
  memcpy(&n_vocab, ptr, sizeof(n_vocab));
  ptr += sizeof(n_vocab);

  // Update the vocabulary size based on whisper.h
  vocab.n_vocab = n_vocab;
  printf("\nn_vocab:%d\n", static_cast<int>(n_vocab));

  // Assuming a maximum word length of 255 characters
  constexpr size_t kMaxBufferSize = 256;
  char word[kMaxBufferSize];
  for (int i = 0; i < n_vocab; i++) {
    uint32_t len;
    memcpy(&len, ptr, sizeof(len));
    ptr += sizeof(len);

    memcpy(word, ptr, len);
    word[len] = '\0';  // Null-terminate the string
    ptr += len;

    vocab.id_to_token[i] = std::string(word);
  }

  // Generate input_features for Audio file
  const char* pcmfilename = argv[3];
  // WAV input
  std::vector<float> pcmf32;
  {
    drwav wav;
    if (!drwav_init_file(&wav, pcmfilename, nullptr)) {
      fprintf(stderr, "%s: failed to open WAV file '%s' - check your input\n",
              argv[0], pcmfilename);
      return 3;
    }

    if (wav.channels != 1 && wav.channels != 2) {
      fprintf(stderr, "%s: WAV file '%s' must be mono or stereo\n", argv[0],
              pcmfilename);
      return 4;
    }

    if (wav.sampleRate !=
        kSampleRate) {  // Update to use the correct sample rate
      fprintf(stderr, "%s: WAV file '%s' must be 16 kHz\n", argv[0],
              pcmfilename);
      return 5;
    }

    if (wav.bitsPerSample != 16) {
      fprintf(stderr, "%s: WAV file '%s' must be 16-bit\n", argv[0],
              pcmfilename);
      return 6;
    }

    std::vector<int16_t> pcm16;
    pcm16.resize(wav.totalPCMFrameCount * wav.channels);
    drwav_read_pcm_frames_s16(&wav, wav.totalPCMFrameCount, pcm16.data());
    drwav_uninit(&wav);
    // convert to mono, float
    pcmf32.resize(wav.totalPCMFrameCount);
    int n = wav.totalPCMFrameCount;
    if (wav.channels == 1) {
      for (int i = 0; i < n; i++) {
        pcmf32[i] = static_cast<float>(pcm16[i]) / 32768.0F;
      }
    } else {
      for (int i = 0; i < n; i++) {
        pcmf32[i] =
            static_cast<float>(pcm16[2 * i] + pcm16[2 * i + 1]) / 65536.0F;
      }
    }
  }

  // Hack if the audio file size is less than 30ms, append with 0's
  pcmf32.resize((kSampleRate * kChunkSize), 0);
  if (!log_mel_spectrogram(pcmf32.data(), pcmf32.size(), kSampleRate, kNFFT,
                           kHopLength, kNMEL, 1, filters, mel)) {
    fprintf(stderr, "%s: failed to compute mel spectrogram\n", __func__);
    return -1;
  }

  printf("\nmel.n_len%d\n",
         mel.n_len);  // Update to use the correct struct members
  printf("\nmel.n_mel:%d\n",
         mel.n_mel);  // Update to use the correct struct members

  // Load tflite model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter with the InterpreterBuilder.
  // Note: all Interpreters should be built with the InterpreterBuilder,
  // which allocates memory for the Interpreter and does various set up
  // tasks so that the Interpreter can read the provided model.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

  // Get information about the memory area to use for the model's input.
  auto* input = interpreter->typed_input_tensor<float>(0);
  // Use the processed audio data as input
  memcpy(input, mel.data.data(),
         mel.n_mel * mel.n_len *
             sizeof(float));  // Update to use the correct struct members

  gettimeofday(&start_time, nullptr);
  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  gettimeofday(&end_time, nullptr);
  printf("Inference time %ld seconds \n",
         (end_time.tv_sec - start_time.tv_sec));

  int output = interpreter->outputs()[0];
  TfLiteTensor* output_tensor = interpreter->tensor(output);
  TfLiteIntArray* output_dims = output_tensor->dims;
  auto output_size = output_dims->data[output_dims->size - 1];
  int* output_int = interpreter->typed_output_tensor<int>(0);
  std::string text;
  auto decode = [&vocab](int token) {
    // Empty
    return vocab.id_to_token.at(token).c_str();
  };

  for (int i = 0; i < output_size; i++) {
    if (output_int[i] == vocab.token_eot) {
      break;
    }
    if (output_int[i] < vocab.token_eot) {
      text += decode(output_int[i]);
    }
  }

  // Remove extra spaces between words
  text = remove_extra_spaces(text);

  printf("\n%s\n", text.c_str());
  printf("\n");

  return 0;
}
