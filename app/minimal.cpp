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

#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/core/model_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "whisper.tflite/wav_util.h"
#include "whisper.tflite/whisper.h"

int main(int argc, char* argv[]) {
  if (argc != 4) {
    fprintf(stderr,
            "Usage: minimal <tflite model> <vocab file> <pcm_file name>\n");
    return 1;
  }

  // NOLINTNEXTLINE
  using namespace whisper;

  const char* filename = argv[1];

  uint64_t vocab_size;
  FILE* vocab_fp = fopen(argv[2], "rb");
  fread(&vocab_size, sizeof(uint64_t), 1, vocab_fp);
  auto vocab_holder = std::make_unique<char[]>(vocab_size);
  fread(vocab_holder.get(), vocab_size, 1, vocab_fp);
  fclose(vocab_fp);

  Filters filters;
  Vocab vocab;
  bool multilingual = false;
  Reader reader(vocab_holder.get(), multilingual);
  reader.read(filters, vocab);

  // Generate input_features for Audio file
  const char* pcmfilename = argv[3];
  // WAV input

  Mel mel;
  // Hack if the audio file size is less than 30ms, append with 0's
  std::vector<float> pcmf32 = wav_read(pcmfilename);
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

  struct timeval start_time;
  struct timeval end_time;
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
  bool omit_special_tokens = true;
  std::string text =
      decode(vocab, output_int, output_int + output_size, omit_special_tokens);

  // Remove extra spaces between words
  text = remove_extra_spaces(text);

  printf("\n%s\n", text.c_str());
  printf("\n");

  return 0;
}
