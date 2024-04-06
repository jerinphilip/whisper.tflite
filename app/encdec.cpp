
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
#include "CLI11/CLI11.hpp"
#include "dr_libs/dr_wav.h"
#include "tensorflow/lite/c/common.h"
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

// core/c/c_api_types.h

/// Types supported by tensor
const char* tf_type_to_name(TfLiteType type) {
  // NOLINTBEGIN
  // clang-format off
	switch(type){
		case 0  : return "kTfLiteNoType     ";
		case 1  : return "kTfLiteFloat32    ";
		case 2  : return "kTfLiteInt32      ";
		case 3  : return "kTfLiteUInt8      ";
		case 4  : return "kTfLiteInt64      ";
		case 5  : return "kTfLiteString     ";
		case 6  : return "kTfLiteBool       ";
		case 7  : return "kTfLiteInt16      ";
		case 8  : return "kTfLiteComplex64  ";
		case 9  : return "kTfLiteInt8       ";
		case 10 : return "kTfLiteFloat16    ";
		case 11 : return "kTfLiteFloat64    ";
		case 12 : return "kTfLiteComplex128 ";
		case 13 : return "kTfLiteUInt64     ";
		case 14 : return "kTfLiteResource   ";
		case 15 : return "kTfLiteVariant    ";
		case 16 : return "kTfLiteUInt32     ";
		case 17 : return "kTfLiteUInt16     ";
		case 18 : return "kTfLiteInt4       ";
		case 19 : return "kTfLiteBFloat16   ";
	}
  // clang-format on
  // NOLINTEND
  return "kUnknown";
}

struct Atom {
 public:
  explicit Atom(const std::string& path)
      : model_(tflite::FlatBufferModel::BuildFromFile(path.c_str())),
        builder_(*model_, resolver_) {
    // Build the interpreter with the InterpreterBuilder.
    // Note: all Interpreters should be built with the InterpreterBuilder,
    // which allocates memory for the Interpreter and does various set up
    // tasks so that the Interpreter can read the provided model.
    builder_(&interpreter_);
    TFLITE_MINIMAL_CHECK(interpreter_ != nullptr);
    TFLITE_MINIMAL_CHECK(interpreter_->AllocateTensors() == kTfLiteOk);
  }

  tflite::Interpreter* interpreter() { return interpreter_.get(); }

 private:
  std::unique_ptr<tflite::FlatBufferModel> model_;
  tflite::ops::builtin::BuiltinOpResolver resolver_;
  tflite::InterpreterBuilder builder_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
};

void inspect_tflite_tensor(const char* name, const TfLiteTensor& tensor) {
  if (std::getenv("DEBUG")) {
    const TfLiteIntArray* dims = tensor.dims;
    fprintf(stderr, "[user::%s] [%s: ", name, tf_type_to_name(tensor.type));
    for (size_t i = 0; i < dims->size; i++) {
      fprintf(stderr, "%s%d", i ? "x" : "", dims->data[i]);
    }
    fprintf(stderr, "] size: %zu [tf::%s]\n", tensor.bytes, tensor.name);
  }
};

struct Encoder {
 public:
  explicit Encoder(const std::string& path) : atom_(path) {}
  std::tuple<TfLiteTensor*, float*> forward(const whisper::Mel& mel) {
    struct timeval start_time;
    struct timeval end_time;
    // Get information about the memory area to use for the model's input.
    auto* interpreter = atom_.interpreter();
    auto* input = interpreter->typed_input_tensor<float>(0);
    // Use the processed audio data as input
    // Update to use the correct struct members
    memcpy(input, mel.data.data(), mel.n_mel * mel.n_len * sizeof(float));
    gettimeofday(&start_time, nullptr);
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
    gettimeofday(&end_time, nullptr);
    // Run inference
    printf("Inference time %ld seconds \n",
           (end_time.tv_sec - start_time.tv_sec));
    int output = interpreter->outputs()[0];

    inspect_tflite_tensor("encoder_in[0]", *(interpreter->input_tensor(0)));
    inspect_tflite_tensor("encoder_out[0]", *(interpreter->output_tensor(0)));
    printf("output: %d\n", output);
    return {
        interpreter->output_tensor(0),             //
        interpreter->typed_input_tensor<float>(0)  //
    };
  }

 private:
  Atom atom_;
};

struct Decoder {
 public:
  explicit Decoder(const std::string& path, const whisper::Vocab& vocab)
      : atom_(path), vocab_(vocab) {}
  std::vector<int64_t> forward(std::tuple<TfLiteTensor*, float*> encoder_out) {
    (void)encoder_out;
    auto* interpreter = atom_.interpreter();
    TfLiteTensor* input = interpreter->input_tensor(0);
    inspect_tflite_tensor("decoder_in[0]", *input);

    auto* data = interpreter->typed_input_tensor<float>(0);
    auto [encoder_tensor, encoder_data] = encoder_out;
    memcpy(data, encoder_data, encoder_tensor->bytes);

    // [50258, 50261, 50359, 50363]
    // [[50261 50359 50363  6142]]
    // NOLINTBEGIN
    std::vector<int64_t> prompt = {
        vocab_.token_sot,
        50261,                    // de
        vocab_.token_transcribe,  // transcribe
        vocab_.token_not,         // no-timestamps
    };
    // NOLINTEND

    auto decoder_ids = [&interpreter]() {
      return interpreter->input_tensor(1);
    };

    inspect_tflite_tensor("decoder_in[1]", *decoder_ids());

    auto argmax = [](const float* begin, const float* end) {
      const float* p = begin;
      float max_value = *p;
      int64_t max_index = std::distance(begin, p);
      ++p;
      while (p < end) {
        int64_t index = std::distance(begin, p);
        if (*p >= max_value) {
          max_value = *p;
          max_index = index;
        }
        ++p;
      }

      return std::make_pair(max_index, max_value);
    };

    int64_t eos_id = vocab_.token_eot;         // NOLINT
    constexpr size_t max_decoder_tokens = 30;  // NOLINT
    size_t vocab_size = vocab_.n_vocab;        // NOLINT
    int64_t num_prime_tokens = prompt.size();
    for (size_t i = prompt.size() - 1; i < max_decoder_tokens; i++) {
      interpreter->ResizeInputTensor(1, {1, static_cast<int>(prompt.size())});
      interpreter->AllocateTensors();
      inspect_tflite_tensor("decoder_in[1]", *decoder_ids());

      auto* prompt_data = interpreter->typed_input_tensor<int64_t>(1);
      std::memcpy(prompt_data, prompt.data(), sizeof(int64_t) * prompt.size());

      interpreter->Invoke();
      TfLiteTensor* output0 = interpreter->output_tensor(0);
      inspect_tflite_tensor("decoder_out[0]", *output0);

      auto* out = interpreter->typed_output_tensor<float>(0);

      std::vector<int64_t> decoded;
      // int64_t suppress = (i + 1 == num_prime_tokens) ? eos_id : -1;
      int64_t suppress = eos_id;
      for (size_t offset = 0; offset <= i; offset++) {
        float* begin = out + offset * vocab_size;
        auto m = argmax(begin, begin + vocab_size);
        if (std::getenv("DEBUG")) {
          fprintf(stderr, "decode[%zu]@%zu = %zu, %f\n", offset, i, m.first,
                  m.second);
        }
        // Last element get added.
        if (offset == i) {
          prompt.push_back(m.first);
        }
      }

      if (prompt.back() == eos_id) {
        break;
      }
    }

    return prompt;
  }

 private:
  Atom atom_;
  const whisper::Vocab& vocab_;
};

int run(const Options& options) {
  // NOLINTNEXTLINE
  using namespace whisper;

  Mel mel;  // Use the correct struct from whisper.h
  Vocab vocab;

  uint64_t vocab_size;
  FILE* vocab_fp = fopen(options.vocab.c_str(), "rb");
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
  // constexpr uint32_t kTFLTExpectedMagic = 0x74666C74;
  // if (magic != kTFLTExpectedMagic) {
  //   printf("Invalid vocab file (bad magic)\n");
  //   return 0;
  // }
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
  transform_vocab_multilingual(vocab);

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
  const char* pcmfilename = options.input.c_str();
  // WAV input
  std::vector<float> pcmf32;
  {
    drwav wav;
    if (!drwav_init_file(&wav, pcmfilename, nullptr)) {
      fprintf(stderr, "failed to open WAV file '%s' - check your input\n",
              pcmfilename);
      return 3;
    }

    if (wav.channels != 1 && wav.channels != 2) {
      fprintf(stderr, "WAV file '%s' must be mono or stereo\n", pcmfilename);
      return 4;
    }

    if (wav.sampleRate !=
        kSampleRate) {  // Update to use the correct sample rate
      fprintf(stderr, "WAV file '%s' must be 16 kHz\n", pcmfilename);
      return 5;
    }

    if (wav.bitsPerSample != 16) {
      fprintf(stderr, "WAV file '%s' must be 16-bit\n", pcmfilename);
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

  printf("Mel{ n_len: %d, n_mel = %d}\n", mel.n_len, mel.n_mel);

  // Allocate tensor buffers.
  //
  Encoder encoder(options.encoder);
  auto encoder_out = encoder.forward(mel);

  Decoder decoder(options.decoder, vocab);
  std::vector<int64_t> decoded = decoder.forward(encoder_out);
  std::string surface;
  fprintf(stderr, "ids: ");
  for (auto& id : decoded) {
    fprintf(stderr, "%zu ", id);
    surface += vocab.id_to_token[id];
  }
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
