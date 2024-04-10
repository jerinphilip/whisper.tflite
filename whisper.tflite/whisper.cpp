#include "whisper.h"

#include <bits/types/struct_timeval.h>
#include <sys/time.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/core/model_builder.h"
#include "wav_util.h"

namespace whisper {

// Print a vector of float values
void print(const std::vector<float>& a) {
  std::cout << "The vector elements are: ";
  for (float i : a) std::cout << i << ' ';
}

// Naive Discrete Fourier Transform
void dft(const std::vector<float>& in, std::vector<float>& out) {
  int N = in.size();  // NOLINT(readability-identifier-naming)
  out.resize(N * 2);

  for (int k = 0; k < N; k++) {
    float re = 0;
    float im = 0;

    for (int n = 0; n < N; n++) {
      float angle = 2 * M_PI * k * n / N;
      re += in[n] * std::cos(angle);
      im -= in[n] * std::sin(angle);
    }

    out[k * 2 + 0] = re;
    out[k * 2 + 1] = im;
  }
}

// Cooley-Tukey FFT
// NOLINTNEXTLINE(misc-no-recursion)
void fft(const std::vector<float>& in, std::vector<float>& out) {
  out.resize(in.size() * 2);

  int N = in.size();  // NOLINT(readability-identifier-naming)

  if (N == 1) {
    out[0] = in[0];
    out[1] = 0;
    return;
  }

  if (N % 2 == 1) {
    dft(in, out);
    return;
  }

  std::vector<float> even;
  std::vector<float> odd;

  for (int i = 0; i < N; i++) {
    if (i % 2 == 0) {
      even.push_back(in[i]);
    } else {
      odd.push_back(in[i]);
    }
  }

  std::vector<float> even_fft;
  std::vector<float> odd_fft;

  fft(even, even_fft);  // NOLINT(misc-no-recursion)
  fft(odd, odd_fft);    // NOLINT(misc-no-recursion)

  for (int k = 0; k < N / 2; k++) {
    float theta = 2 * M_PI * k / N;

    float re = std::cos(theta);
    float im = -std::sin(theta);

    float re_odd = odd_fft[2 * k + 0];
    float im_odd = odd_fft[2 * k + 1];

    out[2 * k + 0] = even_fft[2 * k + 0] + re * re_odd - im * im_odd;
    out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;

    out[2 * (k + N / 2) + 0] = even_fft[2 * k + 0] - re * re_odd + im * im_odd;
    out[2 * (k + N / 2) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
  }
}

// Log mel spectrogram computation
bool log_mel_spectrogram(const float* samples, int n_samples,
                         int /*sample_rate*/, int fft_size, int fft_step,
                         int n_mel, int n_threads, Filters& filters, Mel& mel) {
  std::vector<float> hann;
  hann.resize(fft_size);

  // https://en.wikipedia.org/wiki/Hann_function
  // Hann function is a window function used to perform Hann smoothing
  for (int i = 0; i < fft_size; i++) {
    // NOLINTNEXTLINE(readability-magic-numbers)
    hann[i] = 0.5 * (1.0 - cos((2.0 * M_PI * i) / fft_size));
  }

  mel.n_mel = n_mel;
  mel.n_len = (n_samples) / fft_step;
  mel.data.resize(mel.n_mel * mel.n_len);

  // std::cout << "n_mel: " << mel.n_mel << std::endl;
  // std::cout << "n_len: " << mel.n_len << std::endl;

  const int n_fft = 1 + fft_size / 2;

  std::vector<std::thread> workers(n_threads);
  for (int iw = 0; iw < n_threads; ++iw) {
    workers[iw] = std::thread(
        [&](int ith) {
          std::vector<float> fft_in;
          fft_in.resize(fft_size);
          for (int i = 0; i < fft_size; i++) {
            fft_in[i] = 0.0;
          }

          std::vector<float> fft_out;
          fft_out.resize(2 * fft_size);

          for (int i = ith; i < mel.n_len; i += n_threads) {
            const int offset = i * fft_step;

            // apply Hanning window
            for (int j = 0; j < fft_size; j++) {
              if (offset + j < n_samples) {
                fft_in[j] = hann[j] * samples[offset + j];
              } else {
                fft_in[j] = 0.0;
              }
            }

            // FFT -> mag^2
            fft(fft_in, fft_out);

            for (int j = 0; j < fft_size; j++) {
              fft_out[j] = (fft_out[2 * j + 0] * fft_out[2 * j + 0] +
                            fft_out[2 * j + 1] * fft_out[2 * j + 1]);
            }

            for (int j = 1; j < fft_size / 2; j++) {
              fft_out[j] += fft_out[fft_size - j];
            }

            // mel spectrogram
            for (int j = 0; j < mel.n_mel; j++) {
              double sum = 0.0;

              for (int k = 0; k < n_fft; k++) {
                sum += fft_out[k] * filters.data[j * n_fft + k];
              }

              constexpr float kEPS = 1e-10;

              if (sum < kEPS) {
                sum = kEPS;
              }

              sum = log10(sum);

              mel.data[j * mel.n_len + i] = sum;
            }
          }
        },
        iw);
  }

  // Wait for all threads to finish
  for (auto& worker : workers) {
    worker.join();
  }

  // https://github.com/openai/whisper/blob/ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab/whisper/audio.py#L154
  // clamping and normalization
  double mmax = -1e20;
  for (int i = 0; i < mel.n_mel * mel.n_len; i++) {
    if (mel.data[i] > mmax) {
      mmax = mel.data[i];
    }
  }

  mmax -= 8.0;

  for (int i = 0; i < mel.n_mel * mel.n_len; i++) {
    if (mel.data[i] < mmax) {
      mel.data[i] = mmax;
    }

    mel.data[i] = (mel.data[i] + 4.0) / 4.0;
  }

  return true;
}

void transform_vocab_multilingual(Vocab& vocab) {
  vocab.n_vocab = kVocabMultilingualSize;
  vocab.token_eot++;
  vocab.token_sot++;
  vocab.token_prev++;
  vocab.token_solm++;
  vocab.token_not++;
  vocab.token_beg++;
}

// core/c/c_api_types.h
// https://github.com/tensorflow/tensorflow/blob/b0731769d7c0e1d339fcfce30f46d9a73a6f91f1/tensorflow/lite/core/c/c_api_types.h#L116
/// Types supported by tensor
const char* tf_type_to_name(TfLiteType type) {
  // NOLINTBEGIN
  // clang-format off
	switch(type){
		case kTfLiteNoType     : return "kTfLiteNoType     ";
		case kTfLiteFloat32    : return "kTfLiteFloat32    ";
		case kTfLiteInt32      : return "kTfLiteInt32      ";
		case kTfLiteUInt8      : return "kTfLiteUInt8      ";
		case kTfLiteInt64      : return "kTfLiteInt64      ";
		case kTfLiteString     : return "kTfLiteString     ";
		case kTfLiteBool       : return "kTfLiteBool       ";
		case kTfLiteInt16      : return "kTfLiteInt16      ";
		case kTfLiteComplex64  : return "kTfLiteComplex64  ";
		case kTfLiteInt8       : return "kTfLiteInt8       ";
		case kTfLiteFloat16    : return "kTfLiteFloat16    ";
		case kTfLiteFloat64    : return "kTfLiteFloat64    ";
		case kTfLiteComplex128 : return "kTfLiteComplex128 ";
		case kTfLiteUInt64     : return "kTfLiteUInt64     ";
		case kTfLiteResource   : return "kTfLiteResource   ";
		case kTfLiteVariant    : return "kTfLiteVariant    ";
		case kTfLiteUInt32     : return "kTfLiteUInt32     ";
		case kTfLiteUInt16     : return "kTfLiteUInt16     ";
		case kTfLiteInt4       : return "kTfLiteInt4       ";
		case kTfLiteBFloat16   : return "kTfLiteBFloat16   ";
	}
  // clang-format on
  // NOLINTEND
  return "kUnknown          ";
}

Atom::Atom(const std::string& path)
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

void inspect_tflite_tensor(const char* name, const TfLiteTensor& tensor) {
  if (std::getenv("DEBUG")) {
    const TfLiteIntArray* dims = tensor.dims;
    fprintf(stderr, "[user::%s] [%s: ", name, tf_type_to_name(tensor.type));
    for (size_t i = 0; i < dims->size; i++) {
      fprintf(stderr, "%s%d", i ? "x" : "", dims->data[i]);
    }
    fprintf(stderr, "] size: %zu [tf::%s]\n", tensor.bytes, tensor.name);
  }
}

Encoder::Encoder(const std::string& path) : atom_(path) {}
std::tuple<TfLiteTensor*, float*> Encoder::forward(const whisper::Mel& mel) {
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

Decoder::Decoder(const std::string& path, const whisper::Vocab& vocab)
    : atom_(path), vocab_(vocab) {}

std::vector<int64_t> Decoder::forward(
    std::tuple<TfLiteTensor*, float*> encoder_out) {
  (void)encoder_out;
  auto* interpreter = atom_.interpreter();
  TfLiteTensor* input = interpreter->input_tensor(0);
  inspect_tflite_tensor("decoder_in[0]", *input);

  auto* data = interpreter->typed_input_tensor<float>(0);
  auto [encoder_tensor, encoder_data] = encoder_out;
  memcpy(data, encoder_data, encoder_tensor->bytes);

  // 50259 + idx, because multilingual is shifted.
  // Why is multiligual shifted?
  const int64_t target_lang_id = 50259 + language_id("de");

  std::cout << "de " << target_lang_id << "\n";

  // [50258, 50261, 50359, 50363]
  // [[50261 50359 50363  6142]]
  // NOLINTBEGIN
  std::vector<int64_t> prompt = {
      vocab_.token_sot,         // start of token
      target_lang_id,           // target-lang
      vocab_.token_transcribe,  // transcribe
      vocab_.token_not,         // no-timestamps
  };
  // NOLINTEND

  auto decoder_ids = [&interpreter]() { return interpreter->input_tensor(1); };

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

std::vector<LangKey> language_meta = {
    // clang-format off
      { "en", "english"},
      { "zh", "chinese"},
      { "de", "german"},
      { "es", "spanish"},
      { "ru", "russian"},
      { "ko", "korean"},
      { "fr", "french"},
      { "ja", "japanese"},
      { "pt", "portuguese"},
      { "tr", "turkish"},
      { "pl", "polish"},
      { "ca", "catalan"},
      { "nl", "dutch"},
      { "ar", "arabic"},
      { "sv", "swedish"},
      { "it", "italian"},
      { "id", "indonesian"},
      { "hi", "hindi"},
      { "fi", "finnish"},
      { "vi", "vietnamese"},
      { "he", "hebrew"},
      { "uk", "ukrainian"},
      { "el", "greek"},
      { "ms", "malay"},
      { "cs", "czech"},
      { "ro", "romanian"},
      { "da", "danish"},
      { "hu", "hungarian"},
      { "ta", "tamil"},
      { "no", "norwegian"},
      { "th", "thai"},
      { "ur", "urdu"},
      { "hr", "croatian"},
      { "bg", "bulgarian"},
      { "lt", "lithuanian"},
      { "la", "latin"},
      { "mi", "maori"},
      { "ml", "malayalam"},
      { "cy", "welsh"},
      { "sk", "slovak"},
      { "te", "telugu"},
      { "fa", "persian"},
      { "lv", "latvian"},
      { "bn", "bengali"},
      { "sr", "serbian"},
      { "az", "azerbaijani"},
      { "sl", "slovenian"},
      { "kn", "kannada"},
      { "et", "estonian"},
      { "mk", "macedonian"},
      { "br", "breton"},
      { "eu", "basque"},
      { "is", "icelandic"},
      { "hy", "armenian"},
      { "ne", "nepali"},
      { "mn", "mongolian"},
      { "bs", "bosnian"},
      { "kk", "kazakh"},
      { "sq", "albanian"},
      { "sw", "swahili"},
      { "gl", "galician"},
      { "mr", "marathi"},
      { "pa", "punjabi"},
      { "si", "sinhala"},
      { "km", "khmer"},
      { "sn", "shona"},
      { "yo", "yoruba"},
      { "so", "somali"},
      { "af", "afrikaans"},
      { "oc", "occitan"},
      { "ka", "georgian"},
      { "be", "belarusian"},
      { "tg", "tajik"},
      { "sd", "sindhi"},
      { "gu", "gujarati"},
      { "am", "amharic"},
      { "yi", "yiddish"},
      { "lo", "lao"},
      { "uz", "uzbek"},
      { "fo", "faroese"},
      { "ht", "haitian creole"},
      { "ps", "pashto"},
      { "tk", "turkmen"},
      { "nn", "nynorsk"},
      { "mt", "maltese"},
      { "sa", "sanskrit"},
      { "lb", "luxembourgish"},
      { "my", "myanmar"},
      { "bo", "tibetan"},
      { "tl", "tagalog"},
      { "mg", "malagasy"},
      { "as", "assamese"},
      { "tt", "tatar"},
      { "haw", "hawaiian"},
      { "ln", "lingala"},
      { "ha", "hausa"},
      { "ba", "bashkir"},
      { "jw", "javanese"},
      { "su", "sundanese"},
      { "yue", "cantonese"},
    // clang-format on
};

int language_id(const std::string& code) {
  auto predicate = [&code](const LangKey& key) { return key.first == code; };
  auto needle =
      std::find_if(language_meta.begin(), language_meta.end(), predicate);
  return std::distance(language_meta.begin(), needle);
}

const std::string& lang_code(size_t id) { return language_meta[id].first; }

const char* Reader::read_filters(Filters& filters, const char* head) {
  // Read the magic number
  uint32_t magic = 0;
  memcpy(&magic, head, sizeof(magic));
  // tflt
  // constexpr uint32_t kTFLTExpectedMagic = 0x74666C74;
  // if (magic != kTFLTExpectedMagic) {
  //   printf("Invalid vocab file (bad magic)\n");
  //   return 0;
  // }
  head += sizeof(magic);  // Move the pointer to the next position

  // Filters filters;  // Use the correct struct from whisper.h
  // Load mel filters
  memcpy(&filters.n_mel, head, sizeof(filters.n_mel));
  head += sizeof(filters.n_mel);

  memcpy(&filters.n_fft, head, sizeof(filters.n_fft));
  head += sizeof(filters.n_fft);

  // Allocate memory for the vector and copy data
  filters.data.resize(filters.n_mel * filters.n_fft);
  memcpy(filters.data.data(), head,
         filters.n_mel * filters.n_fft * sizeof(float));
  head += filters.n_mel * filters.n_fft * sizeof(float);
  return head;
}

const char* Reader::read_vocab(Vocab& vocab, bool multilingual,
                               const char* head) {
  int32_t n_vocab = 0;
  memcpy(&n_vocab, head, sizeof(n_vocab));
  head += sizeof(n_vocab);

  // Update the vocabulary size based on whisper.h
  vocab.n_vocab = n_vocab;
  printf("\nn_vocab:%d\n", static_cast<int>(n_vocab));

  // TODO(@jerinphilip): Specialization, fix somehow.
  if (multilingual) {
    transform_vocab_multilingual(vocab);
  }

  // Assuming a maximum word length of 255 characters
  constexpr size_t kMaxBufferSize = 256;
  char word[kMaxBufferSize];
  for (int i = 0; i < n_vocab; i++) {
    uint32_t len;
    memcpy(&len, head, sizeof(len));
    head += sizeof(len);

    memcpy(word, head, len);
    word[len] = '\0';  // Null-terminate the string
    head += len;

    vocab.id_to_token[i] = std::string(word);
  }

  const size_t n_vocab_expected = kVocabEnSize + static_cast<int>(multilingual);
  for (int i = n_vocab; i < n_vocab_expected; i++) {
    std::string word;
    if (i > vocab.token_beg) {
      word = "<|TT" + std::to_string(i - vocab.token_beg) + "|>";
    } else if (i == vocab.token_eot) {
      word = "<|endoftranscript|>";
    } else if (i == vocab.token_sot) {
      word = "<|startoftranscript_|>";
    } else if (i == vocab.token_prev) {
      word = "<|PREV|>";
    } else if (i == vocab.token_not) {
      word = "<|notimestamps|>";
    } else if (i == vocab.token_beg) {
      word = "<|timestampbegin|>";
    } else if (i == vocab.token_translate) {
      word = "<|translate|>";
    } else if (i == vocab.token_transcribe) {
      word = "<|transcribe|>";
    } else if (i > vocab.token_sot && i < vocab.token_translate) {
      int base = vocab.token_sot + 1;
      word = "<|lang-" + lang_code(i - base) + "|>";
    } else {
      word = "<|e" + std::to_string(i) + "|>";
    }
    vocab.id_to_token[i] = word;
    // printf("%s: vocab[%d] = '%s'", __func__, i, word.c_str());
  }
  return head;
}

void Reader::read(Filters& filters, Vocab& vocab) {
  head_ = read_filters(filters, head_);
  head_ = read_vocab(vocab, multilingual_, head_);
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

// Range templates
template <class Int>
std::string decode(const Vocab& vocab, const Int* begin, const Int* end,
                   bool omit_special_tokens /*=false*/) {
  std::string surface;
  for (const Int* p = begin; p != end; p++) {
    int id = *p;
    if (!omit_special_tokens || id < vocab.token_eot) {
      auto query = vocab.id_to_token.find(id);
      assert(query != vocab.id_to_token.end());
      surface += query->second;
    }

    if (id == vocab.token_eot) {
      break;
    }
  }
  return surface;
}

// Template specializations
template std::string decode(const Vocab& vocab, const int* begin,
                            const int* end, bool omit_special_tokens);

template std::string decode(const Vocab& vocab, const int64_t* begin,
                            const int64_t* end, bool omit_special_tokens);

// Convenience on vector, relays into range based functions.
std::string decode(const Vocab& vocab, const std::vector<int64_t>& generated,
                   bool omit_special_tokens) {
  return decode(vocab, generated.data(), generated.data() + generated.size(),
                omit_special_tokens);
}

Monolith::Monolith(const std::string& model_prefix,
                   const std::string& vocab_path, bool multilingual)
    : whisper_(model_prefix + ".tflite"), vocab_file_(vocab_path) {
  /////////////// Load filters and vocab data ///////////////
  FILE* vocab_fp = fopen(vocab_path.c_str(), "rb");
  if (vocab_fp == nullptr) {
    fprintf(stderr, "Unable to open vocabulary file: %s", vocab_path.c_str());
  }

  const char* ptr = static_cast<const char*>(vocab_file_.data());
  int64_t vocab_size;
  std::memcpy(&vocab_size, ptr, sizeof(uint64_t));

  ptr = ptr + sizeof(uint64_t);
  Reader reader(ptr, multilingual);
  reader.read(filters_, vocab_);
}

std::string Monolith::transcribe(const char* waveFile) {
  std::vector<float> pcmf32 = wav_read_legacy(waveFile);
  pcmf32.resize((kSampleRate * kChunkSize), 0);
  std::string text = transcribe(pcmf32);
  return text;
}

std::string Monolith::transcribe(std::vector<float>& samples) {
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

  auto* interpreter = whisper_.interpreter();
  auto* input = interpreter->typed_input_tensor<float>(0);
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
  TfLiteTensor* output_tensor = interpreter->tensor(output);
  TfLiteIntArray* output_dims = output_tensor->dims;
  // assume output dims to be something like (1, 1, ... ,size)
  auto output_size = output_dims->data[output_dims->size - 1];
  int* output_int = interpreter->typed_output_tensor<int>(0);
  bool omit_special_tokens = false;
  std::string text =
      decode(vocab_, output_int, output_int + output_size, omit_special_tokens);

  return text;
}

EncDec::EncDec(const std::string& model_prefix, const std::string& vocab_path,
               bool multilingual)
    : vocab_file_(vocab_path),
      encoder_(model_prefix + ".encoder.tflite"),
      decoder_(model_prefix + ".decoder.tflite", vocab_) {
  // Create a pointer to the start of the unsigned char array
  const char* ptr =
      reinterpret_cast<const char*>(vocab_file_.data()) + sizeof(int64_t);
  Reader reader(ptr, multilingual);
  reader.read(filters_, vocab_);
}

std::string EncDec::transcribe(std::vector<float>& samples) {
  samples.resize((kSampleRate * kChunkSize), 0);
  const auto processor_count = std::thread::hardware_concurrency();

  if (!log_mel_spectrogram(samples.data(), samples.size(), kSampleRate, kNFFT,
                           kHopLength, kNMEL, processor_count, filters_,
                           mel_)) {
    std::cerr << "Failed to compute mel_ spectrogram" << '\n';
    return "";
  }

  auto encoder_out = encoder_.forward(mel_);

  std::vector<int64_t> generated = decoder_.forward(encoder_out);
  bool omit_special_tokens = false;
  std::string surface = decode(vocab_, generated, omit_special_tokens);
  return surface;
}

std::string EncDec::transcribe(const char* waveFile) {
  std::vector<float> pcmf32 = wav_read_legacy(waveFile);
  pcmf32.resize((kSampleRate * kChunkSize), 0);
  std::string text = transcribe(pcmf32);
  return text;
}

Engine* create_engine(EngineType type, const char* model_prefix,
                      const char* vocab_path, bool multilingual) {
  switch (type) {
    case EngineType::Monolith:
      return new Monolith(model_prefix, vocab_path, multilingual);
    case EngineType::EncDec:
      return new EncDec(model_prefix, vocab_path, multilingual);
    default:
      fprintf(stderr, "Unknown engine-type\n");
      break;
  }
  return nullptr;
}

}  // namespace whisper
