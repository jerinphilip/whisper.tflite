#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mmap_file.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/kernels/register.h"

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

#define TIME_DIFF_MS(start, end)                  \
  (((((end).tv_sec - (start).tv_sec) * 1000000) + \
    ((end).tv_usec - (start).tv_usec)) /          \
   1000)

namespace whisper {

// Constants
static constexpr int kNumGoldenGeneratedIDs = 21;
static constexpr int kGoldenGeneratedIDs[kNumGoldenGeneratedIDs] = {
    50257, 50362,                                              //
    1770,  13,    2264, 346, 353, 318, 262,  46329, 286, 262,  //
    3504,  6097,  11,   290, 356, 389, 9675, 284,   7062       //
};

static constexpr int kSampleRate = 16000;
static constexpr int kNFFT = 400;
static constexpr int kNMEL = 80;
static constexpr int kHopLength = 160;
static constexpr int kChunkSize = 30;
static constexpr int kMelLen = 3000;

static constexpr int kVocabEnSize = 51864;
static constexpr int kVocabMultilingualSize = 51865;

struct Vocab {
  std::map<int, std::string> id_to_token;
  // Some explanation available at
  // https://github.com/openai/whisper/blob/ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab/whisper/tokenizer.py#L175
  // https://github.com/openai/whisper/blob/ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab/whisper/tokenizer.py#L339C1-L353C1
  //
  // specials = [
  //     "<|endoftext|>",
  //     "<|startoftranscript|>",
  //     *[f"<|{lang}|>" for lang in list(LANGUAGES.keys())[:num_languages]],
  //     "<|translate|>",
  //     "<|transcribe|>",
  //     "<|startoflm|>",
  //     "<|startofprev|>",
  //     "<|nospeech|>",
  //     "<|notimestamps|>",
  //     *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
  // ]

  // NOLINTBEGIN(readability-magic-numbers)
  // clang-format off
  int n_vocab             = 51864;
  int token_eot           = 50256;   // end of transcript
  int token_sot           = 50257;   // start of transcript
                                     //
  // language-tags come here.
  // https://github.com/openai/whisper/blob/ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab/whisper/tokenizer.py#L10
  // [50259 ... 50358] = 099 languages
  //
  //  en                  = 50259
  //  zh                  = 50260
  //  de                  = 50261
  //  es                  = 50263
  //  ru                  = 50264
  //  ko                  = 50265
  //  fr                  = 50266
  //  ja                  = 50267
  //  ..                  = ...
  //
  // Perhaps this is also detected?
  // TODO(@any): Figure out.
  
  int token_translate     = 50358;   // translate 
  int token_transcribe    = 50359;   // transcribe
  int token_prev          = 50360;   // start of prev?
  int token_solm          = 50361;   // start of LM
  int token_not           = 50362;   // no timestamps
  int token_beg           = 50363;   // timestamp begin <|0.00|>
  // clang-format on
  // NOLINTEND(readability-magic-numbers)
};

struct Filters {
  int n_mel;
  int n_fft;

  std::vector<float> data;
};

struct Mel {
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
                         Filters& filters, Mel& mel);

void transform_vocab_multilingual(Vocab& vocab);
const char* tf_type_to_name(TfLiteType type);

struct Atom {
 public:
  explicit Atom(const std::string& path);
  tflite::Interpreter* interpreter() { return interpreter_.get(); }

 private:
  std::unique_ptr<tflite::FlatBufferModel> model_ = nullptr;
  tflite::ops::builtin::BuiltinOpResolver resolver_;
  tflite::InterpreterBuilder builder_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
};

struct Encoder {
 public:
  explicit Encoder(const std::string& path);
  std::tuple<TfLiteTensor*, float*> forward(const whisper::Mel& mel);

 private:
  Atom atom_;
};

struct Decoder {
 public:
  explicit Decoder(const std::string& path, const whisper::Vocab& vocab);
  std::vector<int64_t> forward(std::tuple<TfLiteTensor*, float*> encoder_out);

 private:
  Atom atom_;
  const whisper::Vocab& vocab_;
};

struct Engine {
  virtual std::string transcribe(std::vector<float>& samples) = 0;
  virtual std::string transcribe(const char* waveFile) = 0;
  virtual ~Engine() = default;
};

struct Monolith : public Engine {
 public:
  Monolith(const std::string& model_prefix, const std::string& vocab_path,
           bool multilingual);
  std::string transcribe(std::vector<float>& samples) final;
  std::string transcribe(const char* waveFile) final;

 private:
  Atom whisper_;
  Vocab vocab_;
  Filters filters_;
  Mel mel_;

  MmapFile vocab_file_;
};

struct EncDec : public Engine {
 public:
  EncDec(const std::string& model_prefix, const std::string& vocab_path,
         bool multilingual);
  std::string transcribe(std::vector<float>& samples) final;
  std::string transcribe(const char* waveFile) final;

 private:
  Encoder encoder_;
  Decoder decoder_;

  Vocab vocab_;
  Filters filters_;
  Mel mel_;

  MmapFile vocab_file_;
};

enum class EngineType {
  // clang-format off
  Monolith = 0,  //
  EncDec   = 1   //
  // clang-format on
};

void inspect_tflite_tensor(const char* name, const TfLiteTensor& tensor);

using LangKey = std::pair<std::string, std::string>;
extern std::vector<LangKey> language_meta;
// https://github.com/openai/whisper/blob/ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab/whisper/tokenizer.py#L10
int language_id(const std::string& code);
const std::string& lang_code(size_t id);

// Vocab file layout.
//
// VocabBin {
//    int magic; // 32-bit?  == 0x5753052
//    filters {
//      n_mel;
//      n_fft;
//      data [ n_mel x n_fft ];
//    }
//    vocab {
//      n_vocab;
//      { token-length <token> } [ n_vocab]
//    }
//
//    extra-vocab {
//        EOT // "[_EOT_]"
//        SOT // "[_SOT_]
//        PREV // "[_PREV_]
//        NOT // "[_NOT_]
//        BEG // "[_BEG_]
//    }
// };
struct Reader {
 public:
  explicit Reader(const char* head, bool multilingual)
      : head_(head), multilingual_(multilingual) {}
  void read(Filters& filters, Vocab& vocab);

 private:
  static const char* read_filters(Filters& filters, const char* head);
  static const char* read_vocab(Vocab& vocab, bool multilingual,
                                const char* head);
  const char* head_;
  bool multilingual_;
};

std::string remove_extra_spaces(const std::string& input);

template <class Int>
std::string decode(const Vocab& vocab, const Int* begin, const Int* end,
                   bool omit_special_tokens);

std::string decode(const Vocab& vocab, const std::vector<int64_t>& generated,
                   bool omit_special_tokens);

Engine* create_engine(EngineType type, const char* model_prefix,
                      const char* vocab_path, bool multilingual);
}  // namespace whisper
