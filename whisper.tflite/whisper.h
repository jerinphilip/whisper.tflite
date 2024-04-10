#ifndef _WHISPER_H_
#define _WHISPER_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/kernels/register.h"

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

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

struct TFLite {
  char* buffer = nullptr;
  int64_t size = 0;
  std::unique_ptr<tflite::FlatBufferModel> model;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  float* input;

  bool is_whisper_tflite_initialized = false;
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
  std::unique_ptr<tflite::FlatBufferModel> model_;
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

void inspect_tflite_tensor(const char* name, const TfLiteTensor& tensor);

// https://github.com/openai/whisper/blob/ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab/whisper/tokenizer.py#L10
int language_id(const std::string& code);

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
  explicit Reader(char* head) : head_(head) {}
  void read(Filters& filters, Vocab& vocab);

 private:
  static char* read_filters(Filters& filters, char* head);
  static char* read_vocab(Vocab& vocab, char* head);
  char* head_;
};

std::string remove_extra_spaces(const std::string& input);

class MmapFile {
 public:
  MmapFile() = default;
  explicit MmapFile(const std::string& filepath);
  ~MmapFile();

  void* data() const { return data_; }
  size_t size() const { return size_; }

  // Disable copy and assignment
  MmapFile(const MmapFile&) = delete;
  MmapFile& operator=(const MmapFile&) = delete;

  MmapFile(MmapFile&& from) noexcept;

  MmapFile& operator=(MmapFile&& from) noexcept;

 private:
  void consume(MmapFile& from);
  void reset();

  int fd_ = -1;
  void* data_ = nullptr;
  size_t size_ = 0;
};

}  // namespace whisper
#endif  // _WHISPER_H_
