#include <cinttypes>
#include <cstdio>
#include <string>

#include "whisper.tflite/filters_vocab_en.h"
#include "whisper.tflite/filters_vocab_multilingual.h"
#include "whisper.tflite/input_features.h"
#include "whisper.tflite/tflt-vocab-mel.h"

int main(int argc, char **argv) {
  if (argc < 2) {
    exit(EXIT_FAILURE);
  }

  std::string asset(argv[1]);
  std::string outdir(argv[2]);
  std::string outpath = outdir + "/" + asset + ".bin";

  uint64_t filters_vocab_en_size = sizeof(filters_vocab_en);
  uint64_t filters_vocab_multilingual_size = sizeof(filters_vocab_multilingual);
  uint64_t tflt_vocab_mel_size = sizeof(tflt_vocab_mel_bin);
  uint64_t input_features_size = sizeof(_content_input_features_bin);

  constexpr uint64_t kExpectedMelSize = 586174;
  if (tflt_vocab_mel_size != kExpectedMelSize) {
    fprintf(stderr, "Error. mismatch\n");
  }

  constexpr uint64_t kExpectedInputFeaturesSize = 960000;
  if (input_features_size != kExpectedInputFeaturesSize) {
    fprintf(stderr, "Error. mismatch %d != %d\n",
            static_cast<int>(input_features_size),
            static_cast<int>(kExpectedInputFeaturesSize));
  }
  // clang-format off
  printf("vocab-en size: %" PRIu64 "\n", filters_vocab_en_size);
  printf("vocab-multilingual size: %" PRIu64 "\n", filters_vocab_multilingual_size);
  printf("tflt_vocab_mel size: %" PRIu64 "\n", tflt_vocab_mel_size);
  // clang-format on

  if (asset == "filters_vocab_en") {
    FILE *out = fopen(outpath.c_str(), "wb");
    fwrite(&filters_vocab_en_size, sizeof(uint64_t), 1, out);
    fwrite(filters_vocab_en, filters_vocab_en_size, 1, out);
    fclose(out);
  } else if (asset == "filters_vocab_multilingual") {
    FILE *out = fopen(outpath.c_str(), "wb");
    fwrite(&filters_vocab_multilingual_size, sizeof(uint64_t), 1, out);
    fwrite(filters_vocab_multilingual, filters_vocab_multilingual_size, 1, out);
    fclose(out);
  } else if (asset == "tflt_vocab_mel") {
    FILE *out = fopen(outpath.c_str(), "wb");
    fwrite(&tflt_vocab_mel_size, sizeof(uint64_t), 1, out);
    fwrite(tflt_vocab_mel_bin, tflt_vocab_mel_size, 1, out);
    fclose(out);
  } else if (asset == "input_features") {
    FILE *out = fopen(outpath.c_str(), "wb");
    fwrite(&input_features_size, sizeof(uint64_t), 1, out);
    fwrite(_content_input_features_bin, input_features_size, 1, out);
    fclose(out);
  } else {
    FILE *out = fopen(outpath.c_str(), "wb");
    fprintf(stderr, "Did not write any assets\n");
    fclose(out);
  }
  return 0;
}
