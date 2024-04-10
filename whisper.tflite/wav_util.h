#include <cstdint>
#include <vector>

namespace whisper {
#pragma pack(push, 1)  // Ensure that struct members are packed tightly

// Define the WAV file header structure
struct WAVHeader {
  char riff_header[4];
  uint32_t wav_size;
  char wave_header[4];
  char fmt_header[4];
  uint32_t fmt_chunk_size;
  uint16_t audio_format;
  uint16_t num_channels;
  uint32_t sample_rate;
  uint32_t byte_rate;
  uint16_t block_align;
  uint16_t bits_per_sample;
};

#pragma pack(pop)  // Restore default struct packing

std::vector<float> wav_read_legacy(const char* filename);
std::vector<float> wav_read(const char* filename);
}  // namespace whisper
