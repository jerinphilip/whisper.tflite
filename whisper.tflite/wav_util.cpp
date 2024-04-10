#include "wav_util.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <ios>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#define DR_WAV_IMPLEMENTATION
#include "dr_libs/dr_wav.h"

namespace whisper {

std::vector<float> wav_read_legacy(const char* filename) {
  // Open the WAV file for binary reading
  std::ifstream wav_file(filename, std::ios::binary);

  if (!wav_file.is_open()) {
    std::cerr << "Failed to open file: " << filename << '\n';
    return std::vector<float>();
  }

  // Read the WAV header
  WAVHeader wav_header;
  wav_file.read(reinterpret_cast<char*>(&wav_header), sizeof(wav_header));

  // Check if it's a valid WAV file
  if (strncmp(wav_header.riff_header, "RIFF", 4) != 0 ||
      strncmp(wav_header.wave_header, "WAVE", 4) != 0 ||
      strncmp(wav_header.fmt_header, "fmt ", 4) != 0) {
    std::cerr << "Not a valid WAV file: " << filename << '\n';
    return std::vector<float>();
  }

  // Determine the audio format
  std::string audio_format_str;
  switch (wav_header.audio_format) {
    case 1:
      audio_format_str = "PCM";
      break;
    case 3:
      audio_format_str = "IEEE Float";
      break;
    // Add more cases for other audio formats as needed
    default:
      audio_format_str = "Unknown";
      break;
  }

  // Print information from the header
  std::cout << "Audio Format: " << audio_format_str << '\n';
  std::cout << "Num Channels: " << wav_header.num_channels << '\n';
  std::cout << "Sample Rate: " << wav_header.sample_rate << '\n';
  std::cout << "Bits Per Sample: " << wav_header.bits_per_sample << '\n';

  // Calculate the number of samples
  uint32_t num_samples = wav_header.wav_size / wav_header.block_align;

  // convert pcm 16 to float
  std::vector<float> float_samples(num_samples);
  if (wav_header.audio_format == 1) {
    // Read audio samples into a vector of int16_t
    std::vector<int16_t> pcm16_samples(num_samples);
    wav_file.read(reinterpret_cast<char*>(pcm16_samples.data()),
                  wav_header.wav_size);

    // Convert int16_t samples to float samples
    for (uint32_t i = 0; i < num_samples; i++) {
      float_samples[i] =
          static_cast<float>(pcm16_samples[i]) / static_cast<float>(INT16_MAX);
    }
  } else {
    // Read audio samples into a vector of float
    wav_file.read(reinterpret_cast<char*>(float_samples.data()),
                  wav_header.wav_size);
  }

  // Close the file
  wav_file.close();

  // Return the float_samples vector
  return float_samples;
}

std::vector<float> wav_read(const char* filename) {
  std::vector<float> pcmf32;
  drwav wav;
  if (!drwav_init_file(&wav, filename, nullptr)) {
    fprintf(stderr, "failed to open WAV file '%s' - check your input\n",
            filename);
  }

  if (wav.channels != 1 && wav.channels != 2) {
    fprintf(stderr, "WAV file '%s' must be mono or stereo\n", filename);
  }

  constexpr int kSampleRate = 16000;
  if (wav.sampleRate != kSampleRate) {  // Update to use the correct sample rate
    fprintf(stderr, "WAV file '%s' must be 16 kHz\n", filename);
  }

  constexpr int kExpectedBitsPerSample = 16;
  if (wav.bitsPerSample != kExpectedBitsPerSample) {
    fprintf(stderr, "WAV file '%s' must be 16-bit\n", filename);
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
      pcmf32[i] = static_cast<float>(pcm16[i]) / static_cast<float>(INT16_MAX);
    }
  } else {
    for (int i = 0; i < n; i++) {
      pcmf32[i] = static_cast<float>(pcm16[2 * i] + pcm16[2 * i + 1]) /
                  static_cast<float>(INT32_MAX);
    }
  }
  return pcmf32;
}
}  // namespace whisper
