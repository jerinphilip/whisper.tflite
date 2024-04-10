#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

int main() {
  // const char* filename = "MicInput_16000_mono_float.pcm";
  const char* filename = "english_test_3_bili_16000_mono_float.pcm";
  const char* output_pcm_file =
      "english_test_3_bili_16000_mono_float_silence_removed.pcm";

  // Open the WAV file
  std::ifstream file(filename, std::ios::binary);

  if (!file.is_open()) {
    std::cerr << "Error: Could not open the WAV file." << '\n';
    return 1;
  }

  std::ofstream pcm_file(output_pcm_file, std::ios::binary);
  if (!pcm_file.is_open()) {
    std::cerr << "Error: Could not open the output PCM file." << '\n';
    return 1;
  }

  const int buffer_size =
      512;  // 32 milliseconds of audio data used for silence detection
  const double silence_threshold_db =
      -35.0;  // Adjust the silence threshold as needed

  int seconds_counter = 0;
  int bytes_read_counter = 0;

  // Read and analyze the audio data
  std::vector<float> buffer(buffer_size);
  while (file.read(reinterpret_cast<char*>(buffer.data()),
                   sizeof(float) * buffer_size)) {
    // For sampling rate 16000, reading 16000 samples equals to 1 second
    bytes_read_counter = bytes_read_counter + buffer_size;
    if (bytes_read_counter > 16000) {
      bytes_read_counter = 0;
      seconds_counter++;
      std::cout << "seconds_counter:===========================> "
                << seconds_counter << '\n';
    }

    double rms = 0.0;
    for (int i = 0; i < buffer_size; i++) {
      float sample = buffer[i];
      rms += sample * sample;
    }

    rms = sqrt(rms / buffer_size);
    double d_b = 20 * log10(rms);

    if (d_b < silence_threshold_db) {
      std::cout << "Silence detected (dB: " << d_b << ")." << '\n';
    } else {
      std::cout << "(dB: " << d_b << ")." << '\n';
      pcm_file.write(reinterpret_cast<char*>(buffer.data()), buffer_size);
    }
  }

  // Close files
  pcm_file.close();
  file.close();

  return 0;
}
