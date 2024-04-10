#include <cmath>
#include <cstdio>
#include <cstdlib>

enum { SAMPLE_RATE = (44100), FRAME_SIZE = (512), BUFFER_SIZE = (2048) };
#define VAD_THRESHOLD (0.01f)

using Frame = struct {
  float buffer[BUFFER_SIZE];
  float energy;
};

int main() {
  FILE *fp = fopen("audio.raw", "rb");  // Replace with your input file
  if (fp == nullptr) {
    fprintf(stderr, "Error opening input file.\n");
    return 1;
  }

  float buffer[BUFFER_SIZE];
  Frame frame = {0};
  int is_speech = 0;
  int num_frames = 0;

  while (fread(buffer, sizeof(float), BUFFER_SIZE, fp) == BUFFER_SIZE) {
    for (int i = 0; i < BUFFER_SIZE; i += FRAME_SIZE) {
      frame.energy = 0.0F;

      for (int j = i; j < i + FRAME_SIZE; j++) {
        frame.buffer[j - i] = buffer[j];
        frame.energy += buffer[j] * buffer[j];
      }

      frame.energy = sqrt(frame.energy / FRAME_SIZE);

      if (frame.energy > VAD_THRESHOLD) {
        is_speech = 1;
      } else {
        is_speech = 0;
      }

      printf("Frame %d: %s\n", num_frames, is_speech ? "Speech" : "Silence");
      num_frames++;
    }
  }

  fclose(fp);
  return 0;
}
