#!/bin/bash

# ./build/app/minimal \
#   ../whisper.tflite/models/whisper-tiny-en.tflite \
#   data/openai.whisper.tiny/tflt_vocab_mel.bin \
#   ./samples/english_test_3_bili.wav

./build/app/encdec \
  --encoder ../hf.whisper/workspace/aware-ai-whisper-tiny-german.encoder.tflite \
  --decoder ../hf.whisper/workspace/aware-ai-whisper-tiny-german.decoder.tflite \
  --vocab data/openai.whisper.tiny/filters_vocab_multilingual.bin \
  --input samples/de/de_9.wav
# --input samples/de/de_9.wav

# --vocab data/openai.whisper.tiny/tflt_vocab_mel.bin \
