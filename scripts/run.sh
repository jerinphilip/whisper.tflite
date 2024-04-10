#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL="3"
WAV=$1
set -x

./build/app/minimal \
  ../whisper.tflite/models/whisper-tiny-en.tflite \
  data/openai.whisper.tiny/filters_vocab_en.bin \
  ./samples/english_test_3_bili.wav
# data/openai.whisper.tiny/tflt_vocab_mel.bin \

./build/app/encdec \
  --encoder "export/workspace/aware-ai-whisper-tiny-german.encoder.tflite" \
  --decoder "export/workspace/aware-ai-whisper-tiny-german.decoder.tflite" \
  --vocab data/openai.whisper.tiny/filters_vocab_multilingual.bin \
  --input samples/de/de_$WAV.wav

# --input samples/de/de_9.wav
# --vocab data/openai.whisper.tiny/tflt_vocab_mel.bin \
