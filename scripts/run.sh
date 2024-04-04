#!/bin/bash

./build/app/minimal \
  ../whisper.tflite/models/whisper-tiny-en.tflite \
  data/openai.whisper.tiny/tflt_vocab_mel.bin \
  ./samples/english_test_3_bili.wav
