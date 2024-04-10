#!/bin/bash

BUILD_DIR="build"
JNI_BUILD_DIR="$BUILD_DIR/bindings/java"
CPP_BUILD_DIR="$BUILD_DIR/whisper.tflite"

find $JNI_BUILD_DIR -iname "*.so"
find $CPP_BUILD_DIR -iname "*.so"

javac $(find io/github/jerinphilip/whisper/ -iname "*.java")

DRIVER_MONO_ARGS=(
  "monolith"                                           # engineType
  "$PWD/../whisper.tflite/models/whisper-tiny-en"      # modelPrefix
  "$PWD/data/openai.whisper.tiny/filters_vocab_en.bin" # vocabPath
  "false"                                              # multilingual
  "$PWD/./samples/english_test_3_bili.wav"             # file
)

DRIVER_ENCDEC_ARGS=(
  "encdec"                                                  # engineType
  "export/workspace/aware-ai-whisper-tiny-german"           # modelPrefix
  "data/openai.whisper.tiny/filters_vocab_multilingual.bin" # vocabPath
  "true"                                                    # multilingual
  "samples/de/de_4.wav"                                     # file
)

LD_LIBRARY_PATH="$JNI_BUILD_DIR:$CPP_BUILD_DIR:$LD_LIBRARY_PATH" \
  java io.github.jerinphilip.whisper.Driver "${DRIVER_MONO_ARGS[@]}"

LD_LIBRARY_PATH="$JNI_BUILD_DIR:$CPP_BUILD_DIR:$LD_LIBRARY_PATH" \
  java io.github.jerinphilip.whisper.Driver "${DRIVER_ENCDEC_ARGS[@]}"
