package io.github.jerinphilip.whisper;

import io.github.jerinphilip.whisper.engine.WhisperEngineNative;

class Driver {
  public static void main(String args[]) {
    System.out.println("Hello World!");
    WhisperEngineNative engine = new WhisperEngineNative();

    String modelPath = args[1];
    String vocabPath = args[2];
    boolean multilingual = args[3] == "true";

    engine.initialize(modelPath, vocabPath, multilingual);
  }
}
