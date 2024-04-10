package io.github.jerinphilip.whisper;

import io.github.jerinphilip.whisper.engine.WhisperEngineNative;

class Driver {
  public static void main(String args[]) {
    System.out.println("Hello World!");
    WhisperEngineNative engine = new WhisperEngineNative();

    for (int i = 0; i < args.length; i++) {
      System.out.print("[" + String.valueOf(i) + ": " + args[i] + "] ");
    }

    System.out.println();

    String modelPath = args[0];
    String vocabPath = args[1];
    boolean multilingual = (args[2] == "true");

    String audioFilePath = args[3];

    System.out.println("modelPath: " + modelPath);
    System.out.println("vocabPath: " + vocabPath);
    System.out.println("multilingual: " + Boolean.toString(multilingual));

    engine.initialize(modelPath, vocabPath, multilingual);
    String transcription = engine.transcribeFile(audioFilePath);
    System.out.println("transcription: " + transcription);
  }
}
