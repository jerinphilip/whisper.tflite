package io.github.jerinphilip.whisper;

import io.github.jerinphilip.whisper.engine.WhisperEngineNative;

class Driver {
  public static void main(String args[]) {
    System.out.println("Hello World!");

    for (int i = 0; i < args.length; i++) {
      System.out.print("[" + String.valueOf(i) + ": " + args[i] + "] ");
    }

    System.out.println();

    long engineType = (args[0].equals("encdec")) ? 1 : 0;
    String modelPath = args[1];
    String vocabPath = args[2];
    boolean multilingual = (args[3].equals("true"));
    String audioFilePath = args[4];

    System.out.println("engineType: " + String.valueOf(engineType));
    System.out.println("modelPath: " + modelPath);
    System.out.println("vocabPath: " + vocabPath);
    System.out.println("multilingual: " + Boolean.toString(multilingual));

    WhisperEngineNative engine =
        new WhisperEngineNative(engineType, modelPath, vocabPath, multilingual);
    String transcription = engine.transcribe(audioFilePath);
    System.out.println("transcription: " + transcription);
  }
}
