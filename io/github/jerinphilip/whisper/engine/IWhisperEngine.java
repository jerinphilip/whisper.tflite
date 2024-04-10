package io.github.jerinphilip.whisper.engine;

import io.github.jerinphilip.whisper.asr.IWhisperListener;
import java.io.IOException;

public interface IWhisperEngine {
  boolean isInitialized();

  void interrupt();

  void setUpdateListener(IWhisperListener listener);

  boolean initialize(String modelPath, String vocabPath, boolean multilingual) throws IOException;

  String transcribeFile(String wavePath);

  String transcribeBuffer(float[] samples);

  // String getTranslation(String wavePath);
}
