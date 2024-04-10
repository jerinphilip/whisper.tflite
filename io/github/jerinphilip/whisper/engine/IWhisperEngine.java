package io.github.jerinphilip.whisper.engine;

import io.github.jerinphilip.whisper.asr.IWhisperListener;

public interface IWhisperEngine {
  void setUpdateListener(IWhisperListener listener);

  String transcribe(String wavePath);

  String transcribe(float[] samples);
}
