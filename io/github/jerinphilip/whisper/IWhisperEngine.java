package io.github.jerinphilip.whisper;


public interface IWhisperEngine {
  void setUpdateListener(IWhisperListener listener);

  String transcribe(String wavePath);

  String transcribe(float[] samples);
}
