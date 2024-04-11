package io.github.jerinphilip.whisper;

public interface IRecorderListener {
  void onUpdateReceived(String message);

  void onDataReceived(float[] samples);
}
