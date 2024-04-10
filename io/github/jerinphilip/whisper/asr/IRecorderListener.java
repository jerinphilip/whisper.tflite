package io.github.jerinphilip.whisper.asr;

public interface IRecorderListener {
  void onUpdateReceived(String message);

  void onDataReceived(float[] samples);
}
