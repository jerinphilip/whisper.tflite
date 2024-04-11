package io.github.jerinphilip.whisper;

public interface IWhisperListener {
  void onUpdateReceived(String message);

  void onResultReceived(String result);
}
