package io.github.jerinphilip.whisper.asr;

public interface IWhisperListener {
  void onUpdateReceived(String message);

  void onResultReceived(String result);
}
