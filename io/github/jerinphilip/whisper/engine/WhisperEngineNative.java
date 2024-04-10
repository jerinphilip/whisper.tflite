package io.github.jerinphilip.whisper.engine;

import io.github.jerinphilip.whisper.asr.IWhisperListener;

public class WhisperEngineNative implements IWhisperEngine {
  private final String TAG = "WhisperEngineNative";
  private final long nativePtr; // Native pointer to the TFLiteEngine instance
  private IWhisperListener mUpdateListener = null;

  public WhisperEngineNative(
      int engineType, String modelPath, String vocabPath, boolean multilingual) {
    nativePtr = create(engineType, modelPath, vocabPath, multilingual);
  }

  @Override
  public void setUpdateListener(IWhisperListener listener) {
    mUpdateListener = listener;
  }

  @Override
  public String transcribe(float[] samples) {
    return transcribeBuffer(nativePtr, samples);
  }

  @Override
  public String transcribe(String waveFile) {
    return transcribeFile(nativePtr, waveFile);
  }

  public void updateStatus(String message) {
    if (mUpdateListener != null) mUpdateListener.onUpdateReceived(message);
  }

  private void destroy() {
    destroy(nativePtr);
  }

  static {
    System.loadLibrary("whisper-tflite");
    System.loadLibrary("whisper-tflite-jni");
  }

  // Native methods
  private native long create();

  private native void destroy(long nativePtr);

  private native String transcribeBuffer(long nativePtr, float[] samples);

  private native String transcribeFile(long nativePtr, String waveFile);
}
