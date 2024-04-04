#include <jni.h>
#include <jni_md.h>

#include <string>
#include <vector>

#include "TFLiteEngine.h"

#define WTJ_JNI_EXPORT(cls, fn) \
  JNICALL java_com_io_github_jerinphilip_##cls##_##fn

extern "C" {

// JNI method to create an instance of TFLiteEngine
JNIEXPORT jlong WTJ_JNI_EXPORT(WhisperEngineNative,
                               createTFLiteEngine)(JNIEnv * /*env*/,
                                                   jobject /*thiz*/) {
  return reinterpret_cast<jlong>(new TFLiteEngine());
}

// JNI method to load the model
JNIEXPORT jint WTJ_JNI_EXPORT(WhisperEngineNative,
                              loadModel)(JNIEnv *env, jobject /*thiz*/,
                                         jlong nativePtr, jstring modelPath,
                                         jstring vocabPath,
                                         jboolean isMultilingual) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto *engine = reinterpret_cast<TFLiteEngine *>(nativePtr);
  const char *c_model_path = env->GetStringUTFChars(modelPath, nullptr);
  const char *c_vocab_path = env->GetStringUTFChars(vocabPath, nullptr);
  int result =
      engine->loadModel(c_model_path, c_vocab_path, isMultilingual != 0U);
  env->ReleaseStringUTFChars(modelPath, c_model_path);
  return static_cast<jint>(result);
}

// JNI method to free the model
JNIEXPORT void WTJ_JNI_EXPORT(WhisperEngineNative, freeModel)(JNIEnv * /*env*/,
                                                              jobject /*thiz*/,
                                                              jlong nativePtr) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto *engine = reinterpret_cast<TFLiteEngine *>(nativePtr);
  engine->freeModel();
  delete engine;
}

// JNI method to transcribe audio buffer
JNIEXPORT jstring WTJ_JNI_EXPORT(WhisperEngineNative, transcribeBuffer)(
    JNIEnv *env, jobject /*thiz*/, jlong nativePtr, jfloatArray samples) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto *engine = reinterpret_cast<TFLiteEngine *>(nativePtr);

  // Convert jfloatArray to std::vector<float>
  jsize len = env->GetArrayLength(samples);
  jfloat *data = env->GetFloatArrayElements(samples, nullptr);
  std::vector<float> sample_vector(data, data + len);
  env->ReleaseFloatArrayElements(samples, data, 0);

  std::string result = engine->transcribeBuffer(sample_vector);
  return env->NewStringUTF(result.c_str());
}

// JNI method to transcribe audio file
JNIEXPORT jstring WTJ_JNI_EXPORT(WhisperEngineNative,
                                 transcribeFile)(JNIEnv *env, jobject /*thiz*/,
                                                 jlong nativePtr,
                                                 jstring waveFile) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto *engine = reinterpret_cast<TFLiteEngine *>(nativePtr);
  const char *c_wave_file = env->GetStringUTFChars(waveFile, nullptr);
  std::string result = engine->transcribeFile(c_wave_file);
  env->ReleaseStringUTFChars(waveFile, c_wave_file);
  return env->NewStringUTF(result.c_str());
}

}  // extern "C"
#undef WTJ_JNI_EXPORT
