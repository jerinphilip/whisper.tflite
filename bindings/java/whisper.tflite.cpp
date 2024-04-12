#include "whisper.h"

#include <jni.h>

#include <string>
#include <vector>

#define WTJ_JNI_EXPORT(cls, fn) \
  JNICALL Java_io_github_jerinphilip_whisper_##cls##_##fn

// NOLINTNEXTLINE
using namespace whisper;

extern "C" {

// JNI method to load the model
JNIEXPORT jlong WTJ_JNI_EXPORT(EngineNative,
                               create)(JNIEnv *env, jobject /*thiz*/,
                                       jlong engineType, jstring modelPath,
                                       jstring vocabPath,
                                       jboolean isMultilingual) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  const char *c_model_path = env->GetStringUTFChars(modelPath, nullptr);
  const char *c_vocab_path = env->GetStringUTFChars(vocabPath, nullptr);
  fprintf(stderr, "model: %s\n", c_model_path);
  fprintf(stderr, "vocab: %s\n", c_vocab_path);
  auto engine_type = static_cast<EngineType>(engineType);
  Engine *engine = create_engine(engine_type, c_model_path, c_vocab_path,
                                 isMultilingual != 0U);
  env->ReleaseStringUTFChars(modelPath, c_model_path);
  env->ReleaseStringUTFChars(vocabPath, c_vocab_path);
  return reinterpret_cast<jlong>(engine);
}

// JNI method to free the model
JNIEXPORT void WTJ_JNI_EXPORT(EngineNative, destroy)(JNIEnv * /*env*/,
                                                     jobject /*thiz*/,
                                                     jlong nativePtr) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto *engine = reinterpret_cast<Engine *>(nativePtr);
  delete engine;
}

// JNI method to transcribe audio buffer
JNIEXPORT jstring WTJ_JNI_EXPORT(EngineNative, transcribeBuffer)(
    JNIEnv *env, jobject /*thiz*/, jlong nativePtr, jfloatArray samples) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto *engine = reinterpret_cast<Engine *>(nativePtr);

  // Convert jfloatArray to std::vector<float>
  jsize len = env->GetArrayLength(samples);
  jfloat *data = env->GetFloatArrayElements(samples, nullptr);
  std::vector<float> sample_vector(data, data + len);
  env->ReleaseFloatArrayElements(samples, data, 0);

  std::string result = engine->transcribe(sample_vector);
  return env->NewStringUTF(result.c_str());
}

// JNI method to transcribe audio file
JNIEXPORT jstring WTJ_JNI_EXPORT(EngineNative,
                                 transcribeFile)(JNIEnv *env, jobject /*thiz*/,
                                                 jlong nativePtr,
                                                 jstring waveFile) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto *engine = reinterpret_cast<Engine *>(nativePtr);
  const char *c_wave_file = env->GetStringUTFChars(waveFile, nullptr);
  std::string result = engine->transcribe(c_wave_file);
  env->ReleaseStringUTFChars(waveFile, c_wave_file);
  return env->NewStringUTF(result.c_str());
}

}  // extern "C"
#undef WTJ_JNI_EXPORT
