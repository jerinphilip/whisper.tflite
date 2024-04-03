#include <jni.h>
#include <jni_md.h>

#include <cstddef>
#include <string>
#include <vector>

#include "TFLiteEngine.h"

extern "C" {

// JNI method to create an instance of TFLiteEngine
JNIEXPORT jlong JNICALL
java_com_whispertflite_engine_whisper_engine_native_create_tf_lite_engine(
    JNIEnv * /*env*/, jobject /*thiz*/) {
  return reinterpret_cast<jlong>(new TFLiteEngine());
}

// JNI method to load the model
JNIEXPORT jint JNICALL
java_com_whispertflite_engine_whisper_engine_native_load_model(
    JNIEnv *env, jobject /*thiz*/, jlong nativePtr, jstring modelPath,
    jboolean isMultilingual) {
  auto *engine = reinterpret_cast<TFLiteEngine *>(nativePtr);
  const char *c_model_path = env->GetStringUTFChars(modelPath, nullptr);
  int result = engine->loadModel(c_model_path, isMultilingual != 0u);
  env->ReleaseStringUTFChars(modelPath, c_model_path);
  return static_cast<jint>(result);
}

// JNI method to free the model
JNIEXPORT void JNICALL
java_com_whispertflite_engine_whisper_engine_native_free_model(
    JNIEnv * /*env*/, jobject /*thiz*/, jlong nativePtr) {
  auto *engine = reinterpret_cast<TFLiteEngine *>(nativePtr);
  engine->freeModel();
  delete engine;
}

// JNI method to transcribe audio buffer
JNIEXPORT jstring JNICALL
java_com_whispertflite_engine_whisper_engine_native_transcribe_buffer(
    JNIEnv *env, jobject /*thiz*/, jlong nativePtr, jfloatArray samples) {
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
JNIEXPORT jstring JNICALL
java_com_whispertflite_engine_whisper_engine_native_transcribe_file(
    JNIEnv *env, jobject /*thiz*/, jlong nativePtr, jstring waveFile) {
  auto *engine = reinterpret_cast<TFLiteEngine *>(nativePtr);
  const char *c_wave_file = env->GetStringUTFChars(waveFile, nullptr);
  std::string result = engine->transcribeFile(c_wave_file);
  env->ReleaseStringUTFChars(waveFile, c_wave_file);
  return env->NewStringUTF(result.c_str());
}

}  // extern "C"
