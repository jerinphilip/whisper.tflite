# whisper (on android)

Trying to repurpose
[nyadla-sys/whisper.tflite](https://github.com/nyadla-sys/whisper.tflite) (MIT
License) to suit my style, mostly. 

My requirements include working with a [German adapted `tiny`
whisper](https://huggingface.co/aware-ai/whisper-tiny-german), so some changes
towards that. This is currently exported via a [rube-goldberg
machine](https://en.wikipedia.org/wiki/Rube_Goldberg_machine) of PyTorch ->
ONNX -> TF -> TFLite. I'm exploring avenues to simplify. 

There are conceptually leaner ways to accomplish what's being done here. This
repository houses an approach that plays fast and loose.

### Build

Clone sources including submodules locally.

```bash 
git clone --recursive https://github.com/jerinphilip/whisper.tflite.git
```

**`tensorflow-lite_flex`** Since some ops that are not standard are added while
doing tflite conversion, the `tensorflowlite_flex.so` library is required to be
built. The only way I found online requires using bazel.

```bash
cd deps/tensorflow

TARGETS=(
    //tensorflow/lite:libtensorflowlite.so 
    //tensorflow/lite/delegates/flex:libtensorflowlite_flex.so
)

# Build for x86-64 (monolithic?) and android_arm64
bazel build -c opt --config=monolithic  "${TARGETS[@]}"
bazel build -c opt --config=android_arm64  "${TARGETS[@]}"

# Look for tensorflowlite, tensorflowlite_flex shared objects (.so)
find -L -iname "*.so" | grep "tensorflowlite"

# Copy select .so to deps/prebuilt
tree deps/prebuilt
# deps/prebuilt
# ├── android
# │   └── arm64-v8a
# │       ├── libtensorflowlite_flex.so
# │       └── libtensorflowlite.so
# └── x86_64
#     ├── libtensorflowlite_flex.so
#     └── libtensorflowlite.so

```

You may alternatively [trust a precompiled
binary](https://github.com/nyadla-sys/whisper.tflite/tree/5eaa87f3af07e580d6b79172433e460fca017224/whisper_android/app/src/main/cpp/tf-lite-api/generated-libs)
and use it as an imported target. 

I use the following scripts to build:

```
bash scripts/build.sh # x86-64 linux desktop

# Test build encdec and monolith, also covering library.
bash scripts/run.sh <sample-no> 

# Compiles the java module and runs a driver to test.
bash scripts/run-java.sh 

# Build libraries for export into android-app
bash scripts/build-android.sh # arm-v8a android
```

Check the scripts to make a sense of specific commands required to run, these
are not intended to work out of the box and will need adapting to your setup.

### Differences

* Some extra Java code for using TFLite via Java is removed.
* Android stuff (`Context`, `Log`, `MediaRecorder` etc) are removed so that the
  JNI bridge can be tested in isolation here also providing a faster
  development feedback loop.
* The vocabulary (data) hardcoded in source is configured to be supplied externally.
* Support for [single tflite](https://colab.research.google.com/github/usefulsensors/openai-whisper/blob/main/notebooks/generate_tflite_from_whisper.ipynb)
  and [encoder-decoder](https://colab.research.google.com/github/usefulsensors/openai-whisper/blob/main/notebooks/whisper_encoder_decoder_tflite.ipynb)
  variations. There are possibly better ways to do this - my German model I could
  export only using the latter method and figured writing C++ code is shorter in
  development time.
 
### Resources

* [tflite build using cmake](https://www.tensorflow.org/lite/guide/build_cmake)
* [openai/whisper: Is there a pure tensorflow implementation of whisper ? #1953](https://github.com/openai/whisper/discussions/1953)
* [whisper-encoder-decoder-tflite.ipynb](https://colab.research.google.com/github/usefulsensors/openai-whisper/blob/main/notebooks/whisper_encoder_decoder_tflite.ipynb)
* [whisper.Android WhisperCppDemo very slow android specific transcibe times for 3s recording a res of 31227ms #1022](https://github.com/ggerganov/whisper.cpp/issues/1022)
* [On-device Whisper inference on Android mobile using whisper.tflite(quantized 40MB model) #506](https://github.com/openai/whisper/discussions/506)
* [Multilingual models converted to TFLite doesn't work #778](https://github.com/openai/whisper/discussions/778)
* [Android Docs:  MediaRecorder overview](https://developer.android.com/media/platform/mediarecorder)
* [tflite: docs](https://www.tensorflow.org/lite/guide)
* [nyadla-sys/whisper.tflite](https://github.com/nyadla-sys/whisper.tflite)
* [HuggingFace audio couse: Spectrogram](https://huggingface.co/learn/audio-course/en/chapter1/audio_data#spectrogram)
* [TF Lite enable Flex delegate C++ API (with CMake of Bazel) #57822](https://github.com/tensorflow/tensorflow/issues/57822#issuecomment-1257127667)
* [Adding Select Tf Ops to Cmake #55536](https://github.com/tensorflow/tensorflow/issues/55536#issuecomment-1286369922)
* [How can I view weights in a .tflite file?](https://stackoverflow.com/a/52174193/4565794)
* [Build TensorFlow Lite for Android](https://www.tensorflow.org/lite/android/lite_build)
* [Build TensorFlow Lite for Android: Configure WORKSPACE and .bazelrc](https://www.tensorflow.org/lite/android/lite_build)
