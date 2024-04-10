# whisper (on android)

Trying to repurpose
[nyadla-sys/whisper.tflite](https://github.com/nyadla-sys/whisper.tflite) (MIT
License) to suit my style, mostly. 

My requirements include working with a [German adapted `tiny`
whisper](https://huggingface.co/aware-ai/whisper-tiny-german), so some changes
towards that. This is currently exported via a [rube-goldberg
machine](https://en.wikipedia.org/wiki/Rube_Goldberg_machine) of PyTorch ->
ONNX -> TF -> TFLite. I'm exploring avenues to simplify. 


```bash git clone --recursive https://github.com/jerinphilip/whisper.tflite.git

# Configure cmake, adjust parallel according to your system.  cmake -B build -S
.  cmake --build build --target all --parallel 28 ```

A plan is to run this locally on my android phone through the GPU or optimized
CPU. Possible to take advantage of the following?

```groovy
...
dependencies {
    ...
    implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly-SNAPSHOT'
    implementation 'org.tensorflow:tensorflow-lite-gpu:0.0.0-nightly-SNAPSHOT'
    implementation 'org.tensorflow:tensorflow-lite-support:0.0.0-nightly-SNAPSHOT'
    ...
}
```

Maybe I can use this using [this](https://stackoverflow.com/a/55144057/4565794) resource?

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
