# whisper (on android)

Trying to repurpose
[nyadla-sys/whisper.tflite](https://github.com/nyadla-sys/whisper.tflite) to
suit my style, mostly. 

My requirements include working with a German adapted `tiny` whisper, so some
changes along those directions as well.

* [tflite build using cmake](https://www.tensorflow.org/lite/guide/build_cmake)

```bash
git clone --recursive https://github.com/jerinphilip/whisper.tflite.git

# Configure cmake, adjust parallel according to your system.
cmake -B build -S .  
cmake --build build --target all --parallel 28
```
