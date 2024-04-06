from tensorflow.lite.python import schema_py_generated as schema_fb
import flatbuffers
import sys


if __name__ == "__main__":
    buf = open(sys.argv[1], "rb").read()
    model_buf = bytearray(buf)
    tflite_model = schema_fb.Model.GetRootAsModel(model_buf, 0)

    # Gets metadata from the model file.
    for i in range(tflite_model.MetadataLength()):
        meta = tflite_model.Metadata(i)
        print(meta.Name().decode("utf-8"))
        buffer_index = meta.Buffer()
        metadata = tflite_model.Buffers(buffer_index)
        byts = metadata.DataAsNumpy().tobytes()
        print(byts)
