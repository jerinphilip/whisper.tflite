import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from transformers import TFWhisperModel, WhisperFeatureExtractor
from datasets import load_dataset
from transformers import WhisperProcessor, TFWhisperForConditionalGeneration
from datasets import load_dataset
from argparse import ArgumentParser


class GenerateModel(tf.Module):
    def __init__(self, model):
        super(GenerateModel, self).__init__()
        self.model = model

    @tf.function(
        # shouldn't need static batch size, but throws exception without it (needs to be fixed)
        input_signature=[
            tf.TensorSpec((1, 80, 3000), tf.float32, name="input_features"),
        ],
    )
    def serving(self, input_features):
        outputs = self.model.generate(
            input_features,
            # max_new_tokens=223,  # change as needed
            return_dict_in_generate=True,
        )
        return {"sequences": outputs["sequences"]}


def extract_input_features(args):
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.tag)
    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )
    inputs = feature_extractor(
        ds[0]["audio"]["array"],
        sampling_rate=ds[0]["audio"]["sampling_rate"],
        return_tensors="tf",
    )
    input_features = inputs.input_features
    return input_features


def save_tf_model(args, save_prefix):
    model = TFWhisperForConditionalGeneration.from_pretrained(args.tag, from_pt=args.pt)
    tf_save = f"{save_prefix}.tf"
    model.save(tf_save)

    generate_model = GenerateModel(model=model)
    tf.saved_model.save(
        generate_model,
        tf_save,
        signatures={"serving_default": generate_model.serving},
    )
    return model, tf_save


def save_tflite_model(args, save_prefix, tf_save):
    # Convert the model
    print("Converting model v2...")
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_save)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
    ]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the model
    tflite_model_path = f"{save_prefix}.tflite"
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

    return tflite_model, tflite_model_path


class Processor:
    def __init__(self, args):
        self.args = args
        self._processor = WhisperProcessor.from_pretrained(args.data)
        self.forced_decoder_ids = self._processor.get_decoder_prompt_ids(
            language=args.lang, task="transcribe"
        )

    def process(self, generated_ids):
        transcription = self._processor.batch_decode(
            generated_ids, skip_special_tokens=self.args.skip_special_tokens
        )[0]
        print(generated_ids)
        print(transcription)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tag", type=str, default="openai/whisper-tiny")
    parser.add_argument("--data", type=str, default="openai/whisper-tiny")
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--skip-special-tokens", action="store_true")
    parser.add_argument("--pt", action="store_true")
    args = parser.parse_args()
    print(vars(args))

    input_features = extract_input_features(args)

    save_prefix = args.tag.replace("/", "-")
    tf_model, tf_save = save_tf_model(args, save_prefix)
    tflite_model, tflite_save = save_tflite_model(args, save_prefix, tf_save)
    processor = Processor(args)

    print("Generating from tf model...")
    generated_ids = tf_model.generate(input_features)
    processor.process(generated_ids)

    # loaded model... now with generate!
    print("Generating tflite model...")
    interpreter = tf.lite.Interpreter(tflite_save)
    print(interpreter.get_input_details())
    tflite_generate = interpreter.get_signature_runner()
    generated_ids = tflite_generate(input_features=input_features)["sequences"]
    processor.process(generated_ids)
