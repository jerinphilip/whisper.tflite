import os
from argparse import ArgumentParser
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


warnings.filterwarnings("ignore")

from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import set_seed, AutoProcessor, WhisperProcessor
from pathlib import Path
import whisper
import torch
import tensorflow as tf
import onnx
from onnx import helper
import numpy as np
import argparse
import tqdm
from onnx_tf.backend import prepare as prepare_tf_from_onnx
from whisper.audio import (
    load_audio,
    log_mel_spectrogram,
    pad_or_trim,
    N_FRAMES,
    SAMPLE_RATE,
)
from transformers import AutoTokenizer


def monkey_patch_onnx(onnx_model_path, name_map):
    onnx_model = onnx.load(onnx_model_path)
    # Initialize a list to hold the new inputs
    new_inputs = []
    # Iterate over the inputs and change their names if needed
    for inp in onnx_model.graph.input:
        if inp.name in name_map:
            # Create a new ValueInfoProto with the new name
            new_inp = helper.make_tensor_value_info(
                name_map[inp.name],
                inp.type.tensor_type.elem_type,
                [dim.dim_value for dim in inp.type.tensor_type.shape.dim],
            )
            new_inputs.append(new_inp)
        else:
            new_inputs.append(inp)

    # Clear the old inputs and add the new ones
    onnx_model.graph.ClearField("input")
    onnx_model.graph.input.extend(new_inputs)

    # Go through all nodes in the model and replace the old input name with the new one
    for node in onnx_model.graph.node:
        for i, input_name in enumerate(node.input):
            if input_name in name_map:
                node.input[i] = name_map[input_name]

    # Save the renamed ONNX model
    onnx.save(onnx_model, onnx_model_path)


# Export vanilla & optimized onnx model
def export_vanilla_optimized_onnx(args):
    set_seed(args.seed)
    processor = AutoProcessor.from_pretrained(args.tag)

    # Vanilla
    model = ORTModelForSpeechSeq2Seq.from_pretrained(
        args.tag, from_transformers=True, use_cache=True
    )

    tag = args.tag.replace("/", "-")
    save_path = os.path.join(args.out_dir, f"{tag}.onnx")
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    return save_path


def export_tf_from_onnx(args, onnx_save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("Using device:", device)

    # load openai->whisper(pytorch)->tiny model
    tiny_model = whisper.load_model("tiny")

    # Export to onnx format
    tag = args.tag.replace("/", "-")
    encoder_tag = f"{tag}.encoder"
    encoder_onnx_save_path = os.path.join(args.out_dir, encoder_tag)
    torch.onnx.export(
        tiny_model.encoder, torch.randn(1, 80, 3000).to(device), encoder_onnx_save_path
    )

    encoder_tf_save_path = os.path.join(args.out_dir, f"{encoder_tag}.tf")
    monkey_patch_onnx(encoder_onnx_save_path, {"x.1": "x_1"})
    encoder_onnx_model = onnx.load(encoder_onnx_save_path)
    encoder_tf = prepare_tf_from_onnx(encoder_onnx_model)
    encoder_tf.export_graph(encoder_tf_save_path)

    decoder_tag = f"{tag}.decoder"
    decoder_onnx_save_path = os.path.join(args.out_dir, decoder_tag)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.onnx.export(
        tiny_model.decoder,
        (
            torch.tensor([[50258, 50259, 50359, 50363]]).to(device),  # sample input ids
            torch.randn(1, 1500, 384).to(device),
        ),  # encoder outputs
        decoder_onnx_save_path,
        opset_version=10,  # opset 13 fails for me with unsupported squeeze sth
        input_names=["tokens", "hidden_states"],  # the model's input names,
        output_names=["output"],  # self-set output node name
        dynamic_axes={
            "tokens": {
                1: "toks"
            },  # variable length axes, inputs ids, tokens are index=1 and we want that dimension
            "output": {1: "toks"},
        },
    )  # variable output axes

    decoder_tf_save_path = os.path.join(args.out_dir, f"{decoder_tag}.tf")
    # monkey_patch_onnx(encoder_onnx_save_path, {"x.1": "x_1"})
    decoder_onnx_model = onnx.load(decoder_onnx_save_path)
    decoder_tf = prepare_tf_from_onnx(
        decoder_onnx_model,
        dynamic_input=["serving_default_tokens"],
        dynamic_output=["PartitionedCall"],
    )
    decoder_tf.export_graph(decoder_tf_save_path)

    return {"encoder": encoder_tf_save_path, "decoder": decoder_tf_save_path}


def export_tflite_from_tf(args, tf_save_path):
    # Convert to tflite(int8) model

    def _tflite_convert(src_path):
        converter = tf.lite.TFLiteConverter.from_saved_model(src_path)
        converter.target_spec.supported_ops = [
            # tf.lite.OpsSet.TFLITE_BUILTINS_INT8, # enable TensorFlow Lite int8 ops.
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
        ]
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,tf.lite.OpsSet.SELECT_TF_OPS]
        # converter.representative_dataset = representative_dataset
        # converter.inference_input_type = tf.int8  # or tf.uint8
        # converter.inference_output_type = tf.int8  # or tf.uint8
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        # Save the model
        tflite_save_path = src_path.replace(".tf", ".tflite")
        with open(tflite_save_path, "wb") as save_file:
            save_file.write(tflite_model)

        return tflite_save_path

    return {
        "encoder": _tflite_convert(tf_save_path["encoder"]),
        "decoder": _tflite_convert(tf_save_path["decoder"]),
    }


def infer_tflite_encoder(args, tflite_save_path):
    # Load the TFLite model and allocate tensors
    interpreter_enc = tf.lite.Interpreter(model_path=tflite_save_path["encoder"])
    interpreter_enc.allocate_tensors()

    if os.environ.get("DEBUG", False):
        print("== Input details ==")
        print("name:", interpreter_enc.get_input_details()[0]["name"])
        print("shape:", interpreter_enc.get_input_details()[0]["shape"])
        print("type:", interpreter_enc.get_input_details()[0]["dtype"])

        print("\nDUMP INPUT")
        print(interpreter_enc.get_input_details()[0])

        print("\n== Output details ==")
        print("name:", interpreter_enc.get_output_details()[0]["name"])
        print("shape:", interpreter_enc.get_output_details()[0]["shape"])
        print("type:", interpreter_enc.get_output_details()[0]["dtype"])

        print("\nDUMP OUTPUT")
        print(interpreter_enc.get_output_details()[0])

    # Get input and output tensors
    input_details = interpreter_enc.get_input_details()
    output_details = interpreter_enc.get_output_details()
    output_tensor = interpreter_enc.get_output_details()[0]["index"]

    # Test the model with random data
    input_shape = input_details[0]["shape"]
    mel_from_file = log_mel_spectrogram(args.wav_path)
    input_tensor = pad_or_trim(mel_from_file, N_FRAMES)
    input_tensor = tf.expand_dims(input_tensor, 0)

    audio = whisper.load_audio(args.wav_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio)
    mel = np.expand_dims(mel, 0)
    # input_tensor = np.array(input_tensor-128, dtype=np.int8)
    interpreter_enc.set_tensor(input_details[0]["index"], mel)

    interpreter_enc.invoke()
    encoder_output_data = interpreter_enc.get_tensor(output_tensor)
    tag = args.tag.replace("/", "-")
    encoder_intermediate_path = os.path.join(args.out_dir, f"{tag}.encoder.out")
    np.savetxt(
        encoder_intermediate_path,
        encoder_output_data.reshape((3, -1)),
        fmt="%s",
        header=str(encoder_output_data.shape),
    )

    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tflite_save_path["decoder"])
    interpreter.allocate_tensors()

    # decoder_input_ids = torch.tensor([50258, 50266, 50358, 50363])

    processor = WhisperProcessor.from_pretrained(args.data)
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=args.lang, task="transcribe"
    )

    decoder_start_token = 50258
    forced_decoder_ids = [vocab_id for idx, vocab_id in forced_decoder_ids]
    forced_decoder_ids.insert(0, decoder_start_token)

    decoder_input_ids = torch.tensor(forced_decoder_ids)
    decoder_input_ids = tf.expand_dims(decoder_input_ids, 0)

    eos_id = 50257
    input_tensor_1 = interpreter.get_input_details()[0]["index"]
    interpreter.set_tensor(input_tensor_1, encoder_output_data)

    input_tensor_2 = interpreter.get_input_details()[1]["index"]
    interpreter.resize_tensor_input(input_tensor_2, decoder_input_ids.shape)
    # Allocate memory for input and output tensors
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_tensor_2, decoder_input_ids)
    output_tensor = interpreter.get_output_details()[0]["index"]
    prompt_ids = [
        50258,  # <|startoftranscript|>
        50266,  # <|ja|>
        50358,  # <|translate|>
        50363,  # <|notimestamps|>
    ]

    tokens = forced_decoder_ids
    print(forced_decoder_ids)
    max_num_tokens = 30
    for i in range(max_num_tokens):
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_tensor)
        cleaned = np.argmax(output_data, axis=-1)
        # print(output_tensor, ": ", cleaned)
        last_token = cleaned[0, -1]
        tokens.append(last_token)
        new_value = tf.constant([last_token], dtype=tf.int64)
        new_value = tf.reshape(new_value, (1, 1))
        decoder_input_ids = tf.concat([decoder_input_ids, new_value], axis=1)
        input_tensor_2 = interpreter.get_input_details()[1]["index"]
        interpreter.resize_tensor_input(input_tensor_2, decoder_input_ids.shape)
        # Allocate memory for input and output tensors
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_tensor_2, decoder_input_ids)
        if last_token == eos_id:
            break

    model_id = args.tag
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    skip_special_tokens = True
    decoded = tokenizer.batch_decode(
        np.expand_dims(tokens, axis=0), skip_special_tokens=skip_special_tokens
    )[0]
    # print(tokens)
    print(decoded)
    print(tokens)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tag", type=str, default="openai/whisper-tiny")
    parser.add_argument("--data", type=str, default="openai/whisper-tiny")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--skip-special-tokens", action="store_true")
    parser.add_argument("--pt", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wav-path", type=str, required=True)
    parser.add_argument("--tflite-prefix", type=str, default=None)
    args = parser.parse_args()
    print(vars(args))

    if not args.tflite_prefix:
        if not os.path.exists(args.out_dir):
            abspath = os.path.abspath(args.out_dir)
            os.makedirs(args.out_dir)
            print(f"Directory {abspath} not found on system, creating.")
        onnx_save_path = export_vanilla_optimized_onnx(args)
        # print(f"Saved ONNX to {onnx_save_path}")
        tf_save_path = export_tf_from_onnx(args, onnx_save_path)
        # print(f"Saved TF Model from ONNX to {tf_save_path}")

        tflite_save_path = export_tflite_from_tf(args, tf_save_path)
        print(f"Saved tflite model from tf to {tflite_save_path}")
    else:
        tflite_save_path = {
            "encoder": f"{args.tflite_prefix}.encoder.tflite",
            "decoder": f"{args.tflite_prefix}.decoder.tflite",
        }

    infer_tflite_encoder(args, tflite_save_path)
