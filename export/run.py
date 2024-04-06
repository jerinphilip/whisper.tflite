import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

if __name__ == "__main__":
    tag = "openai/whisper-tiny.en"
    # tag = "aware-ai/whisper-tiny-german"
    processor = AutoProcessor.from_pretrained(tag)
    model = WhisperForConditionalGeneration.from_pretrained(tag)

    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )

    inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
    input_features = inputs.input_features

    generated_ids = model.generate(inputs=input_features)

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(transcription)
