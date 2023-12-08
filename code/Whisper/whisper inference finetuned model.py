import torch
import librosa
import numpy as np
from datasets import load_dataset, Audio
import evaluate
import random
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq,AutoFeatureExtractor

from transformers import WhisperTokenizer, WhisperProcessor, WhisperFeatureExtractor, WhisperForConditionalGeneration
wers = []
# Function to calculate WER using evaluate
# def calculate_wer_with_evaluate(reference, hypothesis):
#     wer_metric = evaluate.load("wer")
#     wer = 100 * wer_metric.compute(predictions=[hypothesis], references=[reference])
#     return wer

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


feature_extractor = AutoFeatureExtractor.from_pretrained("./whisper-tiny-bn-final")

from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained("./whisper-tiny-bn-final")# , language="Bengali", task="transcribe"

from transformers import WhisperProcessor

processor = AutoProcessor.from_pretrained("./whisper-tiny-bn-final")# , language="Bengali", task="transcribe"

model = WhisperForConditionalGeneration.from_pretrained("./whisper-tiny-bn-final").to(device)
# Load the dataset
# Load the dataset
ds = load_dataset("mozilla-foundation/common_voice_11_0", "bn", split="test")
first_10_rows = ds.select(indices=range(10))
# Print the first 100 rows
print(first_10_rows)
ds = first_10_rows.cast_column("audio", Audio(sampling_rate=16000))

# Function to extract random rows
from jiwer import wer
# Transcribing and calculating WER
for entry in ds:
    audio_input = entry["audio"]
    reference = entry["sentence"]

    # Skip if reference is empty
    if not reference.strip():
        print("Skipping empty reference.")
        continue

    # Process the audio input
    speech_array = audio_input["array"]
    speech_array = librosa.resample(np.asarray(speech_array), orig_sr=audio_input["sampling_rate"], target_sr=16000)
    input_features = feature_extractor(speech_array, sampling_rate=16000, return_tensors="pt").input_features.to(device)

    # Generate token ids
    with torch.no_grad():
        predicted_ids = model.generate(input_features)[0]

    # Decode token ids to text
    transcription = processor.decode(predicted_ids, skip_special_tokens=True)

    # Skip if transcription is empty
    if not transcription.strip():
        print("Skipping empty transcription.")
        continue

    # Calculate WER
    wer1 = wer(reference, transcription)
    wers.append(wer1)
    print(f"Reference: {reference}, Transcription: {transcription}, WER: {wer1}")

wers = np.array(wers)
avg_wer = np.average(wers)
print("average wers", avg_wer)


#"./whisper-tiny-bn-final"