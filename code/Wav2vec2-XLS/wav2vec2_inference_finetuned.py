
from datasets import load_dataset, load_metric, Audio

# common_voice_train =load_dataset("mozilla-foundation/common_voice_11_0", "bn", split="train+validation")
common_voice_test = load_dataset("mozilla-foundation/common_voice_11_0", "bn", split="test")
first_100_rows = common_voice_test.select(indices=range(50))
# Print the first 100 rows
print(first_100_rows)
dataset = first_100_rows.cast_column("audio", Audio(sampling_rate=16000))
from transformers import Wav2Vec2CTCTokenizer
#
# tokenizer = Wav2Vec2CTCTokenizer("/home/ubuntu/bengali_asr/code/Wav2vec2-XLS/vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
# tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("/home/ubuntu/bengali_asr/code/Wav2vec2-XLS/wav2vec2-large-xlsr-bengali")
# #
# from transformers import Wav2Vec2FeatureExtractor
# #
# # feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
#
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("/home/ubuntu/bengali_asr/code/Wav2vec2-XLS/wav2vec2-large-xlsr-bengali")
from transformers import Wav2Vec2Processor

# processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
processor = Wav2Vec2Processor.from_pretrained("./wav2vec2-large-xlsr-bengali")
import torch
from transformers import Wav2Vec2ForCTC
model = Wav2Vec2ForCTC.from_pretrained("./wav2vec2-large-xlsr-bengali")# /checkpoint-4250
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

import soundfile as sf

def map_to_array(batch):
    speech, _ = sf.read(batch["path"])
    batch["speech"] = speech
    return batch

def predict(batch):
    inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)
    inputs = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    batch["predicted"] = processor.batch_decode(predicted_ids)
    return batch


from jiwer import wer

# Preprocess and predict
test_dataset = dataset.map(map_to_array)
result = test_dataset.map(predict, batched=True, batch_size=8)

# Calculate WER
wer_scores = [wer(ref, hyp) for ref, hyp in zip(result["sentence"], result["predicted"])]
average_wer = sum(wer_scores) / len(wer_scores)

print(f"Average WER: {average_wer:.2f}")


