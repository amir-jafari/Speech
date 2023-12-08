import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq,AutoFeatureExtractor
common_voice = DatasetDict()

common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "bn", split="train+validation")
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "bn", split="test")

print(common_voice)

common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])

print(common_voice)

from transformers import WhisperFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained("thesven/whisper-tiny-bn-thesven")

from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained("thesven/whisper-tiny-bn-thesven")# , language="Bengali", task="transcribe"

from transformers import WhisperProcessor

processor = AutoProcessor.from_pretrained("thesven/whisper-tiny-bn-thesven")# , language="Bengali", task="transcribe"

print(common_voice["train"][0])

from datasets import Audio

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

print(common_voice["train"][0])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # Load model and processor
# model_path = "bangla-speech-processing/BanglaASR"
# feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path).to(device)
# tokenizer = WhisperTokenizer.from_pretrained(model_path).to(device)
# processor = WhisperProcessor.from_pretrained(model_path).to(device)
# model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=4)

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

import evaluate

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("thesven/whisper-tiny-bn-thesven").to(device)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

from transformers import Seq2SeqTrainingArguments
#
# training_args = Seq2SeqTrainingArguments(
#     output_dir="./outputs/whisper-small-bn",  # change to a repo name of your choice
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=16,  # increase by 2x for every 2x decrease in batch size
#     learning_rate=1e-5,
#     warmup_steps=50,
#     max_steps=400,
#     gradient_checkpointing=True,
#     fp16=True,
#     evaluation_strategy="steps",
#     per_device_eval_batch_size=8,
#     predict_with_generate=True,
#     generation_max_length=225,
#     save_steps=100,
#     eval_steps=100,
#     logging_steps=25,
#     report_to=["tensorboard"],
#     load_best_model_at_end=True,
#     metric_for_best_model="wer",
#     greater_is_better=False,
#     push_to_hub=False,
# )

# training_args = Seq2SeqTrainingArguments(
#     output_dir="./outputs/whisper-small-bn",  # change to a repo name of your choice
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=16,  # increase by 2x for every 2x decrease in batch size
#     learning_rate=1e-5,
#     warmup_steps=500,
#     max_steps=4000,
#     gradient_checkpointing=True,
#     fp16=True,
#     evaluation_strategy="steps",
#     per_device_eval_batch_size=8,
#     predict_with_generate=True,
#     generation_max_length=225,
#     save_steps=1000,
#     eval_steps=1000,
#     logging_steps=25,
#     report_to=["tensorboard"],
#     load_best_model_at_end=True,
#     metric_for_best_model="wer",
#     greater_is_better=False,
#     push_to_hub=True,
# )

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-tiny-bn-final",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=7e-5,
    warmup_steps=50,
    max_steps=2000,  # Reduced for faster training
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=100,  # Reduced for faster training
    save_steps=500,
    eval_steps=500,  # Increased for less frequent evaluation
    logging_steps=50,  # Increased for less frequent logging
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="eval_wer",
    greater_is_better=False,
    push_to_hub=False,
)

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# model.save()
# tokenizer()


trainer.train()
trainer.save_model(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)
feature_extractor.save_pretrained(training_args.output_dir)
model.save_pretrained(training_args.output_dir)

