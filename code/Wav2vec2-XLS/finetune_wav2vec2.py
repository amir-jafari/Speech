
from datasets import load_dataset, load_metric, Audio

common_voice_train =load_dataset("mozilla-foundation/common_voice_11_0", "bn", split="train+validation")
common_voice_test = load_dataset("mozilla-foundation/common_voice_11_0", "bn", split="test")
# common_voice = DatasetDict()
#
# common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "bn", split="train+validation")
# common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "bn", split="test")

common_voice_train = common_voice_train.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
common_voice_test = common_voice_test.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])

from datasets import ClassLabel
import random
import pandas as pd
# from IPython.display import display, HTML

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    # display(HTML(df.to_html()))

show_random_elements(common_voice_train.remove_columns(["path", "audio"]), num_examples=10)

import re
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'

def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    return batch

common_voice_train = common_voice_train.map(remove_special_characters)
common_voice_test = common_voice_test.map(remove_special_characters)

show_random_elements(common_voice_train.remove_columns(["path","audio"]))

def extract_all_chars(batch):
  all_text = " ".join(batch["sentence"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

vocab_train = common_voice_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_train.column_names)
vocab_test = common_voice_test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_test.column_names)

vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))

vocab_dict = {v: k for k, v in enumerate(vocab_list)}
# vocab_dict

vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
len(vocab_dict)

import json
with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

from transformers import Wav2Vec2CTCTokenizer

tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

from transformers import Wav2Vec2FeatureExtractor

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)


from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# common_voice_train[0]["path"]

# common_voice_train[0]["audio"]

common_voice_train = common_voice_train.cast_column("audio", Audio(sampling_rate=16_000))
common_voice_test = common_voice_test.cast_column("audio", Audio(sampling_rate=16_000))

# common_voice_train[0]["audio"]

# import IPython.display as ipd
import numpy as np
import random

rand_int = random.randint(0, len(common_voice_train)-1)

# ipd.Audio(data=common_voice_train[rand_int]["audio"]["array"], autoplay=True, rate=16000)

rand_int = random.randint(0, len(common_voice_train)-1)

print("Target text:", common_voice_train[rand_int]["sentence"])
print("Input array shape:", common_voice_train[rand_int]["audio"]["array"].shape)
print("Sampling rate:", common_voice_train[rand_int]["audio"]["sampling_rate"])

def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]

    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch

common_voice_train = common_voice_train.map(prepare_dataset, remove_columns=common_voice_train.column_names, num_proc=4)
common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names, num_proc=4)

import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

from transformers import Wav2Vec2ForCTC

model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53",
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)
#
# model.freeze_feature_extractor()
#
# model.gradient_checkpointing_enable()

from transformers import TrainingArguments

training_args = TrainingArguments(
  # output_dir="/content/gdrive/MyDrive/wav2vec2-large-xlsr-turkish-demo",
  output_dir="./wav2vec2-large-xlsr-bengali",
  group_by_length=True,
  per_device_train_batch_size=2,  # Increase if GPU allows
  gradient_accumulation_steps=16,  # Adjusted for new batch size
  evaluation_strategy="steps",
  num_train_epochs=6,  # Reduced for faster training
  fp16=True,  # Enable if supported by hardware
  save_steps=250,  # More frequent saves
  eval_steps=250,  # More frequent evaluations
  logging_steps=250,  # More frequent logging
  learning_rate=7e-4,  # Increased for faster convergence
  warmup_steps=500,  # Reduced warmup steps
  save_total_limit=3,  # Keeping limit to 1
  load_best_model_at_end=True,
  metric_for_best_model='eval_wer',
  greater_is_better=False
)


# from transformers import TrainingArguments
#
# training_args = TrainingArguments(
#     output_dir="./wav2vec2-large-xlsr-bengali",
#     group_by_length=True,
#     per_device_train_batch_size=1,  # Increase batch size if GPU memory allows
#     gradient_accumulation_steps=32,  # Adjust depending on GPU memory and batch size
#     evaluation_strategy="steps",
#     num_train_epochs=3,  # A higher number might improve model quality
#     fp16=True,  # Ensure your hardware supports FP16 for this to be effective
#     save_steps=500,  # Less frequent saves can speed up training
#     eval_steps=500,  # Adjust for quicker evaluations, but less frequent
#     logging_steps=50,  # Less frequent logging
#     learning_rate=5e-4,  # Slightly higher learning rate for faster convergence
#     warmup_steps=1000,  # Adjusted warmup steps for the learning rate
#     save_total_limit=1,  # Limit the number of saved checkpoints
#     load_best_model_at_end=True,  # Load the best model at the end of training
#     metric_for_best_model='accuracy',  # Choose a metric for the best model (if applicable)
# )


"""Now, all instances can be passed to Trainer and we are ready to start training!"""

from transformers import Trainer

# trainer = Trainer(
#     model=model,
#     data_collator=data_collator,
#     args=training_args,
#     compute_metrics=compute_metrics,
#     train_dataset=common_voice_train,
#     eval_dataset=common_voice_test,
#     tokenizer=processor.feature_extractor,
# )

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=common_voice_train,
    eval_dataset=common_voice_test,
    tokenizer=processor.feature_extractor,
)


trainer.train()
processor.save_pretrained(training_args.output_dir)
trainer.save_model(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)
feature_extractor.save_pretrained(training_args.output_dir)
model.save_pretrained(training_args.output_dir)

#
# processor.save_pretrained(training_args.output_dir)
# trainer.train()
#
# trainer.save_model(training_args.output_dir)
# tokenizer.save_pretrained(training_args.output_dir)
# feature_extractor.save_pretrained(training_args.output_dir)
# model.save_pretrained(training_args.output_dir)
#

# """## Evaluation"""
#
# eval_metrics = trainer.evaluate(metric_key_prefix="eval")
#
# print (eval_metrics)
