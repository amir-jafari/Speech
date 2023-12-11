
import torchaudio
import librosa

import numpy as np

import datasets
from datasets import load_dataset, DatasetDict,  Audio
import pandas as pd
import os
import glob
import librosa
import io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, accuracy_score
from transformers import WhisperModel, WhisperFeatureExtractor, AdamW
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.metrics import f1_score, classification_report, accuracy_score
OR_PATH = os.getcwd()
print(OR_PATH)
os.chdir("..")
os.chdir("..")# Change to the parent directory
print(os.getcwd())
audio_df = pd.read_excel("./excel/audio_wav.xlsx")
def get_all_full_paths(parent_directory):
    audio_file_paths = [os.path.join(parent_directory, fname) for fname in os.listdir(parent_directory) if fname.endswith('.wav')]
    file_path_dict = {os.path.basename(path): path for path in audio_file_paths}
    return file_path_dict

audio_files_directory = './data'
file_path_dict = get_all_full_paths(audio_files_directory)

audio_df['full_path'] = audio_df['path'].apply(lambda x: file_path_dict.get(x))


train_df, test_df = train_test_split(audio_df, test_size=0.2, random_state=101, stratify=audio_df["label"])

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_df, temp_df = train_test_split(audio_df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_dataset = datasets.Dataset.from_dict({"audio": train_df["full_path"].tolist(),
                                                  "labels": train_df["label"].tolist()    }
                                                 ).cast_column("audio", Audio(sampling_rate=16_000))

test_dataset  = datasets.Dataset.from_dict({"audio": test_df["full_path"].tolist(),
                                                  "labels": test_df["label"].tolist()    }
                                                 ).cast_column("audio", Audio(sampling_rate=16_000))

eval_dataset = datasets.Dataset.from_dict({"audio": val_df["full_path"].tolist(),
                                                  "labels": val_df["label"].tolist()  }
                                                 ).cast_column("audio", Audio(sampling_rate=16_000))

print(train_dataset)
print(eval_dataset)
#
# # We need to specify the input and output column
# input_column = "path"
output_column = "labels"
#
# # we need to distinguish the unique labels in our SER dataset
label_list = train_dataset.unique(output_column)
label_list.sort()  # Let's sort it for determinism
num_labels = len(label_list)
# print(f"A classification problem with {num_labels} classes: {label_list}")
# # .... till here .......
from transformers import AutoConfig, Wav2Vec2Processor, AutoProcessor

model_name_or_path = OR_PATH+"/wav2vec2-large-xlsr-bengali"
pooling_mode = "mean"

# config
config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=num_labels,
    label2id={label: i for i, label in enumerate(label_list)},
    id2label={i: label for i, label in enumerate(label_list)},
    finetuning_task="wav2vec2_clf",
)
setattr(config, 'pooling_mode', pooling_mode)

processor = AutoProcessor.from_pretrained(model_name_or_path,)
target_sampling_rate = processor.feature_extractor.sampling_rate
print(f"The target sampling rate: {target_sampling_rate}")

# def speech_file_to_array_fn(path):
#     speech_array, sampling_rate = torchaudio.load(path)
#     resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
#     speech = resampler(speech_array).squeeze()
#     # Convert to numpy array if necessary
#     if not isinstance(speech, np.ndarray):
#         speech = speech.numpy()
#     return speech

def label_to_id(label, label_list):
    return label_list.index(label) if label in label_list else -1

def preprocess_function(examples):
    speech_arrays = [audio['array'] for audio in examples["audio"]]
    target_list = [label_to_id(label, label_list) for label in examples[output_column]]

    result = processor(speech_arrays, sampling_rate=target_sampling_rate)
    result["labels"] = target_list # Changed to a list of integers

    return result

train_dataset = train_dataset.map(
    preprocess_function,
    batch_size=1,
    batched=True
)
eval_dataset = eval_dataset.map(
    preprocess_function,
    batch_size=1,
    batched=True
)

# idx = 0
# print(f"Training input_values: {train_dataset[idx]['input_values']}")
# print(f"Training attention_mask: {train_dataset[idx]['attention_mask']}")
# print(f"Training labels: {train_dataset[idx]['labels']} ")

"""Great, now we've successfully read all the audio files, resampled the audio files to 16kHz, and mapped each audio to the corresponding label.

## Model

Before diving into the training part, we need to build our classification model based on the merge strategy.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from transformers.file_utils import ModelOutput


@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)


class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch

import transformers
from transformers import Wav2Vec2Processor


@dataclass
class DataCollatorCTCWithPadding:

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [feature["labels"] for feature in features]

        d_type = torch.long if isinstance(label_features[0], int) else torch.float

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(label_features, dtype=d_type)

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

"""Next, the evaluation metric is defined. There are many pre-defined metrics for classification/regression problems, but in this case, we would continue with just **Accuracy** for classification and **MSE** for regression. You can define other metrics on your own."""

is_regression = False

import numpy as np
from transformers import EvalPrediction


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

    if is_regression:
        return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
    else:
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

"""Now, we can load the pretrained XLSR-Wav2Vec2 checkpoint into our classification model with a pooling strategy."""

model = Wav2Vec2ForSpeechClassification.from_pretrained(
    model_name_or_path,
    config=config,
)


model.freeze_feature_extractor()



# from google.colab import drive

# drive.mount('/gdrive')
outputdir = OR_PATH + "/wav2vec2-xlsr-bengali-classification"
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir=outputdir,
    # output_dir="/content/gdrive/MyDrive/wav2vec2-xlsr-greek-speech-emotion-recognition"
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=1.0,
    fp16=True,
    save_steps=10,
    eval_steps=10,
    logging_steps=10,
    learning_rate=1e-4,
    save_total_limit=2,
)

"""For future use we can create our training script, we do it in a simple way. You can add more on you own."""

from typing import Any, Dict, Union

import torch
from packaging import version
from torch import nn

from transformers import (
    Trainer)

# class CTCTrainer(Trainer):
#     def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
#         """
#         Perform a training step on a batch of inputs.
#
#         Subclass and override to inject custom behavior.
#
#         Args:
#             model (:obj:`nn.Module`):
#                 The model to train.
#             inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
#                 The inputs and targets of the model.
#
#                 The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
#                 argument :obj:`labels`. Check your model's documentation for all accepted arguments.
#
#         Return:
#             :obj:`torch.Tensor`: The tensor with training loss on this batch.
#         """
#
#         model.train()
#         inputs = self._prepare_inputs(inputs)
#
#         loss = self.compute_loss(model, inputs)
#
#         if self.args.gradient_accumulation_steps > 1:
#             loss = loss / self.args.gradient_accumulation_steps
#
#         if self.deepspeed:
#             self.deepspeed.backward(loss)
#         else:
#             loss.backward()
#
#         return loss.detach()

    # Create an instance of the SimpleG5Trainer


trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.feature_extractor,
)

trainer.train()

processor.save_pretrained(training_args.output_dir)
trainer.save_model(training_args.output_dir)
# tokenizer.save_pretrained(training_args.output_dir)
# feature_extractor.save_pretrained(training_args.output_dir)
model.save_pretrained(training_args.output_dir)

import librosa
from sklearn.metrics import classification_report


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model_name_or_path = outputdir
config = AutoConfig.from_pretrained(model_name_or_path )
processor = Wav2Vec2Processor.from_pretrained(model_name_or_path )
model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path ).to(device)

# def speech_file_to_array_fn(batch):
#     # Load the audio file
#     speech_array, sampling_rate = torchaudio.load(batch["path"])
#     speech_array = speech_array.squeeze().numpy()
#
#     # Resample the audio data
#     resampled_array = librosa.resample(np.asarray(speech_array), orig_sr=sampling_rate, target_sr=processor.feature_extractor.sampling_rate)
#
#     # Update the batch
#     batch["speech"] = resampled_array
#     return batch
#
# def preprocess_function(examples):
#     speech_arrays = [audio['array'] for audio in examples["audio"]]
#     target_list = [label_to_id(label, label_list) for label in examples[output_column]]
#
#     result = processor(speech_arrays, sampling_rate=target_sampling_rate)
#     result["labels"] = target_list # Changed to a list of integers
#
#     return result

def predict(batch):
    speech_arrays = [audio['array'] for audio in batch["audio"]]
    features = processor(speech_arrays, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)

    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    batch["predicted"] = pred_ids
    return batch

# test_dataset = test_dataset.map(speech_file_to_array_fn)

result = test_dataset.map(predict, batched=True, batch_size=8)

label_names = ["Negative", "Positive"]


y_true = [name for name in result["labels"]]
y_pred = result["predicted"]

# print(y_true[:5])
# print(y_pred[:5])

print(classification_report(y_true, y_pred, target_names=label_names))
print("ACCURACY SCORE: ",accuracy_score(y_true, y_pred))
