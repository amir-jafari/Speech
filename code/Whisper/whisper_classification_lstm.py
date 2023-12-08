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
import os
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
# /home/ubuntu/bengaliasr/data/79.wav
audio_files_directory = './data'
file_path_dict = get_all_full_paths(audio_files_directory)
# print(file_path_dict)
audio_df['full_path'] = audio_df['path'].apply(lambda x: file_path_dict.get(x))

train_df, temp_df = train_test_split(audio_df, test_size=0.3, random_state=42, shuffle= True)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle= True)

train_audio_dataset = datasets.Dataset.from_dict({"audio": train_df["full_path"].tolist(),
                                                  "labels": train_df["label"].tolist()    }
                                                 ).cast_column("audio", Audio(sampling_rate=16_000))

test_audio_dataset = datasets.Dataset.from_dict({"audio": test_df["full_path"].tolist(),
                                                  "labels": test_df["label"].tolist() }
                                                 ).cast_column("audio", Audio(sampling_rate=16_000))

val_audio_dataset = datasets.Dataset.from_dict({"audio": val_df["full_path"].tolist(),
                                                  "labels": val_df["label"].tolist()  }
                                                 ).cast_column("audio", Audio(sampling_rate=16_000))


from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq,AutoFeatureExtractor,WhisperModel
import torch
model = OR_PATH+"/whisper-tiny-bn-final"
feature_extractor = AutoFeatureExtractor.from_pretrained(model)
encoder = WhisperModel.from_pretrained(model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SpeechClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, audio_data,  text_processor):
        self.audio_data = audio_data
        self.text_processor = text_processor

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, index):

      inputs = self.text_processor(self.audio_data[index]["audio"]["array"],
                                   return_tensors="pt",
                                   sampling_rate=self.audio_data[index]["audio"]["sampling_rate"])
      input_features = inputs.input_features
      decoder_input_ids = torch.tensor([[1, 1]]) * encoder.config.decoder_start_token_id

      labels = np.array(self.audio_data[index]['labels'])

      return input_features, decoder_input_ids, torch.tensor(labels)

train_dataset = SpeechClassificationDataset(train_audio_dataset,  feature_extractor)
test_dataset = SpeechClassificationDataset(test_audio_dataset,  feature_extractor)
val_dataset = SpeechClassificationDataset(val_audio_dataset,  feature_extractor)

batch_size = 8

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

SAVE_PATH = model = OR_PATH+"/best_model_lstm.pt"

import torch
import torch.nn as nn

class SpeechClassifier(nn.Module):
    def __init__(self, num_labels, encoder, hidden_size=256, num_layers=1):
        super(SpeechClassifier, self).__init__()
        self.encoder = encoder

        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.encoder.config.hidden_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        # Linear layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 4096),  # Adjust the input size to match the output of the LSTM
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels)
        )

    def forward(self, input_features, decoder_input_ids):
        # Pass input through the encoder
        outputs = self.encoder(input_features, decoder_input_ids=decoder_input_ids)
        encoder_output = outputs['last_hidden_state']

        # Pass the output through the LSTM
        lstm_output, (hidden, cell) = self.lstm(encoder_output)

        # Use the last hidden state
        last_hidden = hidden[-1]

        # Pass through the classifier
        logits = self.classifier(last_hidden)
        return logits


num_labels = 1

model = SpeechClassifier(num_labels, encoder).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-08)
criterion = nn.BCEWithLogitsLoss()


# Define the training function
def train(model, train_loader, val_loader, optimizer,  criterion, device, num_epochs):

    best_accuracy = 0.0

    for epoch in range(num_epochs):

        model.train()

        for i, batch in enumerate(train_loader):

            input_features, decoder_input_ids, labels = batch

            input_features = input_features.squeeze()
            input_features = input_features.to(device)

            decoder_input_ids = decoder_input_ids.squeeze()
            decoder_input_ids = decoder_input_ids.to(device)

            labels = labels.view(-1)
            labels = labels.to(device).float()

            optimizer.zero_grad()

            logits = model(input_features, decoder_input_ids).squeeze()


            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()

            if (i+1) % 8 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_loader)}, Train Loss: {loss.item() :.4f}')
                train_loss = 0.0

        val_loss, val_accuracy, val_f1, _ , _ = evaluate(model, val_loader, device)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), SAVE_PATH)

        print("========================================================================================")
        print(f'Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, Best Accuracy: {best_accuracy:.4f}')
        print("========================================================================================")

def evaluate(model, data_loader,  device):

    all_labels = []
    all_preds = []
    total_loss = 0.0

    with torch.no_grad():

        for i, batch in enumerate(data_loader):

          input_features, decoder_input_ids, labels = batch

          input_features = input_features.squeeze()
          input_features = input_features.to(device)

          decoder_input_ids = decoder_input_ids.squeeze()
          decoder_input_ids = decoder_input_ids.to(device)

          labels = labels.view(-1)
          labels = labels.to(device).float()

          optimizer.zero_grad()

          logits = model(input_features, decoder_input_ids).squeeze()

          loss = criterion(logits, labels)
          total_loss += loss.item()

          preds = (logits > 0.5).float()
          all_labels.append(labels.cpu().numpy())
          all_preds.append(preds.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return loss, accuracy, f1, all_labels, all_preds

num_epochs = 5
train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs)

state_dict = torch.load(SAVE_PATH)

# Create a new instance of the model and load the state dictionary
num_labels = 1
model = SpeechClassifier(num_labels, encoder).to(device)
model.load_state_dict(state_dict)

_, _, _, all_labels, all_preds = evaluate(model, test_loader, device)


print(classification_report(all_labels, all_preds))
print("ACCURACY SCORE: ",accuracy_score(all_labels, all_preds))
