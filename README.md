# Bengali ASR Project Structure

The repository is organized as follows:

- `code/`: Contains all the source code.
  - `whisper/`: Code related to the Whisper model.
    - `finetune_whisper.py`: Script for fine-tuning the Whisper model.
    - `whisper_inference_base_model.py`: Inference script for the base Whisper model.
    - `whisper_inference_finetuned.py`: Inference script for the fine-tuned Whisper model.
    - `whisper_classification_cnn.py`: Classification using a CNN head.
    - `whisper_classification_linear.py`: Classification using a linear head.
    - `whisper_classification_lstm.py`: Classification using an LSTM head.
  - `wav2vec2/`: Code related to the Wav2Vec2 model.
    - `finetune_wav2vec2.py`: Script for fine-tuning the Wav2Vec2 model.
    - `wav2vec2_classification.py`: Script for Wav2Vec2 classification tasks.
    - `wav2vec2_inference_base.py`: Inference script for the base Wav2Vec2 model.
    - `wav2vec2_inference_finetuned.py`: Inference script for the fine-tuned Wav2Vec2 model.
- `data/`: Directory containing the audio files in .wav format.
- `excel/`: Includes the excel sheet with transcriptions, labels, and file paths for the audio files.

## Data

The `data/` directory contains audio files that are used by the ASR system. Each audio file is a .wav file representing a different sample of spoken Bengali language.

## Excel

The `excel/` directory contains an excel sheet with the following columns:
- File path: The path to the corresponding .wav file in the `data/` directory.
- Transcription: The actual transcription of the audio file.
- Label: The sentiment label (positive or negative) of the transcription.

N.B.: All Whisper models (finetuned for transcription or classification) should be saved in the code/Whisper folder.
N.B.: All Wav2Vec2 models (finetuned for transcription or classification) should be saved in the code/Wav2Vec2 folder.

