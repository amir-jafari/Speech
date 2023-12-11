# FOLDER STRUCTURE:

Bengali_ASR
├── code
│   ├── whisper (should contain all whisper models)
│   │   ├── finetune_whisper.py
│   │   ├── whisper_inference_base_model.py
│   │   ├── whisper_inference_finetuned.py
│   │   ├── whisper_classification_cnn.py
│   │   ├── whisper_classification_linear.py
│   │   └── whisper_classification_lstm.py
│   └── wav2vec2 (should contain all wav2vec2 models)
│       ├── finetune_wav2vec2.py
│       ├── wav2vec2_classification.py
│       ├── wav2vec2_inference_base.py
│       └── wav2vec2_inference_finetuned.py
├── data
│   └── [various .wav files]
└── excel
    └── [excel sheet containing labels, transcriptions of the .wav files, and their paths]


