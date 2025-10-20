## Setup

Tested with [In-the-Wild](https://www.kaggle.com/datasets/abdallamohamed312/in-the-wild-audio-deepfake) Audio Deepfake Dataset.

Implemented pyannote [
speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) for the speaker recognition (SR) model.

Updated DeepSonar to use pytorch over tensorflow, as the core of DeepSonar is a classifier.

```text
CS5647Project/
├── data/
│   ├── release-in-the-wild/
│   ├── feature_scaler.pkl
│   ├── features.npy
│   ├── labels.npy
│   └── trained_model.pth
├── model/
│   └── pyannote-speaker-diarization-community-1/
└── scripts/
    ├── main.py
    ├── feature_extractor.py
    ├── train_classifier.py
    ├── predict_audio.py
    └── utils.py
```

### Create conda environment

To create conda environment:

```python
conda create -n cs5647project python=3.11 -y
conda install "ffmpeg>=7,<8" libiconv libopus lame x264 x265 -y
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install torchcodec==0.7.0 pyannote.audio
```

### Global telemetry configuration

To set telemetry preferences that persist across sessions:

```python
from pyannote.audio.telemetry import set_telemetry_metrics

# disable metrics globally
set_telemetry_metrics(False, save_choice_as_default=True)
```