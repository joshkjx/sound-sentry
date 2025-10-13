import torch, torchaudio
import os, numpy as np
from tqdm import tqdm
from pyannote.audio import Model
import pandas as pd
from utils import register_hooks, load_audio, tkan_feature_for_example, LAYER_NAMES

DATA_DIR = "dataset"
FEATURES_OUT = "features.npy"
LABELS_OUT = "labels.npy"
K = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

audio_dir = DATA_DIR
metadata_path = os.path.join(DATA_DIR, "meta.csv")

model = Model.from_pretrained("pyannote/embedding").to(DEVICE)
model.eval()

activations = {}
hooks = register_hooks(model, LAYER_NAMES, activations)

all_features, all_labels = [], []

# read meta.csv for labels
df = pd.read_csv(metadata_path)
for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    fname = row['file']
    label_str = row['label']
    
    path = os.path.join(audio_dir, fname)
    # skip if the .wav file is not in the folder
    if not os.path.exists(path):
        continue
    
    label = 1 if label_str == "spoof" else 0
    
    waveform = load_audio(path).to(DEVICE)
    activations.clear()
    with torch.no_grad():
        _ = model(waveform)
    
    feat = tkan_feature_for_example(activations, LAYER_NAMES, K)
    all_features.append(feat)
    all_labels.append(label)
    

all_features = np.stack(all_features)
all_labels = np.array(all_labels)

np.save(FEATURES_OUT, all_features)
np.save(LABELS_OUT, all_labels)
