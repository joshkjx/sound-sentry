import torch
import os
import numpy as np
from tqdm import tqdm
from pyannote.audio import Model, Pipeline
import pandas as pd
from .utils import (
    register_hooks, load_audio, apply_tkan, DATA_DIR,
    DATASET, FEATURES_OUTPUT_FILE, LABELS_OUTPUT_FILE,
    LAYER_NAMES, DEVICE, TOP_K
)

# Load pyannote model
# Differences from original DeepSonar:
# - Uses pyannote's pretrained embedding model instead of training Thin-ResNet from scratch.
model = Model.from_pretrained("pyannote/embedding").to(DEVICE)  # type: ignore # No token needed offline
model.eval()

# Setup activations dict and hooks
activations_dict = {}
hooks = register_hooks(model, LAYER_NAMES, activations_dict)

# Prepare to collect features and labels
all_features = []
all_labels = []

# Load metadata
metadata_path = os.path.join(DATA_DIR, DATASET, "meta.csv")
metadata_df = pd.read_csv(metadata_path)

# Extract features and labels for each audio file
for idx, row in tqdm(metadata_df.iterrows(), total=metadata_df.shape[0]):
    file_name = row['file']
    label_str = row['label']

    audio_path = os.path.join(DATA_DIR, DATASET, file_name)
    # skip if the .wav file is not in the folder
    if not os.path.exists(audio_path):
        print(f"Skipping missing file: {audio_path}")
        continue
    
    # Map label: spoof=1 (fake), bona_fide=0 (real)
    label = 1 if label_str == "spoof" else 0
    
    waveform = load_audio(audio_path).to(DEVICE)
    activations_dict.clear()  # Reset for each sample
    with torch.no_grad():
        _ = model(waveform)  # Forward pass to trigger hooks
    
    tkan_features = apply_tkan(activations_dict, LAYER_NAMES, TOP_K)
    all_features.append(tkan_features)
    all_labels.append(label)
    
# Stack and save
all_features_array = np.stack(all_features)
all_labels_array = np.array(all_labels)

# Save to path in DATA_DIR
features_path = os.path.join(DATA_DIR, FEATURES_OUTPUT_FILE)
labels_path = os.path.join(DATA_DIR, LABELS_OUTPUT_FILE)
np.save(features_path, all_features_array)
np.save(labels_path, all_labels_array)

# Cleanup hooks to free memory
for hook in hooks:
    hook.remove()

print(f"Feature and label extraction complete. Saved to {features_path} and {labels_path}.")
