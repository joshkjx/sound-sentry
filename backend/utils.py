from pyexpat import features
import torch
import torchaudio
import numpy as np

# Constants for paths and params
DATA_DIR = "data"
DATASET = "release-in-the-wild"
FEATURES_OUTPUT_FILE = "features.npy"
LABELS_OUTPUT_FILE = "labels.npy"
SCALER_OUTPUT_FILE = "feature_scaler.pkl"
MODEL_OUTPUT_FILE = "trained_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 5  # For TKAN

# List of layers to monitor in the pyannote embedding model.
# Differences from original DeepSonar:
# - Original used Thin-ResNet layers.
# - Here we use pyannote's SincNet + TDNN layers for better pretrained speaker recognition.
LAYER_NAMES = [
    "sincnet.conv1d.0", "sincnet.conv1d.1", "sincnet.conv1d.2",
    "tdnns.0", "tdnns.3", "tdnns.6", "tdnns.9", "tdnns.12",
    "embedding"
]

# Registers forward hooks on specified layers to capture activations.
# - activations_dict: Will store layer_name -> numpy array of activations.
# - Returns list of hook handles for later removal.
# Differences from original DeepSonar:
# - Added to capture mean over time dim if needed, unlike the paper's raw flatten.
def register_hooks(model: torch.nn.Module, layer_names: list, activations_dict: dict) -> list:
    hooks = []
    def get_activation_hook(name):
        def hook(module, input, output):
            act = output.detach().cpu()
            if act.dim() > 2:
                act = act.mean(dim=2)  # Average over time for TDNN-like layers
            activations_dict[name] = act.squeeze(0).numpy()
        return hook
    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(module.register_forward_hook(
                get_activation_hook(name)))
    return hooks

# Applies Top-K Activated Neurons (TKAN) to activations.
# This sorts and takes top-k values per layer.
# - Flattens each layer's activations, sorts, pads if needed.
# - Returns concatenated vector.
# Differences from original DeepSonar:
# - No changes, but handles small arrays with padding.
def apply_tkan(activations_dict: dict, layer_names: list, k: int = 5) -> np.ndarray:
    features = []
    for name in layer_names:
        if name not in activations_dict:
            features.extend([0.0] * k)
            continue  # Skip if layer not activated
        arr = activations_dict[name].astype(np.float32).flatten()
        sorted_arr = np.sort(arr)
        # Pad with 0s if len(sorted_arr) < k
        top_k_values = np.pad(sorted_arr[-k:], (0, max(0, k - len(sorted_arr))))
        features.extend(top_k_values.tolist())
    return np.array(features, dtype=np.float32)

# Loads audio, averages to mono and resamples to 16kHz.
# Ensures mono for consistency, as pyannote expects it.
def load_audio(file_path: str) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(file_path)
    # waveform, sample_rate = torchaudio.load_with_torchcodec(file_path) # depends on ffmpeg (or sth else) version
    waveform = waveform.mean(0, keepdim=True)  # To mono
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

    # Check duration
    duration = waveform.shape[1] / 16000
    if duration < 1.0:
        # Pad with zeros to reach minimum duration
        target_samples = 16000
        padding = target_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))

    return waveform
