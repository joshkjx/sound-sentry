import torch
import torchaudio
import numpy as np

LAYER_NAMES = [
    "sincnet.conv1d.0", "sincnet.conv1d.1", "sincnet.conv1d.2",
    "tdnns.0", "tdnns.3", "tdnns.6", "tdnns.9", "tdnns.12",
    "embedding"
]

def register_hooks(model, layer_names, activations):
    hooks = []
    def get_activation_hook(name):
        def hook(module, input, output):
            a = output.detach().cpu()
            if a.dim() > 2:
                a = a.mean(dim=2)
            activations[name] = a.squeeze(0).numpy()
        return hook
    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(module.register_forward_hook(
                get_activation_hook(name)))
    return hooks


def tkan_feature_for_example(acts_dict, layer_names, k=5):
    feats = []
    for name in layer_names:
        arr = acts_dict[name].flatten()
        topk = np.sort(arr)[-k:]
        feats.extend(topk.tolist())
    return np.array(feats, dtype=np.float32)


def load_audio(path):
    waveform, sr = torchaudio.load(path)
    waveform = waveform.mean(0, keepdim=True)  
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    return waveform
