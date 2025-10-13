import torch
import numpy as np
from pyannote.audio import Model
from utils import register_hooks, load_audio, tkan_feature_for_example, LAYER_NAMES
from train_classifier import BinaryNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_models():
    sr_model = Model.from_pretrained("pyannote/embedding").to(DEVICE)
    sr_model.eval()
    activations = {}
    register_hooks(sr_model, LAYER_NAMES, activations)
    
    feature_dim = len(LAYER_NAMES) * 5
    model = BinaryNN(input_dim=feature_dim)
    model.load_state_dict(torch.load("trained_model.pth", map_location=DEVICE))
    model.to(DEVICE).eval()
    
    return sr_model, model, activations


def predict(audio_path):
    sr_model, model, activations = load_models()
    waveform = load_audio(audio_path).to(DEVICE)
    activations.clear()
    with torch.no_grad():
        _ = sr_model(waveform)
    feat = tkan_feature_for_example(activations, LAYER_NAMES, k=5)
    feat_tensor = torch.tensor(
        feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logit = model(feat_tensor)
        prob_fake = torch.sigmoid(logit).item()
    return prob_fake
