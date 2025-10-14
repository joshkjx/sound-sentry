import torch
import numpy as np
from pyannote.audio import Model
from utils import register_hooks, load_audio, tkan_feature_for_example, LAYER_NAMES
from train_classifier import BinaryNN
import joblib

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


SR_MODEL, MODEL, ACTIVATIONS = load_models()

def predict(audio_path):
    waveform = load_audio(audio_path).to(DEVICE)
    ACTIVATIONS.clear()
    with torch.no_grad():
        _ = SR_MODEL(waveform)
    feat = tkan_feature_for_example(ACTIVATIONS, LAYER_NAMES, k=5)
    
    # scaling logic
    scaler = joblib.load("feature_scaler.pkl")
    feat = scaler.transform(feat.reshape(1, -1))
    
    feat_tensor = torch.tensor(
        feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logit = MODEL(feat_tensor)
        prob_fake = torch.sigmoid(logit).item()
    
    # for debugging
    print("Feature mean:", feat_tensor.mean().item(),
          "std:", feat_tensor.std().item())

    logit = MODEL(feat_tensor)
    print("Logit:", logit.item())
    prob = torch.sigmoid(logit).item()
    print("Probability:", prob)
    
    
    return prob_fake
