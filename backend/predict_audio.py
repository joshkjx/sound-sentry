import torch
import numpy as np
from pyannote.audio import Model, Pipeline
from train_classifier import BinaryClassifier
import joblib
import os
from pydub import AudioSegment
import pandas as pd
from utils import ( 
    register_hooks, load_audio, apply_tkan, 
    DATA_DIR, DATASET, SCALER_OUTPUT_FILE, MODEL_OUTPUT_FILE,
    LAYER_NAMES, DEVICE, TOP_K
)

# Toggle to enable/disable diarization (set to False for single-file prediction)
USE_DIARIZATION = True

# Loads pyannote embedding model, classifier, and setups.
# Differences from original DeepSonar:
# - Uses pretrained pyannote as the Speaker Recognition model.
def load_models():
    sr_model = Model.from_pretrained("" \
    "pyannote/embedding").to(DEVICE)  # type: ignore # No token needed offline
    sr_model.eval()
    activations_dict = {}
    hooks = register_hooks(sr_model, LAYER_NAMES, activations_dict)
    
    # Load classifier
    feature_dim = len(LAYER_NAMES) * TOP_K
    model_path = os.path.join(DATA_DIR, MODEL_OUTPUT_FILE)
    model = BinaryClassifier(input_dim=feature_dim)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE).eval()

    # Load diarization pipeline if enabled
    diar_pipeline = None
    if USE_DIARIZATION:
        diar_pipeline = Pipeline.from_pretrained("" \
        "pyannote/speaker-diarization-community-1").to(torch.device(DEVICE))  # type: ignore # No token needed offline

    return sr_model, model, activations_dict, hooks, diar_pipeline

SR_MODEL, CLASSIFIER_MODEL, ACTIVATIONS_DICT, HOOKS, DIAR_PIPELINE = load_models()
scaler_path = os.path.join(DATA_DIR, SCALER_OUTPUT_FILE)
FEATURE_SCALER = joblib.load(scaler_path)  # Load scaler

# Predicts fake probability for a single audio waveform.
def predict_single_segment(waveform: torch.Tensor) -> float:
    ACTIVATIONS_DICT.clear()
    with torch.no_grad():
        _ = SR_MODEL(waveform.unsqueeze(0))  # Add batch dim if needed
    
    tkan_features = apply_tkan(ACTIVATIONS_DICT, LAYER_NAMES, k=TOP_K)
    scaled_features = FEATURE_SCALER.transform(tkan_features.reshape(1, -1))
    
    features_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        logit = CLASSIFIER_MODEL(features_tensor)
        prob_fake = torch.sigmoid(logit).item()
    
    # Debugging prints
    print(f"Feature mean: {features_tensor.mean().item()}, "
          f"std: {features_tensor.std().item()}")
    print(f"Logit: {logit.item()}, Probability fake: {prob_fake}")
    
    return prob_fake

# Predicts on audio.
# If diarization enabled, segments speakers and predicts per segment. Aggregates all segment results.
# - Returns dict with overall result and per-segment details.
# - Flags as fake if any segment greater than threshold.
# Differences from original DeepSonar:
# - Added diarization integration.
def predict(audio_path: str, threshold: float = 0.4349) -> dict:
    results = {"overall": "Real", "details": []}
    if not USE_DIARIZATION:
        # Single file prediction
        waveform = load_audio(audio_path).to(DEVICE)
        prob_fake = predict_single_segment(waveform)
        result = "Fake" if prob_fake > threshold else "Real"
        results["details"].append({
            "segment": "full",
            "prob_fake": prob_fake,
            "result": result})
        
        if result == "Fake":
            results["overall"] = "Fake"

        return results
    
    # With diarization
    diarization = DIAR_PIPELINE(audio_path)  # type: ignore
    audio = AudioSegment.from_file(audio_path)

    for turn, speaker in diarization.exclusive_speaker_diarization:
        start_ms = int(turn.start * 1000)
        end_ms = int(turn.end * 1000)
        segment = audio[start_ms:end_ms]
        temp_file = f"temp_segment_{speaker}_{start_ms}.wav"
        segment.export(temp_file, format="wav")
        
        waveform = load_audio(temp_file).to(DEVICE)
        prob_fake = predict_single_segment(waveform)
        result = "Fake" if prob_fake > threshold else "Real"
        results["details"].append({
            "speaker": speaker,
            "time": f"{turn.start:.1f}-{turn.end:.1f}s",
            "prob_fake": prob_fake,
            "result": result
        })

        os.remove(temp_file)  # Cleanup
        
        if result == "Fake":
            results["overall"] = "Fake"

    # Load meta.csv and print label for this audio_path
    metadata_path = os.path.join(DATA_DIR, DATASET, "meta.csv")
    metadata_df = pd.read_csv(metadata_path)
    file_name = os.path.basename(audio_path)
    matching_row = metadata_df[metadata_df['file'] == file_name]
    if not matching_row.empty:
        label_str = matching_row['label'].values[0]
        print(f"Ground truth label for {file_name}: {label_str}")
    else:
        print(f"No ground truth label found for {file_name} in meta.csv")
    
    return results

# Example use
if __name__ == "__main__":
    audio_file = "data/release-in-the-wild/33.wav"
    prediction_results = predict(audio_file)
    print(prediction_results)

# Cleanup hooks
for hook in HOOKS:
    hook.remove()


