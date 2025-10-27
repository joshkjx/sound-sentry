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
    DATA_DIR, DATASET, SCALER_OUTPUT_FILE,
    MODEL_OUTPUT_FILE, LAYER_NAMES, DEVICE, TOP_K
)

# Toggle to enable/disable diarization
USE_DIARIZATION = True

# Loads pyannote embedding model, classifier, and setups.
# Differences from original DeepSonar:
# - Uses pretrained pyannote as the Speaker Recognition model.
def load_models():
    sr_model = Model.from_pretrained("pyannote/embedding")
    if sr_model is None:
        raise RuntimeError("Failed to load pyannote/embedding model")
    sr_model = sr_model.to(DEVICE)
    sr_model.eval()
    activations_dict = {}
    hooks = register_hooks(sr_model, LAYER_NAMES, activations_dict)
    
    # Load classifier with metadata
    feature_dim = len(LAYER_NAMES) * TOP_K
    model_path = os.path.join(DATA_DIR, MODEL_OUTPUT_FILE)
    model = BinaryClassifier(input_dim=feature_dim)
    checkpoint = torch.load(model_path, map_location=DEVICE)

    # Extract threshold
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        threshold = checkpoint.get('threshold', 0.5)
        print(f"Loaded threshold: {threshold:.4f} (EER: {checkpoint.get('eer', 'N/A'):.4f})")
        
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Old format (just state_dict)
        threshold = 0.5
        print(f"Old model format, using default threshold: {threshold}")
        
        model.load_state_dict(checkpoint)

    model.to(DEVICE).eval()

    # Load diarization pipeline if enabled
    diar_pipeline = None
    if USE_DIARIZATION:
        diar_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1")
        if diar_pipeline is None:
            raise RuntimeError("Failed to load pyannote/speaker-diarization-community-1 model")
        diar_pipeline = diar_pipeline.to(torch.device(DEVICE))

    return sr_model, model, threshold, activations_dict, hooks, diar_pipeline
    

SR_MODEL = CLASSIFIER_MODEL = THRESHOLD = ACTIVATIONS_DICT = HOOKS = DIAR_PIPELINE = FEATURE_SCALER = None

def initialize():
    global SR_MODEL, CLASSIFIER_MODEL, THRESHOLD, ACTIVATIONS_DICT, HOOKS, DIAR_PIPELINE, FEATURE_SCALER
    SR_MODEL, CLASSIFIER_MODEL, THRESHOLD, ACTIVATIONS_DICT, HOOKS, DIAR_PIPELINE = load_models()
    scaler_path = os.path.join(DATA_DIR, SCALER_OUTPUT_FILE)
    FEATURE_SCALER = joblib.load(scaler_path)

# Predicts fake probability for a single audio waveform.
def predict_single_segment(waveform: torch.Tensor) -> float:
    ACTIVATIONS_DICT.clear()

    print(f"Waveform shape: {waveform.dim()}")
    
    # Ensure waveform has correct shape: [batch, channel, samples]
    if waveform.dim() == 1:
        # [samples] → [1, 1, samples]
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.dim() == 2:
        # [1, samples] → [1, 1, samples]
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 3:
        pass  # Already [batch, channel, samples]
    else:
        raise ValueError(f"Unexpected waveform shape: {waveform.shape}")
    
    # Forward pass to get activations
    with torch.no_grad():
        _ = SR_MODEL(waveform)
    
    # Extract TKAN features
    tkan_features = apply_tkan(ACTIVATIONS_DICT, LAYER_NAMES, k=TOP_K)
    scaled_features = FEATURE_SCALER.transform(tkan_features.reshape(1, -1))
    
    # Convert to tensor and predict
    features_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        logit = CLASSIFIER_MODEL(features_tensor)
        prob_fake = torch.sigmoid(logit).item()

    print(f"Logit: {logit.item():.4f}, Probability fake: {prob_fake:.6f}")

    # # Add small noise (simulates different recording conditions)
    # noise = torch.randn_like(waveform) * 0.005
    # waveform_noisy = waveform + noise
    # waveform_noisy = waveform_noisy / (torch.max(torch.abs(waveform_noisy)) + 1e-8)
    
    # ACTIVATIONS_DICT.clear()
    # with torch.no_grad():
    #     _ = SR_MODEL(waveform_noisy)

    return prob_fake

# Predicts on the supplied audio (inference).
# If diarization enabled, segments speakers and predicts per segment.
# Aggregates all segment results.
# - Returns dict with overall result and per-segment details.
# - Flags as fake if any segment greater than classification threshold
# Differences from original DeepSonar:
# - Added diarization integration.
def predict(audio_path: str) -> dict:
    if SR_MODEL is None:
        initialize()
    results = {"overall": "Real", "details": []}

    if not os.path.exists(audio_path):
        print(f"Error: File not found: {audio_path}")
        return results
    
    if not USE_DIARIZATION or DIAR_PIPELINE is None:
        # Single file prediction
        print(f"Processing: {audio_path}")
        waveform = load_audio(audio_path).to(DEVICE)
        prob_fake = predict_single_segment(waveform)
        result = "Fake" if prob_fake > THRESHOLD else "Real"
        
        results["details"].append({
            "segment": "full",
            "prob_fake": prob_fake,
            "result": result
        })
        
        if result == "Fake":
            results["overall"] = "Fake"
        
        return results
    
    # With diarization
    print(f"Processing with diarization: {audio_path}")
    
    try:
        diarization = DIAR_PIPELINE(audio_path)
        audio = AudioSegment.from_file(audio_path)
        
        segment_count = 0
        for turn, speaker in diarization.exclusive_speaker_diarization:
            start_ms = int(turn.start * 1000)
            end_ms = int(turn.end * 1000)
            
            # Skip very short segments
            if (end_ms - start_ms) < 1000:
                continue

            segment_count += 1
            
            segment = audio[start_ms:end_ms]
            temp_file = f"temp_segment_{speaker}_{start_ms}.wav"
            segment.export(temp_file, format="wav")
            
            try:
                print(f"Processing Segment {segment_count}: {speaker} [{turn.start:.1f}-{turn.end:.1f}s]")

                waveform = load_audio(temp_file).to(DEVICE)
                prob_fake = predict_single_segment(waveform)
                result = "Fake" if prob_fake > THRESHOLD else "Real"

                print(f"→ Result: {result} (p={prob_fake:.4f})\n")  # Summary at the end
                
                results["details"].append({
                    "speaker": speaker,
                    "time": f"{turn.start:.1f}-{turn.end:.1f}s",
                    "prob_fake": prob_fake,
                    "result": result
                })
                
                if result == "Fake":
                    results["overall"] = "Fake"
                    
            except Exception as e:
                print(f"Error processing segment {speaker}: {e}")
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        print(f"Processed {segment_count} segments")
        
    except Exception as e:
        print(f"Diarization failed: {e}")
        print("Falling back to single file prediction...")
        
        waveform = load_audio(audio_path).to(DEVICE)
        prob_fake = predict_single_segment(waveform)
        result = "Fake" if prob_fake > THRESHOLD else "Real"

        results["details"].append({
            "segment": "full",
            "prob_fake": prob_fake,
            "result": result
        })

        if result == "Fake":
            results["overall"] = "Fake"
    
    # Get ground truth if available
    metadata_path = os.path.join(DATA_DIR, DATASET, "meta.csv")
    metadata_df = pd.read_csv(metadata_path)
    file_name = os.path.basename(audio_path)
    matching_row = metadata_df[metadata_df['file'] == file_name]
    if not matching_row.empty:
        label_str = matching_row['label'].values[0]
        results["ground_truth"] = label_str
        
        # Check if prediction is correct
        predicted = results["overall"]
        actual = "Fake" if label_str == "spoof" else "Real"
        is_correct = (predicted == actual)
        results["correct"] = is_correct
    
    return results

# Example usage
if __name__ == "__main__":
    # Test file
    audio_file = os.path.join(DATA_DIR, DATASET, "33.wav")
    
    if not os.path.exists(audio_file):
        print(f"Test file not found: {audio_file}")
        print("Please provide a valid audio file path")
    else:
        print(f"\nTesting on: {audio_file}\n")
        
        # Run prediction
        prediction_results = predict(audio_file)
        
        # Display results
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Overall: {prediction_results['overall']}")
        
        if 'ground_truth' in prediction_results:
            print(f"Ground Truth: {prediction_results['ground_truth']}")
            print(f"Correct: {'✓' if prediction_results.get('correct', False) else '✗'}")
        
        print(f"\nDetails:")
        for detail in prediction_results['details']:
            speaker = detail.get('speaker', 'N/A')
            time = detail.get('time', 'full')
            prob = detail['prob_fake']
            result = detail['result']
            print(f"  [{speaker}] {time}: {result} (p={prob:.4f})")

# Cleanup hooks
if HOOKS:
    for hook in HOOKS:
        hook.remove()
