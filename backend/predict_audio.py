import torch
import torchaudio
import numpy as np
from pyannote.audio import Model, Pipeline
from .train_classifier import BinaryClassifier
import joblib
import os
from pydub import AudioSegment
import pandas as pd
from .utils import ( 
    register_hooks, load_audio, apply_tkan,
    DATA_DIR, MODEL_DIR, DATASET, SCALER_OUTPUT_FILE,
    MODEL_OUTPUT_FILE, LAYER_NAMES, DEVICE, TOP_K
)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from omegaconf.listconfig import ListConfig
from omegaconf.base import ContainerMetadata
from typing import Optional
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"lightning\.pytorch\..*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pyannote\.audio\..*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"numpy\._core\..*")

# Allowlist required globals for safe checkpoint loading
torch.serialization.add_safe_globals([EarlyStopping, ModelCheckpoint, ListConfig, ContainerMetadata])

VERBOSE = True

# Duration-based threshold ADJUSTMENTS (relative to base EER threshold)
# These are multipliers/offsets applied to the model's trained EER threshold
DURATION_ADJUSTMENTS = {
    'very_short': {
        'max_duration': 3.0,
        'threshold_multiplier': 0.70,  # Use 70% of EER threshold
        'use_diarization': False
    },
    'short': {
        'max_duration': 6.0,
        'threshold_multiplier': 0.80,  # Use 80% of EER threshold
        'use_diarization': False
    },
    'normal': {
        'max_duration': float('inf'),
        'threshold_multiplier': 1.00,  # Use full EER threshold
        'use_diarization': True
    }
}

class AudioClassifier:
    # Loads pyannote embedding model, classifier, and setups.
    # Differences from original DeepSonar:
    # - Uses pretrained pyannote as the Speaker Recognition model.
    def __init__(self):
        # Load pyannote embedding model
        local_model_path = os.path.join(MODEL_DIR, "pyannote-embedding", "pytorch_model.bin")
        if os.path.exists(local_model_path):
            if VERBOSE:
                print(f"Loading local model from: {local_model_path}")
            self.sr_model = Model.from_pretrained(local_model_path, strict=False)
        else:
            if VERBOSE:
                print("Local model not found, loading from pyannote/embedding")
            self.sr_model = Model.from_pretrained("pyannote/embedding", strict=False)

        if self.sr_model is None:
            raise RuntimeError("Failed to load embedding model from local or online source")
        self.sr_model = self.sr_model.to(DEVICE)
        self.sr_model.eval()

        # Initialize activations dictionary and hooks
        self.activations_dict = {}
        self.hooks = register_hooks(self.sr_model, LAYER_NAMES, self.activations_dict)

        # Load classifier with metadata
        feature_dim = len(LAYER_NAMES) * TOP_K
        model_path = os.path.join(DATA_DIR, MODEL_OUTPUT_FILE)
        self.classifier_model = BinaryClassifier(input_dim=feature_dim)
        checkpoint = torch.load(model_path, map_location=DEVICE)

        # Extract threshold
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.default_threshold = checkpoint.get('threshold', 0.5)
            if VERBOSE:
                print(f"Loaded threshold: {self.default_threshold:.4f} (EER: {checkpoint.get('eer', 'N/A'):.4f})")
            self.classifier_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.default_threshold = 0.5
            if VERBOSE:
                print(f"Old model format, using default threshold: {self.default_threshold}")
            self.classifier_model.load_state_dict(checkpoint)
        self.classifier_model.to(DEVICE).eval()

        # Load diarization pipeline
        local_diarization_path = os.path.join(MODEL_DIR, "pyannote-speaker-diarization-community-1")
        if os.path.exists(local_diarization_path):
            if VERBOSE:
                print(f"Loading local diarization model from: {local_diarization_path}")
            self.diar_pipeline = Pipeline.from_pretrained(local_diarization_path)
        else:
            if VERBOSE:
                print("Local diarization model not found, loading from pyannote/speaker-diarization-community-1")
            self.diar_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
        
        if self.diar_pipeline is None:
            raise RuntimeError("Failed to load diarization model from local or online source")
        self.diar_pipeline = self.diar_pipeline.to(torch.device(DEVICE))

        # Load feature scaler
        scaler_path = os.path.join(DATA_DIR, SCALER_OUTPUT_FILE)
        self.feature_scaler = joblib.load(scaler_path)

    # Get threshold and diarization setting based on audio duration.
    # Applies relative adjustments to the base EER threshold.
    def get_duration_config(self, duration: float, base_threshold: float) -> dict:
        # Determine category
        if duration < DURATION_ADJUSTMENTS['very_short']['max_duration']:
            category = 'very_short'
        elif duration < DURATION_ADJUSTMENTS['short']['max_duration']:
            category = 'short'
        else:
            category = 'normal'
        
        # Apply adjustment to base threshold
        adjustment_config = DURATION_ADJUSTMENTS[category]
        adjusted_threshold = base_threshold * adjustment_config['threshold_multiplier']
        
        config = {
            'duration': duration,
            'category': category,
            'threshold': adjusted_threshold,
            'use_diarization': adjustment_config['use_diarization'],
            'base_threshold': base_threshold,
            'multiplier': adjustment_config['threshold_multiplier']
        }
        
        if VERBOSE:
            print(f"  Duration: {duration:.1f}s -> {category}")
            print(f"  Base threshold: {base_threshold:.4f}")
            print(f"  Multiplier: {adjustment_config['threshold_multiplier']:.2f}")
            print(f"  Adjusted threshold: {adjusted_threshold:.4f}")
            print(f"  Diarization: {adjustment_config['use_diarization']}")
        
        return config

    # Predicts fake probability for a single audio waveform.
    def predict_single_segment(self, waveform: torch.Tensor) -> float:
        self.activations_dict.clear()
        
        # Ensure waveform is 3D [1, 1, samples]
        waveform = waveform.unsqueeze(0)
        
        # Forward pass to get activations
        with torch.no_grad():
            if self.sr_model is not None:
                _ = self.sr_model(waveform)
        
        # Extract TKAN features
        tkan_features = apply_tkan(self.activations_dict, LAYER_NAMES, k=TOP_K)
        scaled_features = self.feature_scaler.transform(tkan_features.reshape(1, -1))
        
        # Convert to tensor and predict
        features_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            logit = self.classifier_model(features_tensor)
            prob_fake = torch.sigmoid(logit).item()

        return prob_fake
    
    # Get ground truth label from metadata if available.
    def get_ground_truth(self, audio_path: str) -> Optional[str]:
        metadata_path = os.path.join(DATA_DIR, DATASET, "meta.csv")      
        metadata_df = pd.read_csv(metadata_path)
        file_name = os.path.basename(audio_path)
        matching_row = metadata_df[metadata_df['file'] == file_name]
        if not matching_row.empty:
            return matching_row['label'].values[0]
        else:
            return None
        
    # Helper to predict on a waveform and format the result.
    def predict_and_format(self, waveform: torch.Tensor, threshold: float) -> dict:
        prob_fake = self.predict_single_segment(waveform)
        confidence = abs(prob_fake - threshold)
        result = "Fake" if prob_fake > threshold else "Real"
        
        return {
            'prob_fake': prob_fake,
            'confidence': confidence,
            'result': result
        }
    
    # Predict on entire audio file without diarization.
    # Modifies results dict in place.
    def predict_single_file(self, waveform: torch.Tensor, threshold: float, results: dict):
        try:
            pred_info = self.predict_and_format(waveform, threshold)
            
            results["details"].append({
                "segment": "full"
            })
            
            results["mean_probability"] = pred_info['prob_fake']
            results["max_probability"] = pred_info['prob_fake']
            results["min_probability"] = pred_info['prob_fake']
            results["confidence"] = pred_info['confidence']
            
            if pred_info['result'] == "Fake":
                results["overall"] = "Fake"
                
        except Exception as e:
            print(f"Error analyzing audio: {e}")

    # Predict using speaker diarization.
    # Modifies results dict in place.
    def predict_with_diarization(self, audio_path: str, threshold: float, results: dict):
        if self.diar_pipeline is None:
            raise RuntimeError("Diarization pipeline is not initialized")
        diarization = self.diar_pipeline(audio_path)
        audio = AudioSegment.from_file(audio_path)
        
        segment_count = 0
        segment_probs = []
        segment_confidences = []

        for turn, speaker in diarization.exclusive_speaker_diarization:
            start_ms = int(turn.start * 1000)
            end_ms = int(turn.end * 1000)

            # Skip very short segments
            if (end_ms - start_ms) < 1000:
                continue
            
            segment_count += 1
            segment = audio[start_ms:end_ms]
            temp_file = f"temp_segment_{speaker}_{start_ms}.wav"
            
            try:
                segment.export(temp_file, format="wav")
                
                if VERBOSE:
                    print(f"Processing Segment {segment_count}: {speaker} [{turn.start:.1f}-{turn.end:.1f}s]")
                
                waveform, duration = load_audio(temp_file)
                waveform = waveform.to(DEVICE)
                pred_info = self.predict_and_format(waveform, threshold)
                
                if VERBOSE:
                    print(f"-> Result: {pred_info['result']} (p={pred_info['prob_fake']:.4f}, conf={pred_info['confidence']:.4f})")

                results["details"].append({
                    "speaker": speaker,
                    "time": f"{turn.start:.1f}-{turn.end:.1f}s",
                })

                segment_probs.append(pred_info['prob_fake'])
                segment_confidences.append(pred_info['confidence'])

                if pred_info['result'] == "Fake":
                    results["overall"] = "Fake"

            except Exception as e:
                print(f"Error processing segment {speaker}: {e}")
                segment_count -= 1
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

        # Calculate overall statistics
        results["mean_probability"] = np.mean(segment_probs)
        results["max_probability"] = np.max(segment_probs)
        results["min_probability"] = np.min(segment_probs)
        results["confidence"] = np.mean(segment_confidences)

        if VERBOSE:
            print(f"Processed {segment_count} segments")

    # Predicts on the supplied audio (inference).
    # If diarization enabled, segments speakers and predicts per segment.
    # Aggregates all segment results.
    # - Returns dict with overall result and per-segment details.
    # - Flags as fake if any segment greater than classification threshold
    # Differences from original DeepSonar:
    # - Added diarization integration.
    def predict(self, audio_path: str) -> dict:
        results = {
            "overall": "Real",
            "mean_probability": 0.0,
            "max_probability": 0.0,
            "min_probability": 0.0,
            "confidence": 0.0,
            "details": []
        }

        # Load the audio wav file
        waveform, duration = load_audio(audio_path)
        waveform = waveform.to(DEVICE)
        
        # Get duration-aware configuration
        duration_config = self.get_duration_config(duration, self.default_threshold)
        threshold = duration_config['threshold']
        use_diarization = duration_config['use_diarization']
        results["duration_config"] = {
            'duration': duration_config['duration'],
            'category': duration_config['category'],
            'base_threshold': duration_config['base_threshold'],
            'multiplier': duration_config['multiplier'],
            'threshold': threshold,
            'use_diarization': use_diarization
        }

        # Choose prediction strategy based on diarization
        if use_diarization:
            try:
                self.predict_with_diarization(audio_path, threshold, results)
            except Exception as e:
                print(f"Diarization failed: {e}")
                print("Falling back to single file prediction...")
                use_diarization = False

        if not use_diarization:
            self.predict_single_file(waveform, threshold, results)

        # Add ground truth if available
        ground_truth = self.get_ground_truth(audio_path)
        if ground_truth is not None:
            results["ground_truth"] = ground_truth
            predicted = results["overall"]
            actual = "Fake" if ground_truth == "spoof" else "Real"
            results["correct"] = (predicted == actual)

        return results

    # Remove hooks to free resources.
    def cleanup(self):
        if self.hooks:
            for hook in self.hooks:
                hook.remove()
            self.hooks = None
