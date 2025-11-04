import torch
import numpy as np
import joblib
import os
import pandas as pd
import warnings
import logging
import noisereduce as nr
from pyannote.audio import Model, Pipeline, Inference
from train_classifier import BinaryClassifier
from utils import ( 
    register_hooks, load_audio, apply_tkan,
    DATA_DIR, MODEL_DIR, DATASET, SCALER_OUTPUT_FILE,
    MODEL_OUTPUT_FILE, LAYER_NAMES, DEVICE, TOP_K,
    BEST_HIDDEN_SIZE, BEST_DROPOUT_RATE
)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from omegaconf.listconfig import ListConfig
from omegaconf.base import ContainerMetadata
from typing import Optional
from scipy.signal import butter, filtfilt

# Set logging level for lightning to ERROR to reduce verbosity
logging.getLogger("lightning.pytorch.utilities.migration").setLevel(logging.ERROR)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"lightning\.pytorch\..*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pyannote\.audio\..*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"numpy\._core\..*")

# Allowlist required globals for safe checkpoint loading
torch.serialization.add_safe_globals([EarlyStopping, ModelCheckpoint, ListConfig, ContainerMetadata])

VERBOSE = True

# Noise reduction settings
ENABLE_NOISE_REDUCTION = True

# VAD parameters based on pyannote/voice-activity-detection
VAD_HYPER_PARAMETERS = {
    "onset": 0.90,              # Much stricter for cafe noise
    "offset": 0.55,
    "min_duration_on": 0.3,
    "min_duration_off": 0.3
}

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
    # - Added noise reduction preprocessing
    # - Optimized VAD parameters for cafe noise
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
        self.classifier_model = BinaryClassifier(input_dim=feature_dim,
                                                 hidden_size=BEST_HIDDEN_SIZE,
                                                 dropout_rate=BEST_DROPOUT_RATE)
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
        self.diar_pipeline = self.diar_pipeline.to(torch.device("cpu"))

        # Load segmentation model and setup VAD pipeline with optimized parameters
        local_seg_path = os.path.join(MODEL_DIR, "pyannote-segmentation-3", "pytorch_model.bin")
        if os.path.exists(local_seg_path):
            if VERBOSE:
                print(f"Loading local model from: {local_seg_path}")
            self.vad_model = Model.from_pretrained(local_seg_path, strict=False)
        else:
            if VERBOSE:
                print("Local model not found, loading from pyannote/segmentation-3.0")
            self.vad_model = Model.from_pretrained("pyannote/segmentation-3.0", strict=False)

        if self.vad_model is None:
            raise RuntimeError("Failed to load segmentation model from local or online source")
        self.vad_model = self.vad_model.to("cpu")
        self.vad_model.eval()

        # Setup VAD using inference with optimized threshold
        self.vad_infer = Inference(self.vad_model, window="whole", device=torch.device("cpu"))
        
        # Store optimized threshold (higher = less sensitive)
        self.vad_speech_threshold = VAD_HYPER_PARAMETERS['onset']
        
        if VERBOSE:
            print(f"VAD threshold: {self.vad_speech_threshold}")
        
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
    
    # Differences from original DeepSonar:
    # Apply high-pass filter to remove low-frequency cafe rumble and background noise.
    # data: Audio data (numpy array)
    # cutoff: Cutoff frequency in Hz (default: 200 Hz removes most cafe rumble)
    # fs: Sample rate
    # order: Filter order (higher = sharper cutoff)
    def highpass_filter(self, data, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False) # type: ignore
        return filtfilt(b, a, data)

    # Preprocess audio waveform in-memory to reduce background noise.
    # waveform: Audio waveform tensor [channels, samples]
    # method: Noise reduction method ("highpass", "noisereduce", "none")
    # sample_rate: Sample rate of audio
    def preprocess_waveform(self, waveform):
        # Convert to numpy
        audio_np = waveform.cpu().numpy()
        
        # Handle mono/stereo
        if audio_np.ndim == 1:
            audio_np = audio_np.reshape(1, -1)
        
        # Process each channel
        processed_channels = []
        for channel in audio_np:
            processed = self.highpass_filter(channel, cutoff=200, fs=16000)
            processed_channels.append(processed)
        
        # Convert back to tensor
        processed_np = np.array(processed_channels)
        processed_tensor = torch.from_numpy(processed_np).float()
        
        return processed_tensor

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
        
        # Determine result based on threshold
        result = "Fake" if prob_fake >= threshold else "Real"
        
        # Calculate confidence as distance from threshold
        confidence = abs(prob_fake - threshold)
        
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
    def predict_with_diarization(self, audio_path: str, waveform: torch.Tensor,
                                 duration: float, threshold: float, results: dict):
        try:
            if self.diar_pipeline is None:
                raise RuntimeError("Diarization pipeline is not initialized")
            diarization  = self.diar_pipeline(audio_path)
            
            segment_count = 0
            segment_probs = []
            segment_confidences = []

            for turn, speaker in diarization.exclusive_speaker_diarization:
                start = int(turn.start * 16000)
                end = int(turn.end * 16000)

                # Skip very short segments
                if (end - start) < 16000:
                    continue
                
                segment_waveform = waveform[:, start:end].to(DEVICE)

                if VERBOSE:
                    print(f"Processing Segment {segment_count+1}: "
                        f"{speaker} [{turn.start:.1f}-{turn.end:.1f}s]")
                    
                pred_info = self.predict_and_format(segment_waveform, threshold)

                if VERBOSE:
                    print(f"-> Result: {pred_info['result']} "
                        f"(p={pred_info['prob_fake']:.4f}, "
                        f"conf={pred_info['confidence']:.4f})")

                results["details"].append({
                    "speaker": speaker,
                    "time": f"{turn.start:.1f}-{turn.end:.1f}s",
                    "start_time": float(turn.start),
                    "end_time": float(turn.end),
                    "result": pred_info['result'],
                    "probability": float(pred_info['prob_fake']),
                    "confidence": float(pred_info['confidence'])
                })

                segment_probs.append(pred_info['prob_fake'])
                segment_confidences.append(pred_info['confidence'])

                if pred_info['result'] == "Fake":
                    results["overall"] = "Fake"
                    
                segment_count += 1

            # Calculate overall statistics
            if segment_probs:
                results["mean_probability"] = float(np.mean(segment_probs))
                results["max_probability"] = float(np.max(segment_probs))
                results["min_probability"] = float(np.min(segment_probs))
                results["confidence"] = float(np.mean(segment_confidences))

            if VERBOSE:
                print(f"Processed {segment_count} segments "
                    f"(audio length {duration:.1f}s)")
            
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Diarization failed: {e}")
            results["message"] = f"Diarization error: {e}"

    # Uses pyannote segmentation-3.0 model to detect presence of speech.
    # Now uses higher threshold for cafe noise robustness.
    # Expects waveform to already be preprocessed if needed
    def has_speech(self, waveform: torch.Tensor) -> bool:
        try:
            # Additional check: spectral characteristics
            # Background noise tends to have more uniform spectral energy
            audio_np = waveform.cpu().numpy().flatten()
            
            # Check spectral flatness (high = noise-like, low = tonal/speech)
            # This helps distinguish pure noise from speech in noise
            fft = np.fft.rfft(audio_np)
            magnitude = np.abs(fft)
            geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
            arithmetic_mean = np.mean(magnitude)
            spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
            
            if VERBOSE:
                print(f"  Spectral flatness: {spectral_flatness:.3f} (>0.8 suggests noise)")
            # If spectral flatness is too high, likely just noise
            if spectral_flatness > 0.8:
                if VERBOSE:
                    print("  Rejected: too noise-like (high spectral flatness)")
                return False

            # Move to CPU for VAD
            wf = waveform.detach().to("cpu")

            # Run inference
            scores = self.vad_infer({"waveform": wf, "sample_rate": 16000})

            # Extract posterior matrix
            scores_obj = scores[0] if isinstance(scores, tuple) else scores
            post = np.asarray(scores_obj.data) if hasattr(scores_obj, "data") else np.asarray(scores_obj)
            
            # Speech class = column 1 for segmentation-3.0
            if post.ndim == 2 and post.shape[1] >= 2:
                speech_prob = post[:, 1]
            else:
                # fallback: single-prob track
                speech_prob = post.reshape(-1)
            
            # Use adjusted threshold
            speech_ratio = float(np.mean(speech_prob > self.vad_speech_threshold))
            
            if VERBOSE:
                print(f"  Speech ratio (threshold={self.vad_speech_threshold}): {speech_ratio:.3f}")
            
            # Check for continuous background noise vs actual speech
            # Use audio envelope (amplitude variation) rather than just speech probability
            # Speech has natural amplitude modulation, background noise is more constant
            if speech_ratio > 0.95:
                # Calculate envelope variation in the audio signal
                audio_np = waveform.cpu().numpy().flatten()
                
                # Compute short-time energy in 50ms windows
                window_size = int(0.05 * 16000)  # 50ms at 16kHz
                num_windows = len(audio_np) // window_size
                
                if num_windows > 2:
                    energies = []
                    for i in range(num_windows):
                        window = audio_np[i*window_size:(i+1)*window_size]
                        energy = np.mean(window**2)
                        energies.append(energy)
                    
                    # Calculate coefficient of variation (std/mean) of energy
                    energy_std = np.std(energies)
                    energy_mean = np.mean(energies) + 1e-10
                    energy_cv = energy_std / energy_mean
                    
                    if VERBOSE:
                        print(f"  Energy coefficient of variation: {energy_cv:.4f}")
                    
                    # Very low energy variation = likely continuous background noise
                    # Speech (real or fake) has energy modulation from phonemes/words
                    
                    if energy_cv < 0.5:  # Low variation = background noise
                        if VERBOSE:
                            print("  Rejected: low energy variation (continuous background noise)")
                        return False
            
            return bool(speech_ratio > 0.10)

        except Exception as e:
            print(f"VAD inference failed: {e}")
            return False

    # Predicts on the supplied audio (inference).
    # If diarization enabled, segments speakers and predicts per segment.
    # Aggregates all segment results.
    # - Returns dict with overall result and per-segment details.
    # - Flags as fake if any segment greater than classification threshold
    # Differences from original DeepSonar:
    # - Added diarization integration.
    # - Added in-memory noise reduction preprocessing (for VAD only)
    # - Added optional diarization control (default: False)
    def predict(self, audio_path: str, enable_diarization: bool = False) -> dict:
        results = {
            "overall": "Real",
            "message": "",
            "mean_probability": 0.0,
            "max_probability": 0.0,
            "min_probability": 0.0,
            "confidence": 0.0,
            "ground_truth": None,
            "correct": None,
            "details": [],
            "duration_config": []
        }

        # Load the audio wav file
        waveform, duration = load_audio(audio_path)

        # Create preprocessed version for VAD only (don't modify original for classification)
        waveform_for_vad = waveform
        if ENABLE_NOISE_REDUCTION:
            if VERBOSE:
                print(f"Preprocessing audio for VAD with high pass")
            waveform_for_vad = self.preprocess_waveform(waveform)

        waveform = waveform.to(DEVICE)

        # Check for silence (very low energy)
        audio_energy = torch.abs(waveform).mean().item()
        # Return "No Speech" if audio is silent
        if audio_energy < 0.005:
            results["overall"] = "No Speech"
            results["message"] = "Audio is silent or contains no detectable speech"
            if VERBOSE:
                print("  No speech: audio energy too low")
            return results
        
        if duration < 1.0:
            results["overall"] = "No Speech"
            results["message"] = "Audio too short (<1s)"
            if VERBOSE:
                print("  No speech: audio is too short (<1s)")
            return results
        
        # Check for speech using preprocessed waveform (noise-robust VAD)
        if not self.has_speech(waveform_for_vad):
            results["overall"] = "No Speech"
            results["message"] = "No human speech detected"
            if VERBOSE:
                print("  No speech: pyannote VAD found no speaker")
            return results
        
        # Get duration-aware configuration
        duration_config = self.get_duration_config(duration, self.default_threshold)
        threshold = duration_config['threshold']
        # Determine if diarization should be used
        # Priority: user preference > duration-based automatic decision
        use_diarization = duration_config['use_diarization']


        # If user requested diarization, use it
        # Otherwise, default to OFF (enable_diarization parameter default is False)
        if not enable_diarization:
            use_diarization = False

        results["duration_config"].append({
            'duration': duration_config['duration'],
            'category': duration_config['category'],
            'base_threshold': duration_config['base_threshold'],
            'multiplier': duration_config['multiplier'],
            'threshold': threshold,
            'use_diarization': use_diarization
        })

        # Choose prediction strategy based on diarization
        if use_diarization:
            try:
                if self.diar_pipeline is None:
                    raise RuntimeError("Diarization pipeline is not initialized")
                self.predict_with_diarization(audio_path, waveform, duration, threshold, results)
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
