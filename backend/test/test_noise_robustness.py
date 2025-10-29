"""
Test model robustness to various audio corruptions and noise.

This tests how well the model handles:
- Background noise
- Volume changes
- Audio compression
- Microphone quality variations
- Network artifacts

Usage:
    python test_noise_robustness.py
"""

import torch
import numpy as np
from pathlib import Path
from predict_audio import AudioClassifier
import torchaudio
import os
import warnings

# Suppress lightning-related warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"lightning\.pytorch\..*",
    message=r".*"
)

# Suppress pyannote reproducibility warning
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"pyannote\.audio\.utils\.reproducibility",
    message=r"TensorFloat-32 \(TF32\) has been disabled.*"
)

# Suppress pyannote pooling std warning
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"pyannote\.audio\.models\.blocks\.pooling",
    message=r"std\(\): degrees of freedom is <= 0.*"
)

# Suppress NumPy warnings
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module=r"numpy\._core\..*",
    message=r"(Mean of empty slice|invalid value encountered in divide).*"
)

def add_noise(waveform: torch.Tensor, noise_level: float = 0.01) -> torch.Tensor:
    """Add Gaussian noise to audio."""
    noise = torch.randn_like(waveform) * noise_level
    return waveform + noise

def change_volume(waveform: torch.Tensor, scale: float = 0.5) -> torch.Tensor:
    """Change audio volume."""
    return waveform * scale

def add_background_noise(waveform: torch.Tensor, noise_type: str = "white") -> torch.Tensor:
    """Add different types of background noise."""
    if noise_type == "white":
        # White noise (equal power across frequencies)
        noise = torch.randn_like(waveform) * 0.02
    elif noise_type == "pink":
        # Pink noise (1/f power spectrum)
        noise = torch.randn_like(waveform) * 0.02
        # Simple pink noise approximation
        noise = torch.cumsum(noise, dim=-1) / waveform.shape[-1]
    elif noise_type == "brown":
        # Brown noise (1/f^2 power spectrum)
        noise = torch.randn_like(waveform) * 0.02
        noise = torch.cumsum(torch.cumsum(noise, dim=-1), dim=-1) / (waveform.shape[-1] ** 2)
    else:
        noise = torch.zeros_like(waveform)
    
    return waveform + noise

def simulate_compression(waveform: torch.Tensor, quality: str = "medium") -> torch.Tensor:
    """Simulate audio compression artifacts."""
    # Simple simulation: quantize audio
    if quality == "low":
        bits = 8
    elif quality == "medium":
        bits = 12
    else:  # high
        bits = 16
    
    max_val = 2 ** (bits - 1)
    quantized = torch.round(waveform * max_val) / max_val
    return quantized

def apply_corruption(audio_file: str, corruption_type: str, **kwargs) -> str:
    """Apply corruption to audio file and save temporarily."""
    # Load audio
    waveform, sr = torchaudio.load(audio_file)
    
    # Apply corruption
    if corruption_type == "noise":
        corrupted = add_noise(waveform, noise_level=kwargs.get('level', 0.01))
    elif corruption_type == "volume":
        corrupted = change_volume(waveform, scale=kwargs.get('scale', 0.5))
    elif corruption_type == "background":
        corrupted = add_background_noise(waveform, noise_type=kwargs.get('noise_type', 'white'))
    elif corruption_type == "compression":
        corrupted = simulate_compression(waveform, quality=kwargs.get('quality', 'medium'))
    else:
        corrupted = waveform
    
    # Normalize
    corrupted = corrupted / (torch.max(torch.abs(corrupted)) + 1e-8)
    
    # Save temporarily
    temp_file = f"temp_corrupted_{corruption_type}.wav"
    torchaudio.save(temp_file, corrupted, sr)
    
    return temp_file

def test_single_file_robustness(audio_file: str, ground_truth: str):
    """Test robustness on a single file with various corruptions."""
    print(f"\n{'='*60}")
    print(f"Testing: {Path(audio_file).name}")
    print(f"Ground Truth: {ground_truth}")
    print(f"{'='*60}")
    
    results = {}
    
    # 1. Original (clean)
    print("\n1. Original (clean):")
    result = classifier.predict(audio_file)
    prob = result['mean_probability']
    pred = result['overall']
    correct = (pred == ground_truth)
    results['original'] = {'prob': prob, 'correct': correct}
    print(f"   {pred} (p={prob:.4f}) {'✓' if correct else '✗'}")
    
    # 2. Gaussian noise (light)
    print("\n2. Gaussian noise (light, σ=0.005):")
    temp_file = apply_corruption(audio_file, "noise", level=0.005)
    result = classifier.predict(temp_file)
    prob = result['mean_probability']
    pred = result['overall']
    correct = (pred == ground_truth)
    results['noise_light'] = {'prob': prob, 'correct': correct}
    print(f"   {pred} (p={prob:.4f}) {'✓' if correct else '✗'}")
    os.remove(temp_file)
    
    # 3. Gaussian noise (heavy)
    print("\n3. Gaussian noise (heavy, σ=0.02):")
    temp_file = apply_corruption(audio_file, "noise", level=0.02)
    result = classifier.predict(temp_file)
    prob = result['mean_probability']
    pred = result['overall']
    correct = (pred == ground_truth)
    results['noise_heavy'] = {'prob': prob, 'correct': correct}
    print(f"   {pred} (p={prob:.4f}) {'✓' if correct else '✗'}")
    os.remove(temp_file)
    
    # 4. Volume change (quiet)
    print("\n4. Volume reduced (50%):")
    temp_file = apply_corruption(audio_file, "volume", scale=0.5)
    result = classifier.predict(temp_file)
    prob = result['mean_probability']
    pred = result['overall']
    correct = (pred == ground_truth)
    results['volume_quiet'] = {'prob': prob, 'correct': correct}
    print(f"   {pred} (p={prob:.4f}) {'✓' if correct else '✗'}")
    os.remove(temp_file)
    
    # 5. Volume change (loud)
    print("\n5. Volume increased (150%):")
    temp_file = apply_corruption(audio_file, "volume", scale=1.5)
    result = classifier.predict(temp_file)
    prob = result['mean_probability']
    pred = result['overall']
    correct = (pred == ground_truth)
    results['volume_loud'] = {'prob': prob, 'correct': correct}
    print(f"   {pred} (p={prob:.4f}) {'✓' if correct else '✗'}")
    os.remove(temp_file)
    
    # 6. Background noise (white)
    print("\n6. Background noise (white):")
    temp_file = apply_corruption(audio_file, "background", noise_type='white')
    result = classifier.predict(temp_file)
    prob = result['mean_probability']
    pred = result['overall']
    correct = (pred == ground_truth)
    results['bg_white'] = {'prob': prob, 'correct': correct}
    print(f"   {pred} (p={prob:.4f}) {'✓' if correct else '✗'}")
    os.remove(temp_file)
    
    # 7. Compression (low quality)
    print("\n7. Compression (low quality):")
    temp_file = apply_corruption(audio_file, "compression", quality='low')
    result = classifier.predict(temp_file)
    prob = result['mean_probability']
    pred = result['overall']
    correct = (pred == ground_truth)
    results['compression_low'] = {'prob': prob, 'correct': correct}
    print(f"   {pred} (p={prob:.4f}) {'✓' if correct else '✗'}")
    os.remove(temp_file)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    correct_count = sum(1 for r in results.values() if r['correct'])
    total = len(results)
    
    print(f"Robustness: {correct_count}/{total} ({correct_count/total*100:.1f}%)")
    
    # Check if predictions are consistent
    probs = [r['prob'] for r in results.values()]
    prob_std = np.std(probs)
    prob_range = max(probs) - min(probs)
    
    print(f"Probability std: {prob_std:.4f}")
    print(f"Probability range: {prob_range:.4f}")
    
    if prob_std < 0.1:
        print("✓ Model is very robust (low variance)")
    elif prob_std < 0.2:
        print("⚠ Model is moderately robust")
    else:
        print("✗ Model is sensitive to corruptions (high variance)")
    
    return results

def test_dataset_robustness(dataset_dir: str, num_samples: int = 10):
    """Test robustness across multiple files."""
    print("\n" + "="*60)
    print("DATASET ROBUSTNESS TEST")
    print("="*60)
    
    import pandas as pd
    
    # Load metadata
    metadata_path = os.path.join(dataset_dir, "meta.csv")
    if not os.path.exists(metadata_path):
        print(f"Error: metadata not found at {metadata_path}")
        return
    
    metadata_df = pd.read_csv(metadata_path)
    
    # Sample files
    sample_files = metadata_df.head(num_samples)
    
    all_results = []
    
    for idx, row in sample_files.iterrows():
        file_name = row['file']
        label = "Fake" if row['label'] == "spoof" else "Real"
        audio_path = os.path.join(dataset_dir, file_name)
        
        if not os.path.exists(audio_path):
            continue
        
        results = test_single_file_robustness(audio_path, label)
        
        # Store results
        all_results.append({
            'file': file_name,
            'ground_truth': label,
            **{f'{k}_correct': v['correct'] for k, v in results.items()}
        })
    
    # Overall statistics
    print("\n" + "="*60)
    print("OVERALL ROBUSTNESS STATISTICS")
    print("="*60)
    
    df = pd.DataFrame(all_results)
    
    for corruption in ['original', 'noise_light', 'noise_heavy', 'volume_quiet', 
                      'volume_loud', 'bg_white', 'compression_low']:
        col = f'{corruption}_correct'
        if col in df.columns:
            accuracy = df[col].sum() / len(df) * 100
            print(f"{corruption:20s}: {accuracy:5.1f}% accuracy")
    
    print(f"\n{'='*60}")

# Main execution
if __name__ == "__main__":
    from utils import DATA_DIR, DATASET
    
    print("="*60)
    print("NOISE ROBUSTNESS TESTING")
    print("="*60)
    print("\nThis tests how well the model handles:")
    print("  - Gaussian noise")
    print("  - Volume variations")
    print("  - Background noise")
    print("  - Audio compression")
    print("="*60)
    
    # Test on a few sample files
    dataset_path = os.path.join(DATA_DIR, DATASET)

    classifier = AudioClassifier()
    
    # Quick test on 5 files
    test_dataset_robustness(dataset_path, num_samples=5)
