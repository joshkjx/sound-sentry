"""
Test client for the Audio Deepfake Detection API.

Usage:
    python test_app.py <audio_file.wav>
    python test_app.py <audio_file.wav> --threshold 0.3
    python test_app.py <audio_file.wav> --api http://192.168.1.100:8000
    python test_app.py --batch file1.wav file2.wav --api http://localhost:5000
"""

import requests
import sys
import json
import argparse
from pathlib import Path

# Default API URL
DEFAULT_API_URL = "http://127.0.0.1:8000"
API_URL = DEFAULT_API_URL

def test_health():
    """Test if the API is running."""
    try:
        response = requests.get(f"{API_URL}/health")
        print("Health Check:", response.json())
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to API. Is the server running?")
        print("   Start it with: uvicorn api:app --reload")
        return False

def predict_single(audio_path: str):
    """Send a single audio file for prediction."""
    
    if not Path(audio_path).exists():
        print(f"❌ Error: File not found: {audio_path}")
        return
    
    print(f"\n📤 Uploading: {audio_path}")
    
    with open(audio_path, 'rb') as f:
        files = {'file': (Path(audio_path).name, f, 'audio/wav')}
        params = {}
        
        try:
            response = requests.post(
                f"{API_URL}/predict",
                files=files,
                params=params,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                print_result(result)
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            print("❌ Error: Request timed out")
        except Exception as e:
            print(f"❌ Error: {e}")

def print_result(result: dict):
    """Pretty print the prediction result."""
    
    print("\n" + "="*60)
    print(f"📊 PREDICTION RESULTS")
    print("="*60)
    
    # Overall result
    overall = result.get('overall', 'Unknown')
    prob = result.get('mean_probability', 0.0)
    conf = result.get('confidence', 0.0)
    
    if overall == "No Speech":
        print(f"\n   Overall: {overall}")
        message = result.get('message', 'No speech detected in audio')
        print(f"   Message: {message}")
        print(f"\n  This audio does not contain detectable speech.")
        print(f"   Please upload audio with clear human speech.")
    else:
        correct = "F" if overall == "Fake" else "T"
        print(f"\n{correct} Overall: {overall}")
        print(f"   Probability (Fake): {prob:.4f}")
        print(f"   Confidence: {conf:.4f}")
    
    # Duration config
    if 'duration_config' in result and result['duration_config']:
        dc = result['duration_config']
        print(f"\n   Duration: {dc.get('duration', 'N/A'):.1f}s")
        print(f"   Category: {dc.get('category', 'N/A')}")
        if overall != "No Speech":
            print(f"   Threshold: {dc.get('threshold', 'N/A'):.4f}")
            print(f"   Diarization: {dc.get('use_diarization', 'N/A')}")
    
    # Segment details
    if overall != "No Speech":
        details = result.get('details', [])
        if details and len(details) > 1:
            print(f"\n🔍 Segments Analyzed: {len(details)}")
            for i, detail in enumerate(details[:5], 1):  # Show first 5
                speaker = detail.get('speaker', 'N/A')
                time = detail.get('time', 'N/A')
                seg_result = detail.get('result', 'N/A')
                seg_prob = detail.get('prob_fake', 0.0)
                print(f"   {i}. {speaker} [{time}]: {seg_result} (p={seg_prob:.4f})")
            
            if len(details) > 5:
                print(f"   ... and {len(details) - 5} more segments")
    
    print("\n" + "="*60)

def test_batch(audio_files: list[str]):
    """Test batch prediction."""
    
    print(f"\n  Uploading {len(audio_files)} files for batch prediction...")
    
    files = []
    for path in audio_files:
        if not Path(path).exists():
            print(f"   Skipping {path}: File not found")
            continue
        files.append(
            ('files', (Path(path).name, open(path, 'rb'), 'audio/wav'))
        )
    
    if not files:
        print("   No valid files to upload")
        return
    
    try:
        response = requests.post(
            f"{API_URL}/batch-predict",
            files=files,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Batch Results:")
            print(f"   Total: {result['total_files']}")
            print(f"   Successful: {result['successful']}")
            print(f"   Failed: {result.get('failed', 0)}")
            
            for item in result['results']:
                print(f"\n{'='*60}")
                if item.get('status') == 'success':
                    overall = item.get('overall', 'Unknown')
                    if overall == "No Speech":
                        print(f"   {item['filename']}: No Speech")
                        print(f"   {item.get('message', 'No speech detected')}")
                    else:
                        print(f"  {item['filename']}: {overall} (p={item['mean_probability']:.4f})")
                else:
                    print(f"  {item['filename']}: Error - {item.get('error', 'Unknown')}")
        else:
            print(f"  Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"  Error: {e}")
    finally:
        # Close all file handles
        for _, (_, f, _) in files:
            f.close()

def get_model_info():
    """Get information about the model."""
    try:
        response = requests.get(f"{API_URL}/info")
        if response.status_code == 200:
            info = response.json()
            print("\n" + "="*60)
            print("  MODEL INFORMATION")
            print("="*60)
            print(json.dumps(info, indent=2))
            print("="*60)
        else:
            print(f"  Error: {response.status_code}")
    except Exception as e:
        print(f"  Error: {e}")

def main():
    global API_URL
    
    parser = argparse.ArgumentParser(
        description='Test client for Audio Deepfake Detection API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_api.py audio.wav
  python test_api.py audio.wav --threshold 0.3
  python test_api.py audio.wav --api http://192.168.1.100:8000
  python test_api.py --batch file1.wav file2.wav file3.wav
  python test_api.py --info --api http://localhost:5000
        """
    )
    
    parser.add_argument(
        'files',
        nargs='*',
        help='Audio file(s) to analyze'
    )
    parser.add_argument(
        '--api',
        type=str,
        default=DEFAULT_API_URL,
        help=f'API URL (default: {DEFAULT_API_URL})'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        help='Custom classification threshold'
    )
    parser.add_argument(
        '--no-diarization',
        action='store_true',
        help='Disable speaker diarization'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process multiple files in batch mode'
    )
    parser.add_argument(
        '--info',
        action='store_true',
        help='Get model information'
    )
    
    args = parser.parse_args()
    
    # Set API URL
    API_URL = args.api
    print(f"Using API: {API_URL}")
    
    # Check if API is running
    if not test_health():
        sys.exit(1)
    
    # Handle different modes
    if args.info:
        get_model_info()
        return
    
    if not args.files:
        parser.print_help()
        sys.exit(1)
    
    if args.batch:
        test_batch(args.files)
    else:
        # Single file prediction
        audio_file = args.files[0]
        predict_single(audio_file)

if __name__ == "__main__":
    main()
