"""
FastAPI server for audio deepfake detection.

Run with: 
    python api.py
    python api.py --host 0.0.0.0 --port 8000
    python api.py --host 192.168.1.100 --port 5000
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import tempfile
from datetime import datetime
from scripts.predict_audio import AudioClassifier
import logging
import argparse
from pydub import AudioSegment

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Audio Deepfake Detection API",
    description="API for detecting AI-generated/deepfake audio using TKAN features",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize classifier (loaded once at startup)
classifier = None

@app.on_event("startup")
async def startup_event():
    """Load the model when the server starts."""
    global classifier
    try:
        logger.info("Loading audio classifier...")
        classifier = AudioClassifier()
        logger.info("Classifier loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load classifier: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when server shuts down."""
    global classifier
    if classifier:
        classifier.cleanup()
        logger.info("Classifier cleaned up")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "Audio Deepfake Detection API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "classifier_loaded": classifier is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict_audio(
    file: UploadFile = File(...)
):
    """
    Predict if an audio file is real or AI-generated/deepfake.
    
    Args:
        file: Audio file (WAV, MP3, etc.)
        
    Returns:
        JSON with prediction results including:
        - overall: "Real", "Fake" or "No Speech"
        - mean_probability: Average probability of being fake
        - confidence: Confidence score
        - message: Explanation if "No Speech"
        - details: Per-segment details if diarization is used
        - duration_config: Duration-based configuration used
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier not loaded")
    
    # Get file extension
    file_ext = os.path.splitext(file.filename)[1].lower() if file.filename else '.webm'
    
    # Validate file type (accept many formats)
    allowed_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.webm', '.opus', '.aac'}
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported: {allowed_extensions}"
        )
    
    # Create temporary file
    temp_input = None
    temp_wav = None
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            temp_input = tmp.name
            shutil.copyfileobj(file.file, tmp)
        
        # Convert to WAV if needed (model expects WAV)
        if file_ext != '.wav':
            logger.info(f"Converting {file_ext} to WAV...")
            temp_wav = tempfile.mktemp(suffix='.wav')
            
            try:
                # Load audio with pydub (supports many formats)
                audio = AudioSegment.from_file(temp_input)
                
                # Convert to model's expected format
                audio = audio.set_frame_rate(16000)  # Model expects 16kHz
                audio = audio.set_channels(1)        # Mono
                audio = audio.set_sample_width(2)    # 16-bit
                
                # Export as WAV
                audio.export(temp_wav, format='wav')
                audio_path = temp_wav
                
                logger.info(f"Conversion successful")
                
            except Exception as e:
                logger.error(f"Audio conversion failed: {e}")
                raise HTTPException(status_code=400, detail=f"Audio conversion failed: {str(e)}")
        else:
            audio_path = temp_input
        
        logger.info(f"Processing file: {file.filename or 'uploaded_audio'}")
        
        # Run prediction
        results = classifier.predict(audio_path)
        
        # Add metadata
        results["filename"] = file.filename or "audio_blob"
        results["timestamp"] = datetime.now().isoformat()
        results["original_format"] = file_ext
        
        # Remove ground_truth field if present (not relevant for API)
        results.pop("ground_truth", None)
        
        # Log result
        if results['overall'] == "No Speech":
            logger.info(f"No speech detected in {file.filename}")
        else:
            logger.info(f"Prediction complete: {results['overall']} (p={results['mean_probability']:.4f})")
        
        return JSONResponse(content=results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Clean up temporary files
        if temp_input and os.path.exists(temp_input):
            try:
                os.remove(temp_input)
            except Exception as e:
                logger.warning(f"Failed to remove temp input file: {e}")
        
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except Exception as e:
                logger.warning(f"Failed to remove temp wav file: {e}")

@app.post("/batch-predict")
async def batch_predict(
    files: list[UploadFile] = File(...)
):
    """
    Predict multiple audio files in batch.
    
    Args:
        files: List of audio files
        
    Returns:
        JSON array with prediction results for each file.
        Files with no speech will have status="no_speech"
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier not loaded")
    
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 files per batch request"
        )
    
    results = []
    
    for file in files:
        temp_file = None
        try:
            # Validate file type
            allowed_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
            filename = file.filename if file.filename is not None else ""
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext not in allowed_extensions:
                results.append({
                    "filename": file.filename,
                    "error": f"Unsupported file type: {file_ext}",
                    "status": "failed"
                })
                continue
            
            # Save and process file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                temp_file = tmp.name
                shutil.copyfileobj(file.file, tmp)
            
            prediction = classifier.predict(temp_file)
            
            prediction["filename"] = file.filename
            prediction["status"] = "success"
            prediction.pop("ground_truth", None)
            
            results.append(prediction)
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "error": str(e),
                "status": "failed"
            })
            
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

    # Count statuses
    successful = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "failed")
    
    return JSONResponse(content={
        "timestamp": datetime.now().isoformat(),
        "total_files": len(files),
        "successful": successful,
        "failed": failed,
        "results": results
    })

@app.get("/info")
async def model_info():
    """Get information about the loaded model."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier not loaded")
    
    return {
        "model": "TKAN-based Deepfake Audio Detector",
        "threshold": classifier.default_threshold,
        "features": "TKAN (Top-K Activated Neurons)",
        "duration_aware": True,
        "duration_categories": {
            "very_short": "< 3s (threshold multiplier: 0.70)",
            "short": "3-6s (threshold multiplier: 0.80)",
            "normal": "> 6s (threshold multiplier: 1.00)"
        },
        "diarization": "Enabled for audio > 6s",
        "possible_results": {
            "Real": "Audio appears genuine",
            "Fake": "Audio appears AI-generated/deepfake",
            "No Speech": "No speech detected (silent/noise/music)"
        },
        "silence_detection": {
            "enabled": True,
            "threshold": 0.01,
            "description": "Returns 'No Speech' for silent audio or audio without detectable speech"
        },
        "supported_formats": {
            "input": [".wav", ".mp3", ".webm", ".ogg", ".m4a", ".flac", ".opus", ".aac"],
            "auto_conversion": True,
            "output_format": "16kHz, 16-bit, mono WAV (automatic)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Audio Deepfake Detection API Server')
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host IP address (default: 127.0.0.1 for localhost)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port number (default: 8000)'
    )
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload for development'
    )
    
    args = parser.parse_args()
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"API will be available at: http://{args.host}:{args.port}")
    logger.info(f"Interactive docs at: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port,
        reload=args.reload
    )
