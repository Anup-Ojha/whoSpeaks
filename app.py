"""
FastAPI Application for Audio Transcription with Speaker Diarization

This API accepts audio files and returns transcription with speaker diarization.
"""

import os
import torch
import numpy as np
import wave
import librosa
import asyncio
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from TTS.tts.models import setup_model as setup_tts_model
from TTS.config import load_config
from faster_whisper import WhisperModel

# --- Configuration ---
DEVICE = os.getenv("DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")
MODELS_PATH = os.environ.get("COQUI_MODEL_PATH", "models")
XTTS_CHECKPOINT = os.path.join(MODELS_PATH, "v2.0.2")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# Initialize FastAPI app
app = FastAPI(
    title="WhoSpeaks API",
    description="Audio Transcription with Speaker Diarization",
    version="1.0.0"
)
# Initialize placeholders
tts_model = None
whisper_model = None
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)


def load_models():
    """Load TTS and Whisper models with robust logging and LFS checks"""
    global tts_model, whisper_model
    
    print("--- Model Loading Starting ---")
    print(f"Target Device: {DEVICE}")
    print(f"Compute Type: {COMPUTE_TYPE}")
    print(f"Models Path: {os.path.abspath(MODELS_PATH)}")
    
    try:
        # 1. Load Whisper Model
        print("Loading Whisper model (base)...")
        whisper_model = WhisperModel("base", device=DEVICE, compute_type=COMPUTE_TYPE)
        print("✅ Whisper model loaded successfully")
        
        # 2. Check XTTS Checkpoint
        print(f"Checking XTTS checkpoint at: {XTTS_CHECKPOINT}")
        if not os.path.exists(XTTS_CHECKPOINT):
            print(f"❌ ERROR: XTTS checkpoint not found at {XTTS_CHECKPOINT}")
            raise FileNotFoundError(f"XTTS model checkpoint not found at {XTTS_CHECKPOINT}")
            
        # Verify LFS files (not just pointer files)
        model_pth_path = os.path.join(XTTS_CHECKPOINT, "model.pth")
        if os.path.exists(model_pth_path):
            file_size_gb = os.path.getsize(model_pth_path) / (1024**3)
            print(f"Model file size: {file_size_gb:.2f} GB")
            if file_size_gb < 0.1:  # Pointer files are very small
                print("⚠️ WARNING: model.pth seems to be a Git LFS pointer file, not the actual weights.")
                print("Fix: Ensure GIT_LFS_SKIP_SMUDGE=0 is set in your environment variables.")
        
        # 3. Load TTS model for speaker embeddings
        print("Setting up TTS model config...")
        config_path = os.path.join(XTTS_CHECKPOINT, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
            
        config = load_config(config_path)
        tts_model = setup_tts_model(config)
        
        print("Loading TTS checkpoint weights...")
        tts_model.load_checkpoint(
            config, 
            checkpoint_dir=XTTS_CHECKPOINT, 
            eval=True
        )
        
        print(f"Moving TTS model to {DEVICE}...")
        tts_model.to(DEVICE)
        print("✅ TTS model loaded successfully")
        print("--- All models ready for inference ---")
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR during model loading: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan manager for setup and cleanup"""
    load_models()
    yield
    print("Shutting down: cleaning up executor")
    executor.shutdown(wait=True)


# Initialize FastAPI app
app = FastAPI(
    title="WhoSpeaks API",
    description="Audio Transcription with Speaker Diarization",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS to allow all domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


def extract_embedding_sync(audio_path: str) -> np.ndarray:
    """Extract speaker embedding from audio file (synchronous function for thread pool)"""
    try:
        _, embedding = tts_model.get_conditioning_latents(
            audio_path=audio_path, 
            gpt_cond_len=30, 
            max_ref_length=60
        )
        return embedding.view(-1).cpu().detach().numpy()
    except Exception as e:
        print(f"Error extracting embedding: {str(e)}")
        raise





@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "WhoSpeaks API - Audio Transcription with Speaker Diarization",
        "version": "1.0.0",
        "endpoints": {
            "/transcribe": "POST - Upload audio file for transcription with speaker diarization",
            "/health": "GET - Health check endpoint"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    models_loaded = tts_model is not None and whisper_model is not None
    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": models_loaded,
        "device": DEVICE
    }


@app.post("/transcribe")
async def transcribe_with_diarization(
    file: UploadFile = File(...),
    num_speakers: Optional[int] = None
):
    """
    Transcribe audio file with speaker diarization
    
    Args:
        file: Audio file to process (supports common audio formats)
        num_speakers: Optional number of speakers (auto-detected if not provided)
    
    Returns:
        JSON response with transcription segments and speaker labels
    """
    if not tts_model or not whisper_model:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Validate file
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024)}MB")
    
    # Create temporary file for processing
    temp_input_path = None
    temp_segments = []
    
    try:
        # Save uploaded file to temporary location
        suffix = Path(file.filename).suffix if file.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_input:
            content = await file.read()
            temp_input.write(content)
            temp_input_path = temp_input.name
        
        # 1. Transcribe audio using Whisper
        loop = asyncio.get_event_loop()
        print(f"Transcribing audio file: {file.filename}")
        segments, info = await loop.run_in_executor(
            executor, 
            lambda: whisper_model.transcribe(temp_input_path, beam_size=5)
        )
        segments = list(segments)
        print(f"Transcription complete. Found {len(segments)} segments")
        
        # 2. Load full audio for segment extraction
        y_full, sr = librosa.load(temp_input_path, sr=16000)
        
        # 3. Extract embeddings for each segment in parallel
        tasks = []
        valid_results = []
        valid_indices = []
        
        for i, segment in enumerate(segments):
            start_sample = int(segment.start * sr)
            end_sample = int(segment.end * sr)
            y_segment = y_full[start_sample:end_sample]
            
            # Filter out very short segments (min 0.5s)
            if len(y_segment) < 8000:  # 0.5s at 16kHz
                continue
            
            # Save segment to temporary file for embedding extraction
            temp_seg = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_seg_path = temp_seg.name
            temp_segments.append(temp_seg_path)
            
            with wave.open(temp_seg_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes((y_segment * 32767).astype(np.int16).tobytes())
            
            # Store segment info
            valid_results.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip(),
            })
            valid_indices.append(i)
            
            # Schedule embedding extraction
            tasks.append(
                loop.run_in_executor(executor, extract_embedding_sync, temp_seg_path)
            )
        
        # Wait for all embedding extractions to complete
        print(f"Extracting speaker embeddings for {len(tasks)} segments...")
        embeddings = await asyncio.gather(*tasks)
        print("Embedding extraction complete")
        
        # 4. Perform speaker clustering
        if len(embeddings) > 1:
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(np.array(embeddings))
            
            # Determine number of clusters
            if num_speakers and num_speakers > 0:
                n_clusters = min(num_speakers, len(embeddings))
            else:
                # Auto-detect: try 2 speakers first, can be extended
                n_clusters = min(2, len(embeddings))
            
            # Perform K-Means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            speaker_labels = await loop.run_in_executor(
                executor, 
                kmeans.fit_predict, 
                embeddings_scaled
            )
            
            # Assign speaker labels to segments
            for i, result in enumerate(valid_results):
                result["speaker"] = f"Speaker {int(speaker_labels[i])}"
        else:
            # Single segment or no valid segments
            for result in valid_results:
                result["speaker"] = "Speaker 0"
        
        # 5. Prepare response
        response = {
            "status": "success",
            "filename": file.filename,
            "language": info.language if hasattr(info, 'language') else "unknown",
            "language_probability": info.language_probability if hasattr(info, 'language_probability') else None,
            "num_segments": len(valid_results),
            "segments": valid_results
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        # Cleanup temporary files
        if temp_input_path and os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except:
                pass
        
        for temp_seg_path in temp_segments:
            if os.path.exists(temp_seg_path):
                try:
                    os.remove(temp_seg_path)
                except:
                    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

