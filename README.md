# WhoSpeaks

*Audio Transcription with Speaker Diarization API*

A FastAPI-based service that provides audio transcription with automatic speaker diarization. The API accepts audio files and returns transcribed segments with speaker labels.

## Features

- **Fast Transcription**: Uses faster-whisper for efficient audio transcription
- **Speaker Diarization**: Automatically identifies and labels different speakers using TTS embeddings and clustering
- **RESTful API**: Simple HTTP API for easy integration
- **Minimal Dependencies**: Streamlined requirements with only essential packages
- **Automatic Cleanup**: Temporary files are automatically removed after processing

## Quick Start

### Prerequisites

- Python 3.8+
- TTS model files in the `models/v2.0.2/` directory (or set `COQUI_MODEL_PATH` environment variable)

### Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   The requirements file contains only the essential dependencies:
   - FastAPI and uvicorn for the API server
   - faster-whisper for transcription
   - TTS (Coqui TTS) for speaker embeddings
   - librosa, numpy, scikit-learn for audio processing and clustering
   - torch (required by TTS)

2. **Set Environment Variables** (optional):
   ```bash
   # Windows
   set COQUI_MODEL_PATH=models
   set DEVICE=cpu
   set COMPUTE_TYPE=int8
   
   # Linux/Mac
   export COQUI_MODEL_PATH="models"
   export DEVICE="cpu"  # or "cuda" for GPU
   export COMPUTE_TYPE="int8"  # or "float16", "float32"
   ```

3. **Start the API Server**:
   ```bash
   python app.py
   ```
   
   Or using uvicorn directly:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```
   
   The API will be available at `http://localhost:8000`

### Usage

#### API Endpoints

- `GET /` - API information and available endpoints
- `GET /health` - Health check and model status
- `POST /transcribe` - Main endpoint for transcription with speaker diarization

#### Transcribe Audio

**Endpoint**: `POST /transcribe`

**Parameters**:
- `file` (required): Audio file (supports common formats: wav, mp3, m4a, etc.)
- `num_speakers` (optional): Integer specifying the number of speakers (auto-detected if not provided)

**Example using curl**:
```bash
curl -X POST "http://localhost:8000/transcribe?num_speakers=2" \
     -F "file=@your_audio_file.wav"
```

**Example using Python**:
```python
import requests

with open("your_audio_file.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/transcribe",
        files={"file": f},
        params={"num_speakers": 2}
    )

result = response.json()
print(result)
```

**Response Format**:
```json
{
  "status": "success",
  "filename": "your_audio_file.wav",
  "language": "en",
  "language_probability": 0.99,
  "num_segments": 10,
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Hello, how are you?",
      "speaker": "Speaker 0"
    },
    {
      "start": 2.5,
      "end": 5.0,
      "text": "I'm doing well, thank you.",
      "speaker": "Speaker 1"
    }
  ]
}
```

**Health Check**:
```bash
curl http://localhost:8000/health
```

## Project Structure

```
WhoSpeaks/
├── app.py                      # FastAPI main application (primary entry point)
├── requirements.txt            # Minimal dependencies (only essential packages)
├── models/                     # TTS model files (v2.0.2)
│   └── v2.0.2/
│       ├── config.json
│       ├── model.pth
│       ├── speakers_xtts.pth
│       └── vocab.json
├── README.md                   # This file
└── .gitignore                  # Git ignore file
```

## How It Works

WhoSpeaks emerged from the need for better speaker diarization tools. Existing libraries are heavyweight and often fall short in reliability, speed and efficiency. This project offers a more refined alternative.

The core concept:

> **Hint:** *Anybody interested in state-of-the-art voice solutions please also <strong>have a look at [Linguflex](https://github.com/KoljaB/Linguflex)</strong>. It lets you control your environment by speaking and is one of the most capable and sophisticated open-source assistants currently available.*

1. **Transcription**: Audio is transcribed using faster-whisper, which segments the audio into text segments with timestamps.

2. **Voice Characteristic Extraction**: For each transcribed segment, unique voice characteristics are extracted using TTS embeddings, creating audio embeddings.

3. **Speaker Clustering**: The embeddings are normalized and clustered using K-Means to identify distinct speakers. Similar sounding segments are grouped together.

4. **Speaker Assignment**: Each segment is assigned to a speaker based on its embedding similarity to the identified speaker clusters.

This approach allows us to match any segment against the established speaker profiles with remarkable precision.


## Performance

WhoSpeaks has been tested on challenging audio samples with similar voice profiles. In tests, it has demonstrated:

- **High Accuracy**: ~95% precision in speaker assignment
- **Speed**: Efficient processing with parallel embedding extraction
- **Reliability**: Outperforms heavyweight solutions like pyannote audio in both speed and accuracy

The API automatically handles:
- Audio format conversion
- Segment filtering (removes segments shorter than 0.5 seconds)
- Parallel processing for faster results
- Automatic cleanup of temporary files

## Dependencies

The project uses a minimal set of dependencies:

- **FastAPI** & **Uvicorn**: Web framework and ASGI server
- **faster-whisper**: Fast and efficient Whisper transcription
- **TTS (Coqui TTS)**: Speaker embedding extraction
- **librosa**: Audio processing
- **scikit-learn**: Clustering algorithms
- **numpy**: Numerical operations
- **torch**: Deep learning framework (required by TTS)

All other dependencies are automatically installed as transitive dependencies.

## Production Deployment (Railway)

To deploy successfully on Railway, ensure the following configurations:

1. **Environment Variables**:
   - `GIT_LFS_SKIP_SMUDGE = 0`: **Critical** - Forces Railway to download the actual model weights (1.77GB) instead of Git LFS pointer files.
   - `DEVICE = cpu`: Cloud containers usually don't have GPUs unless specifically configured.
   - `COMPUTE_TYPE = int8`: Recommended for CPU environments to save RAM and increase speed.
   - `PORT`: Set automatically by Railway.

2. **Resources**:
   - **RAM**: At least 4GB (8GB recommended) to hold both Whisper and XTTS models in memory.

3. **Deployment**:
   - The repository includes a `railway.json` which configures the start command with `--timeout-keep-alive 300` to allow the heavy models enough time to load without causing a health check timeout.
   - Ensure your `.gitattributes` is pushed to handle the `.pth` files via Git LFS.

## License

This project was initially developed as a personal project and has been released for others to experiment with and improve upon. 
