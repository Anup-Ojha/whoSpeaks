# WhoSpeaks 🎙️

*High-Performance Audio Transcription with Speaker Diarization*

WhoSpeaks is a robust FastAPI-based backend designed to provide state-of-the-art audio transcription and speaker identification. It is optimized for speed, accuracy, and ease of integration into modern frontend applications.

---

## 🏗️ How it Works: The Processing Pipeline

WhoSpeaks follows a sophisticated four-stage pipeline to transform raw audio into structured, speaker-labeled text.

### 1. Intelligence-Driven Transcription (Faster-Whisper)
The system uses `faster-whisper`, a reimplementation of OpenAI’s Whisper model using CTranslate2. 
- **The Magic:** It doesn't just transcribe; it identifies timestamps for every segment and sentence.
- **Why it's better:** It's up to 4x faster than the original Whisper while using significantly less VRAM/RAM.

### 2. Voice Characteristic Extraction (XTTS Embeddings)
Once we have the text segments, we need to know *who* said them.
- **The Process:** We slice the original audio into small chunks based on Whisper's timestamps.
- **The Tech:** We use the `XTTS v2.0.2` encoder to extract a "Voice Fingerprint" (embedding vector) for each audio chunk. This fingerprint represents the unique pitch, tone, and resonance of the speaker.

### 3. Dynamic Speaker Clustering (K-Means)
With a collection of voice fingerprints, the system needs to group them.
- **Normalization:** Fingerprints are passed through a `StandardScaler` to ensure consistency.
- **Clustering:** We use `K-Means` clustering. If the user doesn't specify the number of speakers, the system intelligently analyzes the voice distribution to separate distinct personalities.

### 4. Response Assembly
The final output is a structured JSON object where every line of text is tagged with a speaker ID (e.g., "Speaker 0", "Speaker 1").

---

## 💻 Local Development Guide

### Prerequisites
- **Python:** 3.9 or 3.10 is recommended.
- **Hardware:** 8GB RAM minimum. A GPU (NVIDIA) is supported but not required.
- **Models:** Ensure you have the XTTS v2 model weights in `models/v2.0.2/`.

### 1. Setup Environment
```bash
# Create a virtual environment
python -m venv venv

# Activate it
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration (`.env`)
Create a `.env` file or set these variables in your shell:
```bash
DEVICE=cpu          # 'cuda' for NVIDIA GPU
COMPUTE_TYPE=int8   # 'float16' for GPU, 'int8' for CPU
COQUI_MODEL_PATH=models
MAX_WORKERS=4       # Number of concurrent threads for embedding extraction
```

### 3. Run the Server
```bash
python app.py
```
Visit `http://localhost:8000/docs` to see the interactive API documentation.

---

## 🌐 Frontend Integration Guide

The WhoSpeaks backend is designed to handle multipart form data, making it compatible with standard JavaScript `FormData` objects.

### Typical Workflow
1. **Record/Select Audio:** Capture audio from the user's microphone or file input.
2. **Submit to API:** POST the file to `/transcribe`.
3. **Display Results:** Map over the `segments` array to create a "Chat" style interface.

### Example (JavaScript/React)
```javascript
const formData = new FormData();
formData.append('file', audioBlob, 'recording.wav');
if (numSpeakers) formData.append('num_speakers', 2);

const response = await fetch('https://your-api.railway.app/transcribe', {
  method: 'POST',
  body: formData,
});

const data = await response.json();
// data.segments will contain: { start, end, text, speaker }
```

---

## 🚀 Production Deployment (Railway)

WhoSpeaks is optimized for containerized environments like Railway.

### Critical Deployment Settings
- **Environment Variable:** `GIT_LFS_SKIP_SMUDGE = 0`. This is **mandatory**. Without it, Railway will only download a 1KB "pointer" file instead of the 1.77GB model weights.
- **Memory (RAM):** 4GB is the absolute minimum; **8GB is recommended** for smooth processing of larger files.
- **Health Checks:** The `railway.json` includes a `--timeout-keep-alive 300` flag. This gives the heavy AI models enough time to load into RAM before the server starts accepting traffic.

---

## 📂 Project Structure
```text
WhoSpeaks/
├── app.py              # Main Application Logic
├── railway.json        # Production Config
├── requirements.txt    # Python Packages
├── models/
│   └── v2.0.2/         # XTTS Model Weights (LFS)
└── .gitattributes      # LFS configuration for .pth files
```

## License
Released for experimentation and professional use. Based on the faster-whisper and Coqui TTS ecosystems.
