MamreStream Hebrew TTS — FastAPI + Swagger + GPU Streaming
🚀 Production-ready FastAPI service for Hebrew Text-to-Speech with real-time streaming, GPU acceleration, and comprehensive Swagger documentation.

🛠️ Prerequisites & Setup
1. Clone Required Dependencies
# Clone the main repository
git clone <repository>
cd MamreStream

# Clone the DiffMamba backend (required dependency)
git clone https://github.com/maxmelichov/DiffMamba.git
2. System Requirements
Docker with NVIDIA GPU support
NVIDIA drivers + Container Toolkit
GPU: Minimum 13GB VRAM (RTX 3090 or better recommended)
✨ Key Features
🎙️ Advanced Hebrew TTS with 2+ billion parameter DiffMamba model
⚡ Real-time streaming - audio chunks delivered as they're generated
🚀 GPU acceleration with CUDA 12.4 support
📖 Interactive Swagger UI with complete API documentation
🐳 Docker + Docker Compose ready for production
🎵 Multiple speaker voices with voice cloning support
📱 RESTful JSON API for seamless integration
🌐 Quick Access Links
Service	URL	Description
Swagger UI	http://localhost:8000/docs	Interactive API documentation
ReDoc	http://localhost:8000/redoc	Alternative API docs
Health Check	http://localhost:8000/health	System status & GPU info
API Root	http://localhost:8000/	Service information
🚀 Quick Start (Docker - Recommended)
Build & Launch
# Navigate to project directory
cd MamreStream

# Build with GPU support (first time)
docker compose build --build-arg BUILDKIT_INLINE_CACHE=1 mamre-api

# Start the API service (loads model automatically)
docker compose up mamre-api -d

# Monitor startup progress
docker compose logs mamre-api --tail=20 -f
🎊 That's it! Visit http://localhost:8000/docs to start using the API.

Available Services
# FastAPI Server (recommended)
docker compose up mamre-api -d

# Rebuild API image after code changes and restart
docker compose build mamre-api
docker compose up mamre-api -d

# Original inference script
docker compose up mamre

# Interactive shell
docker compose run --rm mamre-shell
📖 API Reference
🔧 System & Information Endpoints
Endpoint	Method	Description	Response
/	GET	API welcome and service info	JSON metadata
/health	GET	System health, model status, GPU info	Health status
/speakers	GET	List available speaker voice files	Speaker list
🎵 Text-to-Speech Generation
🎯 Complete Audio Generation
Generate full audio file and get download URL when ready.

POST /inference
Content-Type: application/json

{
  "text": "שלום עולם! איך אתה מרגיש היום?",
  "speaker_file": "voices/AUD-20251102-WA0012.mp3",
  "speaker_rate": 9.0,
  "output_format": "wav"
}
✅ Success Response:

{
  "success": true,
  "message": "TTS generation completed successfully",
  "audio_url": "/download/abc123.wav",
  "duration": 2.05,
  "generation_time": 8.2,
  "segments_count": 3
}
⚡ Real-Time Streaming
Stream audio chunks as they're generated - no waiting for completion!

POST /stream
Content-Type: application/json

{
  "text": "זהו טקסט לזרימת קול בזמן אמת",
  "speaker_file": "voices/AUD-20251102-WA0012.mp3", 
  "speaker_rate": 9.0,
  "chunk_overlap": 1,
  "save_to_file": false,
  "output_filename": null
}
📡 Response: Streaming NDJSON (application/x-ndjson) lines in real-time:

{"ts_ms": int, "pcm16_b64": str} – audio chunk payload (PCM16 base64)
{"boundary": true, "text": "..."} – sentence boundary marker
{"file_url": "/download/<name>.wav"} – final line if save_to_file=true
📁 File Management
Endpoint	Method	Description
/download/{filename}	GET	Download generated audio files
💻 Usage Examples
🔍 Quick Testing with cURL
System Status
# Check if service is running and model is loaded
curl -X GET "http://localhost:8000/health" | jq

# List available speaker voices
curl -X GET "http://localhost:8000/speakers" | jq
🎵 Generate Complete Audio
curl -X POST "http://localhost:8000/inference" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "שלום עולם! זהו טסט של המערכת",
    "speaker_file": "voices/AUD-20251102-WA0012.mp3",
    "speaker_rate": 9.0
  }' | jq
⚡ Stream Real-Time Audio
# Stream NDJSON to file (post-process PCM16 base64 into WAV client-side)
curl -X POST "http://localhost:8000/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "טקסט ארוך יותר לזרימה בזמן אמת עם איכות גבוהה",
    "speaker_file": "voices/AUD-20251102-WA0012.mp3",
  "chunk_overlap": 1,
  "save_to_file": true
  }' \
  --output streaming_audio.ndjson

# Monitor file growth in another terminal
watch -n 0.5 'ls -lh streaming_audio.ndjson 2>/dev/null || echo "Waiting for stream..."'
Python Client Example
import requests
import json

# Health check
health = requests.get("http://localhost:8000/health").json()
print(f"Model loaded: {health['model_loaded']}")
print(f"GPU available: {health['gpu_available']}")

# Generate TTS
response = requests.post(
    "http://localhost:8000/inference",
    json={
        "text": "שלום מהעולם של הבינה המלאכותית!",
        "speaker_file": "voices/AUD-20251102-WA0012.mp3",
        "speaker_rate": 9.0
    }
)

if response.status_code == 200:
    result = response.json()
    print(f"✅ Generated: {result['audio_url']}")
    print(f"Duration: {result['duration']:.2f}s")
    print(f"Generation time: {result['generation_time']:.2f}s")
else:
    print(f"❌ Error: {response.text}")

# Stream audio (real-time)
with requests.post(
    "http://localhost:8000/stream", 
    json={
        "text": "זהו טסט לזרימת קול",
        "speaker_file": "voices/AUD-20251102-WA0012.mp3"
    },
    stream=True
) as r:
    with open("streamed_audio.wav", "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                print("📦 Received audio chunk")
JavaScript/Web Example
// Health check
fetch('http://localhost:8000/health')
  .then(r => r.json())
  .then(data => console.log('TTS Status:', data));

// Generate TTS
fetch('http://localhost:8000/inference', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    text: 'שלום מהאתר שלי',
    speaker_file: 'voices/AUD-20251102-WA0012.mp3'
  })
}).then(r => r.json())
  .then(data => {
    console.log('Audio ready:', data.audio_url);
    // Play or download the audio
  });

// Stream real-time audio
fetch('http://localhost:8000/stream', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    text: 'טקסט ארוך לזרימה...',
    speaker_file: 'voices/AUD-20251102-WA0012.mp3'
  })
}).then(response => {
  // Handle streaming audio chunks
  const reader = response.body.getReader();
  // Process chunks as they arrive...
});
🎯 Advanced Features
📊 System Information
# Check model status and GPU info
curl -X GET "http://localhost:8000/health"

# Response includes:
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0", 
  "gpu_available": true
}
🎵 Available Speakers
The system comes with pre-loaded speaker voices:

voices/AUD-20251102-WA0012.mp3 (739KB)
voices/AUD-20251102-WA0013.mp3 (721KB)
# List all available speakers
curl -X GET "http://localhost:8000/speakers"
⚡ Real-Time Streaming Benefits
Low Latency: Audio chunks start streaming immediately
Progressive Playback: No waiting for complete generation
Memory Efficient: Streams chunks as they're produced
Web Compatible: Works with browser audio players
🎵 Real-time Streaming Output
Stream TTS metadata and build WAV from NDJSON (server can also save WAV when save_to_file=true):

# Start streaming NDJSON to file (background process)
curl -X POST "http://localhost:8000/stream" \
  -H "Content-Type: application/json" \
  -d '{
  "text": "ברוכים הבאים למערכת הסינתטיזה הקולית המתקדמת של מעמרה. זהו דגמה של זרימת קול בזמן אמת.",
  "speaker_file": "voices/AUD-20251102-WA0012.mp3",
  "speaker_rate": 9.0,
  "chunk_overlap": 1,
  "save_to_file": true
  }' \
  --output streaming_output.ndjson &

# Monitor file growth in real-time
for i in {1..30}; do 
  if [ -f streaming_output.ndjson ]; then
    size=$(wc -c < streaming_output.ndjson 2>/dev/null || echo "0")
    human_size=$(du -h streaming_output.ndjson 2>/dev/null | cut -f1 || echo "0B")
    echo "Progress $i: $human_size ($size bytes) - $(date +%H:%M:%S)"
  else
    echo "Progress $i: File not created yet - $(date +%H:%M:%S)"
  fi
  sleep 0.8
done

# Alternative: Continuous monitoring with watch
watch -n 0.5 'ls -lh streaming_output.ndjson 2>/dev/null || echo "File not ready yet"'
🐳 Docker Compose Services
Service Overview
# docker-compose.yml defines these services:

mamre-api:       # 🚀 FastAPI server (port 8000) - RECOMMENDED
mamre:           # 📱 Original inference script  
mamre-shell:     # 🐚 Interactive container access
Service Management with GPU Support
# Build with GPU support and caching
docker compose build --build-arg BUILDKIT_INLINE_CACHE=1 mamre-api

# Start API server with GPU access
docker compose up mamre-api -d

# View real-time logs
docker compose logs mamre-api -f

# Stop all services
docker compose down

# Restart API only (preserves GPU access)
docker compose restart mamre-api

# Access interactive shell with GPU
docker compose run --rm mamre-shell

# Force rebuild with GPU support
docker compose build --no-cache --build-arg BUILDKIT_INLINE_CACHE=1 mamre-api
GPU Verification
# Verify GPU access inside container
docker compose exec mamre-api nvidia-smi

# Check CUDA availability
docker compose exec mamre-api python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
⚙️ Local Development Setup
System Prerequisites
# System dependencies (Ubuntu/Debian)
sudo apt-get update && sudo apt-get install -y \
  espeak-ng \
  ffmpeg \
  libsndfile1 \
  build-essential \
  git \
  curl \
  python3-dev \
  python3-venv

# Verify NVIDIA GPU setup
nvidia-smi
Python Environment Setup
Using uv (recommended)
# One-time: install setuptools, wheel, and torch into the venv, then sync (builds flash-attn/causal-conv1d)
./install.sh

# Or manually:
# uv venv && uv pip install setuptools wheel torch torchaudio && uv sync
Using pip
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel

# Install main dependencies (or use uv: see "Using uv" above)
pip install -r requirements.txt

# Install DiffMamba backend (if not already cloned)
if [ ! -d "DiffMamba" ]; then
  git clone https://github.com/maxmelichov/DiffMamba.git
fi
cd DiffMamba
pip install -e .
cd ..
🚀 Launch Development Server
# Method 1: Direct Python execution
python3 api_app.py

# Method 2: Using uvicorn with auto-reload
uvicorn api_app:app --host 0.0.0.0 --port 8000 --reload

# Method 3: Debug mode with detailed logging
PYTHONPATH=. uvicorn api_app:app --host 0.0.0.0 --port 8000 --reload --log-level debug
🔧 Technical Details
Model Information
Parameters: 2,035,174,784 (2+ billion)
Architecture: DiffMamba with Hebrew-specific conditioning
GPU Memory: ~13GB VRAM usage
Sample Rate: 44,100 Hz
Audio Format: 16-bit WAV output
Performance Metrics
Generation Speed: ~4x real-time on RTX 3090
Typical Latency: 8-12 seconds for 2-second audio
Streaming Delay: ~1-2 seconds for first audio chunk
GPU Utilization: 90-100% during inference
Hebrew Text Processing
Input Text → Hebrew UTF-8 string
Phonikud Processing → Diacritization (vowel marks)
IPA Conversion → International Phonetic Alphabet
Model Inference → Audio generation with speaker conditioning
Local Model & Phonikud Weights (Offline Usage)
The service expects all required weights to be present locally under ./weights:
MamreV1_3_epoch3.pt – main TTS model
dac_44khz.safetensors / dac_44khz.pt – autoencoder
phonikud-1.0.onnx – Phonikud ONNX diacritization model
dictabert_tokenizer.json – tokenizer JSON used by Phonikud
At runtime the API never downloads from Hugging Face; if any of these are missing, the model load will fail and /health will report "model_not_loaded".
🚨 Troubleshooting Guide
🔧 Docker & GPU Issues
❌ Docker GPU Error: nvidia-container-cli: initialization error
# 1. Verify NVIDIA Container Toolkit installation
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 2. Test GPU access
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

# 3. Check docker-compose.yml has proper GPU configuration
# Ensure your service includes:
#   deploy:
#     resources:
#       reservations:
#         devices:
#           - driver: nvidia
#             count: 1
#             capabilities: [gpu]
❌ Model Loading Failed: Failed to load model
# 1. Check container logs for detailed error
docker compose logs mamre-api --tail=50

# 2. Verify GPU memory (needs ~13GB free)
nvidia-smi

# 3. Restart with clean build
docker compose down
docker compose build --no-cache mamre-api
docker compose up mamre-api -d
🎵 API & Performance Issues
❌ API Timeouts or Slow Generation
# 1. Test with minimal text first
curl -X POST "http://localhost:8000/inference" \
  -H "Content-Type: application/json" \
  -d '{"text": "שלום", "speaker_file": "voices/AUD-20251102-WA0012.mp3"}'

# 2. Monitor GPU utilization
nvidia-smi -l 1

# 3. Use streaming for longer texts
curl -X POST "http://localhost:8000/stream" \
  -H "Content-Type: application/json" \
  -d '{"text": "טקסט ארוך...", "speaker_file": "voices/AUD-20251102-WA0012.mp3"}' \
  --output test_stream.wav
❌ Speaker File Not Found
# 1. List available speakers
curl -X GET "http://localhost:8000/speakers" | jq

# 2. Use exact filename from the response
# ✅ Correct: "voices/AUD-20251102-WA0012.mp3"
# ❌ Wrong:   "AUD-20251102-WA0012.mp3"
⚡ Performance Optimization Tips
Issue	Solution	Expected Improvement
Slow generation	Use shorter texts (< 100 chars)	2-3x faster
High memory usage	Monitor with nvidia-smi -l 1	Prevent OOM
Long texts	Use /stream endpoint	Real-time output
Cold starts	Keep container running	Skip model loading
Multiple requests	Use batch processing	Better GPU utilization
🧪 Development & Testing
Run Tests
# Test all API endpoints
python3 test_api.py

# Manual testing
curl -X GET "http://localhost:8000/health"
Development Scripts
# Direct inference (bypasses API)
python3 inference.py --text "טקסט לטסט" --target_speaker_path voices/AUD-20251102-WA0012.mp3

# Streaming demo
python3 stream.py
API Development
# Run with auto-reload for development
uvicorn api_app:app --reload --host 0.0.0.0 --port 8000

# Debug mode with detailed logs
docker compose logs mamre-api -f
📝 API Response Formats
Successful Inference Response
{
  "success": true,
  "message": "TTS generation completed successfully", 
  "audio_url": "/download/abc123.wav",
  "duration": 2.05,
  "generation_time": 8.2,
  "segments_count": 3
}
Error Response
{
  "detail": "TTS generation failed: [error details]"
}
Health Response
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0",
  "gpu_available": true
}
📄 License
This repository contains components licensed by their respective owners. See upstream licenses where applicable.

🤝 Contributing
Fork the repository
Create a feature branch
Test your changes with python3 test_api.py
Submit a pull request
🆘 Support
For issues and questions:

Check the troubleshooting section above
Review API documentation at /docs
Check container logs with docker compose logs mamre-api