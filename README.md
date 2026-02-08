# Empathy System - Multimodal Affective Computing Platform

An AI companion system that understands emotions through continuous video and audio analysis, providing proactive empathetic support.

## Architecture

```
empathy-system/
├── agents/          # LiveKit agent workers
├── models/          # ML model inference
│   ├── vision/     # Facial & posture analysis
│   ├── audio/      # Speech & paralinguistics
│   └── fusion/     # Multimodal integration
├── llm/            # LLM integration
├── tts/            # Voice synthesis
├── memory/         # Context & user history
├── privacy/        # Security & data controls
├── config/         # Configuration
└── tests/          # Tests
```

## Technology Stack

- **Vision**: HSEmotion (facial), MediaPipe (posture/gaze)
- **Audio**: SenseVoice (STT), Wav2Vec2 (emotion), Silero VAD
- **LLM**: Llama 3.1 70B (4-bit quantized)
- **TTS**: CosyVoice 2 (emotion-aware)
- **Streaming**: LiveKit (WebRTC)
- **Infrastructure**: GPU servers (A10G/L40S)

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

### 2. Download Models

```bash
# Download pre-trained models (not yet implemented - placeholder)
python scripts/download_models.py
```

### 3. Start LiveKit Server (Local Development)

```bash
# Using Docker
docker run --rm -p 7880:7880 -p 7881:7881 -p 7882:7882/udp \
  livekit/livekit-server --dev --bind 0.0.0.0
```

### 4. Run Agent Worker

```bash
python -m agents.livekit_agent
```

## Configuration

All settings are in `config/config.yaml`:

- **Models**: Paths and inference settings
- **Intervention**: Trigger thresholds
- **Personas**: Response styles
- **Privacy**: Edge processing, encryption
- **Performance**: Latency targets

## Development Status

### ✅ Phase 1: Foundation (Current)
- [x] Project structure
- [x] Configuration system
- [x] Logging infrastructure
- [x] Base agent interface

### 🔄 Phase 2-11: In Progress
See `task.md` for detailed roadmap.

## Privacy & Security

- **Edge Processing**: Video/audio analyzed locally
- **Encryption**: AES-256-GCM for all data
- **Hardware Controls**: Camera shutter, mute button
- **Data Deletion**: "Forget my day" feature

## Target Personas

1. **Remote Workers**: Work-life balance, social connection
2. **Students**: Study habits, exam anxiety, sleep
3. **Young Professionals**: Confidence, imposter syndrome

## Performance Targets

- **Latency**: <500ms end-to-end
- **Throughput**: 10-15 concurrent users per A10G GPU
- **Accuracy**: >85% emotion classification

## License

[To be determined]

## Contact

[To be determined]
