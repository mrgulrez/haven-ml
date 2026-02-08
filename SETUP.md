# Local Setup Guide

This guide will help you set up and test the Empathy System locally on your machine.

## Prerequisites

### Hardware Requirements
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but recommended)
  - For CPU-only: Set `use_mock=True` in agent initialization
- **Storage**: 20GB free space (for models)
- **Webcam**: For video input
- **Microphone**: For audio input

### Software Requirements
- **Python**: 3.9 or 3.10 (3.11+ may have compatibility issues)
- **CUDA**: 11.8 or 12.1 (if using GPU)
- **Git**: For cloning repository

---

## Step 1: Clone Repository

```bash
cd d:/Pet/ML
# Repository should already exist at d:/Pet/ML/empathy-system
```

---

## Step 2: Create Virtual Environment

```bash
cd d:/Pet/ML/empathy-system

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
# source venv/bin/activate
```

---

## Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# If using GPU with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# If using GPU with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# If CPU only
pip install torch torchvision torchaudio
```

### Install Optional Dependencies

```bash
# For Llama 3.1 (LLM)
# GPU version
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# CPU version
# pip install llama-cpp-python

# For CosyVoice TTS
pip install TTS

# For audio playback in demos
pip install sounddevice
```

---

## Step 4: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings (optional for local testing)
# Most features work without API keys for local testing
```

**Minimal `.env` for local testing**:
```env
# Leave empty or use mock clients
LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=
```

---

## Step 5: Download Models (Optional)

For **full functionality** (not using mocks), download these models:

### Llama 3.1 Model
```bash
# Create models directory
mkdir -p models/llm

# Download Llama 3.1 8B GGUF (4-bit quantized, ~4.7GB)
# Visit: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF
# Download: llama-2-7b-chat.Q4_K_M.gguf
# Place in: models/llm/
```

### Update config.yaml
```yaml
models:
  llm:
    model_path: "models/llm/llama-2-7b-chat.Q4_K_M.gguf"
```

**Note**: For initial testing, you can skip model downloads and use mock clients (set `use_mock=True`).

---

## Step 6: Run Tests

Verify installation by running test suites:

```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/test_vision.py -v
pytest tests/test_audio.py -v
pytest tests/test_fusion.py -v
pytest tests/test_llm_memory.py -v
pytest tests/test_tts.py -v
pytest tests/test_agent.py -v
```

**Expected Output**: All tests should pass (using mock models).

---

## Step 7: Run Demo Scripts

### 7.1 Vision Demo (Webcam Required)

```bash
python scripts/demo_vision.py
```

**What to expect**:
- Webcam window opens
- Real-time face emotion detection
- Posture analysis overlay
- Gaze tracking visualization
- Press 'q' to quit

---

### 7.2 Audio Demo (Microphone Required)

```bash
python scripts/demo_audio.py
```

**What to expect**:
- Console shows audio processing
- Speech detection
- Transcription (if speaking)
- Prosody analysis
- Press Ctrl+C to stop

---

### 7.3 Fusion Demo (Webcam + Mic Required)

```bash
python scripts/demo_fusion.py
```

**What to expect**:
- Combined video and audio processing
- Multimodal emotion fusion
- Conflict detection (visual vs audio)
- Temporal smoothing
- Press 'q' to quit

---

### 7.4 LLM Demo

```bash
python scripts/demo_llm.py
```

**What to expect**:
- Scenario-based response generation
- Emotional context injection
- Masking detection in prompts
- Proactive intervention examples

---

### 7.5 TTS Demo

```bash
python scripts/demo_tts.py
```

**What to expect**:
- Voice synthesis with different emotions
- Audio playback (if sounddevice installed)
- Streaming synthesis demonstration

---

### 7.6 **Full Integration Demo** (Webcam + Mic Required)

```bash
python scripts/demo_full_integration.py
```

**What to expect**:
- Complete end-to-end pipeline
- Real-time webcam + microphone
- Emotional state visualization
- Speech transcription
- Contextual voice responses
- Proactive interventions
- Press 'q' to quit, 's' to speak

---

## Step 8: Verify GPU Usage (Optional)

Check if models are using GPU:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError`
**Solution**: Ensure virtual environment is activated and dependencies installed
```bash
venv\Scripts\activate
pip install -r requirements.txt
```

### Issue: Webcam not opening
**Solution**: 
- Check camera permissions
- Try different camera index in demo scripts (change `cv2.VideoCapture(0)` to `(1)`)
- Ensure no other app is using camera

### Issue: Microphone not working
**Solution**:
- Check microphone permissions
- List available devices: `python -c "import sounddevice; print(sounddevice.query_devices())"`
- Update device index in demos

### Issue: CUDA out of memory
**Solution**:
- Use mock models: Set `use_mock=True` in agent initialization
- Reduce batch sizes in config.yaml
- Use smaller model variants

### Issue: Slow performance
**Solution**:
- Ensure GPU is being used (check with `torch.cuda.is_available()`)
- Use mock models for testing
- Close other GPU applications

---

## Using Mock Models (Fast Testing)

For quick testing without downloading models:

```python
from agents import EmpathyAgent

# Initialize with mock models
agent = EmpathyAgent('test_user', use_mock=True)
```

**Mock models**:
- ✅ Fast execution
- ✅ No model downloads needed
- ✅ All tests pass
- ❌ No real AI functionality
- ❌ Canned responses only

---

## Next Steps

Once local testing is successful:

1. **✅ All demos working?** → Ready for deployment
2. **🚀 Deploy to cloud** → See `DEPLOYMENT.md`
3. **🔒 Add privacy features** → See privacy configuration
4. **📊 Monitor performance** → Check latency logs

---

## Performance Expectations (Local)

| Component | Expected FPS/Latency |
|-----------|---------------------|
| Face Detection | 100+ FPS (GPU) / 30+ FPS (CPU) |
| VAD | <100ms |
| Fusion | <200ms |
| LLM Response | 1-3s (depends on model) |
| TTS Synthesis | 0.5-2s |
| **End-to-End** | **2-5s** |

---

## Support

For issues:
1. Check logs in `logs/` directory
2. Review error messages
3. Verify all dependencies installed
4. Ensure hardware meets requirements
