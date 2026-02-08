# Model Download Guide

Complete guide for downloading and setting up production AI models.

---

## Quick Start

**Automated Download** (Recommended):
```bash
cd d:\Pet\ML\empathy-system
venv\Scripts\python.exe scripts\download_models.py
```

**Manual Download**: Follow sections below

---

## 1. LLM - Llama 3.1 (REQUIRED)

### Option A: Llama 3.1 8B Instruct (Recommended)

**Size**: ~4.7GB  
**Quality**: Best  
**Speed**: Moderate

```bash
# Install huggingface_hub
pip install huggingface_hub

# Download via Python
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='bartowski/Meta-Llama-3.1-8B-Instruct-GGUF', filename='Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf', local_dir='./models/llm')"
```

**Manual Download**:
1. Go to: https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
2. Download: `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`
3. Save to: `d:\Pet\ML\empathy-system\models\llm\`

### Option B: Llama 2 7B Chat (Smaller)

**Size**: ~4.1GB  
**Quality**: Good  
**Speed**: Faster

```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='TheBloke/Llama-2-7B-Chat-GGUF', filename='llama-2-7b-chat.Q4_K_M.gguf', local_dir='./models/llm')"
```

**Update config.yaml**:
```yaml
models:
  llm:
    model_path: "models/llm/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    context_length: 8192
    n_gpu_layers: 40  # For GPU, 0 for CPU
```

---

## 2. TTS - CosyVoice (Auto-Download)

**Size**: ~2GB  
**Installation**:
```bash
pip install TTS
```

CosyVoice models download automatically on first use.

**Test**:
```bash
venv\Scripts\python.exe scripts\demo_tts.py
```

---

## 3. Vision Models (Auto-Download)

### HSEmotion

Auto-downloads on first use (~100MB).

**Manual pre-download** (optional):
```python
from hsemotion.facial_emotions import HSEmotionRecognizer
model = HSEmotionRecognizer(model_name='enet_b0_8_best_afew')
```

### MediaPipe

Already installed via pip. Models download automatically.

---

## 4. Audio Models

### Silero VAD (Auto-Download)

Auto-downloads on first use (~1.4MB).

### SenseVoice STT

**Install FunASR**:
```bash
pip install funasr modelscope
```

Models download automatically on first use (~300MB).

### Wav2Vec2 Prosody (Auto-Download)

Auto-downloads from HuggingFace on first use (~1.2GB).

**Model**: `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`

---

## 5. Complete Installation Script

```bash
# Navigate to project
cd d:\Pet\ML\empathy-system

# Activate venv
venv\Scripts\activate

# Install all optional dependencies
pip install huggingface_hub TTS funasr modelscope

# Download models
venv\Scripts\python.exe scripts\download_models.py
```

---

## 6. Verify Installation

**Test LLM**:
```python
from llama_cpp import Llama

llm = Llama(
    model_path="models/llm/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    n_ctx=8192
)

response = llm("Hello, how are you?", max_tokens=50)
print(response)
```

**Test Full System**:
```bash
# Update demo to use real models
# In scripts/demo_full_integration.py, change:
# agent = EmpathyAgent('demo_user', use_mock=False)

venv\Scripts\python.exe scripts\demo_full_integration.py
```

---

## Storage Requirements

| Model | Size | Auto-Download | Required |
|-------|------|---------------|----------|
| Llama 3.1 8B | 4.7GB | No | Yes (LLM) |
| CosyVoice | 2GB | Yes | Yes (TTS) |
| HSEmotion | 100MB | Yes | Yes (Vision) |
| MediaPipe | 10MB | Yes | Yes (Vision) |
| Silero VAD | 1.4MB | Yes | Yes (Audio) |
| Wav2Vec2 | 1.2GB | Yes | Yes (Audio) |
| SenseVoice | 300MB | Yes | Optional (STT) |

**Total**: ~8-9GB

---

## Performance Comparison

### With Mock Models
- Initialization: 2s
- Response time: Instant
- Quality: Demo only
- GPU: Not needed

### With Real Models
- Initialization: 30s
- Response time: 2-5s
- Quality: Production
- GPU: Recommended (8GB+ VRAM)

---

## GPU vs CPU

### GPU (Recommended)
```yaml
models:
  llm:
    n_gpu_layers: 40  # Offload layers to GPU
    device: cuda
  
  vision:
    hsemotion:
      device: cuda
```

**Benefits**: 
- 5-10x faster LLM
- Real-time vision processing

### CPU Only
```yaml
models:
  llm:
    n_gpu_layers: 0  # CPU only
    device: cpu
  
  vision:
    hsemotion:
      device: cpu
```

**Benefits**: 
- Works on any machine
- No GPU required

---

## Troubleshooting

### Download Fails
```bash
# Set HuggingFace token (if needed)
export HF_TOKEN=your_token_here

# Or login
huggingface-cli login
```

### Out of Disk Space
Download only essential models:
- Llama 3.1 (required)
- Skip optional models
- Use smaller Llama 2 variant

### Slow Downloads
Use HuggingFace mirror or download manually from browser.

---

## Next Steps

After downloading:
1. ✅ Update `config.yaml` with model paths
2. ✅ Test with: `venv\Scripts\python.exe main.py`
3. ✅ Run full demo with `use_mock=False`
4. ✅ Deploy to production!
