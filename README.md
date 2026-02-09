# 🤖 Empathy System - Multimodal Emotion AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-orange)](https://huggingface.co/spaces/YOUR_USERNAME/empathy-system)

> **An advanced AI system that understands human emotions through facial expressions and voice tone, providing empathetic and contextually appropriate responses.**

![Empathy System Demo](docs/images/demo_screenshot.png)

## 🎯 Overview

The Empathy System is a complete multimodal affective computing platform that combines **7 state-of-the-art AI models** to achieve comprehensive emotional understanding. It detects emotions from face and voice, fuses these signals intelligently, and generates empathetic responses using large language models.

### Key Features

- 📹 **Real-time Facial Emotion Detection** - 7 emotions with confidence scoring
- 🎵 **Voice Analysis** - Speech transcription + emotional prosody analysis
- 🧠 **Multimodal Fusion** - Combines face and voice for accurate emotional state
- 🎭 **Emotional Masking Detection** - Identifies when face and voice emotions don't match
- 💬 **Empathetic Response Generation** - Context-aware, caring AI responses
- 🔊 **Text-to-Speech Output** - Natural voice synthesis with emotion adjustment
- 💾 **Session Memory** - Maintains conversation context and history
- ⚡ **Real-time Processing** - Sub-2-second end-to-end latency

## 🚀 Quick Start

### Try the Live Demo

**[🌐 Try it on Hugging Face Spaces](https://huggingface.co/spaces/YOUR_USERNAME/empathy-system)**

### Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/empathy-system.git
cd empathy-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the complete demo
python scripts/demo_complete_multimodal.py
```

That's it! The system will:
1. Initialize all AI models (~30-60 seconds first run)
2. Open your webcam for face detection
3. Start listening to your microphone
4. Display an interactive console dashboard
5. Generate and speak empathetic responses

## 📊 Architecture

```
┌─────────────────────┐         ┌─────────────────────┐
│   Camera (Webcam)   │         │  Microphone (Audio) │
└──────────┬──────────┘         └──────────┬──────────┘
           │                               │
           ▼                               ▼
┌──────────────────────┐         ┌────────────────────┐
│  VISION PIPELINE     │         │  AUDIO PIPELINE    │
│  ┌────────────────┐  │         │  ┌──────────────┐  │
│  │ DeepFace       │  │         │  │ Silero VAD   │  │
│  │ (Emotion)      │  │         │  │ (Speech Det) │  │
│  └────────────────┘  │         │  └──────────────┘  │
│  ┌────────────────┐  │         │  ┌──────────────┐  │
│  │ MediaPipe      │  │         │  │ Whisper STT  │  │
│  │ (Posture)      │  │         │  │ (Transcript) │  │
│  └────────────────┘  │         │  └──────────────┘  │
│  ┌────────────────┐  │         │  ┌──────────────┐  │
│  │ Gaze Tracker   │  │         │  │ Wav2Vec2     │  │
│  │ (Eye Contact)  │  │         │  │ (Prosody)    │  │
│  └────────────────┘  │         │  └──────────────┘  │
└──────────┬───────────┘         └──────────┬─────────┘
           │                               │
           └───────────┬───────────────────┘
                       ▼
           ┌───────────────────────┐
           │  MULTIMODAL FUSION    │
           │  (Transformer-based)  │
           └───────────┬───────────┘
                       ▼
           ┌───────────────────────┐
           │   LLM PROCESSING      │
           │   (Llama 3.1 8B)      │
           └───────────┬───────────┘
                       ▼
           ┌───────────────────────┐
           │  EMPATHETIC RESPONSE  │
           │  + TTS Output         │
           └───────────────────────┘
```

## 🔬 Technology Stack

### Vision Models

| Model | Purpose | Size | Accuracy |
|-------|---------|------|----------|
| **DeepFace** | 7-emotion facial detection | ~100MB | 70-80% |
| **MediaPipe** | Posture & body language | ~20MB | 85-90% |
| **Custom Gaze Tracker** | Eye contact patterns | <1MB | 75-85% |

**Detected Emotions**: Happy, Sad, Angry, Fear, Disgust, Surprise, Neutral

### Audio Models

| Model | Purpose | Size | Performance |
|-------|---------|------|-------------|
| **Silero VAD** | Voice activity detection | ~2MB | <100ms latency |
| **Whisper (base)** | Speech-to-text | 140MB | 85-90% accuracy |
| **Wav2Vec2** | Voice emotion analysis | 661MB | Real-time prosody |
| **Audio Event Detector** | Non-speech cues | <1MB | Sigh, breathing detection |

**Prosody Features**: Arousal, Valence, Dominance, Tremor

### Intelligence & Response

| Component | Model/Framework | Purpose |
|-----------|----------------|---------|
| **Fusion Engine** | Transformer-based | Multimodal integration |
| **Language Model** | Llama 3.1 8B | Empathetic response generation |
| **Memory System** | Custom | Session context & history |
| **TTS Engine** | pyttsx3 | Natural voice synthesis |

### Implementation

- **Language**: Python 3.10+
- **ML Frameworks**: PyTorch, TensorFlow, Transformers
- **Web Interface**: Gradio
- **Async Processing**: asyncio
- **Audio**: librosa, sounddevice
- **Video**: OpenCV, cv2

## 💡 Features in Detail

### 1. Real-Time Facial Emotion Detection

```python
# Detects 7 primary emotions with confidence scores
{
    'face_detected': True,
    'primary_emotion': 'happy',
    'confidence': 0.92,
    'valence': 0.8,  # Positive/negative (-1 to +1)
    'arousal': 0.6,  # Calm/excited (0 to 1)
    'all_emotions': {
        'happy': 0.92,
        'neutral': 0.05,
        'sad': 0.02,
        'surprise': 0.01
    }
}
```

### 2. Voice Analysis & Transcription

```python
# Complete audio understanding
{
    'transcribed_text': "I'm feeling great today!",
    'is_speaking': True,
    'audio_emotion': 'positive',
    'valence': 0.7,
    'arousal': 0.8,
    'tremor': 0.1,
    'detected_events': ['speech_start', 'high_energy']
}
```

### 3. Emotional Masking Detection

Identifies when facial expression and voice tone don't align:

```python
# Example: Smiling but stressed voice
{
    'emotional_masking': True,
    'face_emotion': 'happy',
    'voice_emotion': 'stressed',
    'confidence': 0.85,
    'interpretation': 'Positive face but negative voice tone detected'
}
```

### 4. Empathetic Response Generation

```python
# Context-aware, caring responses
User: "I'm feeling overwhelmed with work"
System detects: Fatigue (face) + Stress (voice)
Response: "I can hear the stress in your voice and see the 
          fatigue in your expression. It sounds like you're 
          carrying a heavy load. Would taking a short break 
          help? Sometimes stepping away for even 5 minutes 
          can make a difference."
```

### 5. Multi-Persona Support

Adapts responses based on user context:
- **remote_worker**: Focus on work-life balance, breaks, social connection
- **student**: Academic stress, study tips, time management
- **elderly**: Health, companionship, medication reminders
- **healthcare_patient**: Symptom tracking, medication adherence

### 6. Proactive Intervention

Triggers based on behavioral patterns:
- Prolonged silence (>2 minutes)
- Sustained negative emotion
- Emotional masking detected
- Disengagement signs (no eye contact, slouching)
- Voice tremor (anxiety/stress)

## 📁 Project Structure

```
empathy-system/
├── agents/                 # EmpathyAgent orchestration
│   ├── empathy_agent.py   # Main agent class
│   └── intervention.py    # Proactive intervention logic
├── models/
│   ├── vision/            # Vision processing
│   │   ├── deepface_detector.py
│   │   ├── video_pipeline.py
│   │   ├── posture_analyzer.py
│   │   └── gaze_tracker.py
│   ├── audio/             # Audio processing
│   │   ├── audio_pipeline.py
│   │   ├── vad.py         # Silero VAD
│   │   ├── whisper_stt.py
│   │   ├── prosody.py     # Wav2Vec2
│   │   └── event_detector.py
│   └── tts/               # Text-to-speech
│       ├── voice_synthesis.py
│       └── pyttsx3_wrapper.py
├── fusion/                # Multimodal fusion
│   └── multimodal_fusion.py
├── llm/                   # Language model
│   ├── llama_client.py
│   └── prompt_builder.py
├── memory/                # Session memory
│   ├── session_memory.py
│   └── history_manager.py
├── config/                # Configuration
│   └── config.yaml
├── scripts/               # Demo & testing
│   ├── demo_complete_multimodal.py  # Full demo
│   ├── demo_video_file.py           # No webcam demo
│   ├── test_tts_real.py             # TTS testing
│   └── verify_audio_models.py       # Audio verification
├── utils/                 # Utilities
│   ├── display_dashboard.py
│   └── helpers.py
├── app.py                 # Gradio web app (Hugging Face)
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## 🎯 Use Cases

### Mental Health Support
- Early detection of emotional distress
- Mood tracking over time
- Crisis intervention triggers
- Supportive conversations

### Remote Work & Meetings
- Team member wellbeing monitoring
- Engagement tracking
- Fatigue detection
- Work-life balance prompts

### Education
- Student engagement analysis
- Confusion/frustration detection
- Personalized support
- Learning pace adjustment

### Customer Service
- Real-time satisfaction tracking
- Escalation detection
- Sentiment analysis
- Quality assurance

### Healthcare
- Patient emotional state monitoring
- Medication adherence support
- Chronic pain tracking (via facial micro-expressions)
- Telemedicine enhancement

### Research
- Affective computing studies
- Human-computer interaction
- Multimodal learning
- Emotion AI ethics

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.10 or higher
- Webcam (for vision features)
- Microphone (for audio features)
- 8GB RAM minimum (16GB recommended)
- ~5GB disk space for models

### Step 1: Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/empathy-system.git
cd empathy-system
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: First installation may take 5-10 minutes as models are downloaded.

### Step 4: Configure (Optional)

Edit `config/config.yaml` to customize:
- Model paths
- Processing parameters
- Intervention triggers
- Persona settings

### Step 5: Run Demo

```bash
# Full multimodal demo
python scripts/demo_complete_multimodal.py

# Vision-only demo
python scripts/verify_real_detection.py

# Audio-only demo
python scripts/verify_audio_models.py

# TTS test
python scripts/test_tts_real.py
```

## 🎮 Usage Examples

### Basic Usage

```python
from agents.empathy_agent import EmpathyAgent
import asyncio

# Initialize agent
agent = EmpathyAgent(
    user_id='demo_user',
    persona='remote_worker',
    use_mock=False
)

# Start session
await agent.start_session('session_001')

# Process frame (from webcam)
vision_result = await agent.video_pipeline.process_frame(frame)

# Process audio (from microphone)
audio_result = await agent.audio_pipeline.process_audio(audio_chunk)

# Generate response
response = await agent.generate_response(
    user_input="I'm feeling stressed",
    visual_state=vision_result,
    audio_state=audio_result
)

print(response)
```

### Web Interface (Gradio)

```bash
# Run Gradio app
python app.py
```

Then open `http://localhost:7860` in your browser.

## 📊 Performance

| Metric | Value |
|--------|-------|
| Vision FPS | 10-15 frames/sec |
| Audio Latency | <500ms |
| LLM Response Time | 1-3 seconds |
| End-to-End Latency | <2 seconds |
| RAM Usage | 4-6GB |
| CPU Usage | 50-80% (without GPU) |
| Emotion Accuracy | 70-90% |
| Transcription Accuracy | 85-90% |

### Optimization Tips

- **GPU Acceleration**: Add `device='cuda'` in config for faster processing
- **Model Size**: Use `whisper tiny` instead of `base` for faster STT
- **Frame Rate**: Reduce to 5 FPS if CPU-limited
- **Batch Processing**: Process multiple frames/audio chunks together

## 🧪 Testing

### Run All Tests

```bash
pytest tests/
```

### Individual Component Tests

```bash
# Vision pipeline
python tests/test_vision.py

# Audio pipeline
python tests/test_audio.py

# LLM integration
python tests/test_llm.py

# Complete agent
python tests/test_agent.py
```

## 🚀 Deployment

### Hugging Face Spaces

See [HF Deployment Guide](docs/HF_DEPLOYMENT.md) for step-by-step instructions.

### Docker

```bash
# Build image
docker build -t empathy-system .

# Run container
docker run -p 7860:7860 empathy-system
```

### Cloud Platforms

- **Google Colab**: See [Colab Notebook](notebooks/empathy_system_colab.ipynb)
- **AWS/GCP/Azure**: Standard Python deployment
- **Railway/Render**: See deployment guides in `docs/`

## 🔒 Privacy & Ethics

### Privacy Features

- ✅ **Local Processing**: All models run on-device (no cloud required)
- ✅ **No Data Storage**: Emotions processed in real-time, not stored
- ✅ **Encryption Ready**: AES-256-GCM for sensitive data
- ✅ **Transparent**: Users know what's being detected
- ✅ **Consent-Based**: Explicit user permission required

### Ethical Considerations

⚠️ **Important Notes**:
- This is a **research/demo system**, not a medical device
- Should **augment**, not replace, human empathy
- Requires **informed consent** for use
- May have **bias** - test across demographics
- **Not for surveillance** - designed for support, not monitoring

### Responsible Use

✅ **Do**:
- Use for supportive, helpful purposes
- Get user consent before deployment
- Test for bias and fairness
- Be transparent about capabilities
- Consult professionals for clinical use

❌ **Don't**:
- Use for covert surveillance
- Make critical decisions based solely on AI
- Deploy in high-stakes scenarios without validation
- Ignore privacy concerns
- Claim medical-grade accuracy

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md).

### Areas for Contribution

- 🐛 Bug fixes and testing
- 📚 Documentation improvements
- 🌐 Internationalization (i18n)
- 🎨 UI/UX enhancements
- 🔬 Research & accuracy improvements
- 🚀 Performance optimizations

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 📚 Documentation

- [Installation Guide](docs/INSTALLATION.md)
- [API Documentation](docs/API.md)
- [Architecture Overview](docs/ARCHITECTURE.md)
- [Model Details](docs/MODELS.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [FAQ](docs/FAQ.md)

## 🙏 Acknowledgments

### Models & Libraries

- **DeepFace** - [@serengil](https://github.com/serengil/deepface)
- **Whisper** - OpenAI
- **Wav2Vec2** - Facebook AI / Hugging Face
- **Llama** - Meta AI
- **MediaPipe** - Google
- **Silero VAD** - Silero Team

### Inspiration

- Affective Computing research by Rosalind Picard (MIT)
- Emotional Intelligence work by Daniel Goleman
- Human-Computer Interaction community

## 📧 Contact

- **GitHub**: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- **Email**: your.email@example.com
- **LinkedIn**: [Your Name](https://linkedin.com/in/yourprofile)
- **Twitter**: [@yourhandle](https://twitter.com/yourhandle)

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=YOUR_USERNAME/empathy-system&type=Date)](https://star-history.com/#YOUR_USERNAME/empathy-system&Date)

## 📈 Roadmap

- [ ] Mobile app (iOS/Android)
- [ ] Group emotion analysis
- [ ] Longitudinal emotion tracking
- [ ] Custom model fine-tuning
- [ ] Multi-language support
- [ ] Clinical validation studies
- [ ] Federated learning
- [ ] Real-time collaboration features

---

<div align="center">

**Built with ❤️ for advancing empathetic AI**

[Try Demo](https://huggingface.co/spaces/YOUR_USERNAME/empathy-system) • [Report Bug](https://github.com/YOUR_USERNAME/empathy-system/issues) • [Request Feature](https://github.com/YOUR_USERNAME/empathy-system/issues)

</div>
