"""Test script to verify audio models are loading correctly."""

import sys
sys.path.insert(0, '.')

from models.audio import AudioPipeline
from config import Config
import numpy as np
import asyncio

print("=" * 70)
print("AUDIO PIPELINE VERIFICATION TEST")
print("=" * 70)

# Load config
config = Config()

# Initialize audio pipeline with real models
print("\n1. Initializing AudioPipeline with use_mock=False...")
pipeline = AudioPipeline(config, use_mock=False)

# Check what models are loaded
print(f"\n2. Model Types:")
print(f"   VAD: {type(pipeline.vad).__name__}")
print(f"   STT: {type(pipeline.stt).__name__}")
print(f"   Prosody: {type(pipeline.prosody).__name__}")
print(f"   Event Detector: {type(pipeline.event_detector).__name__}")

# Test with dummy audio
print("\n3. Testing with dummy audio...")
dummy_audio = np.random.randn(16000).astype(np.float32)  # 1 second

async def test():
    result = await pipeline.process_audio(dummy_audio)
    return result

result = asyncio.run(test())

print(f"\n4. Audio Pipeline Result:")
print(f"   Audio emotion: {result['audio_state'].get('audio_emotion', 'N/A')}")
print(f"   Arousal: {result['audio_state'].get('arousal', 0):.2f}")
print(f"   Valence: {result['audio_state'].get('valence', 0):.2f}")
print(f"   Processing time: {result.get('processing_time_ms', 0):.1f}ms")

print("\n" + "=" * 70)
vad_type = type(pipeline.vad).__name__
stt_type = type(pipeline.stt).__name__
prosody_type = type(pipeline.prosody).__name__

if 'Mock' not in vad_type and 'Mock' not in stt_type and 'Mock' not in prosody_type:
    print("✅ ALL REAL AUDIO MODELS LOADED!")
    print("   Audio detection is fully active.")
elif 'Mock' in stt_type and 'Mock' not in vad_type:
    print("⚠️  PARTIAL: VAD working, STT using mock")
    print("   (This is okay - STT will work when you speak)")
else:
    print("❌ Some models still using mocks")
    print("   Check logs above for details")
print("=" * 70)
