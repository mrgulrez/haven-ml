"""Quick audio test - lightweight, no heavy model downloads."""

import sys
sys.path.insert(0, '.')

from models.audio import AudioPipeline
from config import Config
import numpy as np
import asyncio

print("=" * 70)
print("LIGHTWEIGHT AUDIO TEST")
print("=" * 70)

# Load config
config = Config()

print("\nInitializing AudioPipeline...")
print("(This will try to load real models but fallback to mocks if needed)")

try:
    pipeline = AudioPipeline(config, use_mock=False)
    
    print(f"\n✓ Pipeline created!")
    print(f"  VAD type: {type(pipeline.vad).__name__}")
    print(f"  STT type: {type(pipeline.stt).__name__}")
    print(f"  Prosody type: {type(pipeline.prosody).__name__}")
    print(f"  Events type: {type(pipeline.event_detector).__name__}")
    
    # Quick test with minimal audio
    print("\nTesting with 0.1s audio...")
    test_audio = np.random.randn(1600).astype(np.float32)
    
    async def quick_test():
        result = await pipeline.process_audio(test_audio)
        return result
    
    result = asyncio.run(quick_test())
    
    print(f"\n✓ Audio processed!")
    print(f"  Is speech: {result['vad']['is_speech']}")
    print(f"  Audio emotion: {result['audio_state'].get('audio_emotion')}")
    
    print("\n" + "=" * 70)
    if 'Mock' in type(pipeline.vad).__name__:
        print("ℹ Using mock VAD (Silero VAD not loaded)")
    else:
        print("✅ Real VAD working!")
    
    if 'Mock' in type(pipeline.stt).__name__:
        print("ℹ Using mock STT (optional - only used when you speak)")
    else:
        print("✅ Real STT working!")
    
    if 'Mock' in type(pipeline.prosody).__name__:
        print("ℹ Using mock prosody (Wav2Vec2 not loaded)")
    else:
        print("✅ Real prosody working!")
    
except Exception as e:
    print(f"\n❌ Error creating pipeline: {e}")
    import traceback
    traceback.print_exc()

print("=" * 70)
