"""Demo script for testing audio pipeline with microphone.

Run this to see the audio models in action with your microphone.
"""

import asyncio
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.audio import AudioPipeline
from config import config
from utils.logger import setup_logger

try:
    import sounddevice as sd
except ImportError:
    print("sounddevice not installed. Install with: pip install sounddevice")
    sys.exit(1)


async def main():
    """Run audio pipeline demo with microphone."""
    # Setup logging
    setup_logger("INFO")
    
    print("=" * 60)
    print("Audio Pipeline Microphone Demo")
    print("=" * 60)
    print("\nThis demo will:")
    print("- Detect speech with VAD")
    print("- Transcribe what you say")
    print("- Analyze emotional tone (arousal, valence, tremor)")
    print("- Detect events (sighs, breathing, silence)")
    print("\nPress Ctrl+C to quit\n")
    
    # Initialize pipeline
    print("Initializing audio pipeline...")
    pipeline = AudioPipeline(config, use_mock=False)
    print("✓ Pipeline ready!\n")
    
    # Audio settings
    sample_rate = 16000
    chunk_duration = 0.5  # 500ms chunks
    chunk_samples = int(sample_rate * chunk_duration)
    
    print("Starting microphone stream...")
    print("=" * 60)
    
    chunk_count = 0
    
    try:
        # Callback for audio stream
        async def process_stream():
            nonlocal chunk_count
            
            with sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=chunk_samples
            ) as stream:
                while True:
                    # Read audio chunk
                    audio_chunk, overflowed = stream.read(chunk_samples)
                    
                    if overflowed:
                        print("⚠ Audio buffer overflow")
                    
                    # Flatten to 1D
                    audio_chunk = audio_chunk.flatten()
                    
                    # Process
                    timestamp = chunk_count * chunk_duration
                    result = await pipeline.process_audio(audio_chunk, timestamp)
                    
                    # Print results
                    vad = result['vad']
                    audio_state = result['audio_state']
                    
                    # Print on speech events or every 2 seconds
                    if vad.get('event') or chunk_count % 4 == 0:
                        print(f"\n[{timestamp:.1f}s]")
                        
                        if vad.get('is_speaking'):
                            print(f"  🎤 Speaking (confidence: {vad.get('probability', 0):.2f})")
                        
                        if audio_state.get('transcribed_text'):
                            print(f"  📝 Said: \"{audio_state['transcribed_text']}\"")
                        
                        print(f"  😊 Emotion: {audio_state['audio_emotion']}")
                        print(f"  📊 Arousal: {audio_state['arousal']:.2f}, Valence: {audio_state['valence']:.2f}")
                        
                        if audio_state.get('tremor', 0) > 0.6:
                            print(f"  ⚡ Voice tremor detected: {audio_state['tremor']:.2f}")
                        
                        events = audio_state.get('detected_events', [])
                        if events:
                            print(f"  🔔 Events: {', '.join(events)}")
                        
                        if audio_state.get('emotional_masking'):
                            print(f"  😷 Emotional masking detected!")
                        
                        print(f"  ⏱️  Latency: {result['processing_time_ms']:.1f}ms")
                    
                    chunk_count += 1
                    await asyncio.sleep(0.01)  # Small delay
        
        await process_stream()
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        pipeline.reset()
        
        # Print stats
        print("\n" + "=" * 60)
        print("Session Statistics:")
        print("=" * 60)
        stats = pipeline.get_latency_stats()
        if stats:
            print(f"Chunks processed: {stats.get('count', 0)}")
            print(f"Average latency: {stats.get('mean', 0):.1f}ms")
            print(f"P95 latency: {stats.get('p95', 0):.1f}ms")
        print("\n✓ Demo complete!")


if __name__ == '__main__':
    asyncio.run(main())
