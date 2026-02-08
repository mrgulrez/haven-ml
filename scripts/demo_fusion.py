"""Demo script for testing multimodal fusion.

Combines webcam and microphone to demonstrate real-time
emotion fusion with conflict detection.
"""

import cv2
import asyncio
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vision import VideoPipeline
from models.audio import AudioPipeline
from models.fusion import FusionEngine, TemporalSmoother
from config import config
from utils.logger import setup_logger

try:
    import sounddevice as sd
except ImportError:
    print("sounddevice not installed. Install with: pip install sounddevice")
    sys.exit(1)


async def main():
    """Run multimodal fusion demo."""
    setup_logger("INFO")
    
    print("=" * 70)
    print("Multimodal Fusion Demo - Video + Audio Emotion Recognition")
    print("=" * 70)
    print("\nThis demo combines:")
    print("- Video: Face emotion, posture, gaze")
    print("- Audio: Speech transcription, vocal tone, events")
    print("- Fusion: Integrated emotion with conflict detection")
    print("\nPress 'q' to quit\n")
    
    # Initialize pipelines
    print("Initializing pipelines...")
    video_pipeline = VideoPipeline(config, use_mock=False)
    audio_pipeline = AudioPipeline(config, use_mock=False)
    fusion_engine = FusionEngine(device='cpu')  # Use CPU for demo
    temporal_smoother = TemporalSmoother(window_size=30, alpha=0.2)
    print("✓ All systems ready!\n")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Audio settings
    sample_rate = 16000
    chunk_duration = 0.5
    chunk_samples = int(sample_rate * chunk_duration)
    
    # Shared state
    latest_audio_state = {'audio_emotion': 'neutral', 'arousal': 0.5, 'valence': 0.0}
    
    print("Starting multimodal stream...")
    print("=" * 70)
    
    frame_count = 0
    
    async def process_audio():
        """Audio processing coroutine."""
        nonlocal latest_audio_state
        
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=chunk_samples
        ) as stream:
            while True:
                audio_chunk, _ = stream.read(chunk_samples)
                audio_chunk = audio_chunk.flatten()
                
                timestamp = frame_count * 0.033  # ~30 FPS
                result = await audio_pipeline.process_audio(audio_chunk, timestamp)
                latest_audio_state = result['audio_state']
                
                await asyncio.sleep(0.01)
    
    async def process_video():
        """Video processing coroutine."""
        nonlocal frame_count
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process video
            timestamp = frame_count / 30.0
            video_result = await video_pipeline.process_frame(frame_rgb, timestamp)
            visual_state = video_result['visual_state']
            
            # Fuse with latest audio
            fused = fusion_engine.fuse(visual_state, latest_audio_state)
            smoothed = temporal_smoother.smooth(fused)
            
            # Display results every 30 frames
            if frame_count % 30 == 0:
                print(f"\n[{timestamp:.1f}s] FUSED EMOTION:")
                print(f"  Emotion: {smoothed.primary_emotion} (confidence: {smoothed.confidence:.2f})")
                print(f"  Valence: {smoothed.valence:.2f}, Arousal: {smoothed.arousal:.2f}")
                print(f"  Stability: {smoothed.emotion_stability:.2f}, Trend: {smoothed.trend}")
                print(f"  Authenticity: {smoothed.authenticity_score:.2f}", end="")
                
                if smoothed.authenticity_score < 0.5:
                    print(" ⚠️ EMOTIONAL MASKING DETECTED")
                else:
                    print()
                
                print(f"  Visual: {visual_state['overall_state']} ({fused.visual_weight:.2f})")
                print(f"  Audio: {latest_audio_state.get('audio_emotion', 'unknown')} ({fused.audio_weight:.2f})")
            
            # Visualize
            annotated = video_pipeline.visualize(frame_rgb, video_result)
            
            # Add fusion info overlay
            h, w = annotated.shape[:2]
            overlay_text = [
                f"FUSED: {smoothed.primary_emotion}",
                f"V:{smoothed.valence:.2f} A:{smoothed.arousal:.2f}",
                f"Auth:{smoothed.authenticity_score:.2f}"
            ]
            
            y_offset = h - 100
            for i, text in enumerate(overlay_text):
                cv2.putText(
                    annotated,
                    text,
                    (10, y_offset + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 165, 0),
                    2
                )
            
            # Display
            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            cv2.imshow('Multimodal Fusion Demo', annotated_bgr)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
            await asyncio.sleep(0.01)
    
    try:
        # Run both coroutines
        await asyncio.gather(
            process_video(),
            process_audio()
        )
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    finally:
        # Cleanup
        print("\nCleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        video_pipeline.cleanup()
        audio_pipeline.reset()
        print("\n✓ Demo complete!")


if __name__ == '__main__':
    asyncio.run(main())
