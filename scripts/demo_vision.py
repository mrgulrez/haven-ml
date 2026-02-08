"""Demo script for testing vision pipeline with webcam.

Run this to see the vision models in action with your webcam.
"""

import cv2
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vision import VideoPipeline
from config import config
from utils.logger import setup_logger


async def main():
    """Run vision pipeline demo with webcam."""
    # Setup logging
    setup_logger("INFO")
    
    print("=" * 60)
    print("Vision Pipeline Webcam Demo")
    print("=" * 60)
    print("\nThis demo will:")
    print("- Detect facial expressions (8 emotions)")
    print("- Analyze posture (slouching, stillness, head-in-hands)")
    print("- Track gaze (direction, fixation, patterns)")
    print("\nPress 'q' to quit\n")
    
    # Initialize pipeline
    print("Initializing vision pipeline...")
    pipeline = VideoPipeline(config, use_mock=False)
    print("✓ Pipeline ready!\n")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Starting video stream...")
    print("=" * 60)
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            timestamp = frame_count / 30.0  # Assume 30 FPS
            result = await pipeline.process_frame(frame_rgb, timestamp)
            
            # Print results every 30 frames
            if frame_count % 30 == 0:
                visual_state = result['visual_state']
                print(f"\nFrame {frame_count}:")
                print(f"  Emotion: {visual_state['primary_emotion']} ({visual_state['emotion_confidence']:.2f})")
                print(f"  Posture: {visual_state['posture_state']}")
                print(f"  Gaze: {visual_state['gaze_pattern']}")
                print(f"  State: {visual_state['overall_state']} (V:{visual_state['valence']:.2f}, A:{visual_state['arousal']:.2f})")
                print(f"  Latency: {result['processing_time_ms']:.1f}ms")
            
            # Visualize
            annotated = pipeline.visualize(frame_rgb, result)
            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            
            # Display
            cv2.imshow('Vision Pipeline Demo', annotated_bgr)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        pipeline.cleanup()
        
        # Print stats
        print("\n" + "=" * 60)
        print("Session Statistics:")
        print("=" * 60)
        stats = pipeline.get_latency_stats()
        if stats:
            print(f"Frames processed: {stats.get('count', 0)}")
            print(f"Average latency: {stats.get('mean', 0):.1f}ms")
            print(f"P95 latency: {stats.get('p95', 0):.1f}ms")
        print("\n✓ Demo complete!")


if __name__ == '__main__':
    asyncio.run(main())
