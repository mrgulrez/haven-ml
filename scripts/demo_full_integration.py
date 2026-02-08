"""Full integration demo - Complete empathy agent.

Demonstrates the entire pipeline from webcam/mic to empathetic voice response.
"""

import cv2
import asyncio
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import EmpathyAgent
from utils.logger import setup_logger

try:
    import sounddevice as sd
except ImportError:
    print("sounddevice not installed. Install with: pip install sounddevice")
    sd = None


async def main():
    """Run full integration demo."""
    setup_logger("INFO")
    
    print("=" * 80)
    print("EMPATHY AGENT - FULL INTEGRATION DEMO")
    print("=" * 80)
    print("\nThis demo shows the complete system:")
    print("📹 Video: Face emotion, posture, gaze")
    print("🎤 Audio: Speech, tone, events")
    print("🧠 Fusion: Integrated emotional state")
    print("💬 LLM: Context-aware responses")
    print("🔊 TTS: Emotionally-modulated voice")
    print("\nPress 'q' to quit, 's' to speak\n")
    
    # Initialize agent
    print("Initializing EmpathyAgent...")
    print("Attempting to load real AI models...")
    print("(This may take 30-60 seconds on first run)")
    agent = EmpathyAgent('demo_user', persona='remote_worker', use_mock=False)
    
    # CRITICAL: Log which detector is actually being used
    detector_type = type(agent.video_pipeline.face_detector).__name__
    print(f"\n⚠️  DETECTOR TYPE: {detector_type}")
    if 'Mock' in detector_type:
        print("❌ WARNING: Using MOCK detector - real emotions won't work!")
    else:
        print("✅ SUCCESS: Using REAL detector - emotions will update!")
    
    await agent.start_session('demo_session_001')
    print("✓ Agent ready!\n")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Audio settings
    sample_rate = 16000
    chunk_duration = 0.5
    chunk_samples = int(sample_rate * chunk_duration)
    
    # State
    frame_count = 0
    speaking_mode = False
    transcription_buffer = []
    
    print("Starting main loop...")
    print("=" * 80)
    
    async def process_audio_stream():
        """Background audio processing."""
        nonlocal speaking_mode, transcription_buffer
        
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=chunk_samples
        ) as stream:
            while True:
                audio_chunk, _ = stream.read(chunk_samples)
                audio_chunk = audio_chunk.flatten()
                
                timestamp = frame_count * 0.033
                audio_result = await agent.process_audio_chunk(audio_chunk, timestamp)
                
                # Check for speech transcription
                if audio_result.get('transcription') and audio_result['transcription'].get('text'):
                    transcribed = audio_result['transcription']['text']
                    if transcribed:
                        transcription_buffer.append(transcribed)
                        print(f"\n📝 Transcribed: \"{transcribed}\"")
                        
                        # Generate response
                        response_text, response_audio = await agent.handle_user_speech(transcribed)
                        print(f"🤖 Response: \"{response_text}\"")
                        
                        # Play response
                        if sd:
                            sd.play(response_audio, agent.tts.sample_rate)
                            sd.wait()
                
                await asyncio.sleep(0.01)
    
    async def process_video_stream():
        """Main video processing with display."""
        nonlocal frame_count
        
        intervention_check_counter = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process video
            timestamp = frame_count / 30.0
            visual_state = await agent.process_video_frame(frame_rgb, timestamp)
            
            # Get latest audio state (from background thread)
            # For demo simplicity, create dummy audio
            dummy_audio = np.random.randn(chunk_samples).astype(np.float32) * 0.01
            audio_result = await agent.process_audio_chunk(dummy_audio, timestamp)
            
            # Fuse
            fused_state = await agent.process_multimodal(visual_state, audio_result)
            
            # Display info every 30 frames
            if frame_count % 30 == 0:
                print(f"\n[{timestamp:.1f}s] Emotional State:")
                print(f"  Emotion: {fused_state['primary_emotion']}")
                print(f"  Valence: {fused_state['valence']:.2f}, Arousal: {fused_state['arousal']:.2f}")
                print(f"  Stability: {fused_state['emotion_stability']:.2f}, Trend: {fused_state['trend']}")
                
                if fused_state['authenticity_score'] < 0.5:
                    print(f"  ⚠️  MASKING DETECTED (authenticity: {fused_state['authenticity_score']:.2f})")
            
            # Check for proactive interventions (every 90 frames = ~3 seconds)
            intervention_check_counter += 1
            if intervention_check_counter >= 90:
                intervention_check_counter = 0
                
                intervention_result = await agent.proactive_intervention()
                if intervention_result:
                    intervention_text, intervention_audio = intervention_result
                    print(f"\n🚨 PROACTIVE INTERVENTION:")
                    print(f"   {intervention_text}")
                    
                    # Play intervention
                    if sd:
                        sd.play(intervention_audio, agent.tts.sample_rate)
                        sd.wait()
            
            # Visualize (simple overlay)
            display_frame = frame.copy()
            h, w = display_frame.shape[:2]
            
            # Overlay emotional state
            overlay_text = [
                f"Emotion: {fused_state['primary_emotion']}",
                f"V:{fused_state['valence']:.2f} A:{fused_state['arousal']:.2f}",
                f"Auth:{fused_state['authenticity_score']:.2f}"
            ]
            
            y_offset = 30
            for i, text in enumerate(overlay_text):
                cv2.putText(
                    display_frame,
                    text,
                    (10, y_offset + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            
            # Display
            cv2.imshow('Empathy Agent - Full Integration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print("\n🎤 Speak now...")
                speaking_mode = True
            
            frame_count += 1
            await asyncio.sleep(0.01)
    
    try:
        # Run video processing (audio in background if available)
        if sd:
            await asyncio.gather(
                process_video_stream(),
                process_audio_stream()
            )
        else:
            await process_video_stream()
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        
        # Show session summary
        print("\n" + "=" * 80)
        print("SESSION SUMMARY")
        print("=" * 80)
        
        summary = agent.get_session_summary()
        print(f"Dominant emotion: {summary['dominant_emotion']}")
        print(f"Average valence: {summary['avg_valence']:.2f}")
        print(f"Average arousal: {summary['avg_arousal']:.2f}")
        print(f"Trend: {summary['trend']}")
        
        await agent.cleanup()
        print("\n✓ Full integration demo complete!")


if __name__ == '__main__':
    asyncio.run(main())
