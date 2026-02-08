"""Simple main entry point for testing the empathy system."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agents import EmpathyAgent
from utils.logger import setup_logger


async def main():
    """Run empathy agent in testing mode."""
    setup_logger("INFO")
    
    print("=" * 70)
    print("EMPATHY SYSTEM - Quick Test")
    print("=" * 70)
    print("\nInitializing agent with mock models for testing...")
    
    # Initialize with mock models (fast testing)
    agent = EmpathyAgent('test_user', persona='remote_worker', use_mock=True)
    await agent.start_session('test_session')
    
    print("✓ Agent initialized successfully!\n")
    print("Running quick test scenarios...\n")
    
    # Scenario 1: Process some frames
    print("1. Testing video processing...")
    import numpy as np
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    visual_state = await agent.process_video_frame(test_frame, timestamp=0.0)
    print(f"   ✓ Visual state: {visual_state['primary_emotion']}")
    
    # Scenario 2: Process audio
    print("2. Testing audio processing...")
    test_audio = np.random.randn(16000).astype(np.float32) * 0.1
    
    audio_result = await agent.process_audio_chunk(test_audio, timestamp=0.0)
    print(f"   ✓ Audio processed")
    
    # Scenario 3: Fusion
    print("3. Testing multimodal fusion...")
    fused_state = await agent.process_multimodal(visual_state, audio_result)
    print(f"   ✓ Fused emotion: {fused_state['primary_emotion']}")
    print(f"   ✓ Valence: {fused_state['valence']:.2f}, Arousal: {fused_state['arousal']:.2f}")
    
    # Scenario 4: Generate response
    print("4. Testing LLM response generation...")
    response = await agent.generate_response(user_text="Hello, how are you?")
    print(f"   ✓ Response: \"{response}\"")
    
    # Scenario 5: Synthesize speech
    print("5. Testing TTS synthesis...")
    audio = await agent.synthesize_speech("This is a test.")
    print(f"   ✓ Synthesized {len(audio)} audio samples")
    
    # Scenario 6: Check interventions
    print("6. Testing intervention logic...")
    intervention = await agent.check_intervention_triggers()
    print(f"   ✓ Intervention check: {intervention or 'None triggered'}")
    
    # Summary
    print("\n" + "=" * 70)
    summary = agent.get_session_summary()
    print(f"Session Summary:")
    print(f"  Dominant emotion: {summary['dominant_emotion']}")
    print(f"  Average valence: {summary['avg_valence']:.2f}")
    print(f"  Trend: {summary['trend']}")
    
    # Cleanup
    await agent.cleanup()
    
    print("\n✓ All tests passed!")
    print("=" * 70)
    print("\nNext steps:")
    print("  - Run full demos: python scripts/demo_full_integration.py")
    print("  - Run tests: pytest tests/ -v")
    print("  - Deploy: See DEPLOYMENT.md")


if __name__ == '__main__':
    asyncio.run(main())
