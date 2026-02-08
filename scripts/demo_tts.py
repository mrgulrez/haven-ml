"""Demo script for TTS with emotional modulation.

Tests voice synthesis with different emotional tones.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.tts import MockTTS
from utils.logger import setup_logger

try:
    import sounddevice as sd
except ImportError:
    print("sounddevice not installed. Install with: pip install sounddevice")
    sd = None


def main():
    """Run TTS demo."""
    setup_logger("INFO")
    
    print("=" * 70)
    print("Text-to-Speech Demo - Emotional Voice Synthesis")
    print("=" * 70)
    print("\nThis demo shows how the system synthesizes speech with")
    print("different emotional tones based on VAD (Valence-Arousal-Dominance).\n")
    
    # Initialize TTS
    print("Initializing TTS engine...")
    tts = MockTTS()
    print("✓ TTS ready!\n")
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Happy & Energetic',
            'text': "That's wonderful! I'm so glad to hear things are going well for you.",
            'valence': 0.8,
            'arousal': 0.7,
            'description': 'Positive valence, high arousal'
        },
        {
            'name': 'Calm & Supportive',
            'text': "Take your time. There's no rush, and I'm here whenever you need me.",
            'valence': 0.4,
            'arousal': 0.2,
            'description': 'Positive valence, low arousal'
        },
        {
            'name': 'Empathetic & Concerned',
            'text': "I understand this is difficult. Would you like to talk about what's bothering you?",
            'valence': -0.3,
            'arousal': 0.5,
            'description': 'Slightly negative valence, medium arousal'
        },
        {
            'name': 'Gentle & Soothing',
            'text': "It's okay to feel this way. Remember to be kind to yourself.",
            'valence': 0.2,
            'arousal': 0.3,
            'description': 'Mildly positive valence, low arousal'
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario['name']}")
        print(f"   Text: \"{scenario['text']}\"")
        print(f"   Emotion: {scenario['description']}")
        print(f"   VAD: valence={scenario['valence']:.1f}, arousal={scenario['arousal']:.1f}")
        
        # Synthesize
        audio = tts.modulate_emotion(
            scenario['text'],
            valence=scenario['valence'],
            arousal=scenario['arousal']
        )
        
        duration = len(audio) / tts.sample_rate
        print(f"   Generated: {len(audio)} samples ({duration:.2f}s)")
        
        # Play if sounddevice available
        if sd:
            print("   ▶ Playing audio...")
            try:
                sd.play(audio, tts.sample_rate)
                sd.wait()
                print("   ✓ Done")
            except Exception as e:
                print(f"   ✗ Playback error: {e}")
        else:
            print("   (sounddevice not available for playback)")
        
        print()
    
    # Streaming demo
    print("\n" + "=" * 70)
    print("Streaming Synthesis Demo")
    print("=" * 70)
    
    long_text = """
    I want you to know that your feelings are valid. 
    It's completely normal to feel overwhelmed sometimes, 
    especially when you're juggling so many responsibilities.
    Remember that taking breaks isn't a sign of weakness, 
    it's an essential part of maintaining your wellbeing.
    """
    
    print(f"\nText: {long_text.strip()}\n")
    print("Synthesizing in chunks...")
    
    chunks = list(tts.synthesize_stream(long_text, chunk_size=4096))
    total_samples = sum(len(c) for c in chunks)
    duration = total_samples / tts.sample_rate
    
    print(f"✓ Generated {len(chunks)} chunks ({total_samples} samples, {duration:.2f}s)")
    
    if sd:
        print("▶ Playing streamed audio...")
        full_audio = np.concatenate(chunks)
        try:
            sd.play(full_audio, tts.sample_rate)
            sd.wait()
            print("✓ Done")
        except Exception as e:
            print(f"✗ Playback error: {e}")
    
    print("\n" + "=" * 70)
    print("✓ TTS Demo Complete!")
    print("=" * 70)
    print("\nNote: Using MockTTS for demo. Install TTS library and")
    print("download CosyVoice models for real voice synthesis.")


if __name__ == '__main__':
    main()
