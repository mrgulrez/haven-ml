"""Test suite for TTS models."""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.tts import CosyVoiceTTS, MockTTS


# TTS Tests
def test_mock_tts_initialization():
    """Test mock TTS initialization."""
    tts = MockTTS()
    assert tts.sample_rate == 22050


def test_mock_tts_synthesis():
    """Test mock synthesis."""
    tts = MockTTS()
    
    text = "Hello, how are you today?"
    audio = tts.synthesize(text)
    
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert len(audio) > 0


def test_empty_text():
    """Test synthesis with empty text."""
    tts = MockTTS()
    
    audio = tts.synthesize("")
    assert isinstance(audio, np.ndarray)


def test_streaming_synthesis():
    """Test streaming synthesis."""
    tts = MockTTS()
    
    text = "This is a test of streaming synthesis."
    chunks = list(tts.synthesize_stream(text, chunk_size=2048))
    
    assert len(chunks) > 0
    assert all(isinstance(c, np.ndarray) for c in chunks)


def test_emotional_modulation():
    """Test emotional modulation."""
    tts = MockTTS()
    
    text = "I'm feeling great!"
    
    # Happy (positive valence, high arousal)
    audio_happy = tts.modulate_emotion(text, valence=0.8, arousal=0.7)
    assert isinstance(audio_happy, np.ndarray)
    
    # Sad (negative valence, low arousal)
    audio_sad = tts.modulate_emotion(text, valence=-0.6, arousal=0.3)
    assert isinstance(audio_sad, np.ndarray)


def test_audio_duration():
    """Test that duration correlates with text length."""
    tts = MockTTS()
    
    short_text = "Hello"
    long_text = "This is a much longer sentence with many more words to synthesize."
    
    audio_short = tts.synthesize(short_text)
    audio_long = tts.synthesize(long_text)
    
    # Longer text should produce longer audio
    assert len(audio_long) > len(audio_short)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
