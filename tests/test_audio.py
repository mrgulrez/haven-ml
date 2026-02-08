"""Test suite for audio models."""

import pytest
import asyncio
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.audio import (
    SileroVAD,
    SenseVoiceSTT,
    ProsodyAnalyzer,
    AudioEventDetector,
    AudioPipeline,
    MockVAD,
    MockSTT,
    MockProsodyAnalyzer,
    MockEventDetector
)
from config import config


# Test fixtures
@pytest.fixture
def test_audio():
    """Create dummy audio for testing."""
    # Generate 1 second of audio (16kHz)
    duration = 1.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Simple sine wave
    audio = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio


@pytest.fixture
def silence_audio():
    """Create silent audio."""
    return np.zeros(16000, dtype=np.float32)


@pytest.fixture
def audio_pipeline():
    """Create audio pipeline with mock models."""
    return AudioPipeline(config, use_mock=True)


# VAD Tests
def test_mock_vad(test_audio):
    """Test mock VAD."""
    vad = MockVAD()
    result = vad.detect(test_audio)
    
    assert 'is_speech' in result
    assert 'probability' in result
    assert 'event' in result


def test_vad_events(test_audio, silence_audio):
    """Test VAD event detection."""
    vad = MockVAD()
    
    # Silent audio
    result1 = vad.detect(silence_audio)
    assert result1['is_speech'] == False
    
    # Speech audio
    result2 = vad.detect(test_audio)
    # Mock VAD should detect speech in non-silent audio


def test_vad_reset(test_audio):
    """Test VAD reset."""
    vad = MockVAD()
    vad.detect(test_audio)
    vad.reset()
    # Should reset state without errors


# STT Tests
def test_mock_stt(test_audio):
    """Test mock STT."""
    stt = MockSTT()
    result = stt.transcribe(test_audio)
    
    assert 'text' in result
    assert 'language' in result
    assert 'confidence' in result
    assert 'duration_ms' in result


def test_stt_short_audio():
    """Test STT with short audio."""
    stt = MockSTT()
    short_audio = np.zeros(1000, dtype=np.float32)
    result = stt.transcribe(short_audio)
    
    # Short audio may return empty string
    assert isinstance(result['text'], str)


# Prosody Tests
def test_mock_prosody(test_audio):
    """Test mock prosody analyzer."""
    analyzer = MockProsodyAnalyzer()
    result = analyzer.analyze(test_audio)
    
    assert 'arousal' in result
    assert 'valence' in result
    assert 'dominance' in result
    assert 'pitch_mean' in result
    assert 'tempo' in result
    assert 'tremor' in result
    assert 'energy' in result


def test_prosody_values_in_range(test_audio):
    """Test prosody values are in valid ranges."""
    analyzer = MockProsodyAnalyzer()
    result = analyzer.analyze(test_audio)
    
    assert 0.0 <= result['arousal'] <= 1.0
    assert -1.0 <= result['valence'] <= 1.0
    assert -1.0 <= result['dominance'] <= 1.0
    assert 0.0 <= result['tremor'] <= 1.0


def test_emotion_classification(test_audio):
    """Test emotion classification from prosody."""
    analyzer = MockProsodyAnalyzer()
    prosody = analyzer.analyze(test_audio)
    emotion = analyzer.classify_emotion_from_prosody(prosody)
    
    assert emotion in ['excited', 'stressed', 'calm_positive', 'sad', 
                      'positive', 'negative', 'neutral']


# Event Detector Tests
def test_mock_event_detector(test_audio):
    """Test mock event detector."""
    detector = MockEventDetector()
    result = detector.detect(test_audio, timestamp=0.0, is_speech=False)
    
    assert 'events' in result
    assert 'is_silence' in result
    assert 'silence_duration' in result


def test_event_detector_reset():
    """Test event detector reset."""
    detector = MockEventDetector()
    detector.reset()
    # Should reset without errors


# Audio Pipeline Tests
@pytest.mark.asyncio
async def test_audio_pipeline_processing(audio_pipeline, test_audio):
    """Test audio pipeline processing."""
    result = await audio_pipeline.process_audio(test_audio, timestamp=0.0)
    
    assert 'vad' in result
    assert 'transcription' in result
    assert 'prosody' in result
    assert 'events' in result
    assert 'audio_state' in result
    assert 'processing_time_ms' in result


@pytest.mark.asyncio
async def test_audio_pipeline_state(audio_pipeline, test_audio):
    """Test audio state aggregation."""
    result = await audio_pipeline.process_audio(test_audio, timestamp=0.0)
    
    audio_state = result['audio_state']
    assert 'audio_emotion' in audio_state
    assert 'arousal' in audio_state
    assert 'valence' in audio_state
    
    # Check ranges
    assert 0.0 <= audio_state['arousal'] <= 1.0
    assert -1.0 <= audio_state['valence'] <= 1.0


@pytest.mark.asyncio
async def test_audio_pipeline_speech_buffering(audio_pipeline, test_audio):
    """Test speech buffering mechanism."""
    # Simulate speech sequence: start -> continue -> end
    
    # This test uses mocks, so we just verify no errors
    result1 = await audio_pipeline.process_audio(test_audio, timestamp=0.0)
    result2 = await audio_pipeline.process_audio(test_audio, timestamp=1.0)
    result3 = await audio_pipeline.process_audio(test_audio, timestamp=2.0)
    
    # Should complete without errors
    assert result1 is not None
    assert result2 is not None
    assert result3 is not None


def test_audio_pipeline_reset(audio_pipeline):
    """Test pipeline reset."""
    audio_pipeline.reset()
    # Should reset without errors


@pytest.mark.asyncio
async def test_audio_pipeline_latency_tracking(audio_pipeline, test_audio):
    """Test latency tracking."""
    # Process multiple chunks
    for i in range(5):
        await audio_pipeline.process_audio(test_audio, timestamp=float(i))
    
    stats = audio_pipeline.get_latency_stats()
    
    if stats:
        assert 'count' in stats
        assert 'mean' in stats


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
