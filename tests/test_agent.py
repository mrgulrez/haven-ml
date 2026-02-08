"""Test suite for main agent."""

import pytest
import asyncio
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import EmpathyAgent
from config import config


@pytest.fixture
def agent():
    """Create test agent."""
    agent = EmpathyAgent('test_user', persona='remote_worker', use_mock=True)
    return agent


@pytest.fixture
def test_frame():
    """Create test video frame."""
    # 640x480 RGB frame
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def test_audio():
    """Create test audio chunk."""
    # 1 second at 16kHz
    return np.random.randn(16000).astype(np.float32) * 0.1


@pytest.mark.asyncio
async def test_agent_initialization(agent):
    """Test agent initializes successfully."""
    assert agent.user_id == 'test_user'
    assert agent.persona == 'remote_worker'


@pytest.mark.asyncio
async def test_start_session(agent):
    """Test starting a session."""
    await agent.start_session('test_session_123')
    
    assert agent.session_memory.session_id == 'test_session_123'
    assert agent.session_memory.user_id == 'test_user'


@pytest.mark.asyncio
async def test_process_video_frame(agent, test_frame):
    """Test video frame processing."""
    await agent.start_session('test_session')
    
    visual_state = await agent.process_video_frame(test_frame, timestamp=0.0)
    
    assert isinstance(visual_state, dict)
    assert 'valence' in visual_state
    assert 'arousal' in visual_state


@pytest.mark.asyncio
async def test_process_audio_chunk(agent, test_audio):
    """Test audio chunk processing."""
    await agent.start_session('test_session')
    
    audio_result = await agent.process_audio_chunk(test_audio, timestamp=0.0)
    
    assert isinstance(audio_result, dict)
    assert 'audio_state' in audio_result


@pytest.mark.asyncio
async def test_multimodal_fusion(agent, test_frame, test_audio):
    """Test multimodal fusion."""
    await agent.start_session('test_session')
    
    # Process both modalities
    visual_state = await agent.process_video_frame(test_frame, timestamp=0.0)
    audio_result = await agent.process_audio_chunk(test_audio, timestamp=0.0)
    
    # Fuse
    fused_state = await agent.process_multimodal(visual_state, audio_result)
    
    assert isinstance(fused_state, dict)
    assert 'primary_emotion' in fused_state
    assert 'valence' in fused_state
    assert 'arousal' in fused_state
    assert 'authenticity_score' in fused_state


@pytest.mark.asyncio
async def test_generate_response(agent, test_frame, test_audio):
    """Test response generation."""
    await agent.start_session('test_session')
    
    # Set up emotional state
    visual_state = await agent.process_video_frame(test_frame, timestamp=0.0)
    audio_result = await agent.process_audio_chunk(test_audio, timestamp=0.0)
    await agent.process_multimodal(visual_state, audio_result)
    
    # Generate response
    response = await agent.generate_response(user_text="I'm feeling stressed")
    
    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.asyncio
async def test_synthesize_speech(agent, test_frame, test_audio):
    """Test speech synthesis."""
    await agent.start_session('test_session')
    
    # Set up emotional state
    visual_state = await agent.process_video_frame(test_frame, timestamp=0.0)
    audio_result = await agent.process_audio_chunk(test_audio, timestamp=0.0)
    await agent.process_multimodal(visual_state, audio_result)
    
    # Synthesize
    audio = await agent.synthesize_speech("This is a test.")
    
    assert isinstance(audio, np.ndarray)
    assert len(audio) > 0


@pytest.mark.asyncio
async def test_handle_user_speech(agent, test_frame, test_audio):
    """Test end-to-end speech handling."""
    await agent.start_session('test_session')
    
    # Set up emotional state
    visual_state = await agent.process_video_frame(test_frame, timestamp=0.0)
    audio_result = await agent.process_audio_chunk(test_audio, timestamp=0.0)
    await agent.process_multimodal(visual_state, audio_result)
    
    # Handle speech
    response_text, response_audio = await agent.handle_user_speech("Hello")
    
    assert isinstance(response_text, str)
    assert isinstance(response_audio, np.ndarray)


@pytest.mark.asyncio
async def test_intervention_check(agent, test_frame, test_audio):
    """Test intervention trigger checking."""
    await agent.start_session('test_session')
    
    # Process some frames
    for i in range(5):
        visual_state = await agent.process_video_frame(test_frame, timestamp=float(i))
        audio_result = await agent.process_audio_chunk(test_audio, timestamp=float(i))
        await agent.process_multimodal(visual_state, audio_result)
    
    # Check interventions
    intervention_type = await agent.check_intervention_triggers()
    
    # Should return either None or a string
    assert intervention_type is None or isinstance(intervention_type, str)


@pytest.mark.asyncio
async def test_session_summary(agent, test_frame, test_audio):
    """Test session summary generation."""
    await agent.start_session('test_session')
    
    # Process some frames
    for i in range(3):
        visual_state = await agent.process_video_frame(test_frame, timestamp=float(i))
        audio_result = await agent.process_audio_chunk(test_audio, timestamp=float(i))
        await agent.process_multimodal(visual_state, audio_result)
    
    summary = agent.get_session_summary()
    
    assert isinstance(summary, dict)
    assert 'dominant_emotion' in summary


@pytest.mark.asyncio
async def test_cleanup(agent):
    """Test agent cleanup."""
    await agent.start_session('test_session')
    await agent.cleanup()
    
    # Should complete without errors


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
