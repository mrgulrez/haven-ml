"""Test suite for base agent interface."""

import pytest
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent

# Create a concrete implementation for testing
class TestAgent(BaseAgent):
    """Test implementation of BaseAgent."""
    
    async def initialize(self):
        self.initialized = True
    
    async def run(self):
        # Simple run loop that processes for 0.1s then stops
        await asyncio.sleep(0.1)
        await self.stop()
    
    async def cleanup(self):
        self.cleaned_up = True
    
    async def on_video_frame(self, frame):
        return {'emotion': 'neutral'}
    
    async def on_audio_packet(self, audio):
        return {'emotion': 'neutral'}
    
    async def on_multimodal_fusion(self, visual_data, audio_data):
        return {'valence': 0.0, 'arousal': 0.0}
    
    async def generate_response(self, emotion_state, user_text=None):
        return "Hello!"
    
    async def synthesize_speech(self, text, emotion_style):
        return b"audio_bytes"


@pytest.mark.asyncio
async def test_agent_initialization():
    """Test agent initialization."""
    agent = TestAgent(agent_id="test_001", config={})
    assert agent.agent_id == "test_001"
    assert agent.is_running == False

@pytest.mark.asyncio
async def test_agent_lifecycle():
    """Test full agent lifecycle."""
    agent = TestAgent(agent_id="test_002", config={})
    
    # Start agent (will run for 0.1s then stop)
    await agent.start(user_id="user_123", persona="remote_worker")
    
    assert hasattr(agent, 'initialized')
    assert hasattr(agent, 'cleaned_up')
    assert agent.user_id == "user_123"
    assert agent.persona == "remote_worker"

@pytest.mark.asyncio
async def test_agent_methods():
    """Test agent abstract methods."""
    agent = TestAgent(agent_id="test_003", config={})
    
    # Test perception methods
    video_result = await agent.on_video_frame({'frame': 'data'})
    assert 'emotion' in video_result
    
    audio_result = await agent.on_audio_packet({'audio': 'data'})
    assert 'emotion' in audio_result
    
    # Test fusion
    fusion_result = await agent.on_multimodal_fusion(video_result, audio_result)
    assert 'valence' in fusion_result
    assert 'arousal' in fusion_result
    
    # Test generation
    response = await agent.generate_response(fusion_result)
    assert isinstance(response, str)
    
    # Test TTS
    audio = await agent.synthesize_speech("Hello", "neutral")
    assert isinstance(audio, bytes)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
