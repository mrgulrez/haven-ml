"""Test suite for LLM and memory modules."""

import pytest
from pathlib import Path
import sys
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import LlamaClient, MockLlamaClient, PromptBuilder
from memory import SessionMemory, UserProfile, UserPreferences
from config import config


# LLM Tests
def test_mock_llama_client():
    """Test mock Llama client."""
    client = MockLlamaClient()
    
    response = client.generate("Hello")
    assert isinstance(response, str)
    assert len(response) > 0


def test_mock_llama_streaming():
    """Test mock streaming generation."""
    client = MockLlamaClient()
    
    tokens = list(client.generate_stream("Hello"))
    assert len(tokens) > 0
    assert all(isinstance(t, str) for t in tokens)


def test_token_counting():
    """Test token counting."""
    client = MockLlamaClient()
    
    text = "This is a test sentence."
    count = client.count_tokens(text)
    assert count > 0


# Prompt Builder Tests
def test_prompt_builder_initialization():
    """Test prompt builder initialization."""
    builder = PromptBuilder(persona='remote_worker')
    assert builder.persona == 'remote_worker'


def test_build_prompt_with_emotion():
    """Test building prompt with emotional context."""
    builder = PromptBuilder(persona='remote_worker')
    
    emotional_state = {
        'primary_emotion': 'sad',
        'confidence': 0.85,
        'valence': -0.5,
        'arousal': 0.3,
        'authenticity_score': 1.0,
        'trend': 'stable',
        'emotion_stability': 0.7
    }
    
    prompt = builder.build_prompt(
        emotional_state=emotional_state,
        user_text="I'm feeling down today"
    )
    
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert 'sad' in prompt
    assert 'feeling down' in prompt


def test_build_prompt_with_masking():
    """Test prompt includes masking warning."""
    builder = PromptBuilder(persona='remote_worker')
    
    emotional_state = {
        'primary_emotion': 'happy',
        'confidence': 0.7,
        'valence': 0.5,
        'arousal': 0.6,
        'authenticity_score': 0.3,  # Low authenticity
        'trend': 'stable',
        'emotion_stability': 0.5
    }
    
    prompt = builder.build_prompt(
        emotional_state=emotional_state,
        user_text="I'm fine"
    )
    
    # Should include masking warning
    assert 'hiding' in prompt.lower() or 'masking' in prompt.lower()


def test_conversation_history():
    """Test conversation history management."""
    builder = PromptBuilder()
    
    # Add messages
    builder.add_to_history('user', 'Hello')
    builder.add_to_history('assistant', 'Hi there!')
    
    assert len(builder.conversation_history) == 2
    
    # Clear history
    builder.clear_history()
    assert len(builder.conversation_history) == 0


def test_persona_switching():
    """Test switching personas."""
    builder = PromptBuilder(persona='remote_worker')
    
    builder.set_persona('student')
    assert builder.persona == 'student'


# Session Memory Tests
def test_session_memory_initialization():
    """Test session memory initialization."""
    memory = SessionMemory()
    assert memory.session_id is None


def test_start_session():
    """Test starting a session."""
    memory = SessionMemory()
    memory.start_session('session123', 'user456', persona='student')
    
    assert memory.session_id == 'session123'
    assert memory.user_id == 'user456'
    assert memory.persona == 'student'
    assert memory.start_time is not None


def test_add_emotional_snapshot():
    """Test adding emotional snapshots."""
    memory = SessionMemory()
    memory.start_session('s1', 'u1')
    
    memory.add_emotional_snapshot(
        emotion='happy',
        valence=0.7,
        arousal=0.6,
        context='work completed'
    )
    
    assert len(memory.emotional_timeline) == 1


def test_emotional_shift_detection():
    """Test detecting emotional shifts."""
    memory = SessionMemory()
    memory.start_session('s1', 'u1')
    
    # Add contrasting emotions
    memory.add_emotional_snapshot('happy', valence=0.7, arousal=0.6)
    memory.add_emotional_snapshot('sad', valence=-0.6, arousal=0.3)
    
    # Should have recorded shift event
    events = [e for e in memory.significant_events if e['type'] == 'emotional_shift']
    assert len(events) > 0


def test_get_emotional_summary():
    """Test emotional summary generation."""
    memory = SessionMemory()
    memory.start_session('s1', 'u1')
    
    # Add several snapshots
    for i in range(5):
        memory.add_emotional_snapshot('happy', valence=0.5, arousal=0.5)
    
    summary = memory.get_emotional_summary(window_minutes=30)
    
    assert 'dominant_emotion' in summary
    assert summary['dominant_emotion'] == 'happy'


def test_session_duration():
    """Test session duration tracking."""
    memory = SessionMemory()
    memory.start_session('s1', 'u1')
    
    duration = memory.get_session_duration()
    assert isinstance(duration, timedelta)


# User Profile Tests
def test_user_profile_creation():
    """Test creating a new user profile."""
    profile = UserProfile('test_user')
    
    assert profile.user_id == 'test_user'
    assert isinstance(profile.preferences, UserPreferences)


def test_update_preferences():
    """Test updating preferences."""
    profile = UserProfile('test_user')
    
    profile.update_preferences(persona='student', privacy_mode=True)
    
    assert profile.preferences.persona == 'student'
    assert profile.preferences.privacy_mode == True


def test_record_session():
    """Test recording session start."""
    profile = UserProfile('test_user')
    
    initial_count = profile.total_sessions
    profile.record_session_start()
    
    assert profile.total_sessions == initial_count + 1
    assert profile.last_session is not None


def test_learn_pattern():
    """Test learning emotional patterns."""
    profile = UserProfile('test_user')
    
    profile.learn_pattern('morning_stress', 'User shows stress in mornings')
    assert len(profile.emotional_patterns) == 1
    
    # Learn same pattern again
    profile.learn_pattern('morning_stress', 'User shows stress in mornings')
    assert len(profile.emotional_patterns) == 1
    assert profile.emotional_patterns[0].frequency == 2


def test_intervention_effectiveness():
    """Test intervention effectiveness tracking."""
    profile = UserProfile('test_user')
    
    # Record some interventions
    profile.record_intervention('break_reminder', was_effective=True)
    profile.record_intervention('break_reminder', was_effective=True)
    profile.record_intervention('break_reminder', was_effective=False)
    
    effectiveness = profile.get_intervention_effectiveness('break_reminder')
    assert effectiveness is not None
    assert 0.6 < effectiveness < 0.7  # 2/3


def test_should_intervene():
    """Test intervention decision logic."""
    profile = UserProfile('test_user')
    
    # With default preferences
    decision = profile.should_intervene('break_reminder')
    assert isinstance(decision, bool)
    
    # Disable proactive
    profile.update_preferences(enable_proactive=False)
    assert profile.should_intervene('break_reminder') == False


def test_profile_persistence(tmp_path):
    """Test saving and loading profile."""
    # Create and save profile
    profile1 = UserProfile('test_user', storage_path=str(tmp_path))
    profile1.update_preferences(persona='student')
    profile1.learn_pattern('evening_calm', 'User is calm in evenings')
    profile1.save()
    
    # Load in new instance
    profile2 = UserProfile('test_user', storage_path=str(tmp_path))
    
    assert profile2.preferences.persona == 'student'
    assert len(profile2.emotional_patterns) == 1
    assert profile2.emotional_patterns[0].pattern_type == 'evening_calm'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
