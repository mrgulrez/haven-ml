"""Test suite for configuration system."""

import pytest
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

def test_config_loading():
    """Test that configuration loads successfully."""
    config = Config()
    assert config._config is not None
    assert 'system' in config._config

def test_config_dot_notation():
    """Test dot notation access."""
    config = Config()
    log_level = config.get('system.log_level')
    assert log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR']

def test_config_default_values():
    """Test default values for missing keys."""
    config = Config()
    value = config.get('nonexistent.key', 'default_value')
    assert value == 'default_value'

def test_config_properties():
    """Test configuration property accessors."""
    config = Config()
    assert isinstance(config.system, dict)
    assert isinstance(config.models, dict)
    assert isinstance(config.personas, dict)

def test_persona_configs():
    """Test that all required personas are configured."""
    config = Config()
    personas = config.personas
    
    assert 'remote_worker' in personas
    assert 'student' in personas
    assert 'young_professional' in personas
    
    # Check persona structure
    for persona in personas.values():
        assert 'intervention_style' in persona
        assert 'focus_areas' in persona
        assert 'tone' in persona

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
