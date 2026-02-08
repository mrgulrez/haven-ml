"""Configuration management for the Empathy System."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()

class Config:
    """Central configuration manager."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self.load()
    
    def load(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            content = f.read()
            # Replace environment variables in config
            for key, value in os.environ.items():
                content = content.replace(f"${{{key}}}", value)
            
            self._config = yaml.safe_load(content)
        
        logger.info(f"Configuration loaded from {self.config_path}")
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Example: config.get('models.vision.hsemotion.model_name')
        """
        keys = path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, path: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = path.split('.')
        config = self._config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    @property
    def system(self) -> Dict[str, Any]:
        return self._config.get('system', {})
    
    @property
    def livekit(self) -> Dict[str, Any]:
        return self._config.get('livekit', {})
    
    @property
    def models(self) -> Dict[str, Any]:
        return self._config.get('models', {})
    
    @property
    def memory(self) -> Dict[str, Any]:
        return self._config.get('memory', {})
    
    @property
    def intervention(self) -> Dict[str, Any]:
        return self._config.get('intervention', {})
    
    @property
    def privacy(self) -> Dict[str, Any]:
        return self._config.get('privacy', {})
    
    @property
    def personas(self) -> Dict[str, Any]:
        return self._config.get('personas', {})
    
    @property
    def performance(self) -> Dict[str, Any]:
        return self._config.get('performance', {})


# Global configuration instance
config = Config()
