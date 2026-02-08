"""Whisper-based speech-to-text for better compatibility."""

import numpy as np
from typing import Dict, Optional
from loguru import logger

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Whisper not available")


class WhisperSTT:
    """
    OpenAI Whisper speech-to-text.
    
    More compatible than SenseVoice and works better with modern PyTorch.
    """
    
    def __init__(self, model_size: str = 'base', device: str = 'cpu'):
        """
        Initialize Whisper STT.
        
        Args:
            model_size: tiny, base, small, medium, large
            device: cpu or cuda
        """
        if not WHISPER_AVAILABLE:
            raise ImportError("Whisper not installed. Install with: pip install openai-whisper")
        
        self.model_size = model_size
        self.device = device
        
        logger.info(f"Loading Whisper {model_size} model...")
        self.model = whisper.load_model(model_size, device=device)
        logger.info(f"✓ Whisper {model_size} loaded on {device}")
        
        self.sample_rate = 16000
    
    def transcribe(self, audio: np.ndarray) -> Dict:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio samples (float32, 16kHz)
            
        Returns:
            Dictionary with transcription results
        """
        try:
            # Whisper expects audio as float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Transcribe
            result = self.model.transcribe(
                audio,
                language='en',
                fp16=False  # Use FP32 for CPU
            )
            
            return {
                'text': result['text'].strip(),
                'language': result.get('language', 'en'),
                'confidence': 1.0  # Whisper doesn't provide confidence
            }
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return {
                'text': '',
                'language': 'en',
                'confidence': 0.0
            }


# Mock for testing
class MockWhisperSTT:
    """Mock Whisper for testing."""
    
    def __init__(self, model_size: str = 'base', device: str = 'cpu'):
        self.model_size = model_size
        self.device = device
        self.sample_rate = 16000
        logger.info("Using MockWhisperSTT")
    
    def transcribe(self, audio: np.ndarray) -> Dict:
        """Return mock transcription."""
        return {
            'text': 'This is a mock transcription',
            'language': 'en',
            'confidence': 0.8
        }
