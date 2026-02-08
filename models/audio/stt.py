"""Speech-to-Text using SenseVoice.

Fast, multilingual transcription optimized for real-time applications.
"""

import torch
import numpy as np
from typing import Dict, Optional, List
from loguru import logger

from utils.helpers import timeit


class SenseVoiceSTT:
    """
    Speech-to-Text using SenseVoice.
    
    Faster than Whisper with comparable accuracy for streaming applications.
    """
    
    def __init__(
        self,
        model_name: str = "iic/SenseVoiceSmall",
        device: str = "cuda",
        language: str = "auto",
        use_int8: bool = True
    ):
        """
        Initialize SenseVoice STT.
        
        Args:
            model_name: SenseVoice model variant
            device: Device to run on (cuda/cpu)
            language: Language code ('auto', 'en', 'zh', 'ja', 'ko')
            use_int8: Use INT8 quantization for speed
        """
        self.device = device
        self.language = language
        self.model_name = model_name
        
        try:
            # Import SenseVoice
            from funasr import AutoModel
            
            self.model = AutoModel(
                model=model_name,
                device=device,
                disable_update=True,
                disable_pbar=True
            )
            
            logger.info(f"SenseVoice STT initialized with model '{model_name}' on {device}")
            
        except ImportError:
            logger.error("FunASR not installed. Install with: pip install funasr")
            raise
        except Exception as e:
            logger.error(f"Failed to load SenseVoice: {e}")
            raise
    
    @timeit
    def transcribe(
        self, 
        audio: np.ndarray, 
        sample_rate: int = 16000
    ) -> Dict[str, any]:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio samples as numpy array (float32)
            sample_rate: Audio sample rate
            
        Returns:
            Dictionary containing:
            - text: Transcribed text
            - language: Detected language
            - confidence: Transcription confidence
            - duration_ms: Audio duration
        """
        try:
            # Prepare input
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)  # Convert to mono
            
            # Run inference
            result = self.model.generate(
                input=audio,
                language=self.language if self.language != "auto" else None,
                batch_size_s=300
            )
            
            # Extract results
            if isinstance(result, list) and len(result) > 0:
                text = result[0].get('text', '')
                language = result[0].get('lang', 'unknown')
            else:
                text = ''
                language = 'unknown'
            
            # Calculate duration
            duration_ms = len(audio) / sample_rate * 1000
            
            return {
                'text': text.strip(),
                'language': language,
                'confidence': 1.0,  # SenseVoice doesn't provide confidence scores
                'duration_ms': duration_ms
            }
            
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return {
                'text': '',
                'language': 'unknown',
                'confidence': 0.0,
                'duration_ms': 0
            }
    
    def transcribe_streaming(
        self, 
        audio_chunks: List[np.ndarray]
    ) -> str:
        """
        Transcribe streaming audio chunks.
        
        Args:
            audio_chunks: List of audio chunks
            
        Returns:
            Concatenated transcription
        """
        # Concatenate chunks
        if not audio_chunks:
            return ''
        
        full_audio = np.concatenate(audio_chunks)
        result = self.transcribe(full_audio)
        return result['text']


class MockSTT:
    """Mock STT for testing without SenseVoice."""
    
    def __init__(self, *args, **kwargs):
        logger.warning("Using MockSTT - install funasr for real transcription")
    
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict:
        duration_ms = len(audio) / sample_rate * 1000
        
        # Return placeholder based on audio duration
        if duration_ms > 1000:
            text = "This is a mock transcription"
        else:
            text = ""
        
        return {
            'text': text,
            'language': 'en',
            'confidence': 1.0,
            'duration_ms': duration_ms
        }
    
    def transcribe_streaming(self, audio_chunks: List[np.ndarray]) -> str:
        return "Mock streaming transcription"
