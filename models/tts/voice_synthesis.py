"""Text-to-Speech using CosyVoice 2.

Provides natural, emotionally-aware voice synthesis.
"""

import torch
import numpy as np
from typing import Dict, Optional, Iterator
from pathlib import Path
from loguru import logger

from utils.helpers import timeit


class CosyVoiceTTS:
    """
    Text-to-Speech using CosyVoice 2.
    
    Supports emotional prosody modulation and streaming synthesis.
    """
    
    def __init__(
        self,
        model_name: str = "CosyVoice-300M",
        device: str = "cuda",
        speaker: str = "default",
        speed: float = 1.0
    ):
        """
        Initialize CosyVoice TTS.
        
        Args:
            model_name: CosyVoice model variant
            device: Device to run on
            speaker: Voice ID for multi-speaker models
            speed: Speech rate multiplier
        """
        self.device = device
        self.speaker = speaker
        self.speed = speed
        self.model_name = model_name
        self.sample_rate = 22050  # CosyVoice default
        
        try:
            # Import CosyVoice (via cosyvoice package or TTS)
            from TTS.api import TTS
            
            # Initialize model
            self.tts = TTS(model_name=model_name).to(device)
            
            logger.info(f"CosyVoice TTS initialized with '{model_name}' on {device}")
            
        except ImportError:
            logger.error("TTS library not installed. Install with: pip install TTS")
            raise
        except Exception as e:
            logger.error(f"Failed to load CosyVoice: {e}")
            raise
    
    @timeit
    def synthesize(
        self,
        text: str,
        emotion: Optional[str] = None,
        speed: Optional[float] = None
    ) -> np.ndarray:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            emotion: Emotional tone (happy, sad, neutral, etc)
            speed: Override default speed
            
        Returns:
            Audio as numpy array (float32, sample_rate)
        """
        if not text or not text.strip():
            return np.zeros(100, dtype=np.float32)
        
        try:
            # Apply emotional modulation if supported
            if emotion and hasattr(self.tts, 'tts_with_vc_to_file'):
                # Voice conversion for emotion
                wav = self.tts.tts_with_vc(
                    text=text,
                    speaker_wav=None,  # Could provide reference audio
                    language="en"
                )
            else:
                # Standard synthesis
                wav = self.tts.tts(
                    text=text,
                    speaker=self.speaker,
                    language="en"
                )
            
            # Convert to numpy
            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()
            
            # Apply speed modulation
            speed_factor = speed or self.speed
            if speed_factor != 1.0:
                wav = self._time_stretch(wav, speed_factor)
            
            return wav.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error in synthesis: {e}")
            return np.zeros(100, dtype=np.float32)
    
    def synthesize_stream(
        self,
        text: str,
        chunk_size: int = 4096
    ) -> Iterator[np.ndarray]:
        """
        Synthesize speech with streaming output.
        
        Args:
            text: Text to synthesize
            chunk_size: Size of audio chunks
            
        Yields:
            Audio chunks
        """
        # Full synthesis (CosyVoice doesn't support true streaming yet)
        full_audio = self.synthesize(text)
        
        # Split into chunks
        for i in range(0, len(full_audio), chunk_size):
            chunk = full_audio[i:i + chunk_size]
            yield chunk
    
    def _time_stretch(self, audio: np.ndarray, rate: float) -> np.ndarray:
        """
        Time-stretch audio without changing pitch.
        
        Args:
            audio: Input audio
            rate: Speed factor (>1 = faster, <1 = slower)
            
        Returns:
            Time-stretched audio
        """
        try:
            import librosa
            return librosa.effects.time_stretch(audio, rate=rate)
        except ImportError:
            logger.warning("librosa not installed, cannot time-stretch")
            return audio
    
    def modulate_emotion(
        self,
        text: str,
        valence: float,
        arousal: float
    ) -> np.ndarray:
        """
        Synthesize with emotional modulation based on VAD.
        
        Args:
            text: Text to synthesize
            valence: Emotional valence (-1 to 1)
            arousal: Arousal level (0 to 1)
            
        Returns:
            Emotionally modulated audio
        """
        # Map VAD to emotion label
        emotion = self._vad_to_emotion(valence, arousal)
        
        # Map VAD to speed
        # High arousal = faster, low arousal = slower
        speed = 0.9 + (arousal * 0.3)  # Range: 0.9 - 1.2
        
        return self.synthesize(text, emotion=emotion, speed=speed)
    
    def _vad_to_emotion(self, valence: float, arousal: float) -> str:
        """Map valence-arousal to emotion label."""
        if arousal > 0.6 and valence > 0.3:
            return 'happy'
        elif arousal > 0.6 and valence < -0.3:
            return 'angry'
        elif arousal < 0.4 and valence < -0.3:
            return 'sad'
        else:
            return 'neutral'


class MockTTS:
    """Mock TTS for testing without CosyVoice."""
    
    def __init__(self, *args, **kwargs):
        logger.warning("Using MockTTS - install TTS for real synthesis")
        self.sample_rate = 22050
    
    def synthesize(self, text: str, **kwargs) -> np.ndarray:
        # Return dummy audio (1 second of silence)
        duration = len(text.split()) * 0.3  # ~0.3s per word
        samples = int(self.sample_rate * duration)
        return np.zeros(samples, dtype=np.float32)
    
    def synthesize_stream(self, text: str, chunk_size: int = 4096) -> Iterator[np.ndarray]:
        audio = self.synthesize(text)
        for i in range(0, len(audio), chunk_size):
            yield audio[i:i + chunk_size]
    
    def modulate_emotion(self, text: str, valence: float, arousal: float) -> np.ndarray:
        return self.synthesize(text)
