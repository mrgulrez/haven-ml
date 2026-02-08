"""Voice Activity Detection using Silero VAD.

Detects when speech starts and stops with low latency,
critical for barge-in support and speech segmentation.
"""

import torch
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path
from loguru import logger

from utils.helpers import timeit


class SileroVAD:
    """
    Voice Activity Detector using Silero VAD.
    
    Fast, accurate speech detection with <100ms latency.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 500,
        sample_rate: int = 16000
    ):
        """
        Initialize Silero VAD.
        
        Args:
            model_path: Path to VAD model (auto-downloads if None)
            threshold: Voice probability threshold (0.0-1.0)
            min_speech_duration_ms: Minimum speech duration to trigger
            min_silence_duration_ms: Minimum silence to end speech
            sample_rate: Audio sample rate (16000 recommended)
        """
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.sample_rate = sample_rate
        
        # Load model
        try:
            if model_path and Path(model_path).exists():
                self.model = torch.jit.load(model_path)
            else:
                # Auto-download from torch hub
                self.model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False
                )
                self.get_speech_timestamps = utils[0]
            
            logger.info("Silero VAD initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
            raise
        
        # State tracking
        self.is_speaking = False
        self.speech_frames = 0
        self.silence_frames = 0
        
        # Silero VAD requires specific chunk sizes
        self.required_chunk_size = 512 if sample_rate == 16000 else 256
    
    @timeit
    def detect(self, audio: np.ndarray) -> Dict:
        """
        Detect voice activity in audio chunk.
        
        Args:
            audio: Audio samples (float32, mono, 16kHz)
            
        Returns:
            Dictionary with VAD results
        """
        try:
            # Silero VAD requires specific chunk sizes:
            # 512 samples for 16kHz, 256 for 8kHz
            
            # If audio is longer than required, process in chunks and average
            if len(audio) > self.required_chunk_size:
                # Split into chunks
                num_chunks = len(audio) // self.required_chunk_size
                confidences = []
                
                for i in range(num_chunks):
                    start = i * self.required_chunk_size
                    end = start + self.required_chunk_size
                    chunk = audio[start:end]
                    
                    # Convert to torch tensor
                    audio_tensor = torch.from_numpy(chunk).float()
                    
                    # Run VAD on chunk
                    with torch.no_grad():
                        conf = self.model(audio_tensor, self.sample_rate).item()
                    confidences.append(conf)
                
                # Average confidence across chunks
                confidence = sum(confidences) / len(confidences) if confidences else 0.0
            else:
                # Pad if too short
                if len(audio) < self.required_chunk_size:
                    audio = np.pad(audio, (0, self.required_chunk_size - len(audio)))
                
                # Convert to torch tensor
                audio_tensor = torch.from_numpy(audio).float()
                
                # Run VAD
                with torch.no_grad():
                    confidence = self.model(audio_tensor, self.sample_rate).item()
            
            is_speech = confidence > self.threshold
            
            # Update state and determine event
            event = self._update_state(is_speech)
            
            return {
                'is_speech': is_speech,
                'confidence': confidence,
                'event': event,  # 'start', 'continue', 'end', 'none'
                'is_speaking': self.is_speaking
            }
            
        except Exception as e:
            logger.error(f"Error in VAD detection: {e}")
            return {
                'is_speech': False,
                'confidence': 0.0,
                'event': 'none',
                'is_speaking': False
            }
    
    def _update_state(self, is_speech: bool) -> str:
        """
        Update VAD state and return event type.
        
        Returns:
            'start', 'continue', 'end', or 'none'
        """
        event = 'none'
        
        if is_speech:
            self.speech_frames += 1
            self.silence_frames = 0
            
            if not self.is_speaking and self.speech_frames >= 2:
                # Speech started
                event = 'start'
                self.is_speaking = True
                logger.debug("Speech started")
            elif self.is_speaking:
                event = 'continue'
        else:
            self.silence_frames += 1
            self.speech_frames = 0
            
            if self.is_speaking and self.silence_frames >= 3:
                # Speech ended
                event = 'end'
                self.is_speaking = False
                logger.debug("Speech ended")
        
        return event
    
    def reset(self):
        """Reset VAD state."""
        self.is_speaking = False
        self.speech_frames = 0
        self.silence_frames = 0
        logger.debug("VAD state reset")
    
    def get_speech_segments(
        self, 
        audio: np.ndarray, 
        return_seconds: bool = False
    ) -> List[Dict]:
        """
        Get all speech segments from audio.
        
        Args:
            audio: Full audio array
            return_seconds: Return timestamps in seconds vs samples
            
        Returns:
            List of speech segments with 'start' and 'end' times
        """
        try:
            audio_tensor = torch.from_numpy(audio).float()
            
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.model,
                sampling_rate=self.sample_rate,
                threshold=self.threshold,
                min_speech_duration_ms=self.min_speech_duration_ms,
                min_silence_duration_ms=self.min_silence_duration_ms,
                return_seconds=return_seconds
            )
            
            return speech_timestamps
            
        except Exception as e:
            logger.error(f"Error getting speech segments: {e}")
            return []


class MockVAD:
    """Mock VAD for testing without Silero."""
    
    def __init__(self, *args, **kwargs):
        logger.warning("Using MockVAD - install silero-vad for real detection")
        self.is_speaking = False
    
    def detect(self, audio: np.ndarray) -> Dict:
        # Simple energy-based mock
        energy = np.mean(np.abs(audio))
        is_speech = energy > 0.01
        
        event = 'none'
        if is_speech and not self.is_speaking:
            event = 'start'
            self.is_speaking = True
        elif not is_speech and self.is_speaking:
            event = 'end'
            self.is_speaking = False
        
        return {
            'is_speech': is_speech,
            'confidence': float(energy * 10),
            'event': event,
            'is_speaking': self.is_speaking
        }
    
    def reset(self):
        self.is_speaking = False
    
    def get_speech_segments(self, audio: np.ndarray, return_seconds: bool = False) -> List[Dict]:
        return []
