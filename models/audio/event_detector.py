"""Audio event detection for non-speech sounds.

Detects emotional cues from non-speech audio:
- Sighs (frustration, relief)
- Heavy breathing (stress, exertion)
- Long silences (thinking, avoidance)
- Ambient noise patterns
"""

import numpy as np
import librosa
from typing import Dict, List, Optional
from dataclasses import dataclass
from loguru import logger

from utils.helpers import timeit


@dataclass
class AudioEvent:
    """Container for detected audio event."""
    event_type: str  # 'sigh', 'breath', 'silence', 'noise'
    timestamp: float  # Seconds
    duration: float  # Seconds
    confidence: float  # 0.0-1.0


class AudioEventDetector:
    """
    Detects non-speech audio events.
    
    Uses signal processing techniques to identify emotional cues
    from non-verbal sounds.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        silence_threshold_db: float = -40.0,
        min_silence_duration: float = 2.0
    ):
        """
        Initialize event detector.
        
        Args:
            sample_rate: Audio sample rate
            silence_threshold_db: dB threshold for silence detection
            min_silence_duration: Minimum silence duration to report (seconds)
        """
        self.sample_rate = sample_rate
        self.silence_threshold_db = silence_threshold_db
        self.min_silence_duration = min_silence_duration
        
        # History tracking
        self.last_sound_time: Optional[float] = None
        self.silence_start: Optional[float] = None
        self.detected_events: List[AudioEvent] = []
        
        logger.info("AudioEventDetector initialized")
    
    @timeit
    def detect(
        self, 
        audio: np.ndarray, 
        timestamp: float,
        is_speech: bool = False
    ) -> Dict[str, any]:
        """
        Detect audio events in chunk.
        
        Args:
            audio: Audio samples (float32, normalized)
            timestamp: Current timestamp in seconds
            is_speech: Whether VAD detected speech
            
        Returns:
            Dictionary containing:
            - events: List of detected AudioEvent objects
            - is_silence: Whether current chunk is silence
            - silence_duration: Duration of ongoing silence
            - ambient_noise_level: Background noise level (dB)
        """
        try:
            # Calculate RMS energy
            rms = np.sqrt(np.mean(audio**2))
            db = 20 * np.log10(rms + 1e-10)
            
            # Detect silence
            is_silence = db < self.silence_threshold_db and not is_speech
            
            events = []
            
            # Track silence duration
            if is_silence:
                if self.silence_start is None:
                    self.silence_start = timestamp
                
                silence_duration = timestamp - self.silence_start
                
                # Report prolonged silence
                if silence_duration >= self.min_silence_duration:
                    # Only report once when threshold is crossed
                    if not any(e.event_type == 'prolonged_silence' and 
                             abs(e.timestamp - self.silence_start) < 0.5 
                             for e in self.detected_events):
                        event = AudioEvent(
                            event_type='prolonged_silence',
                            timestamp=self.silence_start,
                            duration=silence_duration,
                            confidence=1.0
                        )
                        events.append(event)
                        self.detected_events.append(event)
            else:
                if self.silence_start is not None:
                    silence_duration = timestamp - self.silence_start
                else:
                    silence_duration = 0.0
                
                self.silence_start = None
                self.last_sound_time = timestamp
            
            # Detect non-speech sounds (when not speaking)
            if not is_speech and not is_silence:
                # Detect sighs (sharp energy increase then decrease)
                sigh_event = self._detect_sigh(audio, timestamp)
                if sigh_event:
                    events.append(sigh_event)
                    self.detected_events.append(sigh_event)
                
                # Detect breathing patterns
                breath_event = self._detect_breathing(audio, timestamp)
                if breath_event:
                    events.append(breath_event)
                    self.detected_events.append(breath_event)
            
            # Calculate ambient noise
            ambient_noise_level = float(db) if is_silence else None
            
            # Clean old events (keep last 60 seconds)
            self.detected_events = [
                e for e in self.detected_events 
                if timestamp - e.timestamp < 60
            ]
            
            return {
                'events': events,
                'is_silence': is_silence,
                'silence_duration': silence_duration if is_silence else 0.0,
                'ambient_noise_level': ambient_noise_level
            }
            
        except Exception as e:
            logger.error(f"Error in event detection: {e}")
            return {
                'events': [],
                'is_silence': False,
                'silence_duration': 0.0,
                'ambient_noise_level': None
            }
    
    def _detect_sigh(self, audio: np.ndarray, timestamp: float) -> Optional[AudioEvent]:
        """
        Detect sigh sounds.
        
        Sighs have characteristic spectral and temporal patterns:
        - Sharp energy increase then gradual decrease
        - Lower frequency content
        - Duration ~1-2 seconds
        """
        # Calculate energy envelope
        envelope = np.abs(librosa.effects.harmonic(audio))
        
        # Look for sharp peak followed by decay
        if len(envelope) < 100:
            return None
        
        # Find peaks
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(envelope, prominence=0.1)
        
        if len(peaks) == 0:
            return None
        
        # Check for sigh pattern
        for peak in peaks:
            if peak > len(envelope) - 50:
                continue
            
            # Check decay after peak
            decay_region = envelope[peak:peak+50]
            if len(decay_region) > 0:
                decay_rate = (decay_region[0] - decay_region[-1]) / len(decay_region)
                
                # Sigh has gradual decay
                if 0.001 < decay_rate < 0.01:
                    return AudioEvent(
                        event_type='sigh',
                        timestamp=timestamp + peak / self.sample_rate,
                        duration=1.0,
                        confidence=0.7
                    )
        
        return None
    
    def _detect_breathing(self, audio: np.ndarray, timestamp: float) -> Optional[AudioEvent]:
        """
        Detect heavy breathing patterns.
        
        Heavy breathing shows:
        - Regular periodic patterns
        - Low frequency content
        - Moderate energy
        """
        # Use zero-crossing rate to detect breathing rhythm
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        
        # Breathing has low ZCR
        avg_zcr = np.mean(zcr)
        
        if avg_zcr < 0.05:  # Low frequency content
            # Check energy (not silence, but not loud speech)
            rms = np.sqrt(np.mean(audio**2))
            
            if 0.01 < rms < 0.1:
                return AudioEvent(
                    event_type='heavy_breathing',
                    timestamp=timestamp,
                    duration=len(audio) / self.sample_rate,
                    confidence=0.6
                )
        
        return None
    
    def get_recent_events(self, window_seconds: float = 30) -> List[AudioEvent]:
        """Get events from recent time window."""
        if not self.detected_events:
            return []
        
        latest_time = self.detected_events[-1].timestamp
        return [
            e for e in self.detected_events
            if latest_time - e.timestamp <= window_seconds
        ]
    
    def reset(self):
        """Reset detector state."""
        self.last_sound_time = None
        self.silence_start = None
        self.detected_events.clear()


class MockEventDetector:
    """Mock event detector for testing."""
    
    def __init__(self, *args, **kwargs):
        logger.warning("Using MockEventDetector")
    
    def detect(
        self, 
        audio: np.ndarray, 
        timestamp: float, 
        is_speech: bool = False
    ) -> Dict:
        return {
            'events': [],
            'is_silence': False,
            'silence_duration': 0.0,
            'ambient_noise_level': -50.0
        }
    
    def get_recent_events(self, window_seconds: float = 30) -> List[AudioEvent]:
        return []
    
    def reset(self):
        pass
