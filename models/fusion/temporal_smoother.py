"""Temporal smoothing for emotion predictions.

Smooths emotion predictions over time to reduce jitter
and detect sustained emotional states.
"""

import numpy as np
from typing import Dict, List, Optional, Deque
from collections import deque
from dataclasses import dataclass
from loguru import logger

from .fusion_transformer import FusedEmotion


@dataclass
class SmoothedEmotion:
    """Container for smoothed emotional state."""
    valence: float
    arousal: float
    dominance: float
    primary_emotion: str
    confidence: float
    authenticity_score: float
    emotion_stability: float  # 0.0-1.0, high = stable over time
    trend: str  # 'improving', 'declining', 'stable'


class TemporalSmoother:
    """
    Smooths emotion predictions over time.
    
    Uses exponential moving average and trend detection
    to produce stable, reliable emotion assessments.
    """
    
    def __init__(
        self,
        window_size: int = 30,  # Number of frames to smooth over
        alpha: float = 0.3,  # EMA smoothing factor
        stability_threshold: float = 0.1  # Variance threshold for stability
    ):
        """
        Initialize temporal smoother.
        
        Args:
            window_size: Number of past predictions to consider
            alpha: Smoothing factor (0=smooth, 1=responsive)
            stability_threshold: Max variance for "stable" classification
        """
        self.window_size = window_size
        self.alpha = alpha
        self.stability_threshold = stability_threshold
        
        # History buffers
        self.valence_history: Deque[float] = deque(maxlen=window_size)
        self.arousal_history: Deque[float] = deque(maxlen=window_size)
        self.dominance_history: Deque[float] = deque(maxlen=window_size)
        self.emotion_history: Deque[str] = deque(maxlen=window_size)
        self.authenticity_history: Deque[float] = deque(maxlen=window_size)
        
        # Smoothed values
        self.smoothed_valence: Optional[float] = None
        self.smoothed_arousal: Optional[float] = None
        self.smoothed_dominance: Optional[float] = None
        
        logger.info(f"TemporalSmoother initialized (window={window_size}, alpha={alpha})")
    
    def smooth(self, emotion: FusedEmotion) -> SmoothedEmotion:
        """
        Apply temporal smoothing to emotion prediction.
        
        Args:
            emotion: Current fused emotion
            
        Returns:
            SmoothedEmotion with temporal filtering applied
        """
        # Add to history
        self.valence_history.append(emotion.valence)
        self.arousal_history.append(emotion.arousal)
        self.dominance_history.append(emotion.dominance)
        self.emotion_history.append(emotion.primary_emotion)
        self.authenticity_history.append(emotion.authenticity_score)
        
        # Initialize smoothed values if first prediction
        if self.smoothed_valence is None:
            self.smoothed_valence = emotion.valence
            self.smoothed_arousal = emotion.arousal
            self.smoothed_dominance = emotion.dominance
        
        # Exponential moving average
        self.smoothed_valence = (
            self.alpha * emotion.valence + 
            (1 - self.alpha) * self.smoothed_valence
        )
        self.smoothed_arousal = (
            self.alpha * emotion.arousal + 
            (1 - self.alpha) * self.smoothed_arousal
        )
        self.smoothed_dominance = (
            self.alpha * emotion.dominance + 
            (1 - self.alpha) * self.smoothed_dominance
        )
        
        # Determine primary emotion from smoothed values
        primary_emotion = self._get_majority_emotion()
        
        # Calculate stability
        stability = self._calculate_stability()
        
        # Detect trend
        trend = self._detect_trend()
        
        # Smooth authenticity
        smoothed_authenticity = float(np.mean(self.authenticity_history))
        
        # Calculate confidence based on stability
        confidence = emotion.confidence * (0.5 + 0.5 * stability)
        
        return SmoothedEmotion(
            valence=self.smoothed_valence,
            arousal=self.smoothed_arousal,
            dominance=self.smoothed_dominance,
            primary_emotion=primary_emotion,
            confidence=confidence,
            authenticity_score=smoothed_authenticity,
            emotion_stability=stability,
            trend=trend
        )
    
    def _get_majority_emotion(self) -> str:
        """Get most common emotion in recent history."""
        if not self.emotion_history:
            return 'unknown'
        
        # Count occurrences
        from collections import Counter
        emotion_counts = Counter(self.emotion_history)
        
        # Get most common
        most_common = emotion_counts.most_common(1)[0][0]
        return most_common
    
    def _calculate_stability(self) -> float:
        """
        Calculate emotion stability score.
        
        Returns:
            0.0-1.0, where 1.0 = very stable
        """
        if len(self.valence_history) < 3:
            return 0.5
        
        # Calculate variance of recent predictions
        valence_var = float(np.var(self.valence_history))
        arousal_var = float(np.var(self.arousal_history))
        
        # Combine variances
        total_var = (valence_var + arousal_var) / 2
        
        # Convert to stability score (inverse of variance)
        # High variance = low stability
        stability = 1.0 / (1.0 + total_var * 10)
        
        return float(np.clip(stability, 0.0, 1.0))
    
    def _detect_trend(self) -> str:
        """
        Detect emotional trend over time.
        
        Returns:
            'improving', 'declining', or 'stable'
        """
        if len(self.valence_history) < 10:
            return 'stable'
        
        # Calculate trend using linear regression on valence
        recent_valence = list(self.valence_history)[-10:]
        x = np.arange(len(recent_valence))
        
        # Simple linear fit
        coeffs = np.polyfit(x, recent_valence, 1)
        slope = coeffs[0]
        
        # Classify trend
        if slope > 0.02:
            return 'improving'
        elif slope < -0.02:
            return 'declining'
        else:
            return 'stable'
    
    def get_sustained_emotion(self, duration_frames: int = 15) -> Optional[str]:
        """
        Get emotion if it's been sustained for a duration.
        
        Args:
            duration_frames: Minimum duration in frames
            
        Returns:
            Emotion if sustained, None otherwise
        """
        if len(self.emotion_history) < duration_frames:
            return None
        
        # Check last N frames
        recent_emotions = list(self.emotion_history)[-duration_frames:]
        
        # Check if all the same
        if len(set(recent_emotions)) == 1:
            return recent_emotions[0]
        
        # Check if majority is same
        from collections import Counter
        counts = Counter(recent_emotions)
        most_common_emotion, count = counts.most_common(1)[0]
        
        if count >= duration_frames * 0.8:  # 80% threshold
            return most_common_emotion
        
        return None
    
    def reset(self):
        """Reset smoother state."""
        self.valence_history.clear()
        self.arousal_history.clear()
        self.dominance_history.clear()
        self.emotion_history.clear()
        self.authenticity_history.clear()
        
        self.smoothed_valence = None
        self.smoothed_arousal = None
        self.smoothed_dominance = None
        
        logger.debug("Temporal smoother reset")
