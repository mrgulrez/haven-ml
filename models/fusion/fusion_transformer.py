"""Transformer-based multimodal fusion model.

Combines visual and audio emotional signals to produce accurate,
conflict-aware emotional assessments.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from utils.helpers import timeit


@dataclass
class FusedEmotion:
    """Container for fused emotional state."""
    valence: float  # -1.0 to 1.0
    arousal: float  # 0.0 to 1.0
    dominance: float  # -1.0 to 1.0
    primary_emotion: str
    confidence: float
    authenticity_score: float  # 0.0 to 1.0 (low = emotional masking)
    visual_weight: float  # Contribution of visual signal
    audio_weight: float  # Contribution of audio signal


class MultimodalFusionTransformer(nn.Module):
    """
    Transformer-based fusion of visual and audio emotional signals.
    
    Architecture:
    - Separate encoders for visual and audio features
    - Cross-attention to detect conflicts/agreements
    - Fusion layer outputs valence, arousal, dominance
    - Authenticity detector for emotional masking
    """
    
    def __init__(
        self,
        visual_dim: int = 16,
        audio_dim: int = 16,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize fusion transformer.
        
        Args:
            visual_dim: Dimension of visual feature embeddings
            audio_dim: Dimension of audio feature embeddings
            hidden_dim: Hidden dimension for transformer
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        
        # Feature projection layers
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        
        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, 
            num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # Self-attention for joint representation
        self.self_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Output heads
        self.valence_head = nn.Linear(hidden_dim, 1)
        self.arousal_head = nn.Linear(hidden_dim, 1)
        self.dominance_head = nn.Linear(hidden_dim, 1)
        
        # Authenticity detector (conflict detection)
        self.authenticity_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),  # Takes both modalities
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Modality weight predictor
        self.weight_head = nn.Sequential(
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        logger.info("MultimodalFusionTransformer initialized")
    
    def forward(
        self,
        visual_features: torch.Tensor,
        audio_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through fusion network.
        
        Args:
            visual_features: (batch, visual_dim)
            audio_features: (batch, audio_dim)
            
        Returns:
            valence, arousal, dominance, authenticity, weights
        """
        # Project to hidden dimension
        visual_hidden = self.visual_proj(visual_features).unsqueeze(1)  # (batch, 1, hidden)
        audio_hidden = self.audio_proj(audio_features).unsqueeze(1)  # (batch, 1, hidden)
        
        # Cross-attention: Visual attending to Audio
        visual_attended, _ = self.cross_attention(
            visual_hidden,
            audio_hidden,
            audio_hidden
        )
        
        # Cross-attention: Audio attending to Visual
        audio_attended, _ = self.cross_attention(
            audio_hidden,
            visual_hidden,
            visual_hidden
        )
        
        # Concatenate attended features
        combined = torch.cat([visual_attended, audio_attended], dim=1)  # (batch, 2, hidden)
        
        # Self-attention on combined
        fused, _ = self.self_attention(combined, combined, combined)
        
        # Feed-forward
        fused = fused + self.ffn(fused)
        
        # Pool to single representation
        fused_pooled = fused.mean(dim=1)  # (batch, hidden)
        
        # Predict valence, arousal, dominance
        valence = torch.tanh(self.valence_head(fused_pooled))  # [-1, 1]
        arousal = torch.sigmoid(self.arousal_head(fused_pooled))  # [0, 1]
        dominance = torch.tanh(self.dominance_head(fused_pooled))  # [-1, 1]
        
        # Predict authenticity (conflict detection)
        concat_features = torch.cat([
            visual_hidden.squeeze(1),
            audio_hidden.squeeze(1)
        ], dim=-1)
        authenticity = self.authenticity_head(concat_features)
        
        # Predict modality weights
        weights = self.weight_head(fused_pooled)
        
        return valence, arousal, dominance, authenticity, weights


class FusionEngine:
    """
    High-level fusion engine combining visual and audio signals.
    
    Handles feature extraction, fusion, and emotion classification.
    """
    
    EMOTIONS = [
        'excited', 'happy', 'calm_positive', 'neutral',
        'sad', 'frustrated', 'anxious', 'stressed', 'angry'
    ]
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        confidence_threshold: float = 0.6
    ):
        """
        Initialize fusion engine.
        
        Args:
            model_path: Path to pre-trained fusion model
            device: Device to run on
            confidence_threshold: Minimum confidence for predictions
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # Initialize model
        self.model = MultimodalFusionTransformer(
            visual_dim=16,
            audio_dim=16,
            hidden_dim=64,
            num_heads=4
        ).to(device)
        
        # Load pre-trained weights if available
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path))
            logger.info(f"Loaded fusion model from {model_path}")
        else:
            logger.warning("No pre-trained model found, using random initialization")
        
        self.model.eval()
    
    @timeit
    def fuse(
        self,
        visual_state: Dict,
        audio_state: Dict
    ) -> FusedEmotion:
        """
        Fuse visual and audio emotional states.
        
        Args:
            visual_state: Output from video pipeline
            audio_state: Output from audio pipeline
            
        Returns:
            FusedEmotion object with combined assessment
        """
        try:
            # Extract features
            visual_features = self._extract_visual_features(visual_state)
            audio_features = self._extract_audio_features(audio_state)
            
            # Convert to tensors
            visual_tensor = torch.FloatTensor(visual_features).unsqueeze(0).to(self.device)
            audio_tensor = torch.FloatTensor(audio_features).unsqueeze(0).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                valence, arousal, dominance, authenticity, weights = self.model(
                    visual_tensor,
                    audio_tensor
                )
            
            # Extract values
            valence_val = float(valence[0, 0].cpu())
            arousal_val = float(arousal[0, 0].cpu())
            dominance_val = float(dominance[0, 0].cpu())
            authenticity_val = float(authenticity[0, 0].cpu())
            visual_weight = float(weights[0, 0].cpu())
            audio_weight = float(weights[0, 1].cpu())
            
            # Classify primary emotion
            primary_emotion, confidence = self._classify_emotion(
                valence_val, arousal_val, dominance_val
            )
            
            return FusedEmotion(
                valence=valence_val,
                arousal=arousal_val,
                dominance=dominance_val,
                primary_emotion=primary_emotion,
                confidence=confidence,
                authenticity_score=authenticity_val,
                visual_weight=visual_weight,
                audio_weight=audio_weight
            )
            
        except Exception as e:
            logger.error(f"Error in fusion: {e}")
            return self._empty_fusion()
    
    def _extract_visual_features(self, visual_state: Dict) -> np.ndarray:
        """Extract feature vector from visual state."""
        features = np.zeros(16, dtype=np.float32)
        
        # Valence, arousal from visual
        features[0] = visual_state.get('valence', 0.0)
        features[1] = visual_state.get('arousal', 0.5)
        
        # One-hot encode primary emotion (8 emotions)
        emotion = visual_state.get('primary_emotion', 'neutral')
        emotion_map = {
            'happy': 2, 'sad': 3, 'angry': 4, 'fear': 5,
            'surprise': 6, 'disgust': 7, 'neutral': 8
        }
        if emotion in emotion_map:
            features[emotion_map[emotion]] = 1.0
        
        # Posture state
        posture = visual_state.get('posture_state', 'normal')
        if posture == 'slouching':
            features[9] = 1.0
        elif posture == 'frustrated':
            features[10] = 1.0
        
        # Gaze pattern
        gaze = visual_state.get('gaze_pattern', 'normal')
        if gaze == 'blank_stare':
            features[11] = 1.0
        elif gaze == 'looking_down':
            features[12] = 1.0
        
        # Confidence
        features[13] = visual_state.get('emotion_confidence', 0.5)
        
        return features
    
    def _extract_audio_features(self, audio_state: Dict) -> np.ndarray:
        """Extract feature vector from audio state."""
        features = np.zeros(16, dtype=np.float32)
        
        # Arousal, valence from audio
        features[0] = audio_state.get('arousal', 0.5)
        features[1] = audio_state.get('valence', 0.0)
        
        # Tremor
        features[2] = audio_state.get('tremor', 0.0)
        
        # One-hot encode audio emotion
        emotion = audio_state.get('audio_emotion', 'neutral')
        emotion_map = {
            'excited': 3, 'positive': 4, 'neutral': 5,
            'negative': 6, 'sad': 7, 'stressed': 8, 'nervous': 9
        }
        if emotion in emotion_map:
            features[emotion_map[emotion]] = 1.0
        
        # Events
        events = audio_state.get('detected_events', [])
        if 'sigh' in events:
            features[10] = 1.0
        if 'heavy_breathing' in events:
            features[11] = 1.0
        if 'prolonged_silence' in events:
            features[12] = 1.0
        
        # Speaking state
        features[13] = 1.0 if audio_state.get('is_speaking', False) else 0.0
        
        # Silence duration (normalized)
        features[14] = min(audio_state.get('silence_duration', 0.0) / 10.0, 1.0)
        
        return features
    
    def _classify_emotion(
        self,
        valence: float,
        arousal: float,
        dominance: float
    ) -> Tuple[str, float]:
        """
        Classify emotion from VAD values.
        
        Uses Russell's circumplex model with dominance.
        """
        # High arousal, positive valence
        if arousal > 0.6 and valence > 0.4:
            return 'excited', 0.9
        elif arousal > 0.5 and valence > 0.2:
            return 'happy', 0.8
        
        # Low arousal, positive valence
        elif arousal < 0.4 and valence > 0.3:
            return 'calm_positive', 0.8
        
        # High arousal, negative valence
        elif arousal > 0.6 and valence < -0.3:
            if dominance > 0.2:
                return 'angry', 0.9
            else:
                return 'stressed', 0.85
        
        # Medium arousal, negative valence
        elif arousal > 0.4 and valence < -0.2:
            return 'anxious', 0.8
        
        # Low arousal, negative valence
        elif arousal < 0.4 and valence < -0.3:
            return 'sad', 0.8
        
        # Medium valence, medium arousal
        elif abs(valence) < 0.2 and 0.3 < arousal < 0.6:
            return 'neutral', 0.7
        
        # Frustrated (medium-high arousal, negative valence, low dominance)
        elif 0.4 < arousal < 0.7 and -0.4 < valence < -0.1 and dominance < -0.2:
            return 'frustrated', 0.75
        
        else:
            return 'neutral', 0.5
    
    def _empty_fusion(self) -> FusedEmotion:
        """Return empty fusion result."""
        return FusedEmotion(
            valence=0.0,
            arousal=0.5,
            dominance=0.0,
            primary_emotion='unknown',
            confidence=0.0,
            authenticity_score=1.0,
            visual_weight=0.5,
            audio_weight=0.5
        )


from pathlib import Path

class MockFusionEngine:
    """Mock fusion engine for testing."""
    
    def __init__(self, *args, **kwargs):
        logger.warning("Using MockFusionEngine")
    
    def fuse(self, visual_state: Dict, audio_state: Dict) -> FusedEmotion:
        # Simple averaging
        valence = (visual_state.get('valence', 0.0) + audio_state.get('valence', 0.0)) / 2
        arousal = (visual_state.get('arousal', 0.5) + audio_state.get('arousal', 0.5)) / 2
        
        return FusedEmotion(
            valence=valence,
            arousal=arousal,
            dominance=0.0,
            primary_emotion='neutral',
            confidence=0.7,
            authenticity_score=1.0,
            visual_weight=0.5,
            audio_weight=0.5
        )
