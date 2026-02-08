"""Paralinguistics analysis using Wav2Vec2.

Analyzes HOW things are said, not just WHAT is said.
Extracts emotional cues from voice: pitch, tempo, tremor, arousal, valence.
"""

import torch
import numpy as np
import librosa
from typing import Dict, Optional, Tuple
from loguru import logger

from utils.helpers import timeit


class ProsodyAnalyzer:
    """
    Analyzes emotional paralinguistics from audio.
    
    Uses Wav2Vec2 fine-tuned for emotion recognition to extract
    arousal, valence, and dominance from speech.
    """
    
    def __init__(
        self,
        model_name: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
        device: str = "cuda",
        sample_rate: int = 16000
    ):
        """
        Initialize prosody analyzer.
        
        Args:
            model_name: Wav2Vec2 emotion model from HuggingFace
            device: Device to run on
            sample_rate: Audio sample rate
        """
        self.device = device
        self.sample_rate = sample_rate
        self.model_name = model_name
        
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
            
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
            
            logger.info(f"ProsodyAnalyzer initialized with '{model_name}' on {device}")
            
        except ImportError:
            logger.error("Transformers not installed. Install with: pip install transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load Wav2Vec2 model: {e}")
            raise
    
    @timeit
    def analyze(self, audio: np.ndarray) -> Dict[str, any]:
        """
        Analyze paralinguistic features from audio.
        
        Args:
            audio: Audio samples as numpy array (float32, normalized)
            
        Returns:
            Dictionary containing:
            - arousal: Calm (0.0) to Excited (1.0)
            - valence: Negative (-1.0) to Positive (1.0)
            - dominance: Submissive (-1.0) to Dominant (1.0)
            - pitch_mean: Average pitch in Hz
            - pitch_std: Pitch variation
            - tempo: Speech rate (syllables per second)
            - tremor: Voice instability (0.0-1.0)
            - energy: Overall voice energy
        """
        try:
            # Extract Wav2Vec2 features (arousal, valence, dominance)
            wav2vec_features = self._extract_wav2vec_features(audio)
            
            # Extract acoustic features
            acoustic_features = self._extract_acoustic_features(audio)
            
            # Combine results
            return {
                **wav2vec_features,
                **acoustic_features
            }
            
        except Exception as e:
            logger.error(f"Error in prosody analysis: {e}")
            return self._empty_result()
    
    def _extract_wav2vec_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract arousal, valence, dominance using Wav2Vec2."""
        try:
            # Process audio
            inputs = self.processor(
                audio, 
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0].cpu().numpy()
            
            # Model outputs arousal, valence, dominance
            # Values are typically in [-1, 1] range
            arousal = float(logits[0])  # Convert to [0, 1]
            valence = float(logits[1])  # Keep [-1, 1]
            dominance = float(logits[2]) if len(logits) > 2 else 0.0
            
            # Normalize arousal to [0, 1]
            arousal = (arousal + 1) / 2
            
            return {
                'arousal': np.clip(arousal, 0.0, 1.0),
                'valence': np.clip(valence, -1.0, 1.0),
                'dominance': np.clip(dominance, -1.0, 1.0)
            }
            
        except Exception as e:
            logger.error(f"Error extracting Wav2Vec2 features: {e}")
            return {'arousal': 0.5, 'valence': 0.0, 'dominance': 0.0}
    
    def _extract_acoustic_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract acoustic features using librosa."""
        try:
            # Pitch (F0) analysis
            pitch, voiced_flag, voiced_prob = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate
            )
            
            # Remove NaN values
            pitch = pitch[~np.isnan(pitch)]
            
            if len(pitch) > 0:
                pitch_mean = float(np.mean(pitch))
                pitch_std = float(np.std(pitch))
                tremor = float(pitch_std / (pitch_mean + 1e-6))  # Normalized variation
            else:
                pitch_mean = 0.0
                pitch_std = 0.0
                tremor = 0.0
            
            # Tempo (speech rate) estimation
            # Use onset detection as proxy for syllable rate
            onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
            tempo = float(librosa.beat.tempo(onset_envelope=onset_env, sr=self.sample_rate)[0])
            tempo = tempo / 60.0  # Convert BPM to per second
            
            # Energy (RMS)
            rms = librosa.feature.rms(y=audio)[0]
            energy = float(np.mean(rms))
            
            return {
                'pitch_mean': pitch_mean,
                'pitch_std': pitch_std,
                'tempo': tempo,
                'tremor': np.clip(tremor, 0.0, 1.0),
                'energy': energy
            }
            
        except Exception as e:
            logger.error(f"Error extracting acoustic features: {e}")
            return {
                'pitch_mean': 0.0,
                'pitch_std': 0.0,
                'tempo': 0.0,
                'tremor': 0.0,
                'energy': 0.0
            }
    
    def classify_emotion_from_prosody(self, prosody: Dict) -> str:
        """
        Classify emotion from prosody features.
        
        Uses arousal-valence circumplex model.
        """
        arousal = prosody['arousal']
        valence = prosody['valence']
        
        # High arousal, positive valence
        if arousal > 0.6 and valence > 0.3:
            return 'excited'
        # High arousal, negative valence
        elif arousal > 0.6 and valence < -0.3:
            return 'stressed'
        # Low arousal, positive valence
        elif arousal < 0.4 and valence > 0.3:
            return 'calm_positive'
        # Low arousal, negative valence
        elif arousal < 0.4 and valence < -0.3:
            return 'sad'
        # Neutral arousal, positive valence
        elif valence > 0.3:
            return 'positive'
        # Neutral arousal, negative valence
        elif valence < -0.3:
            return 'negative'
        else:
            return 'neutral'
    
    def detect_voice_tremor(self, prosody: Dict) -> bool:
        """Detect if voice shows signs of tremor (nervousness, fear)."""
        return prosody['tremor'] > 0.7
    
    def _empty_result(self) -> Dict:
        """Return empty result for errors."""
        return {
            'arousal': 0.5,
            'valence': 0.0,
            'dominance': 0.0,
            'pitch_mean': 0.0,
            'pitch_std': 0.0,
            'tempo': 0.0,
            'tremor': 0.0,
            'energy': 0.0
        }


class MockProsodyAnalyzer:
    """Mock prosody analyzer for testing."""
    
    def __init__(self, *args, **kwargs):
        logger.warning("Using MockProsodyAnalyzer - install transformers for real analysis")
    
    def analyze(self, audio: np.ndarray) -> Dict:
        # Simple energy-based mock
        energy = np.mean(np.abs(audio))
        
        return {
            'arousal': 0.5,
            'valence': 0.0,
            'dominance': 0.0,
            'pitch_mean': 200.0,
            'pitch_std': 20.0,
            'tempo': 3.0,
            'tremor': 0.1,
            'energy': float(energy)
        }
    
    def classify_emotion_from_prosody(self, prosody: Dict) -> str:
        return 'neutral'
    
    def detect_voice_tremor(self, prosody: Dict) -> bool:
        return False
