"""Facial expression detection using HSEmotion.

HSEmotion is a high-speed emotion recognition model that can process
100+ FPS and detect 8 emotions: happy, sad, angry, surprise, fear,
disgust, contempt, and neutral.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
import torch
from loguru import logger

try:
    from hsemotion.facial_emotions import HSEmotionRecognizer
except ImportError:
    logger.warning("HSEmotion not installed. Install with: pip install hsemotion")
    HSEmotionRecognizer = None

from utils.helpers import timeit


class FaceEmotionDetector:
    """
    Detects facial expressions and emotions from video frames.
    
    Uses HSEmotion for fast, accurate emotion recognition.
    """
    
    EMOTIONS = [
        'angry', 'contempt', 'disgust', 'fear',
        'happy', 'neutral', 'sad', 'surprise'
    ]
    
    def __init__(
        self,
        model_name: str = "enet_b0_8_best_afew",
        device: str = "cuda",
        confidence_threshold: float = 0.3
    ):
        """
        Initialize the face emotion detector.
        
        Args:
            model_name: HSEmotion model variant
            device: Device to run inference on (cuda/cpu)
            confidence_threshold: Minimum confidence for emotion detection
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model_name = model_name
        
        if HSEmotionRecognizer is None:
            raise ImportError("HSEmotion not installed")
        
        # Initialize the recognizer
        self.recognizer = HSEmotionRecognizer(model_name=model_name)
        logger.info(f"FaceEmotionDetector initialized with model '{model_name}' on {device}")
    
    @timeit
    def detect(self, frame: np.ndarray) -> Dict[str, any]:
        """
        Detect emotions from a video frame.
        
        Args:
            frame: RGB image as numpy array (H, W, 3)
            
        Returns:
            Dictionary containing:
            - emotions: Dict mapping emotion names to probabilities
            - primary_emotion: Most confident emotion
            - confidence: Confidence score for primary emotion
            - face_detected: Whether a face was found
            - face_bbox: Bounding box coordinates (x, y, w, h) if face found
        """
        try:
            # Detect face and emotions
            emotion_scores, face_bbox = self.recognizer.predict_emotions(
                frame, 
                logits=False
            )
            
            if emotion_scores is None or len(emotion_scores) == 0:
                return {
                    'emotions': {},
                    'primary_emotion': 'unknown',
                    'confidence': 0.0,
                    'face_detected': False,
                    'face_bbox': None
                }
            
            # Convert to dictionary
            emotions_dict = {
                emotion: float(score)
                for emotion, score in zip(self.EMOTIONS, emotion_scores)
            }
            
            # Get primary emotion
            primary_emotion = max(emotions_dict.items(), key=lambda x: x[1])
            
            # Filter low confidence
            if primary_emotion[1] < self.confidence_threshold:
                primary_emotion = ('neutral', primary_emotion[1])
            
            return {
                'emotions': emotions_dict,
                'primary_emotion': primary_emotion[0],
                'confidence': primary_emotion[1],
                'face_detected': True,
                'face_bbox': face_bbox
            }
            
        except Exception as e:
            logger.error(f"Error in face emotion detection: {e}")
            return {
                'emotions': {},
                'primary_emotion': 'error',
                'confidence': 0.0,
                'face_detected': False,
                'face_bbox': None
            }
    
    def detect_batch(self, frames: list) -> list:
        """
        Detect emotions from multiple frames.
        
        Args:
            frames: List of RGB image arrays
            
        Returns:
            List of detection results
        """
        return [self.detect(frame) for frame in frames]
    
    def get_emotion_valence_arousal(self, emotion: str) -> Tuple[float, float]:
        """
        Map emotion to valence-arousal coordinates.
        
        Valence: negative (-1) to positive (+1)
        Arousal: calm (0) to excited (1)
        
        Args:
            emotion: Emotion label
            
        Returns:
            (valence, arousal) tuple
        """
        # Based on Russell's circumplex model of affect
        mapping = {
            'happy': (0.8, 0.7),
            'surprise': (0.3, 0.9),
            'angry': (-0.7, 0.8),
            'fear': (-0.6, 0.8),
            'disgust': (-0.7, 0.5),
            'contempt': (-0.5, 0.3),
            'sad': (-0.8, 0.2),
            'neutral': (0.0, 0.0)
        }
        
        return mapping.get(emotion, (0.0, 0.0))
    
    def visualize(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """
        Draw emotion detection results on frame.
        
        Args:
            frame: Input frame
            result: Detection result from detect()
            
        Returns:
            Annotated frame
        """
        frame_copy = frame.copy()
        
        if not result['face_detected']:
            return frame_copy
        
        # Draw bounding box
        if result['face_bbox'] is not None:
            x, y, w, h = result['face_bbox']
            cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw emotion text
            emotion_text = f"{result['primary_emotion']}: {result['confidence']:.2f}"
            cv2.putText(
                frame_copy,
                emotion_text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        
        return frame_copy


# Fallback implementation if HSEmotion is not available
class MockFaceEmotionDetector:
    """Mock detector for development/testing without HSEmotion."""
    
    def __init__(self, *args, **kwargs):
        logger.warning("Using MockFaceEmotionDetector - install hsemotion for real detection")
    
    def detect(self, frame: np.ndarray) -> Dict:
        return {
            'emotions': {'neutral': 1.0},
            'primary_emotion': 'neutral',
            'confidence': 1.0,
            'face_detected': True,
            'face_bbox': None
        }
    
    def detect_batch(self, frames: list) -> list:
        return [self.detect(frame) for frame in frames]
    
    def get_emotion_valence_arousal(self, emotion: str) -> Tuple[float, float]:
        return (0.0, 0.0)
    
    def visualize(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        return frame
