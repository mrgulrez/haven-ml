"""DeepFace-based emotion detector.

More compatible alternative to HSEmotion.
"""

import cv2
import numpy as np
from typing import Dict, Optional
from loguru import logger

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    logger.warning("DeepFace not available. Install with: pip install deepface")


class DeepFaceEmotionDetector:
    """
    Face emotion detection using DeepFace.
    
    More accurate and compatible than HSEmotion.
    Detects: angry, disgust, fear, happy, sad, surprise, neutral
    """
    
    def __init__(self, device: str = 'cpu'):
        """Initialize DeepFace detector."""
        if not DEEPFACE_AVAILABLE:
            raise ImportError("DeepFace not installed")
        
        self.device = device
        
        # Emotion mapping to valence/arousal
        self.emotion_map = {
            'angry': (-0.8, 0.8),
            'disgust': (-0.7, 0.5),
            'fear': (-0.6, 0.7),
            'happy': (0.8, 0.6),
            'sad': (-0.7, 0.3),
            'surprise': (0.3, 0.8),
            'neutral': (0.0, 0.3)
        }
        
        logger.info("DeepFace emotion detector initialized")
    
    def detect(self, frame: np.ndarray) -> Dict:
        """
        Detect emotions in frame.
        
        Args:
            frame: RGB image (H, W, 3)
            
        Returns:
            Dictionary with emotion results
        """
        try:
            # DeepFace expects BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Analyze emotions
            result = DeepFace.analyze(
                frame_bgr,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv',
                silent=True
            )
            
            # Handle list or dict response
            if isinstance(result, list):
                result = result[0] if result else {}
            
            if not result or 'emotion' not in result:
                return self._empty_result()
            
            # Extract emotions
            emotions = result['emotion']
            dominant_emotion = result.get('dominant_emotion', 'neutral')
            confidence = emotions.get(dominant_emotion, 0) / 100.0
            
            # Get face region
            region = result.get('region', {})
            face_box = [
                region.get('x', 0),
                region.get('y', 0),
                region.get('w', 0),
                region.get('h', 0)
            ] if region else None
            
            return {
                'face_detected': True,
                'primary_emotion': dominant_emotion,
                'confidence': confidence,
                'all_emotions': emotions,
                'face_box': face_box,
                'valence': self.emotion_map[dominant_emotion][0],
                'arousal': self.emotion_map[dominant_emotion][1]
            }
            
        except Exception as e:
            logger.debug(f"DeepFace detection failed: {e}")
            return self._empty_result()
    
    def _empty_result(self) -> Dict:
        """Return empty result when no face detected."""
        return {
            'face_detected': False,
            'primary_emotion': 'unknown',
            'confidence': 0.0,
            'all_emotions': {},
            'face_box': None,
            'valence': 0.0,
            'arousal': 0.0
        }
    
    def get_emotion_valence_arousal(self, emotion: str) -> tuple[float, float]:
        """Get valence and arousal for emotion."""
        return self.emotion_map.get(emotion, (0.0, 0.0))
    
    def visualize(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """
        Draw emotion detection on frame.
        
        Args:
            frame: Input frame
            result: Detection result
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        if not result['face_detected']:
            cv2.putText(
                annotated,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            return annotated
        
        # Draw face box
        if result['face_box']:
            x, y, w, h = result['face_box']
            cv2.rectangle(
                annotated,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )
        
        # Draw emotion text
        emotion = result['primary_emotion']
        confidence = result['confidence']
        
        text = f"{emotion.upper()} ({confidence:.2f})"
        cv2.putText(
            annotated,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )
        
        # Draw top 3 emotions
        all_emotions = result['all_emotions']
        sorted_emotions = sorted(
            all_emotions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        y_offset = 70
        for emo, score in sorted_emotions:
            text = f"{emo}: {score:.1f}%"
            cv2.putText(
                annotated,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
            y_offset += 25
        
        return annotated


# Mock for testing
class MockDeepFaceDetector:
    """Mock detector for testing."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        logger.info("Using MockDeepFaceDetector")
    
    def detect(self, frame: np.ndarray) -> Dict:
        """Return mock results."""
        return {
            'face_detected': True,
            'primary_emotion': 'neutral',
            'confidence': 0.7,
            'all_emotions': {
                'neutral': 70.0,
                'happy': 15.0,
                'sad': 10.0,
                'angry': 5.0
            },
            'face_box': [100, 100, 200, 200],
            'valence': 0.0,
            'arousal': 0.3
        }
    
    def get_emotion_valence_arousal(self, emotion: str) -> tuple[float, float]:
        return (0.0, 0.3)
    
    def visualize(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        annotated = frame.copy()
        cv2.putText(
            annotated,
            f"MOCK: {result['primary_emotion']}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2
        )
        return annotated
