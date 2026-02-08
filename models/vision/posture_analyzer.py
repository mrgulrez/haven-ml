"""Postural analysis using MediaPipe Pose.

Detects body pose and analyzes posture to identify:
- Slouching (fatigue indicator)
- Pacing (anxiety indicator)
- Stillness (deep focus or dissociation)
- Head-in-hands (frustration/stress)
"""

import cv2
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from loguru import logger

try:
    import mediapipe as mp
except ImportError:
    logger.warning("MediaPipe not installed. Install with: pip install mediapipe")
    mp = None

from utils.helpers import timeit


@dataclass
class PostureMetrics:
    """Container for posture analysis metrics."""
    shoulder_slope: float  # Degrees from horizontal
    head_tilt: float  # Degrees from vertical
    spine_angle: float  # Degrees from vertical
    head_in_hands: bool
    is_slouching: bool
    movement_score: float  # 0=still, 1=high movement


class PostureAnalyzer:
    """
    Analyzes body posture and movement from video frames.
    
    Uses MediaPipe Pose for skeleton tracking.
    """
    
    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        smoothing_window: int = 5
    ):
        """
        Initialize the posture analyzer.
        
        Args:
            model_complexity: 0=lite, 1=full, 2=heavy
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
            smoothing_window: Number of frames to smooth over
        """
        if mp is None:
            raise ImportError("MediaPipe not installed")
        
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            smooth_landmarks=True
        )
        
        self.smoothing_window = smoothing_window
        self.pose_history: List[any] = []
        
        logger.info(f"PostureAnalyzer initialized with complexity={model_complexity}")
    
    @timeit
    def analyze(self, frame: np.ndarray) -> Dict[str, any]:
        """
        Analyze posture from a video frame.
        
        Args:
            frame: RGB image as numpy array (H, W, 3)
            
        Returns:
            Dictionary containing:
            - pose_detected: Whether pose was found
            - metrics: PostureMetrics object
            - landmarks: MediaPipe landmarks (33 points)
            - posture_state: Overall posture classification
        """
        try:
            # Process frame
            results = self.pose.process(frame)
            
            if not results.pose_landmarks:
                return {
                    'pose_detected': False,
                    'metrics': None,
                    'landmarks': None,
                    'posture_state': 'unknown'
                }
            
            # Extract landmarks
            landmarks = results.pose_landmarks.landmark
            
            # Calculate posture metrics
            metrics = self._calculate_metrics(landmarks, frame.shape)
            
            # Store in history for movement analysis
            self.pose_history.append(landmarks)
            if len(self.pose_history) > self.smoothing_window:
                self.pose_history.pop(0)
            
            # Classify posture state
            posture_state = self._classify_posture(metrics)
            
            return {
                'pose_detected': True,
                'metrics': metrics,
                'landmarks': landmarks,
                'posture_state': posture_state
            }
            
        except Exception as e:
            logger.error(f"Error in posture analysis: {e}")
            return {
                'pose_detected': False,
                'metrics': None,
                'landmarks': None,
                'posture_state': 'error'
            }
    
    def _calculate_metrics(
        self, 
        landmarks: any, 
        frame_shape: Tuple[int, int, int]
    ) -> PostureMetrics:
        """Calculate posture metrics from landmarks."""
        h, w, _ = frame_shape
        
        # Get key points
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        
        # Calculate shoulder slope
        shoulder_slope = self._calculate_angle(
            (left_shoulder.x * w, left_shoulder.y * h),
            (right_shoulder.x * w, right_shoulder.y * h),
            horizontal=True
        )
        
        # Calculate head tilt (nose to shoulder midpoint)
        shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2 * w
        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2 * h
        head_tilt = self._calculate_angle(
            (nose.x * w, nose.y * h),
            (shoulder_mid_x, shoulder_mid_y),
            horizontal=False
        )
        
        # Calculate spine angle (shoulder to hip)
        hip_mid_x = (left_hip.x + right_hip.x) / 2 * w
        hip_mid_y = (left_hip.y + right_hip.y) / 2 * h
        spine_angle = self._calculate_angle(
            (shoulder_mid_x, shoulder_mid_y),
            (hip_mid_x, hip_mid_y),
            horizontal=False
        )
        
        # Detect head-in-hands (hands near face)
        head_in_hands = self._detect_head_in_hands(
            nose, left_wrist, right_wrist
        )
        
        # Detect slouching (forward head, rounded shoulders)
        is_slouching = abs(head_tilt) > 15 or abs(spine_angle) > 20
        
        # Calculate movement score
        movement_score = self._calculate_movement()
        
        return PostureMetrics(
            shoulder_slope=shoulder_slope,
            head_tilt=head_tilt,
            spine_angle=spine_angle,
            head_in_hands=head_in_hands,
            is_slouching=is_slouching,
            movement_score=movement_score
        )
    
    def _calculate_angle(
        self, 
        point1: Tuple[float, float], 
        point2: Tuple[float, float],
        horizontal: bool = False
    ) -> float:
        """Calculate angle between two points."""
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        
        if horizontal:
            # Angle from horizontal
            angle = np.degrees(np.arctan2(dy, dx))
        else:
            # Angle from vertical
            angle = 90 - np.degrees(np.arctan2(dy, dx))
        
        return angle
    
    def _detect_head_in_hands(
        self, 
        nose: any, 
        left_wrist: any, 
        right_wrist: any
    ) -> bool:
        """Detect if hands are near face (head-in-hands gesture)."""
        # Calculate distance from each wrist to nose
        left_dist = np.sqrt(
            (nose.x - left_wrist.x)**2 + 
            (nose.y - left_wrist.y)**2
        )
        right_dist = np.sqrt(
            (nose.x - right_wrist.x)**2 + 
            (nose.y - right_wrist.y)**2
        )
        
        # If either hand is within threshold distance
        threshold = 0.2  # Normalized coordinates
        return left_dist < threshold or right_dist < threshold
    
    def _calculate_movement(self) -> float:
        """Calculate movement score from pose history."""
        if len(self.pose_history) < 2:
            return 0.0
        
        # Compare current pose to previous poses
        total_movement = 0.0
        current = self.pose_history[-1]
        
        for prev in self.pose_history[:-1]:
            # Calculate average landmark displacement
            displacement = sum(
                np.sqrt(
                    (c.x - p.x)**2 + 
                    (c.y - p.y)**2 + 
                    (c.z - p.z)**2
                )
                for c, p in zip(current, prev)
            ) / len(current)
            
            total_movement += displacement
        
        # Normalize
        movement_score = min(total_movement / len(self.pose_history), 1.0)
        return movement_score
    
    def _classify_posture(self, metrics: PostureMetrics) -> str:
        """Classify overall posture state."""
        if metrics.head_in_hands:
            return "frustrated"  # Head in hands indicates frustration/stress
        elif metrics.is_slouching:
            return "slouching"  # Poor posture, fatigue
        elif metrics.movement_score > 0.5:
            return "pacing"  # High movement, anxiety/restlessness
        elif metrics.movement_score < 0.05:
            return "still"  # Very low movement, deep focus or dissociation
        else:
            return "normal"
    
    def visualize(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """
        Draw posture analysis results on frame.
        
        Args:
            frame: Input frame
            result: Analysis result from analyze()
            
        Returns:
            Annotated frame
        """
        frame_copy = frame.copy()
        
        if not result['pose_detected']:
            return frame_copy
        
        # Draw pose landmarks
        self.mp_drawing.draw_landmarks(
            frame_copy,
            result['landmarks'],
            self.mp_pose.POSE_CONNECTIONS
        )
        
        # Draw posture state
        metrics = result['metrics']
        state_text = f"Posture: {result['posture_state']}"
        cv2.putText(
            frame_copy,
            state_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        
        # Draw metrics
        if metrics:
            metrics_text = f"Slouch: {metrics.is_slouching}, Movement: {metrics.movement_score:.2f}"
            cv2.putText(
                frame_copy,
                metrics_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        
        return frame_copy
    
    def cleanup(self):
        """Release resources."""
        self.pose.close()


class MockPostureAnalyzer:
    """Mock analyzer for development/testing without MediaPipe."""
    
    def __init__(self, *args, **kwargs):
        logger.warning("Using MockPostureAnalyzer - install mediapipe for real detection")
    
    def analyze(self, frame: np.ndarray) -> Dict:
        return {
            'pose_detected': True,
            'metrics': PostureMetrics(
                shoulder_slope=0.0,
                head_tilt=0.0,
                spine_angle=0.0,
                head_in_hands=False,
                is_slouching=False,
                movement_score=0.1
            ),
            'landmarks': None,
            'posture_state': 'normal'
        }
    
    def visualize(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        return frame
    
    def cleanup(self):
        pass
