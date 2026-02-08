"""Gaze tracking using MediaPipe Face Mesh.

Detects eye gaze direction to identify:
- Blank staring (dissociation, fatigue)
- Looking down (sadness, avoidance)
- Rapid scanning (anxiety, distraction)
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from loguru import logger

try:
    import mediapipe as mp
except ImportError:
    logger.warning("MediaPipe not installed")
    mp = None

from utils.helpers import timeit


@dataclass
class GazeMetrics:
    """Container for gaze analysis metrics."""
    gaze_direction: str  # 'center', 'left', 'right', 'up', 'down'
    gaze_yaw: float  # Horizontal angle (-1=left, 0=center, 1=right)
    gaze_pitch: float  # Vertical angle (-1=down, 0=center, 1=up)
    blink_rate: float  # Blinks per minute
    fixation_duration: float  # Seconds looking at same point
    is_staring: bool  # Prolonged fixation without movement


class GazeTracker:
    """
    Tracks eye gaze direction and patterns.
    
    Uses MediaPipe Face Mesh for eye landmark detection.
    """
    
    def __init__(
        self,
        smoothing_window: int = 5,
        stare_threshold: float = 3.0  # seconds
    ):
        """
        Initialize the gaze tracker.
        
        Args:
            smoothing_window: Number of frames to smooth over
            stare_threshold: Duration to classify as staring
        """
        if mp is None:
            raise ImportError("MediaPipe not installed")
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.smoothing_window = smoothing_window
        self.stare_threshold = stare_threshold
        
        # History tracking
        self.gaze_history: List[Tuple[float, float]] = []
        self.blink_events: List[float] = []  # Timestamps
        self.fixation_start: Optional[float] = None
        self.frame_count = 0
        
        # Eye landmarks indices (MediaPipe Face Mesh)
        self.LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.LEFT_IRIS_INDICES = [474, 475, 476, 477]
        self.RIGHT_IRIS_INDICES = [469, 470, 471, 472]
        
        logger.info("GazeTracker initialized")
    
    @timeit
    def track(self, frame: np.ndarray, timestamp: float) -> Dict[str, any]:
        """
        Track gaze from a video frame.
        
        Args:
            frame: RGB image as numpy array (H, W, 3)
            timestamp: Current timestamp in seconds
            
        Returns:
            Dictionary containing:
            - face_detected: Whether face was found
            - metrics: GazeMetrics object
            - gaze_pattern: Overall gaze pattern classification
        """
        try:
            results = self.face_mesh.process(frame)
            
            if not results.multi_face_landmarks:
                return {
                    'face_detected': False,
                    'metrics': None,
                    'gaze_pattern': 'unknown'
                }
            
            landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape
            
            # Calculate gaze direction
            gaze_yaw, gaze_pitch = self._estimate_gaze(landmarks, w, h)
            
            # Store in history
            self.gaze_history.append((gaze_yaw, gaze_pitch))
            if len(self.gaze_history) > self.smoothing_window:
                self.gaze_history.pop(0)
            
            # Detect blinks
            if self._detect_blink(landmarks):
                self.blink_events.append(timestamp)
                # Remove old events (older than 60s)
                self.blink_events = [t for t in self.blink_events if timestamp - t < 60]
            
            # Calculate blink rate (per minute)
            blink_rate = len(self.blink_events)
            
            # Calculate fixation
            fixation_duration, is_staring = self._calculate_fixation(timestamp)
            
            # Classify gaze direction
            gaze_direction = self._classify_direction(gaze_yaw, gaze_pitch)
            
            metrics = GazeMetrics(
                gaze_direction=gaze_direction,
                gaze_yaw=gaze_yaw,
                gaze_pitch=gaze_pitch,
                blink_rate=blink_rate,
                fixation_duration=fixation_duration,
                is_staring=is_staring
            )
            
            # Classify overall pattern
            gaze_pattern = self._classify_pattern(metrics)
            
            return {
                'face_detected': True,
                'metrics': metrics,
                'gaze_pattern': gaze_pattern
            }
            
        except Exception as e:
            logger.error(f"Error in gaze tracking: {e}")
            return {
                'face_detected': False,
                'metrics': None,
                'gaze_pattern': 'error'
            }
    
    def _estimate_gaze(
        self, 
        landmarks: any, 
        w: int, 
        h: int
    ) -> Tuple[float, float]:
        """
        Estimate gaze direction from iris position.
        
        Returns:
            (yaw, pitch) in range [-1, 1]
        """
        # Get eye corners and iris centers
        left_eye_left = landmarks[33]
        left_eye_right = landmarks[133]
        left_iris_center = self._get_center(
            [landmarks[i] for i in self.LEFT_IRIS_INDICES]
        )
        
        right_eye_left = landmarks[362]
        right_eye_right = landmarks[263]
        right_iris_center = self._get_center(
            [landmarks[i] for i in self.RIGHT_IRIS_INDICES]
        )
        
        # Calculate iris position relative to eye corners
        left_ratio_x = (left_iris_center.x - left_eye_left.x) / (left_eye_right.x - left_eye_left.x)
        right_ratio_x = (right_iris_center.x - right_eye_left.x) / (right_eye_right.x - right_eye_left.x)
        
        # Average horizontal gaze
        gaze_yaw = ((left_ratio_x + right_ratio_x) / 2 - 0.5) * 2  # Normalize to [-1, 1]
        
        # Vertical gaze (simplified - use iris y-position)
        eye_top = landmarks[159]  # Top of eye
        eye_bottom = landmarks[145]  # Bottom of eye
        iris_ratio_y = (left_iris_center.y - eye_top.y) / (eye_bottom.y - eye_top.y)
        gaze_pitch = (iris_ratio_y - 0.5) * 2  # Normalize to [-1, 1]
        
        return float(gaze_yaw), float(gaze_pitch)
    
    def _get_center(self, points: List) -> any:
        """Calculate center point of landmarks."""
        avg_x = sum(p.x for p in points) / len(points)
        avg_y = sum(p.y for p in points) / len(points)
        avg_z = sum(p.z for p in points) / len(points)
        
        class Point:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z
        
        return Point(avg_x, avg_y, avg_z)
    
    def _detect_blink(self, landmarks: any) -> bool:
        """Detect eye blink using Eye Aspect Ratio (EAR)."""
        def eye_aspect_ratio(eye_points):
            # Calculate vertical distances
            v1 = np.linalg.norm(np.array([eye_points[1].x, eye_points[1].y]) - 
                               np.array([eye_points[5].x, eye_points[5].y]))
            v2 = np.linalg.norm(np.array([eye_points[2].x, eye_points[2].y]) - 
                               np.array([eye_points[4].x, eye_points[4].y]))
            # Calculate horizontal distance
            h = np.linalg.norm(np.array([eye_points[0].x, eye_points[0].y]) - 
                              np.array([eye_points[3].x, eye_points[3].y]))
            
            return (v1 + v2) / (2.0 * h)
        
        left_eye = [landmarks[i] for i in self.LEFT_EYE_INDICES]
        right_eye = [landmarks[i] for i in self.RIGHT_EYE_INDICES]
        
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        
        # Average EAR
        ear = (left_ear + right_ear) / 2.0
        
        # Blink threshold (typically < 0.2)
        return ear < 0.2
    
    def _calculate_fixation(self, timestamp: float) -> Tuple[float, bool]:
        """Calculate how long gaze has been fixed."""
        if len(self.gaze_history) < 2:
            return 0.0, False
        
        # Check if gaze is relatively stable
        recent_gaze = self.gaze_history[-min(10, len(self.gaze_history)):]
        variance = np.var([g[0]**2 + g[1]**2 for g in recent_gaze])
        
        is_stable = variance < 0.01  # Low variance = stable gaze
        
        if is_stable:
            if self.fixation_start is None:
                self.fixation_start = timestamp
            fixation_duration = timestamp - self.fixation_start
        else:
            self.fixation_start = None
            fixation_duration = 0.0
        
        is_staring = fixation_duration > self.stare_threshold
        
        return fixation_duration, is_staring
    
    def _classify_direction(self, yaw: float, pitch: float) -> str:
        """Classify gaze direction."""
        # Thresholds
        YAW_THRESHOLD = 0.3
        PITCH_THRESHOLD = 0.3
        
        if abs(yaw) < YAW_THRESHOLD and abs(pitch) < PITCH_THRESHOLD:
            return 'center'
        elif pitch < -PITCH_THRESHOLD:
            return 'down'  # Looking down (sadness indicator)
        elif pitch > PITCH_THRESHOLD:
            return 'up'
        elif yaw < -YAW_THRESHOLD:
            return 'left'
        elif yaw > YAW_THRESHOLD:
            return 'right'
        else:
            return 'center'
    
    def _classify_pattern(self, metrics: GazeMetrics) -> str:
        """Classify overall gaze pattern."""
        if metrics.is_staring:
            return 'blank_stare'  # Dissociation, fatigue
        elif metrics.gaze_direction == 'down' and metrics.fixation_duration > 1.0:
            return 'looking_down'  # Sadness, avoidance
        elif len(self.gaze_history) >= 5:
            # Check for rapid scanning (high variance in recent gaze)
            recent = self.gaze_history[-5:]
            variance = np.var([g[0]**2 + g[1]**2 for g in recent])
            if variance > 0.1:
                return 'rapid_scanning'  # Anxiety, distraction
        
        return 'normal'
    
    def visualize(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Draw gaze tracking results on frame."""
        frame_copy = frame.copy()
        
        if not result['face_detected']:
            return frame_copy
        
        metrics = result['metrics']
        if metrics:
            # Draw gaze direction arrow
            h, w, _ = frame.shape
            center_x, center_y = w // 2, h // 2
            
            arrow_length = 100
            end_x = int(center_x + metrics.gaze_yaw * arrow_length)
            end_y = int(center_y + metrics.gaze_pitch * arrow_length)
            
            cv2.arrowedLine(
                frame_copy,
                (center_x, center_y),
                (end_x, end_y),
                (0, 255, 255),
                3
            )
            
            # Draw pattern text
            pattern_text = f"Gaze: {result['gaze_pattern']}"
            cv2.putText(
                frame_copy,
                pattern_text,
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )
        
        return frame_copy
    
    def cleanup(self):
        """Release resources."""
        self.face_mesh.close()


class MockGazeTracker:
    """Mock tracker for development/testing."""
    
    def __init__(self, *args, **kwargs):
        logger.warning("Using MockGazeTracker - install mediapipe for real tracking")
    
    def track(self, frame: np.ndarray, timestamp: float) -> Dict:
        return {
            'face_detected': True,
            'metrics': GazeMetrics(
                gaze_direction='center',
                gaze_yaw=0.0,
                gaze_pitch=0.0,
                blink_rate=15.0,
                fixation_duration=0.5,
                is_staring=False
            ),
            'gaze_pattern': 'normal'
        }
    
    def visualize(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        return frame
    
    def cleanup(self):
        pass
