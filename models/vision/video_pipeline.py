"""Unified video processing pipeline.

Combines facial expression, posture, and gaze analysis
into a single async pipeline for real-time processing.
"""

import asyncio
import cv2
import numpy as np
from typing import Dict, Optional
from loguru import logger
import time

from .face_emotion import FaceEmotionDetector, MockFaceEmotionDetector
from .posture_analyzer import PostureAnalyzer, MockPostureAnalyzer
from .gaze_tracker import GazeTracker, MockGazeTracker
from utils.helpers import timeit, LatencyTracker


class VideoPipeline:
    """
    Processes video frames through all vision models in parallel.
    
    Architecture:
    - Runs face emotion, posture, and gaze analysis concurrently
    - Aggregates results into a unified emotional state
    - Tracks latency for optimization
    """
    
    def __init__(
        self,
        config: Dict,
        use_mock: bool = False
    ):
        """
        Initialize the video pipeline.
        
        Args:
            config: Configuration dictionary
            use_mock: Use mock models for testing
        """
        self.config = config
        self.latency_tracker = LatencyTracker()
        
        # Initialize models
        if use_mock:
            logger.info("Using mock models (use_mock=True)")
            from .deepface_detector import MockDeepFaceDetector
            self.face_detector = MockDeepFaceDetector()
            self.posture_analyzer = MockPostureAnalyzer()
            self.gaze_tracker = MockGazeTracker()
        else:
            logger.info("Attempting to load real vision models...")
            
            # Try DeepFace for emotion detection
            face_detector_loaded = False
            try:
                from models.vision.deepface_detector import DeepFaceEmotionDetector
                self.face_detector = DeepFaceEmotionDetector(device='cpu')
                face_detector_loaded = True
                logger.info("✓ DeepFace emotion detector loaded successfully")
            except Exception as e:
                logger.warning(f"DeepFace failed to load: {e}")
                logger.warning("Trying HSEmotion fallback...")
                try:
                    from .face_emotion import FaceEmotionDetector
                    self.face_detector = FaceEmotionDetector(
                        model_name=config.get('models.vision.hsemotion.model_name', 'enet_b0_8_best_afew'),
                        device='cpu',
                        confidence_threshold=0.3
                    )
                    face_detector_loaded = True
                    logger.info("✓ HSEmotion detector loaded")
                except Exception as e2:
                    logger.error(f"HSEmotion also failed: {e2}")
            
            # Load MediaPipe models
            posture_loaded = False
            try:
                self.posture_analyzer = PostureAnalyzer(
                    model_complexity=config.get('models.vision.mediapipe.model_complexity', 1),
                    min_detection_confidence=config.get('models.vision.mediapipe.min_detection_confidence', 0.5),
                    min_tracking_confidence=config.get('models.vision.mediapipe.min_tracking_confidence', 0.5),
                    smoothing_window=config.get('models.vision.gaze.smoothing_window', 5)
                )
                posture_loaded = True
                logger.info("✓ Posture analyzer loaded")
            except Exception as e:
                logger.warning(f"Posture analyzer failed: {e}")
            
            # Load gaze tracker
            gaze_loaded = False
            try:
                self.gaze_tracker = GazeTracker(
                    smoothing_window=config.get('models.vision.gaze.smoothing_window', 5),
                    stare_threshold=3.0
                )
                gaze_loaded = True
                logger.info("✓ Gaze tracker loaded")
            except Exception as e:
                logger.warning(f"Gaze tracker failed: {e}")
            
            # Fallback to mocks if any failed
            if not face_detector_loaded:
                logger.warning("Using MockDeepFaceDetector")
                from .deepface_detector import MockDeepFaceDetector
                self.face_detector = MockDeepFaceDetector()
            
            if not posture_loaded:
                logger.warning("Using MockPostureAnalyzer")
                self.posture_analyzer = MockPostureAnalyzer()
            
            if not gaze_loaded:
                logger.warning("Using MockGazeTracker")
                self.gaze_tracker = MockGazeTracker()
            
            if face_detector_loaded and posture_loaded and gaze_loaded:
                logger.info("✓ All real vision models loaded successfully!")
            else:
                logger.warning("⚠ Some models using mocks")
        
        self.frame_count = 0
        self.fps_target = config.get('models.vision.hsemotion.fps_target', 30)
    
    @timeit
    async def process_frame(
        self, 
        frame: np.ndarray, 
        timestamp: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Process a single video frame through all vision models.
        
        Args:
            frame: RGB image as numpy array (H, W, 3)
            timestamp: Frame timestamp in seconds
            
        Returns:
            Dictionary containing:
            - face_emotion: Facial expression results
            - posture: Posture analysis results
            - gaze: Gaze tracking results
            - visual_state: Aggregated visual emotional state
            - processing_time_ms: Total processing time
        """
        if timestamp is None:
            timestamp = time.time()
        
        start_time = time.perf_counter()
        
        # Skip frames based on FPS target
        self.frame_count += 1
        if self.frame_count % (60 // self.fps_target) != 0:
            return self._empty_result()
        
        try:
            # Run all models in parallel
            face_task = asyncio.create_task(
                self._run_face_detection(frame)
            )
            posture_task = asyncio.create_task(
                self._run_posture_analysis(frame)
            )
            gaze_task = asyncio.create_task(
                self._run_gaze_tracking(frame, timestamp)
            )
            
            # Wait for all tasks
            face_result = await face_task
            posture_result = await posture_task
            gaze_result = await gaze_task
            
            # Aggregate results
            visual_state = self._aggregate_results(
                face_result, 
                posture_result, 
                gaze_result
            )
            
            # Track latency
            processing_time = (time.perf_counter() - start_time) * 1000
            self.latency_tracker.record('video_pipeline', processing_time)
            
            return {
                'face_emotion': face_result,
                'posture': posture_result,
                'gaze': gaze_result,
                'visual_state': visual_state,
                'processing_time_ms': processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in video pipeline: {e}")
            return self._empty_result()
    
    async def _run_face_detection(self, frame: np.ndarray) -> Dict:
        """Run face emotion detection."""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.face_detector.detect,
                frame
            )
            # Log for debugging
            if result.get('face_detected'):
                logger.debug(f"Face detected: {result.get('primary_emotion')} (conf: {result.get('confidence', 0):.2f})")
            return result
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return {
                'face_detected': False,
                'primary_emotion': 'unknown',
                'confidence': 0.0,
                'valence': 0.0,
                'arousal': 0.0
            }
    
    async def _run_posture_analysis(self, frame: np.ndarray) -> Dict:
        """Run posture analysis (async wrapper)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.posture_analyzer.analyze, frame)
    
    async def _run_gaze_tracking(self, frame: np.ndarray, timestamp: float) -> Dict:
        """Run gaze tracking (async wrapper)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.gaze_tracker.track, frame, timestamp)
    
    def _aggregate_results(
        self, 
        face: Dict, 
        posture: Dict, 
        gaze: Dict
    ) -> Dict:
        """
        Aggregate vision results into unified emotional state.
        
        Combines insights from face, posture, and gaze to produce
        a holistic visual emotional assessment.
        """
        # Extract primary signals
        primary_emotion = face.get('primary_emotion', 'unknown')
        face_confidence = face.get('confidence', 0.0)
        posture_state = posture.get('posture_state', 'unknown')
        gaze_pattern = gaze.get('gaze_pattern', 'unknown')
        
        # Calculate valence and arousal
        if face.get('face_detected', False) and primary_emotion != 'unknown':
            valence, arousal = self.face_detector.get_emotion_valence_arousal(primary_emotion)
        else:
            valence, arousal = 0.0, 0.0
        
        # Adjust based on posture
        if posture_state == 'slouching':
            valence -= 0.2  # Slouching suggests negativity
            arousal -= 0.1  # Lower energy
        elif posture_state == 'frustrated':
            valence -= 0.3
            arousal += 0.2
        elif posture_state == 'pacing':
            arousal += 0.3  # High movement = high arousal
        
        # Adjust based on gaze
        if gaze_pattern == 'blank_stare':
            arousal -= 0.3  # Low engagement
        elif gaze_pattern == 'looking_down':
            valence -= 0.2  # Sadness indicator
        elif gaze_pattern == 'rapid_scanning':
            arousal += 0.2  # Anxiety indicator
        
        # Clamp values
        valence = max(-1.0, min(1.0, valence))
        arousal = max(0.0, min(1.0, arousal))
        
        # Determine overall state
        overall_state = self._classify_overall_state(
            primary_emotion, posture_state, gaze_pattern, valence, arousal
        )
        
        return {
            'primary_emotion': primary_emotion,
            'emotion_confidence': face_confidence,
            'posture_state': posture_state,
            'gaze_pattern': gaze_pattern,
            'valence': valence,
            'arousal': arousal,
            'overall_state': overall_state
        }
    
    def _classify_overall_state(
        self,
        emotion: str,
        posture: str,
        gaze: str,
        valence: float,
        arousal: float
    ) -> str:
        """Classify overall emotional state from multimodal signals."""
        # High-priority patterns (strong indicators)
        if posture == 'frustrated' or gaze == 'blank_stare':
            return 'distressed'
        
        if posture == 'slouching' and valence < -0.3:
            return 'fatigued'
        
        if gaze == 'looking_down' and emotion in ['sad', 'neutral']:
            return 'sad'
        
        if gaze == 'rapid_scanning' or posture == 'pacing':
            return 'anxious'
        
        # Use valence-arousal classification
        if valence > 0.5 and arousal > 0.5:
            return 'excited'
        elif valence > 0.3:
            return 'positive'
        elif valence < -0.5 and arousal > 0.5:
            return 'stressed'
        elif valence < -0.3:
            return 'negative'
        elif arousal < 0.2:
            return 'calm'
        else:
            return 'neutral'
    
    def _empty_result(self) -> Dict:
        """Return empty result for skipped/failed frames."""
        return {
            'face_emotion': {'face_detected': False},
            'posture': {'pose_detected': False},
            'gaze': {'face_detected': False},
            'visual_state': {
                'primary_emotion': 'unknown',
                'overall_state': 'unknown',
                'valence': 0.0,
                'arousal': 0.0
            },
            'processing_time_ms': 0.0
        }
    
    def visualize(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """
        Draw all vision analysis results on frame.
        
        Args:
            frame: Input frame
            result: Pipeline result from process_frame()
            
        Returns:
            Annotated frame with all visual overlays
        """
        annotated = frame.copy()
        
        # Draw face emotion
        if result['face_emotion'].get('face_detected'):
            annotated = self.face_detector.visualize(
                annotated, 
                result['face_emotion']
            )
        
        # Draw posture
        if result['posture'].get('pose_detected'):
            annotated = self.posture_analyzer.visualize(
                annotated, 
                result['posture']
            )
        
        # Draw gaze
        if result['gaze'].get('face_detected'):
            annotated = self.gaze_tracker.visualize(
                annotated, 
                result['gaze']
            )
        
        # Draw overall state
        visual_state = result['visual_state']
        state_text = f"State: {visual_state['overall_state']} (V:{visual_state['valence']:.2f}, A:{visual_state['arousal']:.2f})"
        cv2.putText(
            annotated,
            state_text,
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )
        
        return annotated
    
    def get_latency_stats(self) -> Dict:
        """Get latency statistics for the pipeline."""
        return self.latency_tracker.get_stats('video_pipeline')
    
    def cleanup(self):
        """Release all resources."""
        try:
            self.posture_analyzer.cleanup()
            self.gaze_tracker.cleanup()
            logger.info("VideoPipeline cleaned up")
        except:
            pass
