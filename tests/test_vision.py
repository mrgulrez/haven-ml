"""Test suite for vision models."""

import pytest
import asyncio
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vision import (
    FaceEmotionDetector,
    PostureAnalyzer,
    GazeTracker,
    VideoPipeline,
    MockFaceEmotionDetector,
    MockPostureAnalyzer,
    MockGazeTracker
)
from config import config


# Test fixtures
@pytest.fixture
def test_frame():
    """Create a dummy RGB frame for testing."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def video_pipeline():
    """Create a video pipeline with mock models."""
    return VideoPipeline(config, use_mock=True)


# Face Emotion Tests
def test_mock_face_emotion_detector(test_frame):
    """Test mock face emotion detector."""
    detector = MockFaceEmotionDetector()
    result = detector.detect(test_frame)
    
    assert 'face_detected' in result
    assert 'primary_emotion' in result
    assert 'confidence' in result
    assert result['primary_emotion'] in ['neutral', 'happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'contempt']


def test_emotion_valence_arousal():
    """Test valence-arousal mapping."""
    detector = MockFaceEmotionDetector()
    
    # Test various emotions
    happy_va = detector.get_emotion_valence_arousal('happy')
    assert happy_va[0] > 0  # Positive valence
    assert happy_va[1] > 0  # High arousal
    
    sad_va = detector.get_emotion_valence_arousal('sad')
    assert sad_va[0] < 0  # Negative valence
    assert sad_va[1] < 0.5  # Low arousal


# Posture Analyzer Tests
def test_mock_posture_analyzer(test_frame):
    """Test mock posture analyzer."""
    analyzer = MockPostureAnalyzer()
    result = analyzer.analyze(test_frame)
    
    assert 'pose_detected' in result
    assert 'posture_state' in result
    assert result['posture_state'] in ['normal', 'slouching', 'frustrated', 'pacing', 'still']


def test_posture_metrics():
    """Test posture metrics structure."""
    analyzer = MockPostureAnalyzer()
    result = analyzer.analyze(np.zeros((480, 640, 3), dtype=np.uint8))
    
    if result['metrics']:
        metrics = result['metrics']
        assert hasattr(metrics, 'shoulder_slope')
        assert hasattr(metrics, 'head_tilt')
        assert hasattr(metrics, 'spine_angle')
        assert hasattr(metrics, 'head_in_hands')
        assert hasattr(metrics, 'is_slouching')
        assert hasattr(metrics, 'movement_score')


# Gaze Tracker Tests
def test_mock_gaze_tracker(test_frame):
    """Test mock gaze tracker."""
    tracker = MockGazeTracker()
    result = tracker.track(test_frame, timestamp=0.0)
    
    assert 'face_detected' in result
    assert 'gaze_pattern' in result
    assert result['gaze_pattern'] in ['normal', 'blank_stare', 'looking_down', 'rapid_scanning']


def test_gaze_metrics():
    """Test gaze metrics structure."""
    tracker = MockGazeTracker()
    result = tracker.track(np.zeros((480, 640, 3), dtype=np.uint8), timestamp=0.0)
    
    if result['metrics']:
        metrics = result['metrics']
        assert hasattr(metrics, 'gaze_direction')
        assert hasattr(metrics, 'gaze_yaw')
        assert hasattr(metrics, 'gaze_pitch')
        assert hasattr(metrics, 'blink_rate')
        assert hasattr(metrics, 'fixation_duration')
        assert hasattr(metrics, 'is_staring')


# Video Pipeline Tests
@pytest.mark.asyncio
async def test_video_pipeline_processing(video_pipeline, test_frame):
    """Test video pipeline frame processing."""
    result = await video_pipeline.process_frame(test_frame, timestamp=0.0)
    
    assert 'face_emotion' in result
    assert 'posture' in result
    assert 'gaze' in result
    assert 'visual_state' in result
    assert 'processing_time_ms' in result


@pytest.mark.asyncio
async def test_video_pipeline_visual_state(video_pipeline, test_frame):
    """Test visual state aggregation."""
    result = await video_pipeline.process_frame(test_frame, timestamp=0.0)
    
    visual_state = result['visual_state']
    assert 'overall_state' in visual_state
    assert 'valence' in visual_state
    assert 'arousal' in visual_state
    assert 'primary_emotion' in visual_state
    
    # Check value ranges
    assert -1.0 <= visual_state['valence'] <= 1.0
    assert 0.0 <= visual_state['arousal'] <= 1.0


@pytest.mark.asyncio
async def test_video_pipeline_parallel_processing(video_pipeline, test_frame):
    """Test that pipeline processes models in parallel."""
    import time
    
    start = time.perf_counter()
    result = await video_pipeline.process_frame(test_frame, timestamp=0.0)
    duration = time.perf_counter() - start
    
    # Processing should be relatively fast with mocks
    assert duration < 1.0  # Should complete in < 1 second
    assert result['processing_time_ms'] >= 0


def test_video_pipeline_visualization(video_pipeline, test_frame):
    """Test visualization overlay."""
    # Process frame first
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(
        video_pipeline.process_frame(test_frame, timestamp=0.0)
    )
    
    # Visualize
    annotated = video_pipeline.visualize(test_frame, result)
    
    # Should return image of same shape
    assert annotated.shape == test_frame.shape


def test_video_pipeline_latency_tracking(video_pipeline, test_frame):
    """Test latency tracking."""
    loop = asyncio.get_event_loop()
    
    # Process multiple frames
    for i in range(5):
        loop.run_until_complete(
            video_pipeline.process_frame(test_frame, timestamp=float(i))
        )
    
    # Get stats
    stats = video_pipeline.get_latency_stats()
    
    if stats:
        assert 'count' in stats
        assert 'mean' in stats
        assert 'p50' in stats
        assert 'p95' in stats


def test_video_pipeline_cleanup(video_pipeline):
    """Test pipeline cleanup."""
    video_pipeline.cleanup()
    # Should not raise exceptions


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
