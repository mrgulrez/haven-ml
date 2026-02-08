"""Simple test - check if video pipeline is using real or mock models."""

import sys
sys.path.insert(0, '.')

from models.vision import VideoPipeline
from config import Config
import cv2
import numpy as np

print("=" * 70)
print("VIDEO PIPELINE DEBUG TEST")
print("=" * 70)

# Load config
config = Config()

# Initialize pipeline with use_mock=False
print("\n1. Initializing VideoPipeline with use_mock=False...")
pipeline = VideoPipeline(config, use_mock=False)

# Check what detector is being used
print(f"\n2. Face detector type: {type(pipeline.face_detector).__name__}")
print(f"   Detector class: {pipeline.face_detector.__class__}")

# Test with webcam frame
print("\n3. Opening webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam")
    sys.exit(1)

ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Cannot read frame")
    sys.exit(1)

print(f"✅ Frame captured: {frame.shape}")

# Convert to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Test face detector directly
print("\n4. Testing face detector directly...")
result = pipeline.face_detector.detect(frame_rgb)
print(f"   Result: {result}")

# Test full pipeline
print("\n5. Testing full pipeline...")
import asyncio

async def test():
    result = await pipeline.process_frame(frame_rgb, 0.0)
    return result

pipeline_result = asyncio.run(test())
print(f"   Pipeline result: {pipeline_result}")

print("\n" + "=" * 70)
print("If detector is 'DeepFaceEmotionDetector' and face_detected=True,")
print("then DeepFace is working!")
print("=" * 70)
