"""Verification script to confirm real DeepFace is being used."""

import sys
sys.path.insert(0, '.')

from models.vision import VideoPipeline
from config import Config
import cv2
import numpy as np
import asyncio

print("=" * 70)
print("REAL vs MOCK VERIFICATION TEST")
print("=" * 70)

config = Config()

# Initialize pipeline
print("\n1. Creating VideoPipeline with use_mock=False...")
pipeline = VideoPipeline(config, use_mock=False)

# Check detector type
detector_type = type(pipeline.face_detector).__name__
print(f"\n2. Face detector type: {detector_type}")

if 'Mock' in detector_type:
    print("   ❌ USING MOCK DETECTOR")
    print("   This means real DeepFace failed to load")
else:
    print("   ✅ USING REAL DETECTOR")
    print("   DeepFace is properly loaded!")

# Check if it has the DeepFace analyze method
print(f"\n3. Detector class: {pipeline.face_detector.__class__.__module__}.{pipeline.face_detector.__class__.__name__}")
print(f"   Has detect method: {hasattr(pipeline.face_detector, 'detect')}")

# Test with multiple frames to see if values change
print("\n4. Testing with webcam frames...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("   ❌ Cannot open webcam")
    sys.exit(1)

results = []
for i in range(3):
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pipeline.face_detector.detect(frame_rgb)
        results.append(result)
        print(f"   Frame {i+1}: emotion={result.get('primary_emotion', 'N/A')}, "
              f"conf={result.get('confidence', 0):.2f}, "
              f"face_detected={result.get('face_detected', False)}")

cap.release()

# Check if values vary (real) or are static (mock)
if len(results) >= 2:
    emotions = [r.get('primary_emotion') for r in results]
    confidences = [r.get('confidence', 0) for r in results]
    
    print(f"\n5. Analysis:")
    print(f"   Emotions detected: {emotions}")
    print(f"   All same? {len(set(emotions)) == 1}")
    
    # Mock always returns same values
    if emotions[0] == emotions[1] and confidences[0] == confidences[1]:
        print("   ⚠️  Values are IDENTICAL - might be mock or truly neutral")
    else:
        print("   ✅ Values VARY - definitely using real detection!")

print("\n" + "=" * 70)
print("VERDICT:")
if 'Mock' not in detector_type:
    print("✅ REAL DeepFace emotion detection is ACTIVE!")
    print("   Emotions should update when you change expressions.")
else:
    print("❌ Still using MOCK detector")
    print("   Need to debug why DeepFace import failed")
print("=" * 70)
