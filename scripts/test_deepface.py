"""Quick test to verify DeepFace is working with real emotion detection."""

import cv2
import numpy as np
from models.vision.deepface_detector import DeepFaceEmotionDetector
from loguru import logger

def test_deepface():
    """Test DeepFace emotion detection."""
    print("=" * 70)
    print("DEEPFACE EMOTION DETECTION TEST")
    print("=" * 70)
    print("\nThis will test real-time emotion detection with DeepFace.")
    print("Try different expressions:")
    print("  😊 Happy - Smile")
    print("  😢 Sad - Frown") 
    print("  😠 Angry - Furrow brows")
    print("  😲 Surprise - Open mouth, raise eyebrows")
    print("  😨 Fear - Wide eyes")
    print("\nPress 'q' to quit\n")
    
    # Initialize detector
    detector = DeepFaceEmotionDetector(device='cpu')
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    print("✅ Webcam opened. Starting detection...\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB for detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect emotions (every 5 frames to avoid slowdown)
        if frame_count % 5 == 0:
            result = detector.detect(frame_rgb)
            
            if result['face_detected']:
                emotion = result['primary_emotion']
                confidence = result['confidence']
                valence = result['valence']
                arousal = result['arousal']
                
                print(f"\r🎭 Emotion: {emotion.upper():10s} | Confidence: {confidence:.2f} | V: {valence:+.2f} A: {arousal:+.2f}", end='', flush=True)
                
                # Visualize
                annotated = detector.visualize(frame, result)
                cv2.imshow('DeepFace Emotion Test', annotated)
            else:
                print("\r❌ No face detected" + " " * 50, end='', flush=True)
                cv2.imshow('DeepFace Emotion Test', frame)
        else:
            cv2.imshow('DeepFace Emotion Test', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n\n✅ Test complete!")
    print("\nIf emotions changed when you made different facial expressions,")
    print("DeepFace is working correctly!")


if __name__ == '__main__':
    test_deepface()
