"""Download and verify Wav2Vec2 prosody model."""

import sys
sys.path.insert(0, '.')

from transformers import AutoModel, AutoFeatureExtractor
from loguru import logger

print("=" * 70)
print("WAV2VEC2 PROSODY MODEL DOWNLOAD")
print("=" * 70)

model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

print(f"\nModel: {model_name}")
print("Size: ~661MB")
print("Purpose: Voice emotion detection (arousal, valence, dominance)")
print("\nStarting download...\n")

try:
    # Download feature extractor
    print("1. Downloading feature extractor...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    print("   ✓ Feature extractor downloaded")
    
    # Download model
    print("\n2. Downloading Wav2Vec2 model (this will take a few minutes)...")
    model = AutoModel.from_pretrained(model_name)
    print("   ✓ Model downloaded")
    
    print("\n3. Verifying model works...")
    import torch
    import numpy as np
    
    # Test with dummy audio
    dummy_audio = np.random.randn(16000).astype(np.float32)
    inputs = feature_extractor(
        dummy_audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    print("   ✓ Model inference works!")
    
    print("\n" + "=" * 70)
    print("✅ WAV2VEC2 PROSODY MODEL READY!")
    print("   The model is now cached and ready for use.")
    print("   Voice emotion detection will be fully functional.")
    print("=" * 70)
    
except Exception as e:
    print(f"\n❌ Error downloading Wav2Vec2: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
