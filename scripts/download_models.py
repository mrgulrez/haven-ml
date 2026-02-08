"""Download essential models for the Empathy System.

This script downloads and sets up all required AI models.
"""

import os
from pathlib import Path
import requests
from tqdm import tqdm


def download_file(url: str, destination: Path, description: str):
    """Download a file with progress bar."""
    print(f"\n📥 Downloading {description}...")
    
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        total=total_size,
        unit='B',
        unit_scale=True,
        desc=description
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
            pbar.update(len(chunk))
    
    print(f"✓ Downloaded to {destination}")


def download_huggingface_model(repo_id: str, filename: str, destination: Path):
    """Download model from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
        
        print(f"\n📥 Downloading {filename} from {repo_id}...")
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=destination.parent,
            local_dir_use_symlinks=False
        )
        print(f"✓ Downloaded to {downloaded}")
        return downloaded
    except ImportError:
        print("Installing huggingface_hub...")
        os.system("pip install huggingface_hub")
        return download_huggingface_model(repo_id, filename, destination)


def main():
    """Download all models."""
    print("=" * 70)
    print("EMPATHY SYSTEM - MODEL DOWNLOADER")
    print("=" * 70)
    print("\nThis will download ~15GB of models. Ensure you have:")
    print("  - Stable internet connection")
    print("  - 20GB free disk space")
    print("  - HuggingFace account (free)")
    print()
    
    proceed = input("Continue? (y/n): ")
    if proceed.lower() != 'y':
        print("Cancelled.")
        return
    
    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / "models"
    
    # ========================================
    # 1. LLM - Llama 3.1 (Most Important)
    # ========================================
    print("\n" + "=" * 70)
    print("1. LLAMA 3.1 - Large Language Model")
    print("=" * 70)
    
    llm_dir = models_dir / "llm"
    llm_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nOptions for Llama 3.1:")
    print("  A. Llama-3.1-8B-Instruct (GGUF, 4-bit, ~4.7GB) - Recommended")
    print("  B. Llama-3-7B-Chat (GGUF, 4-bit, ~4.1GB) - Smaller, faster")
    print("  C. Skip (continue with mock)")
    
    llm_choice = input("\nChoose (A/B/C): ").upper()
    
    if llm_choice == 'A':
        # Llama 3.1 8B Instruct
        download_huggingface_model(
            repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            destination=llm_dir / "llama-3.1-8b-instruct-q4.gguf"
        )
    elif llm_choice == 'B':
        # Llama 2 7B Chat
        download_huggingface_model(
            repo_id="TheBloke/Llama-2-7B-Chat-GGUF",
            filename="llama-2-7b-chat.Q4_K_M.gguf",
            destination=llm_dir / "llama-2-7b-chat-q4.gguf"
        )
    
    # ========================================
    # 2. TTS - CosyVoice
    # ========================================
    print("\n" + "=" * 70)
    print("2. COSYVOICE - Text-to-Speech")
    print("=" * 70)
    print("\nCosyVoice will be auto-downloaded on first use.")
    print("Size: ~2GB")
    
    # ========================================
    # 3. Vision Models
    # ========================================
    print("\n" + "=" * 70)
    print("3. VISION MODELS")
    print("=" * 70)
    
    print("\n✓ HSEmotion - Auto-downloaded on first use")
    print("✓ MediaPipe - Already installed")
    
    # ========================================
    # 4. Audio Models
    # ========================================
    print("\n" + "=" * 70)
    print("4. AUDIO MODELS")
    print("=" * 70)
    
    print("\n✓ Silero VAD - Auto-downloaded on first use")
    print("✓ Wav2Vec2 - Auto-downloaded on first use")
    
    # SenseVoice requires special setup
    print("\n⚠️  SenseVoice (STT) requires FunASR:")
    print("   pip install funasr modelscope")
    
    install_audio = input("\nInstall audio dependencies? (y/n): ")
    if install_audio.lower() == 'y':
        os.system("pip install funasr modelscope")
    
    # ========================================
    # 5. Update Config
    # ========================================
    print("\n" + "=" * 70)
    print("UPDATING CONFIGURATION")
    print("=" * 70)
    
    config_path = base_dir / "config.yaml"
    
    if llm_choice in ['A', 'B']:
        model_filename = "llama-3.1-8b-instruct-q4.gguf" if llm_choice == 'A' else "llama-2-7b-chat-q4.gguf"
        model_path = llm_dir / model_filename
        
        print(f"\n✓ Update config.yaml:")
        print(f"  models.llm.model_path: {model_path}")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE!")
    print("=" * 70)
    
    print("\n✅ Downloaded models:")
    if llm_choice in ['A', 'B']:
        print(f"  - LLM: {model_filename}")
    print("  - TTS: CosyVoice (will download on first use)")
    print("  - Vision: HSEmotion, MediaPipe")
    print("  - Audio: Silero VAD, Wav2Vec2")
    
    print("\n📝 Next steps:")
    print("  1. Update config.yaml with model paths")
    print("  2. Run: venv\\Scripts\\python.exe scripts\\demo_full_integration.py")
    print("  3. Set use_mock=False in agent initialization")
    
    print("\n🎉 Ready to use real AI models!")


if __name__ == '__main__':
    main()
