"""
Gradio Web App for Hugging Face Spaces

Interactive demo of the Empathy System with:
- Image upload for face emotion detection
- Audio upload for voice analysis
- Real-time emotional state display
- Empathetic response generation
"""

import gradio as gr
import numpy as np
from PIL import Image
import cv2
import asyncio
from loguru import logger

# Import empathy system components
from agents.empathy_agent import EmpathyAgent
from models.vision.deepface_detector import DeepFaceEmotionDetector
from models.audio.whisper_stt import WhisperSTT
from models.audio.prosody import ProsodyAnalyzer

# Initialize components (singleton pattern)
agent = None

def initialize_agent():
    """Initialize the empathy agent (called once)."""
    global agent
    if agent is None:
        logger.info("Initializing Empathy Agent...")
        agent = EmpathyAgent('web_user', persona='remote_worker', use_mock=False)
        asyncio.run(agent.start_session('web_demo_001'))
        logger.info("✓ Agent initialized")
    return agent

def analyze_image(image):
    """
    Analyze facial emotion from uploaded image.
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        str: Emotion analysis results
    """
    if image is None:
        return "❌ Please upload an image"
    
    try:
        # Initialize agent
        agent = initialize_agent()
        
        # Convert to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Process through vision pipeline
        result = asyncio.run(agent.video_pipeline.process_frame(image_bgr))
        
        # Format results
        if result.get('face_detected'):
            emotion = result.get('primary_emotion', 'unknown')
            confidence = result.get('confidence', 0)
            valence = result.get('valence', 0)
            arousal = result.get('arousal', 0)
            
            output = f"""
## 😊 Face Emotion Detected!

**Primary Emotion**: {emotion.upper()}  
**Confidence**: {confidence:.2%}

**Emotional State**:
- Valence (positive/negative): {valence:+.2f}
- Arousal (calm/excited): {arousal:.2f}

**Interpretation**:
"""
            # Add interpretation
            if emotion == 'happy':
                output += "You appear to be in a positive, joyful state. 🌟"
            elif emotion == 'sad':
                output += "You seem to be experiencing sadness. I'm here to support you. 💙"
            elif emotion == 'angry':
                output += "I detect frustration or anger. Would you like to talk about it? 🤝"
            elif emotion == 'fear':
                output += "You appear anxious or fearful. It's okay, I'm here to help. 🫂"
            elif emotion == 'surprise':
                output += "Something surprising! Tell me more. ✨"
            elif emotion == 'neutral':
                output += "You appear calm and neutral. 😌"
            else:
                output += f"Detected emotion: {emotion}"
            
            return output
        else:
            return "❌ No face detected in the image. Please upload a clear image with a visible face."
            
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        return f"❌ Error processing image: {str(e)}"

def analyze_audio(audio_file):
    """
    Analyze voice emotion from uploaded audio.
    
    Args:
        audio_file: Path to audio file
        
    Returns:
        str: Voice analysis results
    """
    if audio_file is None:
        return "❌ Please upload an audio file"
    
    try:
        # Initialize agent
        agent = initialize_agent()
        
        # Load audio
        import librosa
        audio, sr = librosa.load(audio_file, sr=16000, mono=True)
        
        # Process through audio pipeline
        result = asyncio.run(agent.audio_pipeline.process_audio(audio))
        
        # Format results
        audio_state = result.get('audio_state', {})
        
        output = f"""
## 🎵 Voice Analysis Complete!

**Transcription**: "{audio_state.get('transcribed_text', 'N/A')}"

**Voice Emotion**: {audio_state.get('audio_emotion', 'unknown')}

**Emotional Metrics**:
- Valence: {audio_state.get('valence', 0):+.2f}
- Arousal: {audio_state.get('arousal', 0):.2f}
- Tremor: {audio_state.get('tremor', 0):.2f}

**Detected Events**: {', '.join(audio_state.get('detected_events', [])) or 'None'}
"""
        
        return output
        
    except Exception as e:
        logger.error(f"Error analyzing audio: {e}")
        return f"❌ Error processing audio: {str(e)}"

def analyze_multimodal(image, audio_file):
    """
    Analyze both face and voice for complete emotional understanding.
    
    Args:
        image: PIL Image
        audio_file: Path to audio file
        
    Returns:
        str: Complete multimodal analysis
    """
    if image is None and audio_file is None:
        return "❌ Please provide at least an image or audio file"
    
    try:
        # Initialize agent
        agent = initialize_agent()
        
        results = []
        
        # Analyze image if provided
        if image is not None:
            vision_output = analyze_image(image)
            results.append("### 📹 Vision Analysis\n" + vision_output)
        
        # Analyze audio if provided
        if audio_file is not None:
            audio_output = analyze_audio(audio_file)
            results.append("### 🎵 Audio Analysis\n" + audio_output)
        
        # Add fusion insights
        if image is not None and audio_file is not None:
            results.append("""
### 🧠 Multimodal Fusion

By combining facial expressions and voice tone, I can detect:
- **Emotional Masking**: When your face shows one emotion but voice shows another
- **True Emotional State**: More accurate understanding of how you really feel
- **Context**: Better grasp of the situation

This holistic view allows for more empathetic and appropriate responses.
""")
        
        return "\n\n".join(results)
        
    except Exception as e:
        logger.error(f"Error in multimodal analysis: {e}")
        return f"❌ Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Empathy System - Multimodal Emotion AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 Empathy System - Multimodal Emotion AI
    
    An advanced AI system that understands emotions through facial expressions and voice tone.
    
    **Upload an image or audio file** to analyze emotional state using:
    - 📹 **DeepFace**: 7-emotion facial detection
    - 🎵 **Whisper**: Speech transcription  
    - 🧠 **Wav2Vec2**: Voice emotion analysis
    - 🤝 **Llama 3.1**: Empathetic understanding
    """)
    
    with gr.Tabs():
        with gr.Tab("Image Analysis"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="Upload Image with Face")
                    image_btn = gr.Button("Analyze Face Emotion", variant="primary")
                with gr.Column():
                    image_output = gr.Markdown(label="Results")
            
            image_btn.click(analyze_image, inputs=image_input, outputs=image_output)
            
            gr.Examples(
                examples=[],  # Add example images here
                inputs=image_input
            )
        
        with gr.Tab("Audio Analysis"):
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(type="filepath", label="Upload Audio")
                    audio_btn = gr.Button("Analyze Voice Emotion", variant="primary")
                with gr.Column():
                    audio_output = gr.Markdown(label="Results")
            
            audio_btn.click(analyze_audio, inputs=audio_input, outputs=audio_output)
        
        with gr.Tab("Multimodal Analysis"):
            gr.Markdown("### Combine face and voice for complete emotional understanding")
            with gr.Row():
                with gr.Column():
                    multi_image = gr.Image(type="pil", label="Upload Image")
                    multi_audio = gr.Audio(type="filepath", label="Upload Audio")
                    multi_btn = gr.Button("Analyze Complete Emotional State", variant="primary")
                with gr.Column():
                    multi_output = gr.Markdown(label="Results")
            
            multi_btn.click(analyze_multimodal, inputs=[multi_image, multi_audio], outputs=multi_output)
    
    gr.Markdown("""
    ---
    ### 🔬 Technology Stack
    - **Vision**: DeepFace (7 emotions: happy, sad, angry, fear, disgust, surprise, neutral)
    - **Audio**: Whisper STT + Wav2Vec2 prosody analysis
    - **Fusion**: Transformer-based multimodal integration
    - **LLM**: Llama 3.1 8B for contextual understanding
    
    ### 📊 What It Detects
    - Facial emotions with confidence scores
    - Voice tone (valence, arousal, tremor)
    - Speech transcription
    - Emotional masking (face vs. voice mismatch)
    - Audio events (sighs, breathing patterns)
    
    ### 🔒 Privacy
    All processing happens on the server. No data is stored or shared.
    """)

if __name__ == "__main__":
    demo.launch()
