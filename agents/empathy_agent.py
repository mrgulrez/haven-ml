"""Main empathy agent orchestrating all components.

Brings together perception, fusion, memory, LLM, and TTS
into a unified conversational agent.
"""

import asyncio
import numpy as np
from typing import Dict, Optional
from datetime import datetime
from loguru import logger

from models.vision import VideoPipeline
from models.audio import AudioPipeline
from models.fusion import FusionEngine, TemporalSmoother
from llm import LlamaClient, MockLlamaClient, PromptBuilder
from memory import SessionMemory, UserProfile
from models.tts import CosyVoiceTTS, MockTTS
from utils.helpers import LatencyTracker
from config import config


class EmpathyAgent:
    """
    Main empathy agent coordinating all components.
    
    Architecture:
    1. Video/Audio pipelines → Emotional state
    2. Fusion engine → Unified emotion
    3. LLM + Memory → Contextual response
    4. TTS → Voice output
    """
    
    def __init__(
        self,
        user_id: str,
        persona: str = 'remote_worker',
        use_mock: bool = False
    ):
        """
        Initialize empathy agent.
        
        Args:
            user_id: User identifier
            persona: User persona (remote_worker, student, young_professional)
            use_mock: Use mock models for testing
        """
        self.user_id = user_id
        self.persona = persona
        self.use_mock = use_mock
        
        # Latency tracking
        self.latency_tracker = LatencyTracker()
        
        # Initialize components
        logger.info(f"Initializing EmpathyAgent for user {user_id} (persona: {persona})")
        
        # Perception
        self.video_pipeline = VideoPipeline(config, use_mock=use_mock)
        self.audio_pipeline = AudioPipeline(config, use_mock=use_mock)
        
        # Fusion
        import torch
        device = 'cpu' if (use_mock or not torch.cuda.is_available()) else 'cuda'
        self.fusion_engine = FusionEngine(device=device)
        self.temporal_smoother = TemporalSmoother(window_size=30, alpha=0.2)
        
        # Memory
        self.session_memory = SessionMemory()
        self.user_profile = UserProfile(
            user_id,
            storage_path=config.get('memory.user_profiles_path', './data/profiles')
        )
        
        # LLM
        try:
            if use_mock:
                raise ImportError("Using mock")
            
            model_path = config.get('models.llm.model_path')
            self.llm = LlamaClient(
                model_path=model_path,
                context_length=config.get('models.llm.context_length', 8192),
                n_gpu_layers=config.get('models.llm.n_gpu_layers', 40),
                temperature=config.get('models.llm.temperature', 0.7)
            )
        except:
            logger.warning("Using MockLlamaClient")
            self.llm = MockLlamaClient()
        
        self.prompt_builder = PromptBuilder(persona=persona)
        
        # TTS
        try:
            if use_mock:
                raise ImportError("Using mock")
            
            device = 'cpu' if not torch.cuda.is_available() else 'cuda'
            self.tts = CosyVoiceTTS(device=device)
        except:
            logger.warning("Using MockTTS")
            self.tts = MockTTS()
        
        # State
        self.current_emotional_state: Optional[Dict] = None
        self.last_intervention_time: Optional[datetime] = None
        
        logger.info("✓ EmpathyAgent initialized successfully")
    
    async def start_session(self, session_id: str):
        """Start a new session."""
        self.session_memory.start_session(session_id, self.user_id, self.persona)
        self.user_profile.record_session_start()
        logger.info(f"Session {session_id} started")
    
    async def process_video_frame(
        self,
        frame: np.ndarray,
        timestamp: float
    ) -> Dict:
        """
        Process video frame.
        
        Args:
            frame: RGB image frame
            timestamp: Frame timestamp
            
        Returns:
            Visual state
        """
        result = await self.video_pipeline.process_frame(frame, timestamp)
        return result['visual_state']
    
    async def process_audio_chunk(
        self,
        audio: np.ndarray,
        timestamp: float
    ) -> Dict:
        """
        Process audio chunk.
        
        Args:
            audio: Audio samples
            timestamp: Audio timestamp
            
        Returns:
            Audio result with transcription if available
        """
        result = await self.audio_pipeline.process_audio(audio, timestamp)
        return result
    
    async def process_multimodal(
        self,
        visual_state: Dict,
        audio_result: Dict
    ) -> Dict:
        """
        Fuse visual and audio states.
        
        Args:
            visual_state: From video pipeline
            audio_result: From audio pipeline
            
        Returns:
            Fused emotional state
        """
        # Extract audio state
        audio_state = audio_result['audio_state']
        
        # Fuse
        fused = self.fusion_engine.fuse(visual_state, audio_state)
        
        # Temporal smoothing
        smoothed = self.temporal_smoother.smooth(fused)
        
        # Record in memory
        self.session_memory.add_emotional_snapshot(
            emotion=smoothed.primary_emotion,
            valence=smoothed.valence,
            arousal=smoothed.arousal,
            authenticity=smoothed.authenticity_score,
            context=f"fused from visual and audio"
        )
        
        # Store current state
        self.current_emotional_state = {
            'primary_emotion': smoothed.primary_emotion,
            'valence': smoothed.valence,
            'arousal': smoothed.arousal,
            'dominance': smoothed.dominance,
            'confidence': smoothed.confidence,
            'authenticity_score': smoothed.authenticity_score,
            'emotion_stability': smoothed.emotion_stability,
            'trend': smoothed.trend
        }
        
        return self.current_emotional_state
    
    async def generate_response(
        self,
        user_text: Optional[str] = None,
        intervention_type: Optional[str] = None
    ) -> str:
        """
        Generate contextual response.
        
        Args:
            user_text: User's spoken text (if any)
            intervention_type: Proactive intervention type (if any)
            
        Returns:
            Generated text response
        """
        if not self.current_emotional_state:
            logger.warning("No emotional state available for response generation")
            return "I'm here for you."
        
        # Build prompt
        prompt = self.prompt_builder.build_prompt(
            emotional_state=self.current_emotional_state,
            user_text=user_text,
            intervention_type=intervention_type,
            time_of_day=datetime.now()
        )
        
        # Generate
        response = self.llm.generate(prompt)
        
        # Add to history
        if user_text:
            self.prompt_builder.add_to_history('user', user_text)
        self.prompt_builder.add_to_history('assistant', response)
        
        # Record intervention if proactive
        if intervention_type:
            self.session_memory.record_intervention(
                intervention_type,
                response,
                self.current_emotional_state
            )
            self.last_intervention_time = datetime.now()
        
        return response
    
    async def synthesize_speech(self, text: str) -> np.ndarray:
        """
        Synthesize speech with emotional modulation.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio samples
        """
        if not self.current_emotional_state:
            return self.tts.synthesize(text)
        
        # Use emotional modulation
        audio = self.tts.modulate_emotion(
            text,
            valence=self.current_emotional_state['valence'],
            arousal=self.current_emotional_state['arousal']
        )
        
        return audio
    
    async def handle_user_speech(
        self,
        transcribed_text: str
    ) -> tuple[str, np.ndarray]:
        """
        Handle user speech end-to-end.
        
        Args:
            transcribed_text: Transcribed user speech
            
        Returns:
            (response_text, response_audio)
        """
        # Generate response
        response_text = await self.generate_response(user_text=transcribed_text)
        
        # Synthesize
        response_audio = await self.synthesize_speech(response_text)
        
        return response_text, response_audio
    
    async def check_intervention_triggers(self) -> Optional[str]:
        """
        Check if proactive intervention should be triggered.
        
        Returns:
            Intervention type or None
        """
        if not self.current_emotional_state:
            return None
        
        # Emotional masking
        if self.current_emotional_state['authenticity_score'] < 0.5:
            if self.user_profile.should_intervene('emotional_masking'):
                return 'emotional_masking'
        
        # Prolonged negative state
        if self.current_emotional_state['valence'] < -0.4:
            sustained = self.temporal_smoother.get_sustained_emotion(duration_frames=30)
            if sustained in ['sad', 'stressed', 'anxious']:
                if self.user_profile.should_intervene('prolonged_negative_state'):
                    return 'prolonged_negative_state'
        
        # Declining trend
        if self.current_emotional_state['trend'] == 'declining':
            summary = self.session_memory.get_emotional_summary(window_minutes=15)
            if summary['avg_valence'] < -0.3:
                if self.user_profile.should_intervene('declining_mood'):
                    return 'declining_mood'
        
        return None
    
    async def proactive_intervention(self) -> Optional[tuple[str, np.ndarray]]:
        """
        Execute proactive intervention if triggered.
        
        Returns:
            (intervention_text, intervention_audio) or None
        """
        intervention_type = await self.check_intervention_triggers()
        
        if not intervention_type:
            return None
        
        logger.info(f"Triggering proactive intervention: {intervention_type}")
        
        # Generate intervention
        intervention_text = await self.generate_response(
            intervention_type=intervention_type
        )
        
        # Synthesize
        intervention_audio = await self.synthesize_speech(intervention_text)
        
        return intervention_text, intervention_audio
    
    def get_session_summary(self) -> Dict:
        """Get current session summary."""
        return self.session_memory.get_emotional_summary(window_minutes=60)
    
    def save_profile(self):
        """Save user profile to disk."""
        self.user_profile.save()
    
    async def cleanup(self):
        """Cleanup resources."""
        self.video_pipeline.cleanup()
        self.audio_pipeline.reset()
        self.session_memory.clear()
        self.save_profile()
        logger.info("EmpathyAgent cleanup complete")
