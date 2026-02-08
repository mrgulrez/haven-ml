"""Base agent interface for the Empathy System."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import asyncio
from loguru import logger
from datetime import datetime

class BaseAgent(ABC):
    """
    Abstract base class for all empathy agents.
    
    Defines the core interface and event loop structure that all
    agents must implement.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for this agent instance
            config: Configuration dictionary
        """
        self.agent_id = agent_id
        self.config = config
        self.is_running = False
        self.session_start = None
        self.user_id: Optional[str] = None
        self.persona: str = "remote_worker"  # Default persona
        
        logger.info(f"Agent {agent_id} initialized")
    
    async def start(self, user_id: str, persona: str = "remote_worker") -> None:
        """
        Start the agent for a specific user session.
        
        Args:
            user_id: User identifier
            persona: User persona (remote_worker, student, young_professional)
        """
        self.user_id = user_id
        self.persona = persona
        self.session_start = datetime.now()
        self.is_running = True
        
        logger.info(f"Agent {self.agent_id} started for user {user_id} with persona '{persona}'")
        
        # Initialize subcomponents
        await self.initialize()
        
        # Start the main event loop
        await self.run()
    
    async def stop(self) -> None:
        """Stop the agent and cleanup resources."""
        self.is_running = False
        await self.cleanup()
        
        session_duration = (datetime.now() - self.session_start).total_seconds() if self.session_start else 0
        logger.info(f"Agent {self.agent_id} stopped. Session duration: {session_duration:.1f}s")
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize agent resources (models, connections, etc.).
        
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    async def run(self) -> None:
        """
        Main event loop for processing audio/video packets.
        
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """
        Cleanup resources when agent stops.
        
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    async def on_video_frame(self, frame: Any) -> Dict[str, Any]:
        """
        Process a single video frame.
        
        Args:
            frame: Video frame data
            
        Returns:
            Dictionary containing visual emotion analysis
        """
        pass
    
    @abstractmethod
    async def on_audio_packet(self, audio: Any) -> Dict[str, Any]:
        """
        Process a single audio packet.
        
        Args:
            audio: Audio data
            
        Returns:
            Dictionary containing audio emotion analysis
        """
        pass
    
    @abstractmethod
    async def on_multimodal_fusion(
        self, 
        visual_data: Dict[str, Any], 
        audio_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fuse visual and audio emotion data.
        
        Args:
            visual_data: Results from video analysis
            audio_data: Results from audio analysis
            
        Returns:
            Fused emotion assessment with valence, arousal, and primary emotion
        """
        pass
    
    @abstractmethod
    async def generate_response(
        self, 
        emotion_state: Dict[str, Any],
        user_text: Optional[str] = None
    ) -> str:
        """
        Generate empathetic response using LLM.
        
        Args:
            emotion_state: Current emotional state from fusion
            user_text: Optional text transcription from user
            
        Returns:
            Generated response text
        """
        pass
    
    @abstractmethod
    async def synthesize_speech(
        self, 
        text: str, 
        emotion_style: str
    ) -> bytes:
        """
        Convert text to speech with emotional inflection.
        
        Args:
            text: Text to synthesize
            emotion_style: Emotion style for TTS (e.g., "soothing", "excited")
            
        Returns:
            Audio bytes
        """
        pass
    
    async def check_intervention_triggers(
        self, 
        emotion_state: Dict[str, Any]
    ) -> Optional[str]:
        """
        Check if proactive intervention is needed.
        
        Args:
            emotion_state: Current emotional state
            
        Returns:
            Intervention message if triggered, None otherwise
        """
        # This can be overridden by subclasses for custom logic
        return None
    
    def get_persona_config(self) -> Dict[str, Any]:
        """Get configuration for the current persona."""
        return self.config.get('personas', {}).get(self.persona, {})
