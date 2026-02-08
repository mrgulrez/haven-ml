"""Dynamic prompt builder with emotional context injection.

Constructs prompts that include emotional state, persona,
and conversation history for empathetic responses.
"""

from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger


class PromptBuilder:
    """
    Builds prompts with emotional context for the LLM.
    
    Injects current emotional state, persona, and memory
    to generate contextually appropriate responses.
    """
    
    SYSTEM_PROMPTS = {
        'remote_worker': """You are a supportive AI companion for remote workers. Your role is to:
- Encourage healthy work-life boundaries
- Celebrate accomplishments, both big and small
- Detect and address burnout and isolation
- Suggest breaks and social connections when needed
- Be professional yet warm and encouraging

Respond naturally and empathetically. Keep responses concise (2-3 sentences max).""",
        
        'student': """You are a motivational study companion for students. Your role is to:
- Help with study focus and productivity
- Reduce exam anxiety through encouragement
- Promote healthy sleep and study habits
- Break down overwhelming tasks into manageable steps
- Be friendly, supportive, and understanding

Respond naturally and empathetically. Keep responses concise (2-3 sentences max).""",
        
        'young_professional': """You are a confidence coach for young professionals. Your role is to:
- Combat imposter syndrome with validation
- Provide encouragement before presentations and meetings
- Help process workplace challenges
- Build professional confidence
- Be empowering and constructive

Respond naturally and empathetically. Keep responses concise (2-3 sentences max)."""
    }
    
    def __init__(self, persona: str = 'remote_worker'):
        """
        Initialize prompt builder.
        
        Args:
            persona: User persona (remote_worker, student, young_professional)
        """
        self.persona = persona
        self.conversation_history: List[Dict[str, str]] = []
        
        logger.info(f"PromptBuilder initialized for persona '{persona}'")
    
    def build_prompt(
        self,
        emotional_state: Dict,
        user_text: Optional[str] = None,
        intervention_type: Optional[str] = None,
        time_of_day: Optional[datetime] = None
    ) -> str:
        """
        Build a prompt with emotional context.
        
        Args:
            emotional_state: Fused emotional state from fusion engine
            user_text: User's spoken text (if any)
            intervention_type: Type of proactive intervention (if any)
            time_of_day: Current time for context
            
        Returns:
            Complete prompt for LLM
        """
        # Get system prompt for persona
        system_prompt = self.SYSTEM_PROMPTS.get(
            self.persona,
            self.SYSTEM_PROMPTS['remote_worker']
        )
        
        # Build emotional context
        emotion_context = self._build_emotion_context(emotional_state)
        
        # Build time context
        time_context = self._build_time_context(time_of_day)
        
        # Build conversation history
        history_context = self._build_history_context()
        
        # Determine response type
        if intervention_type:
            # Proactive intervention
            user_section = f"\n\nSituation: {intervention_type}\n{emotion_context}\n{time_context}"
            instruction = "\n\nProvide a brief, supportive intervention based on the situation."
        elif user_text:
            # Reactive response
            user_section = f"\n\nUser said: \"{user_text}\"\n{emotion_context}\n{time_context}"
            instruction = "\n\nRespond empathetically to what the user said, considering their emotional state."
        else:
            # Ambient check-in
            user_section = f"\n{emotion_context}\n{time_context}"
            instruction = "\n\nProvide a brief check-in based on their current state."
        
        # Construct full prompt
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{history_context}{user_section}{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        return prompt
    
    def _build_emotion_context(self, emotional_state: Dict) -> str:
        """Build emotional context string."""
        context_parts = []
        
        # Primary emotion
        emotion = emotional_state.get('primary_emotion', 'unknown')
        confidence = emotional_state.get('confidence', 0.0)
        context_parts.append(f"Current emotion: {emotion} (confidence: {confidence:.2f})")
        
        # Valence and arousal
        valence = emotional_state.get('valence', 0.0)
        arousal = emotional_state.get('arousal', 0.5)
        
        if valence < -0.3:
            context_parts.append("Mood appears negative")
        elif valence > 0.3:
            context_parts.append("Mood appears positive")
        
        if arousal > 0.7:
            context_parts.append("High energy/stress level")
        elif arousal < 0.3:
            context_parts.append("Low energy/calm state")
        
        # Authenticity (emotional masking)
        authenticity = emotional_state.get('authenticity_score', 1.0)
        if authenticity < 0.5:
            context_parts.append("⚠️ IMPORTANT: User may be hiding their true emotions (emotional masking detected). They might say they're fine but show signs of distress.")
        
        # Trend
        trend = emotional_state.get('trend', 'stable')
        if trend == 'declining':
            context_parts.append("Emotional state has been declining")
        elif trend == 'improving':
            context_parts.append("Emotional state has been improving")
        
        # Stability
        stability = emotional_state.get('emotion_stability', 0.5)
        if stability < 0.3:
            context_parts.append("Emotional state is volatile")
        
        return "Emotional Context:\n- " + "\n- ".join(context_parts)
    
    def _build_time_context(self, time_of_day: Optional[datetime]) -> str:
        """Build time-based context."""
        if not time_of_day:
            return ""
        
        hour = time_of_day.hour
        
        if 5 <= hour < 12:
            time_str = "morning"
        elif 12 <= hour < 17:
            time_str = "afternoon"
        elif 17 <= hour < 21:
            time_str = "evening"
        else:
            time_str = "late night"
        
        context = f"\nTime: {time_str}"
        
        # Add relevant concerns
        if hour < 6:
            context += " (very early - sleep concern?)"
        elif hour > 23 or hour < 5:
            context += " (very late - sleep deprivation?)"
        
        return context
    
    def _build_history_context(self, max_turns: int = 3) -> str:
        """Build recent conversation history."""
        if not self.conversation_history:
            return ""
        
        # Get last N turns
        recent = self.conversation_history[-max_turns:]
        
        history_lines = []
        for turn in recent:
            if turn['role'] == 'user':
                history_lines.append(f"User: {turn['content']}")
            else:
                history_lines.append(f"Assistant: {turn['content']}")
        
        if history_lines:
            return "Recent conversation:\n" + "\n".join(history_lines) + "\n\n"
        return ""
    
    def add_to_history(self, role: str, content: str):
        """Add message to conversation history."""
        self.conversation_history.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now()
        })
        
        # Keep only last 20 messages
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history.clear()
        logger.debug("Conversation history cleared")
    
    def set_persona(self, persona: str):
        """Change user persona."""
        if persona in self.SYSTEM_PROMPTS:
            self.persona = persona
            logger.info(f"Persona changed to '{persona}'")
        else:
            logger.warning(f"Unknown persona '{persona}', keeping '{self.persona}'")
