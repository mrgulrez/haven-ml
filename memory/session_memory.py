"""Session memory for tracking current conversation context.

Maintains short-term emotional timeline and conversation state.
"""

from typing import Dict, List, Optional, Deque
from collections import deque
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger


@dataclass
class EmotionalSnapshot:
    """Single point in emotional timeline."""
    timestamp: datetime
    emotion: str
    valence: float
    arousal: float
    authenticity: float
    context: str  # What triggered this emotion


class SessionMemory:
    """
    Maintains short-term memory for the current session.
    
    Tracks emotional timeline, significant events, and
    conversation context within a single session.
    """
    
    def __init__(
        self,
        max_duration_hours: int = 8,
        timeline_capacity: int = 1000
    ):
        """
        Initialize session memory.
        
        Args:
            max_duration_hours: Maximum session duration
            timeline_capacity: Maximum timeline snapshots to keep
        """
        self.max_duration = timedelta(hours=max_duration_hours)
        self.timeline_capacity = timeline_capacity
        
        # Session metadata
        self.session_id: Optional[str] = None
        self.user_id: Optional[str] = None
        self.start_time: Optional[datetime] = None
        self.persona: str = 'remote_worker'
        
        # Emotional timeline
        self.emotional_timeline: Deque[EmotionalSnapshot] = deque(maxlen=timeline_capacity)
        
        # Significant events
        self.significant_events: List[Dict] = []
        
        # Intervention history
        self.interventions: List[Dict] = []
        
        logger.info("SessionMemory initialized")
    
    def start_session(
        self,
        session_id: str,
        user_id: str,
        persona: str = 'remote_worker'
    ):
        """Start a new session."""
        self.session_id = session_id
        self.user_id = user_id
        self.start_time = datetime.now()
        self.persona = persona
        
        logger.info(f"Session {session_id} started for user {user_id} with persona '{persona}'")
    
    def add_emotional_snapshot(
        self,
        emotion: str,
        valence: float,
        arousal: float,
        authenticity: float = 1.0,
        context: str = ""
    ):
        """
        Add emotional state to timeline.
        
        Args:
            emotion: Primary emotion
            valence: Emotional valence
            arousal: Arousal level
            authenticity: Authenticity score
            context: What triggered this state
        """
        snapshot = EmotionalSnapshot(
            timestamp=datetime.now(),
            emotion=emotion,
            valence=valence,
            arousal=arousal,
            authenticity=authenticity,
            context=context
        )
        
        self.emotional_timeline.append(snapshot)
        
        # Detect significant changes
        if len(self.emotional_timeline) >= 2:
            prev = self.emotional_timeline[-2]
            
            # Significant valence shift
            if abs(snapshot.valence - prev.valence) > 0.5:
                self.record_event(
                    'emotional_shift',
                    f"Valence shifted from {prev.valence:.2f} to {snapshot.valence:.2f}"
                )
            
            # Emotional masking detected
            if snapshot.authenticity < 0.5 and prev.authenticity >= 0.5:
                self.record_event(
                    'masking_detected',
                    f"User may be hiding emotions ({snapshot.emotion})"
                )
    
    def record_event(self, event_type: str, description: str, metadata: Optional[Dict] = None):
        """Record a significant event."""
        event = {
            'timestamp': datetime.now(),
            'type': event_type,
            'description': description,
            'metadata': metadata or {}
        }
        
        self.significant_events.append(event)
        logger.info(f"Event recorded: {event_type} - {description}")
    
    def record_intervention(
        self,
        intervention_type: str,
        message: str,
        emotional_state: Dict
    ):
        """Record a proactive intervention."""
        intervention = {
            'timestamp': datetime.now(),
            'type': intervention_type,
            'message': message,
            'emotional_state': emotional_state
        }
        
        self.interventions.append(intervention)
        logger.info(f"Intervention recorded: {intervention_type}")
    
    def get_emotional_summary(self, window_minutes: int = 30) -> Dict:
        """
        Get emotional summary for recent time window.
        
        Args:
            window_minutes: Time window in minutes
            
        Returns:
            Summary statistics
        """
        if not self.emotional_timeline:
            return {
                'dominant_emotion': 'unknown',
                'avg_valence': 0.0,
                'avg_arousal': 0.5,
                'trend': 'stable'
            }
        
        # Filter to time window
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        recent = [s for s in self.emotional_timeline if s.timestamp >= cutoff]
        
        if not recent:
            recent = list(self.emotional_timeline)[-10:]  # Fallback to last 10
        
        # Calculate statistics
        from collections import Counter
        emotion_counts = Counter(s.emotion for s in recent)
        dominant_emotion = emotion_counts.most_common(1)[0][0]
        
        avg_valence = sum(s.valence for s in recent) / len(recent)
        avg_arousal = sum(s.arousal for s in recent) / len(recent)
        
        # Determine trend (simple)
        if len(recent) >= 3:
            early_valence = sum(s.valence for s in recent[:len(recent)//2]) / (len(recent)//2)
            late_valence = sum(s.valence for s in recent[len(recent)//2:]) / (len(recent) - len(recent)//2)
            
            if late_valence > early_valence + 0.2:
                trend = 'improving'
            elif late_valence < early_valence - 0.2:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return {
            'dominant_emotion': dominant_emotion,
            'avg_valence': avg_valence,
            'avg_arousal': avg_arousal,
            'trend': trend,
            'num_snapshots': len(recent)
        }
    
    def get_recent_events(self, count: int = 5) -> List[Dict]:
        """Get most recent significant events."""
        return self.significant_events[-count:] if self.significant_events else []
    
    def get_session_duration(self) -> timedelta:
        """Get current session duration."""
        if not self.start_time:
            return timedelta(0)
        return datetime.now() - self.start_time
    
    def should_end_session(self) -> bool:
        """Check if session should end based on duration."""
        return self.get_session_duration() >= self.max_duration
    
    def clear(self):
        """Clear session memory."""
        self.emotional_timeline.clear()
        self.significant_events.clear()
        self.interventions.clear()
        self.session_id = None
        self.user_id = None
        self.start_time = None
        
        logger.info("Session memory cleared")
