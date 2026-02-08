"""User profile storage for long-term preferences and patterns.

Maintains persistent user data across sessions.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from loguru import logger


@dataclass
class UserPreferences:
    """User preferences and settings."""
    persona: str = 'remote_worker'
    intervention_frequency: str = 'medium'  # low, medium, high
    preferred_response_style: str = 'balanced'  # brief, balanced, detailed
    privacy_mode: bool = False
    enable_proactive: bool = True


@dataclass
class EmotionalPattern:
    """Learned emotional pattern."""
    pattern_type: str  # e.g., 'morning_stress', 'evening_calm'
    description: str
    frequency: int  # How many times observed
    last_seen: datetime


class UserProfile:
    """
    Maintains long-term user profile and learned patterns.
    
    Stores preferences, intervention history, and
    emotional patterns across sessions.
    """
    
    def __init__(
        self,
        user_id: str,
        storage_path: Optional[str] = None
    ):
        """
        Initialize user profile.
        
        Args:
            user_id: Unique user identifier
            storage_path: Directory to store profiles
        """
        self.user_id = user_id
        
        # Storage
        if storage_path:
            self.storage_dir = Path(storage_path)
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self.profile_path = self.storage_dir / f"{user_id}.json"
        else:
            self.storage_dir = None
            self.profile_path = None
        
        # User data
        self.preferences = UserPreferences()
        self.created_at: datetime = datetime.now()
        self.last_session: Optional[datetime] = None
        self.total_sessions: int = 0
        
        # Learned patterns
        self.emotional_patterns: List[EmotionalPattern] = []
        
        # Intervention history
        self.intervention_history: List[Dict] = []
        
        # Load existing profile if available
        if self.profile_path and self.profile_path.exists():
            self.load()
        else:
            logger.info(f"Created new user profile for {user_id}")
    
    def update_preferences(self, **kwargs):
        """Update user preferences."""
        for key, value in kwargs.items():
            if hasattr(self.preferences, key):
                setattr(self.preferences, key, value)
                logger.info(f"Updated preference {key} = {value}")
    
    def record_session_start(self):
        """Record that a new session started."""
        self.last_session = datetime.now()
        self.total_sessions += 1
    
    def record_intervention(
        self,
        intervention_type: str,
        was_effective: Optional[bool] = None
    ):
        """
        Record an intervention for learning.
        
        Args:
            intervention_type: Type of intervention
            was_effective: Whether it was effective (if known)
        """
        record = {
            'timestamp': datetime.now(),
            'type': intervention_type,
            'effective': was_effective
        }
        
        self.intervention_history.append(record)
        
        # Keep only last 100
        if len(self.intervention_history) > 100:
            self.intervention_history = self.intervention_history[-100:]
    
    def learn_pattern(
        self,
        pattern_type: str,
        description: str
    ):
        """
        Learn a new emotional pattern.
        
        Args:
            pattern_type: Pattern identifier
            description: Human-readable description
        """
        # Check if pattern already exists
        for pattern in self.emotional_patterns:
            if pattern.pattern_type == pattern_type:
                # Update existing
                pattern.frequency += 1
                pattern.last_seen = datetime.now()
                logger.info(f"Updated pattern '{pattern_type}' (seen {pattern.frequency} times)")
                return
        
        # Add new pattern
        pattern = EmotionalPattern(
            pattern_type=pattern_type,
            description=description,
            frequency=1,
            last_seen=datetime.now()
        )
        self.emotional_patterns.append(pattern)
        logger.info(f"Learned new pattern: {pattern_type}")
    
    def get_intervention_effectiveness(
        self,
        intervention_type: str
    ) -> Optional[float]:
        """
        Calculate how effective a type of intervention has been.
        
        Args:
            intervention_type: Type to check
            
        Returns:
            Effectiveness rate (0.0-1.0) or None if no data
        """
        relevant = [
            i for i in self.intervention_history
            if i['type'] == intervention_type and i['effective'] is not None
        ]
        
        if not relevant:
            return None
        
        effective_count = sum(1 for i in relevant if i['effective'])
        return effective_count / len(relevant)
    
    def should_intervene(self, intervention_type: str) -> bool:
        """
        Decide if intervention should be triggered based on history.
        
        Args:
            intervention_type: Type to check
            
        Returns:
            Whether to proceed with intervention
        """
        # Check frequency preference
        freq_map = {'low': 0.3, 'medium': 0.6, 'high': 0.9}
        base_probability = freq_map.get(
            self.preferences.intervention_frequency,
            0.6
        )
        
        # Adjust based on effectiveness
        effectiveness = self.get_intervention_effectiveness(intervention_type)
        if effectiveness is not None:
            if effectiveness < 0.3:
                # This type has been ineffective, reduce
                base_probability *= 0.5
            elif effectiveness > 0.7:
                # This type works well, increase
                base_probability *= 1.2
        
        # Check if proactive is enabled
        if not self.preferences.enable_proactive:
            return False
        
        import random
        return random.random() < base_probability
    
    def save(self):
        """Save profile to disk."""
        if not self.profile_path:
            logger.warning("No storage path configured, cannot save")
            return
        
        try:
            data = {
                'user_id': self.user_id,
                'preferences': asdict(self.preferences),
                'created_at': self.created_at.isoformat(),
                'last_session': self.last_session.isoformat() if self.last_session else None,
                'total_sessions': self.total_sessions,
                'emotional_patterns': [
                    {
                        'pattern_type': p.pattern_type,
                        'description': p.description,
                        'frequency': p.frequency,
                        'last_seen': p.last_seen.isoformat()
                    }
                    for p in self.emotional_patterns
                ],
                'intervention_history': [
                    {
                        'timestamp': i['timestamp'].isoformat(),
                        'type': i['type'],
                        'effective': i['effective']
                    }
                    for i in self.intervention_history
                ]
            }
            
            with open(self.profile_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Profile saved to {self.profile_path}")
            
        except Exception as e:
            logger.error(f"Failed to save profile: {e}")
    
    def load(self):
        """Load profile from disk."""
        if not self.profile_path or not self.profile_path.exists():
            return
        
        try:
            with open(self.profile_path, 'r') as f:
                data = json.load(f)
            
            # Load preferences
            pref_data = data.get('preferences', {})
            self.preferences = UserPreferences(**pref_data)
            
            # Load metadata
            self.created_at = datetime.fromisoformat(data['created_at'])
            self.last_session = datetime.fromisoformat(data['last_session']) if data.get('last_session') else None
            self.total_sessions = data.get('total_sessions', 0)
            
            # Load patterns
            self.emotional_patterns = [
                EmotionalPattern(
                    pattern_type=p['pattern_type'],
                    description=p['description'],
                    frequency=p['frequency'],
                    last_seen=datetime.fromisoformat(p['last_seen'])
                )
                for p in data.get('emotional_patterns', [])
            ]
            
            # Load intervention history
            self.intervention_history = [
                {
                    'timestamp': datetime.fromisoformat(i['timestamp']),
                    'type': i['type'],
                    'effective': i.get('effective')
                }
                for i in data.get('intervention_history', [])
            ]
            
            logger.info(f"Profile loaded for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"Failed to load profile: {e}")
