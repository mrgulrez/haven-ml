"""Demo script for LLM integration with emotional context.

Tests the full pipeline: perception → fusion → memory → LLM response.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import MockLlamaClient, PromptBuilder
from memory import SessionMemory, UserProfile
from utils.logger import setup_logger


async def main():
    """Run LLM integration demo."""
    setup_logger("INFO")
    
    print("=" * 70)
    print("LLM Integration Demo - Emotional Context-Aware Responses")
    print("=" * 70)
    print("\nThis demo shows how the system generates empathetic responses")
    print("based on emotional state, conversation history, and user patterns.\n")
    
    # Initialize components
    print("Initializing components...")
    llm_client = MockLlamaClient()
    prompt_builder = PromptBuilder(persona='remote_worker')
    session_memory = SessionMemory()
    user_profile = UserProfile('demo_user')
    
    # Start session
    session_memory.start_session('demo_session', 'demo_user', persona='remote_worker')
    user_profile.record_session_start()
    print("✓ Components ready!\n")
    
    # Scenario 1: User feeling stressed
    print("=" * 70)
    print("SCENARIO 1: Detecting Stress")
    print("=" * 70)
    
    emotional_state_1 = {
        'primary_emotion': 'stressed',
        'confidence': 0.85,
        'valence': -0.4,
        'arousal': 0.8,
        'authenticity_score': 1.0,
        'trend': 'declining',
        'emotion_stability': 0.6
    }
    
    session_memory.add_emotional_snapshot(
        emotion='stressed',
        valence=-0.4,
        arousal=0.8,
        context='Work deadline approaching'
    )
    
    user_text_1 = "I have so much work to do today"
    
    print(f"\nUser says: \"{user_text_1}\"")
    print(f"Detected emotion: {emotional_state_1['primary_emotion']}")
    print(f"Valence: {emotional_state_1['valence']:.2f}, Arousal: {emotional_state_1['arousal']:.2f}")
    print(f"Trend: {emotional_state_1['trend']}\n")
    
    # Build prompt
    prompt_1 = prompt_builder.build_prompt(
        emotional_state=emotional_state_1,
        user_text=user_text_1,
        time_of_day=datetime.now()
    )
    
    print("Generated Prompt (excerpt):")
    print("-" * 70)
    print(prompt_1[-500:])  # Show last 500 chars
    print("-" * 70)
    
    # Generate response
    response_1 = llm_client.generate(prompt_1)
    print(f"\nAssistant: {response_1}\n")
    
    # Record in history
    prompt_builder.add_to_history('user', user_text_1)
    prompt_builder.add_to_history('assistant', response_1)
    
    # Scenario 2: Emotional Masking Detection
    print("\n" + "=" * 70)
    print("SCENARIO 2: Detecting Emotional Masking")
    print("=" * 70)
    
    emotional_state_2 = {
        'primary_emotion': 'happy',
        'confidence': 0.65,
        'valence': 0.3,
        'arousal': 0.6,
        'authenticity_score': 0.25,  # LOW - masking detected!
        'trend': 'stable',
        'emotion_stability': 0.5
    }
    
    session_memory.add_emotional_snapshot(
        emotion='happy',
        valence=0.3,
        arousal=0.6,
        authenticity=0.25,
        context='Saying fine but voice trembling'
    )
    
    user_text_2 = "I'm fine, everything is okay"
    
    print(f"\nUser says: \"{user_text_2}\"")
    print(f"Detected emotion: {emotional_state_2['primary_emotion']}")
    print(f"⚠️  AUTHENTICITY: {emotional_state_2['authenticity_score']:.2f} - EMOTIONAL MASKING DETECTED!")
    print(f"The system notices:")
    print(f"  - Positive words ('fine', 'okay')")
    print(f"  - But: Low authenticity score")
    print(f"  - Likely hiding true feelings\n")
    
    # Build prompt
    prompt_2 = prompt_builder.build_prompt(
        emotional_state=emotional_state_2,
        user_text=user_text_2,
        time_of_day=datetime.now()
    )
    
    print("Prompt includes masking warning:")
    print("-" * 70)
    # Find and show masking warning
    if 'hiding' in prompt_2.lower() or 'masking' in prompt_2.lower():
        lines = prompt_2.split('\n')
        for i, line in enumerate(lines):
            if 'hiding' in line.lower() or 'masking' in line.lower():
                print(line)
                if i + 1 < len(lines):
                    print(lines[i + 1])
    print("-" * 70)
    
    response_2 = llm_client.generate(prompt_2)
    print(f"\nAssistant (aware of masking): {response_2}\n")
    
    prompt_builder.add_to_history('user', user_text_2)
    prompt_builder.add_to_history('assistant', response_2)
    
    # Scenario 3: Proactive Check-in (no user input)
    print("\n" + "=" * 70)
    print("SCENARIO 3: Proactive Check-in")
    print("=" * 70)
    
    emotional_state_3 = {
        'primary_emotion': 'sad',
        'confidence': 0.75,
        'valence': -0.6,
        'arousal': 0.3,
        'authenticity_score': 1.0,
        'trend': 'declining',
        'emotion_stability': 0.4
    }
    
    session_memory.add_emotional_snapshot(
        emotion='sad',
        valence=-0.6,
        arousal=0.3,
        context='User has been quiet and looking down'
    )
    
    print("\nSituation: User has been quiet for 5 minutes, looking down")
    print(f"Detected emotion: {emotional_state_3['primary_emotion']}")
    print(f"Trend: {emotional_state_3['trend']} (getting worse)")
    print("\nSystem decides to proactively check in...\n")
    
    # Build prompt for proactive intervention
    prompt_3 = prompt_builder.build_prompt(
        emotional_state=emotional_state_3,
        intervention_type='prolonged_negative_state',
        time_of_day=datetime.now()
    )
    
    response_3 = llm_client.generate(prompt_3)
    print(f"Assistant (proactive): {response_3}\n")
    
    # Record intervention
    session_memory.record_intervention(
        'prolonged_negative_state',
        response_3,
        emotional_state_3
    )
    user_profile.record_intervention('prolonged_negative_state', was_effective=None)
    
    # Show session summary
    print("\n" + "=" * 70)
    print("SESSION SUMMARY")
    print("=" * 70)
    
    summary = session_memory.get_emotional_summary(window_minutes=30)
    print(f"\nDominant emotion: {summary['dominant_emotion']}")
    print(f"Average valence: {summary['avg_valence']:.2f}")
    print(f"Average arousal: {summary['avg_arousal']:.2f}")
    print(f"Trend: {summary['trend']}")
    print(f"Snapshots captured: {summary['num_snapshots']}")
    
    print(f"\nSignificant events:")
    for event in session_memory.get_recent_events(count=5):
        print(f"  - [{event['type']}] {event['description']}")
    
    print(f"\nInterventions: {len(session_memory.interventions)}")
    
    # Learn pattern
    user_profile.learn_pattern(
        'work_stress_pattern',
        'User shows stress when discussing work deadlines'
    )
    
    print(f"\nLearned patterns:")
    for pattern in user_profile.emotional_patterns:
        print(f"  - {pattern.pattern_type}: {pattern.description} (seen {pattern.frequency}x)")
    
    print("\n" + "=" * 70)
    print("✓ Demo complete!")
    print("=" * 70)


if __name__ == '__main__':
    asyncio.run(main())
