"""Unified audio processing pipeline.

Combines VAD, STT, prosody analysis, and event detection
into a single async pipeline for real-time processing.
"""

import asyncio
import numpy as np
from typing import Dict, Optional, List
from loguru import logger
import time

from .vad import SileroVAD, MockVAD
from .stt import SenseVoiceSTT, MockSTT
from .prosody import ProsodyAnalyzer, MockProsodyAnalyzer
from .event_detector import AudioEventDetector, MockEventDetector
from utils.helpers import timeit, LatencyTracker


class AudioPipeline:
    """
    Processes audio through all audio models in parallel.
    
    Architecture:
    - VAD detects speech boundaries
    - STT transcribes speech segments
    - Prosody analyzes emotional tone
    - Event detector finds non-speech cues (sighs, breathing)
    """
    
    def __init__(
        self,
        config: Dict,
        use_mock: bool = False
    ):
        """
        Initialize the audio pipeline.
        
        Args:
            config: Configuration dictionary
            use_mock: Use mock models for testing
        """
        self.config = config
        self.latency_tracker = LatencyTracker()
        self.sample_rate = 16000
        
        # Speech buffer for STT
        self.speech_buffer: List[np.ndarray] = []
        self.is_buffering = False
        
        
        # Initialize models
        if use_mock:
            logger.info("Using mock audio models (use_mock=True)")
            self.vad = MockVAD()
            self.stt = MockSTT()
            self.prosody = MockProsodyAnalyzer()
            self.event_detector = MockEventDetector()
        else:
            logger.info("Attempting to load real audio models...")
            
            # Silero VAD (lightweight, always try)
            vad_loaded = False
            try:
                self.vad = SileroVAD(
                    model_path=config.get('models.audio.silero_vad.model_path'),
                    threshold=config.get('models.audio.silero_vad.threshold', 0.5),
                    min_speech_duration_ms=config.get('models.audio.silero_vad.min_speech_duration_ms', 250),
                    min_silence_duration_ms=config.get('models.audio.silero_vad.min_silence_duration_ms', 500),
                    sample_rate=self.sample_rate
                )
                vad_loaded = True
                logger.info("✓ Silero VAD loaded")
            except Exception as e:
                logger.warning(f"Silero VAD failed: {e}")
            
            # Whisper STT (try Whisper first, fallback to SenseVoice)
            stt_loaded = False
            try:
                from models.audio.whisper_stt import WhisperSTT
                self.stt = WhisperSTT(
                    model_size=config.get('models.audio.whisper.model_size', 'base'),
                    device='cpu'  # Always use CPU for compatibility
                )
                stt_loaded = True
                logger.info("✓ Whisper STT loaded")
            except Exception as e:
                logger.warning(f"Whisper failed: {e}")
                logger.warning("Trying SenseVoice fallback...")
                try:
                    self.stt = SenseVoiceSTT(
                        model_name=config.get('models.audio.sensevoice.model_name', 'iic/SenseVoiceSmall'),
                        device='cpu',
                        language=config.get('models.audio.sensevoice.language', 'auto'),
                        use_int8=config.get('models.audio.sensevoice.use_int8', True)
                    )
                    stt_loaded = True
                    logger.info("✓ SenseVoice STT loaded")
                except Exception as e2:
                    logger.error(f"SenseVoice also failed: {e2}")
            
            # Wav2Vec2 Prosody
            prosody_loaded = False
            try:
                self.prosody = ProsodyAnalyzer(
                    model_name=config.get('models.audio.wav2vec2.model_name', 
                                         'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'),
                    device='cpu',  # Use CPU for compatibility
                    sample_rate=self.sample_rate
                )
                prosody_loaded = True
                logger.info("✓ Wav2Vec2 prosody analyzer loaded")
            except Exception as e:
                logger.warning(f"Prosody analyzer failed: {e}")
            
            # Event Detector
            event_loaded = False
            try:
                self.event_detector = AudioEventDetector(
                    sample_rate=self.sample_rate,
                    silence_threshold_db=-40.0,
                    min_silence_duration=2.0
                )
                event_loaded = True
                logger.info("✓ Audio event detector loaded")
            except Exception as e:
                logger.warning(f"Event detector failed: {e}")
            
            # Fallback to mocks for any failed components
            if not vad_loaded:
                logger.warning("Using MockVAD")
                self.vad = MockVAD()
            
            if not stt_loaded:
                logger.warning("Using MockSTT")
                self.stt = MockSTT()
            
            if not prosody_loaded:
                logger.warning("Using MockProsodyAnalyzer")
                self.prosody = MockProsodyAnalyzer()
            
            if not event_loaded:
                logger.warning("Using MockEventDetector")
                self.event_detector = MockEventDetector()
            
            if vad_loaded and stt_loaded and prosody_loaded and event_loaded:
                logger.info("✓ All real audio models loaded successfully!")
            else:
                logger.warning("⚠ Some audio models using mocks")
        
        
        self.packet_count = 0
    
    @timeit
    async def process_audio(
        self, 
        audio_chunk: np.ndarray, 
        timestamp: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Process a single audio chunk through all audio models.
        
        Args:
            audio_chunk: Audio samples (float32, normalized, mono)
            timestamp: Audio timestamp in seconds
            
        Returns:
            Dictionary containing:
            - vad: Voice activity detection result
            - transcription: Speech-to-text result (if speech ended)
            - prosody: Paralinguistic features
            - events: Audio events (sighs, breathing, silence)
            - audio_state: Aggregated audio emotional state
            - processing_time_ms: Total processing time
        """
        if timestamp is None:
            timestamp = time.time()
        
        start_time = time.perf_counter()
        self.packet_count += 1
        
        try:
            # 1. VAD (always runs first)
            vad_result = await self._run_vad(audio_chunk)
            is_speech = vad_result['is_speech']
            vad_event = vad_result['event']
            
            # 2. Handle speech buffering for STT
            transcription_result = None
            if vad_event == 'start':
                # Start buffering
                self.is_buffering = True
                self.speech_buffer = [audio_chunk]
            elif vad_event == 'continue' and self.is_buffering:
                # Continue buffering
                self.speech_buffer.append(audio_chunk)
            elif vad_event == 'end' and self.is_buffering:
                # End buffering, transcribe
                self.speech_buffer.append(audio_chunk)
                full_speech = np.concatenate(self.speech_buffer)
                transcription_result = await self._run_stt(full_speech)
                self.speech_buffer.clear()
                self.is_buffering = False
            
            # 3. Run prosody and events in parallel
            prosody_task = asyncio.create_task(
                self._run_prosody(audio_chunk)
            )
            events_task = asyncio.create_task(
                self._run_event_detection(audio_chunk, timestamp, is_speech)
            )
            
            prosody_result = await prosody_task
            events_result = await events_task
            
            # 4. Aggregate results
            audio_state = self._aggregate_results(
                vad_result,
                transcription_result,
                prosody_result,
                events_result
            )
            
            # Track latency
            processing_time = (time.perf_counter() - start_time) * 1000
            self.latency_tracker.record('audio_pipeline', processing_time)
            
            return {
                'vad': vad_result,
                'transcription': transcription_result,
                'prosody': prosody_result,
                'events': events_result,
                'audio_state': audio_state,
                'processing_time_ms': processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in audio pipeline: {e}")
            return self._empty_result()
    
    async def _run_vad(self, audio: np.ndarray) -> Dict:
        """Run VAD (async wrapper)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.vad.detect, audio)
    
    async def _run_stt(self, audio: np.ndarray) -> Dict:
        """Run STT (async wrapper)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.stt.transcribe, audio, self.sample_rate)
    
    async def _run_prosody(self, audio: np.ndarray) -> Dict:
        """Run prosody analysis (async wrapper)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.prosody.analyze, audio)
    
    async def _run_event_detection(self, audio: np.ndarray, timestamp: float, is_speech: bool) -> Dict:
        """Run event detection (async wrapper)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.event_detector.detect, audio, timestamp, is_speech)
    
    def _aggregate_results(
        self,
        vad: Dict,
        transcription: Optional[Dict],
        prosody: Dict,
        events: Dict
    ) -> Dict:
        """
        Aggregate audio results into unified emotional state.
        
        Combines insights from speech content, prosody, and events.
        """
        # Extract key signals
        is_speaking = vad.get('is_speaking', False)
        transcribed_text = transcription.get('text', '') if transcription else ''
        
        arousal = prosody.get('arousal', 0.5)
        valence = prosody.get('valence', 0.0)
        tremor = prosody.get('tremor', 0.0)
        
        detected_events = events.get('events', [])
        is_silence = events.get('is_silence', False)
        silence_duration = events.get('silence_duration', 0.0)
        
        # Adjust based on events
        event_types = [e.event_type for e in detected_events]
        
        if 'sigh' in event_types:
            valence -= 0.2  # Sighs indicate frustration/relief
            arousal -= 0.1
        
        if 'heavy_breathing' in event_types:
            arousal += 0.3  # Stress/exertion
            valence -= 0.1
        
        if 'prolonged_silence' in event_types or silence_duration > 3.0:
            arousal -= 0.2  # Low engagement
        
        # Detect emotional masking
        emotional_masking = False
        if transcribed_text:
            # Simple heuristic: "I'm fine" with negative valence/high tremor
            if any(phrase in transcribed_text.lower() for phrase in ["i'm fine", "i'm okay", "no problem"]):
                if valence < -0.2 or tremor > 0.6:
                    emotional_masking = True
        
        # Clamp values
        valence = max(-1.0, min(1.0, valence))
        arousal = max(0.0, min(1.0, arousal))
        
        # Classify overall audio state
        audio_emotion = self._classify_audio_state(
            arousal, valence, tremor, detected_events, is_silence
        )
        
        return {
            'is_speaking': is_speaking,
            'transcribed_text': transcribed_text,
            'arousal': arousal,
            'valence': valence,
            'tremor': tremor,
            'audio_emotion': audio_emotion,
            'detected_events': event_types,
            'is_silence': is_silence,
            'silence_duration': silence_duration,
            'emotional_masking': emotional_masking
        }
    
    def _classify_audio_state(
        self,
        arousal: float,
        valence: float,
        tremor: float,
        events: List,
        is_silence: bool
    ) -> str:
        """Classify overall audio emotional state."""
        # High-priority indicators
        if tremor > 0.7:
            return 'nervous'
        
        event_types = [e.event_type for e in events]
        
        if 'sigh' in event_types:
            return 'frustrated'
        
        if 'heavy_breathing' in event_types:
            return 'stressed'
        
        if is_silence or 'prolonged_silence' in event_types:
            return 'withdrawn'
        
        # Use arousal-valence classification
        if arousal > 0.6 and valence > 0.3:
            return 'excited'
        elif arousal > 0.6 and valence < -0.3:
            return 'agitated'
        elif arousal < 0.4 and valence > 0.3:
            return 'calm_positive'
        elif arousal < 0.4 and valence < -0.3:
            return 'sad'
        elif valence > 0.3:
            return 'positive'
        elif valence < -0.3:
            return 'negative'
        else:
            return 'neutral'
    
    def _empty_result(self) -> Dict:
        """Return empty result for errors."""
        return {
            'vad': {'is_speech': False},
            'transcription': None,
            'prosody': {},
            'events': {'events': []},
            'audio_state': {
                'audio_emotion': 'unknown',
                'arousal': 0.5,
                'valence': 0.0
            },
            'processing_time_ms': 0.0
        }
    
    def get_latency_stats(self) -> Dict:
        """Get latency statistics for the pipeline."""
        return self.latency_tracker.get_stats('audio_pipeline')
    
    def reset(self):
        """Reset pipeline state."""
        self.vad.reset()
        self.event_detector.reset()
        self.speech_buffer.clear()
        self.is_buffering = False
        logger.info("AudioPipeline reset")
