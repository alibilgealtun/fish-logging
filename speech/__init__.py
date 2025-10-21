"""Speech recognition package initialization with lazy loading.

This package provides various speech recognition implementations optimized for
fish species and measurement recognition in marine environments. It uses lazy
loading to avoid importing heavy dependencies during package import time,
which keeps pytest collection and IDE indexing fast.

Available recognizers:
- WhisperRecognizer: CPU-only offline recognition using faster-whisper
- VoskRecognizer: Offline recognition with vocabulary constraints
- AssemblyAIRecognizer: Cloud-based recognition with word boosting
- GoogleSpeechRecognizer: Google Cloud Speech-to-Text integration
- GeminiRecognizer: Google Gemini AI recognition
- Wav2Vec2Recognizer: Facebook's Wav2Vec2 model recognition
- WhisperXRecognizer: Enhanced Whisper with speaker diarization

All recognizers inherit from BaseSpeechRecognizer and provide consistent
PyQt signal interfaces for integration with the application's GUI.
"""
from .base_recognizer import BaseSpeechRecognizer

__all__ = [
    "BaseSpeechRecognizer",
]

# Lazy attribute access to avoid importing heavy modules at package import time
# This keeps pytest collection and IDE indexing fast while allowing dynamic
# imports when specific recognizers are needed.
def __getattr__(name: str):
    """Dynamically import speech recognizer classes on first access.

    This lazy loading pattern ensures that heavy dependencies like torch,
    transformers, vosk, etc. are only loaded when actually needed, rather
    than at package import time.

    Args:
        name: Name of the recognizer class to import

    Returns:
        The requested recognizer class

    Raises:
        AttributeError: If the requested recognizer is not available
    """
    if name == "WhisperRecognizer":
        from .faster_whisper_recognizer import WhisperRecognizer
        return WhisperRecognizer
    if name == "VoskRecognizer":
        from .vosk_recognizer import VoskRecognizer
        return VoskRecognizer
    if name == "WhisperXRecognizer":
        from .whisperx_recognizer import WhisperXRecognizer
        return WhisperXRecognizer
    if name == "GoogleSpeechRecognizer":
        from .google_speech_recognizer import GoogleSpeechRecognizer
        return GoogleSpeechRecognizer
    if name == "AssemblyAIRecognizer":
        from .assemblyai_recognizer import AssemblyAIRecognizer
        return AssemblyAIRecognizer
    if name == "GeminiRecognizer":
        from .gemini_recognizer import GeminiRecognizer
        return GeminiRecognizer
    if name == "ChirpRecognizer":
        # Legacy alias or alternative recognizer
        from .chirp_recognizer import ChirpRecognizer
        return ChirpRecognizer
    if name == "Wav2Vec2Recognizer":
        from .wav2vec2_recognizer import Wav2Vec2Recognizer
        return Wav2Vec2Recognizer

    # Raise AttributeError for unknown recognizers
    raise AttributeError(f"module 'speech' has no attribute {name!r}")
