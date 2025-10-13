from .base_recognizer import BaseSpeechRecognizer

__all__ = [
    "BaseSpeechRecognizer",
]

# Lazy attribute access to avoid importing heavy modules at package import time
# This keeps pytest collection and IDE indexing fast.
def __getattr__(name: str):
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
        from .chirp_recognizer import ChirpRecognizer
        return ChirpRecognizer
    raise AttributeError(f"module 'speech' has no attribute {name!r}")
