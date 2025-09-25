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
    raise AttributeError(f"module 'speech' has no attribute {name!r}")
