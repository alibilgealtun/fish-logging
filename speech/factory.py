"""Factory for creating speech recognizer instances."""
from __future__ import annotations

from typing import Callable

from speech import BaseSpeechRecognizer


def create_recognizer(engine: str, numbers_only: bool = False) -> BaseSpeechRecognizer:
    """Factory for a speech recognizer instance.

    Parameters:
        engine: One of {whisper, whisperx, vosk, google}.
        numbers_only: Passed to recognizers that support it (e.g., Google).

    Returns:
        An instance implementing BaseSpeechRecognizer.
    """
    eng = (engine or "").lower().strip()

    def _make_whisper():
        from speech import WhisperRecognizer  # type: ignore
        return WhisperRecognizer()

    def _make_whisperx():
        from speech import WhisperXRecognizer  # type: ignore
        return WhisperXRecognizer()

    def _make_vosk():
        from speech import VoskRecognizer  # type: ignore
        return VoskRecognizer()

    def _make_google():
        try:
            from speech import GoogleSpeechRecognizer  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "GoogleSpeechRecognizer not available. Install google-cloud-speech and retry."
            ) from exc
        return GoogleSpeechRecognizer(numbers_only=numbers_only)  # type: ignore[call-arg]

    registry: dict[str, Callable[[], BaseSpeechRecognizer]] = {
        "whisper": _make_whisper,
        "whisperx": _make_whisperx,
        "vosk": _make_vosk,
        "google": _make_google,
    }

    try:
        factory = registry[eng]
    except KeyError:
        raise ValueError(f"Unknown speech recognition engine: {engine}")
    return factory()
