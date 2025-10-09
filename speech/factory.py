"""Factory for creating speech recognizer instances."""
from __future__ import annotations

from typing import Callable, Optional

from speech import BaseSpeechRecognizer


def create_recognizer(engine: str, numbers_only: bool = False, noise_profile: Optional[str] = None) -> BaseSpeechRecognizer:
    """Factory for a speech recognizer instance.

    Parameters:
        engine: One of {whisper, whisperx, vosk, google}.
        numbers_only: Passed to recognizers that support it (e.g., Google).
        noise_profile: Optional noise profile name (clean|human|engine|mixed) applied to NoiseController.

    Returns:
        An instance implementing BaseSpeechRecognizer.
    """
    eng = (engine or "").lower().strip()

    def _make_whisper():
        from speech import WhisperRecognizer  # type: ignore
        return WhisperRecognizer(noise_profile=noise_profile)

    def _make_whisperx():
        from speech import WhisperXRecognizer  # type: ignore
        return WhisperXRecognizer(noise_profile=noise_profile)

    def _make_vosk():
        from speech import VoskRecognizer  # type: ignore
        return VoskRecognizer(noise_profile=noise_profile)

    def _make_google():
        try:
            from speech import GoogleSpeechRecognizer  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "GoogleSpeechRecognizer not available. Install google-cloud-speech and retry."
            ) from exc
        return GoogleSpeechRecognizer(numbers_only=numbers_only, noise_profile=noise_profile)  # type: ignore[call-arg]

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
