"""Factory for creating speech recognizer instances with registry pattern."""
from __future__ import annotations

from typing import Callable, Optional, Dict, Any

from speech import BaseSpeechRecognizer
from core.exceptions import RecognizerError


class RecognizerRegistry:
    """Registry for speech recognizer factories following Open/Closed Principle.

    New recognizers can register themselves without modifying this class.
    """

    _factories: Dict[str, Callable[..., BaseSpeechRecognizer]] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        factory: Callable[..., BaseSpeechRecognizer],
        description: str = "",
        requires: list[str] = None
    ) -> None:
        """Register a recognizer factory.

        Args:
            name: Engine name (e.g., 'whisper', 'google')
            factory: Factory function that creates the recognizer
            description: Human-readable description
            requires: List of required packages
        """
        cls._factories[name.lower()] = factory
        cls._metadata[name.lower()] = {
            "description": description,
            "requires": requires or [],
        }

    @classmethod
    def create(
        cls,
        name: str,
        numbers_only: bool = False,
        noise_profile: Optional[str] = None,
        **kwargs
    ) -> BaseSpeechRecognizer:
        """Create a recognizer instance.

        Args:
            name: Engine name
            numbers_only: Whether to optimize for numbers only
            noise_profile: Noise profile to use
            **kwargs: Additional arguments for the recognizer

        Returns:
            Recognizer instance

        Raises:
            RecognizerError: If engine is unknown or creation fails
        """
        engine = name.lower().strip()

        if engine not in cls._factories:
            available = ", ".join(cls._factories.keys())
            raise RecognizerError(
                f"Unknown speech recognition engine: '{name}'. "
                f"Available engines: {available}"
            )

        try:
            factory = cls._factories[engine]
            return factory(numbers_only=numbers_only, noise_profile=noise_profile, **kwargs)
        except Exception as e:
            raise RecognizerError(f"Failed to create {engine} recognizer: {e}") from e

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if an engine is registered."""
        return name.lower() in cls._factories

    @classmethod
    def get_available_engines(cls) -> list[str]:
        """Get list of available engine names."""
        return list(cls._factories.keys())

    @classmethod
    def get_metadata(cls, name: str) -> Dict[str, Any]:
        """Get metadata for a registered engine."""
        return cls._metadata.get(name.lower(), {})


# Factory functions for each recognizer
def _make_whisper(numbers_only: bool = False, noise_profile: Optional[str] = None, **kwargs) -> BaseSpeechRecognizer:
    from speech import WhisperRecognizer  # type: ignore
    return WhisperRecognizer(noise_profile=noise_profile, **kwargs)


def _make_whisperx(numbers_only: bool = False, noise_profile: Optional[str] = None, **kwargs) -> BaseSpeechRecognizer:
    from speech import WhisperXRecognizer  # type: ignore
    return WhisperXRecognizer(noise_profile=noise_profile, **kwargs)


def _make_vosk(numbers_only: bool = False, noise_profile: Optional[str] = None, **kwargs) -> BaseSpeechRecognizer:
    from speech import VoskRecognizer  # type: ignore
    return VoskRecognizer(noise_profile=noise_profile, **kwargs)


def _make_google(numbers_only: bool = False, noise_profile: Optional[str] = None, **kwargs) -> BaseSpeechRecognizer:
    try:
        from speech import GoogleSpeechRecognizer  # type: ignore
    except ImportError as exc:
        raise RecognizerError(
            "GoogleSpeechRecognizer not available. Install google-cloud-speech and retry."
        ) from exc
    return GoogleSpeechRecognizer(numbers_only=numbers_only, noise_profile=noise_profile, **kwargs)


def _make_assemblyai(numbers_only: bool = False, noise_profile: Optional[str] = None, **kwargs) -> BaseSpeechRecognizer:
    try:
        from speech import AssemblyAIRecognizer  # type: ignore
    except ImportError as exc:
        raise RecognizerError(
            "AssemblyAIRecognizer not available. Install websocket-client and retry."
        ) from exc
    return AssemblyAIRecognizer(noise_profile=noise_profile, **kwargs)


def _make_gemini(numbers_only: bool = False, noise_profile: Optional[str] = None, **kwargs) -> BaseSpeechRecognizer:
    try:
        from speech import GeminiRecognizer  # type: ignore
    except ImportError as exc:
        raise RecognizerError(
            "GeminiRecognizer not available. Install google-generativeai and retry."
        ) from exc
    return GeminiRecognizer(noise_profile=noise_profile, **kwargs)


def _make_chirp(numbers_only: bool = False, noise_profile: Optional[str] = None, **kwargs) -> BaseSpeechRecognizer:
    try:
        from speech import ChirpRecognizer  # type: ignore
    except ImportError as exc:
        raise RecognizerError(
            "ChirpRecognizer not available. Install google-cloud-speech v2 and retry."
        ) from exc
    return ChirpRecognizer(noise_profile=noise_profile, **kwargs)


def _make_wav2vec2(numbers_only: bool = False, noise_profile: Optional[str] = None, **kwargs) -> BaseSpeechRecognizer:
    try:
        from speech import Wav2Vec2Recognizer  # type: ignore
    except ImportError as exc:
        raise RecognizerError(
            "Wav2Vec2Recognizer not available. Install transformers and torch and retry."
        ) from exc
    return Wav2Vec2Recognizer(noise_profile=noise_profile, **kwargs)


# Register all recognizers
RecognizerRegistry.register("whisper", _make_whisper, "OpenAI Whisper (local)", [])
RecognizerRegistry.register("whisperx", _make_whisperx, "WhisperX (enhanced)", ["whisperx"])
RecognizerRegistry.register("vosk", _make_vosk, "Vosk (offline)", ["vosk"])
RecognizerRegistry.register("google", _make_google, "Google Cloud Speech", ["google-cloud-speech"])
RecognizerRegistry.register("assemblyai", _make_assemblyai, "AssemblyAI", ["websocket-client"])
RecognizerRegistry.register("gemini", _make_gemini, "Google Gemini", ["google-generativeai"])
RecognizerRegistry.register("chirp", _make_chirp, "Google Chirp", ["google-cloud-speech"])
RecognizerRegistry.register("wav2vec2", _make_wav2vec2, "Wav2Vec2", ["transformers", "torch"])


def create_recognizer(engine: str, numbers_only: bool = False, noise_profile: Optional[str] = None) -> BaseSpeechRecognizer:
    """Factory for a speech recognizer instance (backwards compatible).

    Parameters:
        engine: One of {whisper, whisperx, vosk, google, assemblyai, gemini, chirp, wav2vec2}.
        numbers_only: Passed to recognizers that support it (e.g., Google).
        noise_profile: Optional noise profile name (clean|human|engine|mixed) applied to NoiseController.

    Returns:
        An instance implementing BaseSpeechRecognizer.
    """
    return RecognizerRegistry.create(engine, numbers_only=numbers_only, noise_profile=noise_profile)
