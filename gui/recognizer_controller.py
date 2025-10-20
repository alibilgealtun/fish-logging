"""Recognizer controller to manage speech recognizer lifecycle."""
from __future__ import annotations

from typing import Protocol, Optional

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class SpeechRecognizer(Protocol):
    """Protocol for speech recognizer."""

    def isRunning(self) -> bool:
        """Check if recognizer is running."""
        ...

    def begin(self) -> None:
        """Begin recognition (if supported)."""
        ...

    def start(self) -> None:
        """Start recognition."""
        ...

    def stop(self) -> None:
        """Stop recognition."""
        ...

    def set_last_species(self, species: str) -> None:
        """Set the last detected species."""
        ...

    def set_noise_profile(self, profile: str) -> None:
        """Set the noise profile."""
        ...


class RecognizerController:
    """Controls speech recognizer lifecycle and operations.

    This class encapsulates all recognizer management logic,
    following Single Responsibility Principle.
    """

    def __init__(self, recognizer: SpeechRecognizer):
        """Initialize the recognizer controller.

        Args:
            recognizer: The speech recognizer to control
        """
        self.recognizer = recognizer

    def start(self) -> bool:
        """Start the recognizer if not already running.

        Returns:
            True if started successfully, False otherwise
        """
        if not self.recognizer.isRunning():
            try:
                if hasattr(self.recognizer, "begin"):
                    self.recognizer.begin()
                else:
                    self.recognizer.start()
                return True
            except Exception as e:
                logger.error(f"Failed to start recognizer: {e}")
                return False
        return True  # Already running

    def stop(self) -> bool:
        """Stop the recognizer.

        Returns:
            True if stopped successfully, False otherwise
        """
        try:
            self.recognizer.stop()
            return True
        except Exception as e:
            logger.error(f"Failed to stop recognizer: {e}")
            return False

    def restart(self) -> bool:
        """Restart the recognizer.

        Returns:
            True if restarted successfully, False otherwise
        """
        self.stop()
        return self.start()

    def is_running(self) -> bool:
        """Check if recognizer is running.

        Returns:
            True if running, False otherwise
        """
        return self.recognizer.isRunning()

    def set_species(self, species: str) -> None:
        """Set the current species for the recognizer.

        Args:
            species: Species name
        """
        if hasattr(self.recognizer, 'set_last_species'):
            try:
                self.recognizer.set_last_species(species)
            except Exception as e:
                logger.warning(f"Failed to set species: {e}")

    def set_noise_profile(self, profile: str) -> None:
        """Set the noise profile for the recognizer.

        Args:
            profile: Noise profile name
        """
        if hasattr(self.recognizer, 'set_noise_profile'):
            try:
                self.recognizer.set_noise_profile(profile)
            except Exception as e:
                logger.warning(f"Failed to set noise profile: {e}")

    def ensure_started(self) -> None:
        """Ensure the recognizer is started (start if not running)."""
        if not self.is_running():
            self.start()

