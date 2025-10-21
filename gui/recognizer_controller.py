"""
Speech Recognizer Controller for Fish Logging Application.

This module contains the RecognizerController class which manages the lifecycle
and operations of speech recognition services. It provides a clean abstraction
layer for speech recognizer management.

Classes:
    SpeechRecognizer: Protocol defining speech recognizer interface
    RecognizerController: Controls speech recognizer lifecycle and operations

Architecture:
    - Protocol-based design for flexibility
    - Single Responsibility Principle
    - Robust error handling and recovery
    - Clean API for recognizer management
"""
from __future__ import annotations

from typing import Protocol, Optional

# Optional logger: prefer loguru if available, else fallback to stdlib logging
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class SpeechRecognizer(Protocol):
    """
    Protocol defining the interface for speech recognizers.

    This protocol ensures that any speech recognizer implementation
    provides the necessary methods for lifecycle management and configuration.

    Design Benefits:
        - Type safety with static analysis
        - Duck typing compatibility
        - Clear contract definition
        - Easy testing with mock objects
    """

    def isRunning(self) -> bool:
        """
        Check if the recognizer is currently running.

        Returns:
            bool: True if recognizer is active, False otherwise
        """
        ...

    def begin(self) -> None:
        """
        Begin recognition process (if supported by implementation).

        Note:
            Not all recognizers support this method. Some use start() directly.
        """
        ...

    def start(self) -> None:
        """
        Start the speech recognition process.

        Raises:
            Exception: If recognition cannot be started
        """
        ...

    def stop(self) -> None:
        """
        Stop the speech recognition process.

        Note:
            Should be safe to call even if recognizer is not running.
        """
        ...

    def set_last_species(self, species: str) -> None:
        """
        Set context for the last detected species.

        Args:
            species: Name of the species to set as context

        Note:
            Optional feature for context-aware recognition.
        """
        ...

    def set_noise_profile(self, profile: str) -> None:
        """
        Configure the noise profile for recognition.

        Args:
            profile: Identifier for the noise profile to use

        Note:
            Used for environment-specific noise cancellation.
        """
        ...


class RecognizerController:
    """
    Controls speech recognizer lifecycle and operations.

    This class encapsulates all speech recognizer management logic,
    providing a clean interface for starting, stopping, and configuring
    speech recognition services while following the Single Responsibility Principle.

    Key Responsibilities:
        - Manage recognizer lifecycle (start/stop/restart)
        - Provide robust error handling and recovery
        - Abstract recognizer implementation details
        - Ensure thread-safe operations
        - Handle different recognizer types gracefully

    Design Patterns:
        - Facade pattern: Simplifies recognizer interaction
        - Strategy pattern: Works with different recognizer implementations
        - Defensive programming: Handles errors gracefully

    Attributes:
        recognizer: The speech recognizer instance being controlled
    """

    def __init__(self, recognizer: SpeechRecognizer) -> None:
        """
        Initialize the recognizer controller.

        Args:
            recognizer: The speech recognizer instance to control

        Design:
            Uses dependency injection to work with any recognizer
            implementation that follows the SpeechRecognizer protocol.
        """
        self.recognizer = recognizer
        logger.debug(f"Initialized recognizer controller with {type(recognizer).__name__}")

    def start(self) -> bool:
        """
        Start the speech recognizer if not already running.

        Handles different recognizer implementations by checking for the
        existence of 'begin' method and falling back to 'start' method.

        Returns:
            bool: True if started successfully, False otherwise

        Implementation Details:
            - Checks if already running to avoid conflicts
            - Handles different recognizer startup methods
            - Provides comprehensive error handling
            - Logs operations for debugging

        Error Handling:
            - Catches all exceptions during startup
            - Logs detailed error information
            - Returns failure status instead of raising
        """
        if self.is_running():
            logger.debug("Recognizer already running, skipping start")
            return True

        try:
            logger.info("Starting speech recognizer")

            # Handle different recognizer implementations
            if hasattr(self.recognizer, "begin"):
                # Some recognizers use 'begin' method
                logger.debug("Using 'begin' method for startup")
                self.recognizer.begin()
            else:
                # Standard recognizers use 'start' method
                logger.debug("Using 'start' method for startup")
                self.recognizer.start()

            logger.info("Speech recognizer started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start recognizer: {e}", exc_info=True)
            return False

    def stop(self) -> bool:
        """
        Stop the speech recognizer safely.

        Provides a safe way to stop the recognizer with comprehensive
        error handling and logging.

        Returns:
            bool: True if stopped successfully, False otherwise

        Implementation Details:
            - Safe to call even if recognizer is not running
            - Handles all exceptions gracefully
            - Provides detailed logging
            - Always attempts to stop (no pre-check)

        Design Philosophy:
            "Always try to stop" - better to attempt and fail than to
            leave resources hanging due to state inconsistencies.
        """
        try:
            logger.info("Stopping speech recognizer")
            self.recognizer.stop()
            logger.info("Speech recognizer stopped successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to stop recognizer: {e}", exc_info=True)
            return False

    def restart(self) -> bool:
        """
        Restart the speech recognizer (stop then start).

        Provides a convenient way to restart the recognizer, useful for
        recovering from errors or applying configuration changes.

        Returns:
            bool: True if restarted successfully, False otherwise

        Implementation:
            - Always attempts to stop first (ignores stop failures)
            - Only reports success if start succeeds
            - Provides comprehensive logging

        Use Cases:
            - Error recovery
            - Configuration changes
            - Manual user restart
        """
        logger.info("Restarting speech recognizer")

        # Always attempt to stop, but don't fail restart if stop fails
        stop_success = self.stop()
        if not stop_success:
            logger.warning("Stop failed during restart, continuing with start")

        # Restart is only successful if start succeeds
        start_success = self.start()

        if start_success:
            logger.info("Speech recognizer restarted successfully")
        else:
            logger.error("Failed to restart speech recognizer")

        return start_success

    def is_running(self) -> bool:
        """
        Check if the speech recognizer is currently running.

        Provides a safe way to check recognizer status with error handling.

        Returns:
            bool: True if running, False if stopped or error occurred

        Error Handling:
            - Returns False if any exception occurs
            - Logs errors for debugging
            - Defensive programming approach
        """
        try:
            running = self.recognizer.isRunning()
            logger.debug(f"Recognizer running status: {running}")
            return running

        except Exception as e:
            logger.error(f"Failed to check recognizer status: {e}")
            return False  # Assume stopped if we can't determine status

    def ensure_started(self) -> bool:
        """
        Ensure the recognizer is running, starting it if necessary.

        Convenience method that guarantees the recognizer is in a running
        state, starting it only if needed.

        Returns:
            bool: True if recognizer is running after operation, False otherwise

        Use Cases:
            - Application startup
            - Recovery from errors
            - Lazy initialization
        """
        if self.is_running():
            logger.debug("Recognizer already running")
            return True

        logger.info("Recognizer not running, starting now")
        return self.start()

    def ensure_stopped(self) -> bool:
        """
        Ensure the recognizer is stopped, stopping it if necessary.

        Convenience method that guarantees the recognizer is in a stopped
        state, stopping it only if needed.

        Returns:
            bool: True if recognizer is stopped after operation, False otherwise

        Use Cases:
            - Application shutdown
            - Resource conservation
            - Configuration changes
        """
        if not self.is_running():
            logger.debug("Recognizer already stopped")
            return True

        logger.info("Recognizer running, stopping now")
        return self.stop()

    def configure_species_context(self, species: str) -> bool:
        """
        Configure species context for improved recognition accuracy.

        Args:
            species: Name of the species to set as context

        Returns:
            bool: True if configuration successful, False otherwise

        Features:
            - Graceful handling if recognizer doesn't support this feature
            - Comprehensive error handling
            - Debug logging
        """
        try:
            if hasattr(self.recognizer, 'set_last_species'):
                logger.debug(f"Setting species context to: {species}")
                self.recognizer.set_last_species(species)
                return True
            else:
                logger.debug("Recognizer does not support species context")
                return True  # Not an error, just not supported

        except Exception as e:
            logger.error(f"Failed to set species context: {e}")
            return False

    def configure_noise_profile(self, profile: str) -> bool:
        """
        Configure noise profile for environment-specific recognition.

        Args:
            profile: Identifier for the noise profile to use

        Returns:
            bool: True if configuration successful, False otherwise

        Features:
            - Support for different noise environments
            - Graceful handling if not supported
            - Error recovery
        """
        try:
            if hasattr(self.recognizer, 'set_noise_profile'):
                logger.debug(f"Setting noise profile to: {profile}")
                self.recognizer.set_noise_profile(profile)
                return True
            else:
                logger.debug("Recognizer does not support noise profiles")
                return True  # Not an error, just not supported

        except Exception as e:
            logger.error(f"Failed to set noise profile: {e}")
            return False

    def get_status_info(self) -> dict:
        """
        Get comprehensive status information about the recognizer.

        Returns:
            dict: Status information including running state and capabilities

        Information Provided:
            - Running status
            - Supported features
            - Recognizer type
        """
        try:
            info = {
                'running': self.is_running(),
                'type': type(self.recognizer).__name__,
                'supports_begin': hasattr(self.recognizer, 'begin'),
                'supports_species_context': hasattr(self.recognizer, 'set_last_species'),
                'supports_noise_profiles': hasattr(self.recognizer, 'set_noise_profile'),
            }

            logger.debug(f"Recognizer status: {info}")
            return info

        except Exception as e:
            logger.error(f"Failed to get status info: {e}")
            return {
                'running': False,
                'type': 'unknown',
                'error': str(e)
            }
