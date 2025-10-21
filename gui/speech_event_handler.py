"""
Speech Event Handler for Fish Logging Application.

This module contains the SpeechEventHandler class which serves as a mediator between
the speech recognition system and the business logic layer. It implements the
Observer pattern and follows Clean Architecture principles.

Classes:
    SpeechEventHandler: Coordinates speech recognition events with business logic

Architecture:
    - Decouples GUI from speech recognition logic
    - Implements callback-based communication pattern
    - Delegates business operations to use cases
    - Provides clean separation of concerns
"""
from __future__ import annotations

from typing import Callable, Optional

# Optional logger: prefer loguru if available, else fallback to stdlib logging
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from app.use_cases import (
    ProcessFinalTextUseCase,
    LogFishEntryUseCase,
    CancelLastEntryUseCase,
)
from core.error_handler import handle_exceptions


class SpeechEventHandler:
    """
    Handles speech recognition events and coordinates business logic.

    This class acts as a mediator between speech recognition events and the
    application's business logic, implementing the Single Responsibility Principle
    and maintaining clean separation of concerns.

    Key Responsibilities:
        - Process partial speech recognition results for real-time feedback
        - Handle final speech recognition results and trigger business logic
        - Coordinate fish entry logging workflow
        - Manage entry cancellation operations
        - Provide callback-based communication with UI components

    Architecture:
        - Uses dependency injection for use cases
        - Implements callback pattern for UI communication
        - Delegates business logic to dedicated use cases
        - Provides robust error handling

    Attributes:
        process_text: Use case for processing speech text
        log_entry: Use case for logging fish entries
        cancel_entry: Use case for cancelling entries
        on_*: Callback functions for UI communication
    """

    def __init__(
        self,
        process_text_use_case: ProcessFinalTextUseCase,
        log_entry_use_case: LogFishEntryUseCase,
        cancel_entry_use_case: CancelLastEntryUseCase,
    ) -> None:
        """
        Initialize the speech event handler with required use cases.

        Args:
            process_text_use_case: Use case for processing speech text into fish data
            log_entry_use_case: Use case for logging fish entries to storage
            cancel_entry_use_case: Use case for cancelling previous entries

        Design:
            Uses dependency injection to maintain loose coupling and
            enable easy testing and modification of business logic.
        """
        # Business logic dependencies (Clean Architecture)
        self.process_text = process_text_use_case
        self.log_entry = log_entry_use_case
        self.cancel_entry = cancel_entry_use_case

        # UI callback functions (Observer pattern)
        # These will be set by the GUI layer to handle presentation updates
        self.on_entry_logged: Optional[Callable] = None
        self.on_entry_cancelled: Optional[Callable] = None
        self.on_cancel_failed: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        self.on_partial_update: Optional[Callable] = None
        self.on_species_detected: Optional[Callable] = None

    def handle_partial_text(self, text: str) -> None:
        """
        Handle partial text from speech recognition for real-time feedback.

        Processes incomplete speech recognition results to provide immediate
        visual feedback to users, enhancing the user experience with live
        transcription display.

        Args:
            text: Partial recognition text from speech engine

        Features:
            - Real-time transcription updates
            - Non-blocking operation
            - Optional callback execution
            - Debug logging for troubleshooting
        """
        logger.debug(f"[GUI partial] {text}")

        # Notify UI layer if callback is available
        if self.on_partial_update:
            self.on_partial_update(text)

    @handle_exceptions(message="Error handling final text")
    def handle_final_text(
        self,
        text: str,
        confidence: float,
        boat_name: str,
        station_id: str
    ) -> None:
        """
        Handle complete speech recognition results and trigger business workflow.

        This is the main entry point for processing complete speech recognition
        results. It orchestrates the entire fish logging workflow from text
        processing to data persistence.

        Args:
            text: Final recognized speech text
            confidence: Recognition confidence score (0.0-1.0)
            boat_name: Current boat name from settings
            station_id: Current station ID from settings

        Workflow:
            1. Process speech text to extract fish data
            2. Validate extracted information
            3. Log entry to persistent storage
            4. Notify UI of success/failure
            5. Handle any errors gracefully

        Error Handling:
            - Comprehensive exception catching
            - User-friendly error messages
            - Graceful degradation
            - Detailed logging for debugging
        """
        try:
            logger.info(f"Processing final text: '{text}' (confidence: {confidence:.2f})")

            # Step 1: Process speech text to extract fish data
            result = self.process_text.execute(text)

            if not result.is_success:
                # Text processing failed - notify user
                error_msg = result.error_message or "Failed to parse fish information from speech"
                logger.warning(f"Text processing failed: {error_msg}")

                if self.on_error:
                    self.on_error(error_msg)
                return

            # Step 2: Extract validated fish data
            fish_data = result.value

            # Step 3: Log the fish entry with context information
            log_result = self.log_entry.execute(
                species=fish_data.species,
                length_cm=fish_data.length_cm,
                confidence=confidence,
                boat_name=boat_name,
                station_id=station_id,
                raw_text=text
            )

            if log_result.is_success:
                # Step 4a: Notify UI of successful logging
                logger.info(f"Successfully logged: {fish_data.species} ({fish_data.length_cm:.1f}cm)")

                if self.on_entry_logged:
                    self.on_entry_logged(
                        species=fish_data.species,
                        length=fish_data.length_cm,
                        confidence=confidence,
                        boat=boat_name,
                        station=station_id
                    )

                # Notify UI of species detection for context
                if self.on_species_detected and fish_data.species:
                    self.on_species_detected(fish_data.species)

            else:
                # Step 4b: Handle logging failure
                error_msg = log_result.error_message or "Failed to save fish entry"
                logger.error(f"Logging failed: {error_msg}")

                if self.on_error:
                    self.on_error(error_msg)

        except Exception as e:
            # Step 5: Handle unexpected errors
            error_msg = f"Unexpected error processing speech: {str(e)}"
            logger.error(error_msg, exc_info=True)

            if self.on_error:
                self.on_error(error_msg)

    @handle_exceptions(message="Error handling cancellation")
    def handle_cancellation_request(self) -> None:
        """
        Handle requests to cancel the last logged fish entry.

        Provides users with the ability to undo their last entry, useful for
        correcting mistakes or handling misrecognized speech.

        Features:
            - Safe cancellation with validation
            - User feedback on success/failure
            - Maintains data integrity
            - Comprehensive error handling

        Error Handling:
            - Validates cancellation is possible
            - Provides meaningful error messages
            - Maintains system stability
        """
        try:
            logger.info("Processing cancellation request")

            # Attempt to cancel the last entry
            result = self.cancel_entry.execute()

            if result.is_success:
                # Notify UI of successful cancellation
                logger.info("Successfully cancelled last entry")

                if self.on_entry_cancelled:
                    self.on_entry_cancelled()
            else:
                # Handle cancellation failure
                error_msg = result.error_message or "Failed to cancel last entry"
                logger.warning(f"Cancellation failed: {error_msg}")

                if self.on_cancel_failed:
                    self.on_cancel_failed(error_msg)

        except Exception as e:
            # Handle unexpected errors
            error_msg = f"Unexpected error during cancellation: {str(e)}"
            logger.error(error_msg, exc_info=True)

            if self.on_cancel_failed:
                self.on_cancel_failed(error_msg)

    def set_callbacks(
        self,
        on_entry_logged: Optional[Callable] = None,
        on_entry_cancelled: Optional[Callable] = None,
        on_cancel_failed: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        on_partial_update: Optional[Callable] = None,
        on_species_detected: Optional[Callable] = None,
    ) -> None:
        """
        Set callback functions for UI communication.

        Provides a convenient way to configure all callback functions at once,
        following the Builder pattern for optional configuration.

        Args:
            on_entry_logged: Called when fish entry is successfully logged
            on_entry_cancelled: Called when entry is successfully cancelled
            on_cancel_failed: Called when cancellation fails
            on_error: Called when any error occurs
            on_partial_update: Called for real-time transcription updates
            on_species_detected: Called when species is detected in speech

        Design:
            - Optional parameters for flexible configuration
            - None values preserved existing callbacks
            - Enables fluent configuration style
        """
        if on_entry_logged is not None:
            self.on_entry_logged = on_entry_logged
        if on_entry_cancelled is not None:
            self.on_entry_cancelled = on_entry_cancelled
        if on_cancel_failed is not None:
            self.on_cancel_failed = on_cancel_failed
        if on_error is not None:
            self.on_error = on_error
        if on_partial_update is not None:
            self.on_partial_update = on_partial_update
        if on_species_detected is not None:
            self.on_species_detected = on_species_detected

    def clear_callbacks(self) -> None:
        """
        Clear all callback functions.

        Useful for cleanup operations or when switching between different
        UI contexts. Prevents memory leaks and stale references.
        """
        self.on_entry_logged = None
        self.on_entry_cancelled = None
        self.on_cancel_failed = None
        self.on_error = None
        self.on_partial_update = None
        self.on_species_detected = None

    def is_configured(self) -> bool:
        """
        Check if the handler has essential callbacks configured.

        Returns:
            bool: True if essential callbacks are set, False otherwise

        Essential callbacks for basic functionality:
            - on_error: For error reporting
            - on_entry_logged: For success feedback
        """
        return self.on_error is not None and self.on_entry_logged is not None
