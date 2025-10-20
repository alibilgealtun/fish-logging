"""Speech event handler for managing speech recognition events."""
from __future__ import annotations

from typing import Callable, Optional

from loguru import logger

from app.use_cases import (
    ProcessFinalTextUseCase,
    LogFishEntryUseCase,
    CancelLastEntryUseCase,
)
from core.error_handler import handle_exceptions


class SpeechEventHandler:
    """Handles speech recognition events and coordinates business logic.

    This class decouples the GUI from the speech recognition logic,
    implementing the Single Responsibility Principle.
    """

    def __init__(
        self,
        process_text_use_case: ProcessFinalTextUseCase,
        log_entry_use_case: LogFishEntryUseCase,
        cancel_entry_use_case: CancelLastEntryUseCase,
    ):
        self.process_text = process_text_use_case
        self.log_entry = log_entry_use_case
        self.cancel_entry = cancel_entry_use_case

        # Callbacks to be set by the GUI
        self.on_entry_logged: Optional[Callable] = None
        self.on_entry_cancelled: Optional[Callable] = None
        self.on_cancel_failed: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        self.on_partial_update: Optional[Callable] = None
        self.on_species_detected: Optional[Callable] = None

    def handle_partial_text(self, text: str) -> None:
        """Handle partial text from speech recognition.

        Args:
            text: Partial recognition text
        """
        logger.debug(f"[GUI partial] {text}")
        if self.on_partial_update:
            self.on_partial_update(text)

    @handle_exceptions(message="Error handling final text")
    def handle_final_text(
        self,
        text: str,
        confidence: float,
        boat: str = "",
        station_id: str = ""
    ) -> None:
        """Handle final text from speech recognition.

        Args:
            text: Final recognition text
            confidence: Recognition confidence score
            boat: Boat name from settings
            station_id: Station ID from settings
        """
        logger.info(f"[GUI final] conf={confidence:.2f} text={text}")

        # Update partial text display
        if self.on_partial_update:
            self.on_partial_update(text)

        # Process the text
        result = self.process_text.execute(text, confidence)

        if result.is_failure():
            if self.on_error:
                self.on_error(f"Failed to process: {result.error}")
            return

        parsed = result.unwrap()

        # Handle cancel command
        if parsed.cancel:
            self._handle_cancel()
            return

        # Handle species detection
        if parsed.species and self.on_species_detected:
            self.on_species_detected(parsed.species)

        # Log if we have complete data
        if parsed.species and parsed.length_cm is not None:
            self._handle_complete_entry(
                parsed.species,
                parsed.length_cm,
                confidence,
                boat,
                station_id
            )

    def _handle_cancel(self) -> None:
        """Handle cancel command."""
        result = self.cancel_entry.execute()

        if result.is_success() and result.unwrap():
            logger.info("Entry cancelled successfully")
            if self.on_entry_cancelled:
                self.on_entry_cancelled()
        else:
            logger.info("Nothing to cancel")
            if self.on_cancel_failed:
                self.on_cancel_failed()

    def _handle_complete_entry(
        self,
        species: str,
        length_cm: float,
        confidence: float,
        boat: str,
        station_id: str
    ) -> None:
        """Handle logging a complete fish entry.

        Args:
            species: Fish species
            length_cm: Length in centimeters
            confidence: Recognition confidence
            boat: Boat name
            station_id: Station ID
        """
        result = self.log_entry.execute(
            species=species,
            length_cm=length_cm,
            confidence=confidence,
            boat=boat,
            station_id=station_id
        )

        if result.is_success():
            entry = result.unwrap()
            logger.info(f"Entry logged: {species} {length_cm}cm")
            if self.on_entry_logged:
                self.on_entry_logged(entry)
        else:
            logger.error(f"Failed to log entry: {result.error}")
            if self.on_error:
                self.on_error(f"Failed to log entry: {result.error}")

    def handle_error(self, error_message: str) -> None:
        """Handle speech recognition errors.

        Args:
            error_message: Error message from recognizer
        """
        logger.error(f"Speech recognition error: {error_message}")
        if self.on_error:
            self.on_error(error_message)

    def handle_status_change(self, status: str) -> None:
        """Handle status changes from speech recognizer.

        Args:
            status: Status message
        """
        logger.debug(f"Speech recognizer status: {status}")

