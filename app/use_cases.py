"""Use cases for fish logging business logic.

Implements the use case layer following Clean Architecture principles,
encapsulating business rules and orchestrating data flow between
the presentation layer and data layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from loguru import logger

from core.exceptions import LoggingError, ParsingError
from core.result import Success, Failure, Result
from parser.parser import FishParser, ParserResult


@dataclass
class FishEntry:
    """Represents a fish logging entry.

    Attributes:
        species: Fish species name
        length_cm: Length in centimeters
        confidence: Recognition confidence score (0.0 to 1.0)
        boat: Boat identifier
        station_id: Station identifier
        timestamp: Entry timestamp (defaults to current time)
    """
    species: str
    length_cm: float
    confidence: float
    boat: str = ""
    station_id: str = ""
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ProcessFinalTextUseCase:
    """Use case for processing final speech recognition text.

    Takes raw recognized text and parses it into structured fish data.
    """

    def __init__(self, fish_parser: FishParser):
        self.fish_parser = fish_parser

    def execute(self, text: str, confidence: float) -> Result[Optional[ParserResult], ParsingError]:
        """Process final text and return parsed result.

        Args:
            text: The recognized text
            confidence: Recognition confidence score

        Returns:
            Result containing ParserResult on success or ParsingError on failure
        """
        try:
            result = self.fish_parser.parse_text(text)
            logger.info(f"[parsed] species={result.species} length_cm={result.length_cm}")
            return Success(result)
        except Exception as e:
            logger.error(f"Failed to parse text '{text}': {e}")
            return Failure(ParsingError(f"Failed to parse: {e}"))


class LogFishEntryUseCase:
    """Use case for logging a fish entry.

    Persists fish catch data to Excel and optionally session logs.
    """

    def __init__(self, excel_logger, session_logger=None):
        self.excel_logger = excel_logger
        self.session_logger = session_logger

    def execute(
        self,
        species: str,
        length_cm: float,
        confidence: float,
        boat: str = "",
        station_id: str = ""
    ) -> Result[FishEntry, LoggingError]:
        """Log a fish entry.

        Args:
            species: Fish species name
            length_cm: Length in centimeters
            confidence: Recognition confidence
            boat: Boat name
            station_id: Station identifier

        Returns:
            Result containing FishEntry on success or LoggingError on failure
        """
        try:
            # Create entry
            entry = FishEntry(
                species=species,
                length_cm=length_cm,
                confidence=confidence,
                boat=boat,
                station_id=station_id
            )

            # Log to Excel
            self.excel_logger.log_entry(species, length_cm, confidence, boat, station_id)
            logger.info(
                f"[excel] Logged {species} {length_cm:.1f} cm "
                f"conf={confidence:.2f} boat={boat} station={station_id}"
            )

            return Success(entry)
        except Exception as e:
            logger.error(f"Failed to log fish entry: {e}")
            return Failure(LoggingError(f"Failed to log: {e}"))


class CancelLastEntryUseCase:
    """Use case for canceling the last logged entry.

    Removes the most recent entry from the Excel log, useful for
    correcting mistakes or handling mis-recognitions.
    """

    def __init__(self, excel_logger):
        self.excel_logger = excel_logger

    def execute(self) -> Result[bool, LoggingError]:
        """Cancel the last entry.

        Returns:
            Result with True if entry was canceled, False if nothing to cancel,
            or LoggingError on failure
        """
        try:
            ok = self.excel_logger.cancel_last()
            if ok:
                logger.info("[parsed] CANCEL -> removed last entry")
            else:
                logger.info("[parsed] CANCEL -> nothing to remove")
            return Success(ok)
        except Exception as e:
            logger.error(f"Failed to cancel last entry: {e}")
            return Failure(LoggingError(f"Failed to cancel: {e}"))
