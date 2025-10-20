"""Unit tests for use cases."""
import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime

from app.use_cases import (
    ProcessFinalTextUseCase,
    LogFishEntryUseCase,
    CancelLastEntryUseCase,
    FishEntry,
)
from parser.parser import ParserResult
from core.exceptions import ParsingError, LoggingError


class TestProcessFinalTextUseCase:
    """Tests for ProcessFinalTextUseCase."""

    def test_successful_parsing(self):
        # Arrange
        mock_parser = Mock()
        mock_parser.parse_text.return_value = ParserResult(
            cancel=False,
            species="Trout",
            length_cm=25.5
        )
        use_case = ProcessFinalTextUseCase(mock_parser)

        # Act
        result = use_case.execute("trout twenty five centimeters", 0.95)

        # Assert
        assert result.is_success()
        parsed = result.unwrap()
        assert parsed.species == "Trout"
        assert parsed.length_cm == 25.5
        mock_parser.parse_text.assert_called_once()

    def test_cancel_command(self):
        # Arrange
        mock_parser = Mock()
        mock_parser.parse_text.return_value = ParserResult(
            cancel=True,
            species=None,
            length_cm=None
        )
        use_case = ProcessFinalTextUseCase(mock_parser)

        # Act
        result = use_case.execute("cancel", 0.95)

        # Assert
        assert result.is_success()
        parsed = result.unwrap()
        assert parsed.cancel is True

    def test_parsing_failure(self):
        # Arrange
        mock_parser = Mock()
        mock_parser.parse_text.side_effect = Exception("Parse error")
        use_case = ProcessFinalTextUseCase(mock_parser)

        # Act
        result = use_case.execute("invalid text", 0.95)

        # Assert
        assert result.is_failure()
        assert isinstance(result.error, ParsingError)


class TestLogFishEntryUseCase:
    """Tests for LogFishEntryUseCase."""

    def test_successful_logging(self):
        # Arrange
        mock_excel_logger = Mock()
        use_case = LogFishEntryUseCase(mock_excel_logger)

        # Act
        result = use_case.execute("Salmon", 30.5, 0.98, "Boat1", "ST1")

        # Assert
        assert result.is_success()
        entry = result.unwrap()
        assert isinstance(entry, FishEntry)
        assert entry.species == "Salmon"
        assert entry.length_cm == 30.5
        assert entry.confidence == 0.98
        assert entry.boat == "Boat1"
        assert entry.station_id == "ST1"
        assert isinstance(entry.timestamp, datetime)

        mock_excel_logger.log_entry.assert_called_once_with(
            "Salmon", 30.5, 0.98, "Boat1", "ST1"
        )

    def test_logging_with_defaults(self):
        # Arrange
        mock_excel_logger = Mock()
        use_case = LogFishEntryUseCase(mock_excel_logger)

        # Act
        result = use_case.execute("Trout", 25.0, 0.90)

        # Assert
        assert result.is_success()
        entry = result.unwrap()
        assert entry.boat == ""
        assert entry.station_id == ""

    def test_logging_failure(self):
        # Arrange
        mock_excel_logger = Mock()
        mock_excel_logger.log_entry.side_effect = Exception("Excel error")
        use_case = LogFishEntryUseCase(mock_excel_logger)

        # Act
        result = use_case.execute("Bass", 20.0, 0.85)

        # Assert
        assert result.is_failure()
        assert isinstance(result.error, LoggingError)


class TestCancelLastEntryUseCase:
    """Tests for CancelLastEntryUseCase."""

    def test_successful_cancel(self):
        # Arrange
        mock_excel_logger = Mock()
        mock_excel_logger.cancel_last.return_value = True
        use_case = CancelLastEntryUseCase(mock_excel_logger)

        # Act
        result = use_case.execute()

        # Assert
        assert result.is_success()
        assert result.unwrap() is True
        mock_excel_logger.cancel_last.assert_called_once()

    def test_nothing_to_cancel(self):
        # Arrange
        mock_excel_logger = Mock()
        mock_excel_logger.cancel_last.return_value = False
        use_case = CancelLastEntryUseCase(mock_excel_logger)

        # Act
        result = use_case.execute()

        # Assert
        assert result.is_success()
        assert result.unwrap() is False

    def test_cancel_failure(self):
        # Arrange
        mock_excel_logger = Mock()
        mock_excel_logger.cancel_last.side_effect = Exception("Cancel error")
        use_case = CancelLastEntryUseCase(mock_excel_logger)

        # Act
        result = use_case.execute()

        # Assert
        assert result.is_failure()
        assert isinstance(result.error, LoggingError)


class TestFishEntry:
    """Tests for FishEntry dataclass."""

    def test_creation_with_defaults(self):
        # Act
        entry = FishEntry(
            species="Cod",
            length_cm=35.0,
            confidence=0.92
        )

        # Assert
        assert entry.species == "Cod"
        assert entry.length_cm == 35.0
        assert entry.confidence == 0.92
        assert entry.boat == ""
        assert entry.station_id == ""
        assert isinstance(entry.timestamp, datetime)

    def test_creation_with_all_fields(self):
        # Arrange
        timestamp = datetime(2025, 10, 20, 12, 0, 0)

        # Act
        entry = FishEntry(
            species="Halibut",
            length_cm=50.0,
            confidence=0.99,
            boat="Ocean Voyager",
            station_id="ST5",
            timestamp=timestamp
        )

        # Assert
        assert entry.species == "Halibut"
        assert entry.length_cm == 50.0
        assert entry.confidence == 0.99
        assert entry.boat == "Ocean Voyager"
        assert entry.station_id == "ST5"
        assert entry.timestamp == timestamp

