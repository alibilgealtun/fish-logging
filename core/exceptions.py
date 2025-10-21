"""Custom exception hierarchy for the application."""
from __future__ import annotations


class FishLoggingException(Exception):
    """Base exception for all fish logging errors."""
    pass


class ParsingError(FishLoggingException):
    """Raised when text parsing fails."""
    pass


class RecognizerError(FishLoggingException):
    """Raised when speech recognition fails."""
    pass


class LoggingError(FishLoggingException):
    """Raised when logging operations fail."""
    pass


class ConfigurationError(FishLoggingException):
    """Raised when configuration is invalid or missing."""
    pass
