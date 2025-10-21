"""
GUI Package for Fish Logging Application.

This package contains all graphical user interface components for the fish logging
application, built with PyQt6. It follows a modular architecture with clear
separation of concerns.

Modules:
    MainWindow: Main application window and orchestration
    speech_event_handler: Event handling for speech recognition
    recognizer_controller: Speech recognizer lifecycle management
    status_presenter: Status display logic
    settings_provider: Settings management interface
    table_manager: Table data management
    widgets: Custom UI components and widgets

Architecture:
    - MVP (Model-View-Presenter) pattern for separation of concerns
    - Dependency injection for testability
    - Event-driven communication between components
    - Clean architecture principles with use cases
"""

from __future__ import annotations

# Explicit imports for better IDE support and clarity
from .MainWindow import MainWindow
from .speech_event_handler import SpeechEventHandler
from .recognizer_controller import RecognizerController
from .status_presenter import StatusPresenter
from .settings_provider import SettingsProvider
from .table_manager import TableManager

__all__ = [
    "MainWindow",
    "SpeechEventHandler",
    "RecognizerController",
    "StatusPresenter",
    "SettingsProvider",
    "TableManager",
]

__version__ = "1.0.0"
__author__ = "Ali Altun"

