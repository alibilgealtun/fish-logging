"""Status presentation logic for the main window."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class StatusConfig:
    """Configuration for a status state."""
    color: str
    bg_color: str
    label: str
    status_message: str


class StatusDisplay(Protocol):
    """Protocol for objects that can display status."""

    def set_status_color(self, color: str, bg_color: str) -> None:
        """Set the status indicator colors."""
        ...

    def setText(self, text: str) -> None:
        """Set the status label text."""
        ...


class StatusBar(Protocol):
    """Protocol for status bar."""

    def showMessage(self, message: str, timeout: int = 0) -> None:
        """Show message in status bar."""
        ...


class StatusPresenter:
    """Handles status presentation logic following Single Responsibility Principle.

    This class centralizes all status configuration and presentation logic,
    removing this responsibility from MainWindow.
    """

    # Status configurations as class constants
    STATUS_CONFIGS = {
        "listening": StatusConfig(
            color="#2196F3",
            bg_color="#1976D2",
            label="🎧 Listening for speech...",
            status_message="Ready to capture your words"
        ),
        "capturing": StatusConfig(
            color="#4CAF50",
            bg_color="#388E3C",
            label="🎤 Capturing speech...",
            status_message="Recording your fishing story"
        ),
        "finishing": StatusConfig(
            color="#FF9800",
            bg_color="#F57C00",
            label="⚡ Processing...",
            status_message="Analyzing your catch data"
        ),
        "stopped": StatusConfig(
            color="#9E9E9E",
            bg_color="#616161",
            label="⏸️ Listening stopped",
            status_message="Voice recognition paused"
        ),
        "error": StatusConfig(
            color="#f44336",
            bg_color="#d32f2f",
            label="❌ Error",
            status_message="An error occurred"
        ),
    }

    def __init__(self, status_panel: StatusDisplay, status_label: StatusDisplay, status_bar: StatusBar):
        """Initialize the status presenter.

        Args:
            status_panel: Widget that displays status color
            status_label: Widget that displays status text
            status_bar: Application status bar
        """
        self.status_panel = status_panel
        self.status_label = status_label
        self.status_bar = status_bar

    def show_status(self, status_key: str) -> None:
        """Show a predefined status state.

        Args:
            status_key: One of: listening, capturing, finishing, stopped, error
        """
        config = self.STATUS_CONFIGS.get(status_key, self.STATUS_CONFIGS["listening"])
        self.status_panel.set_status_color(config.color, config.bg_color)
        self.status_label.setText(config.label)
        self.status_bar.showMessage(config.status_message)

    def show_error(self, message: str) -> None:
        """Show an error status.

        Args:
            message: Error message to display
        """
        config = self.STATUS_CONFIGS["error"]
        self.status_panel.set_status_color(config.color, config.bg_color)
        self.status_label.setText(config.label)
        self.status_bar.showMessage(f"❌ Error: {message}")

    def show_custom_message(self, message: str, timeout: int = 0) -> None:
        """Show a custom message in the status bar.

        Args:
            message: Message to display
            timeout: Timeout in milliseconds (0 = no timeout)
        """
        self.status_bar.showMessage(message, timeout)

