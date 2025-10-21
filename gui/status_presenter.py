"""
Status Presenter for Fish Logging Application.

This module contains the StatusPresenter class which manages the display of
application status information across multiple UI components. It implements
the Presenter pattern to separate status display logic from business logic.

Classes:
    StatusPresenter: Manages status display across multiple UI components

Architecture:
    - Presenter pattern for clean separation of concerns
    - Centralized status management
    - Multiple display target coordination
    - Visual feedback with animations and styling
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol
from PyQt6.QtWidgets import QStatusBar

# Optional logger: prefer loguru if available, else fallback to stdlib logging
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


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
    """
    Manages status display across multiple UI components.

    This class centralizes all status-related display logic, ensuring
    consistent status presentation across different UI elements while
    maintaining separation of concerns from business logic.

    Key Responsibilities:
        - Coordinate status display across multiple UI components
        - Provide visual feedback with appropriate styling
        - Handle different status types (success, error, warning, info)
        - Manage status animations and visual effects
        - Ensure consistent user experience

    Design Patterns:
        - Presenter pattern: Separates display logic from business logic
        - Observer pattern: Responds to status changes
        - Strategy pattern: Different display strategies for different status types

    Attributes:
        status_panel: Visual status indicator (e.g., pulsing light)
        status_label: Text label for status messages
        status_bar: Application status bar for additional context
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

    def __init__(
        self,
        status_panel=None,
        status_label=None,
        status_bar: Optional[QStatusBar] = None
    ) -> None:
        """
        Initialize the status presenter with UI components.

        Args:
            status_panel: Visual status indicator widget (optional)
            status_label: Text label widget for status messages (optional)
            status_bar: Application status bar widget (optional)

        Design:
            - Flexible initialization allowing partial UI component sets
            - Graceful degradation when components are missing
            - No hard dependencies on specific widget types
        """
        self.status_panel = status_panel
        self.status_label = status_label
        self.status_bar = status_bar

        # Status state tracking
        self.current_status = "idle"
        self.last_message = ""

        logger.debug(f"StatusPresenter initialized with components: "
                    f"panel={status_panel is not None}, "
                    f"label={status_label is not None}, "
                    f"bar={status_bar is not None}")

    def show_status(self, status: str, message: Optional[str] = None) -> None:
        """
        Display status across all configured UI components.

        Updates all available status display components with appropriate
        visual indicators and messages based on the status type.

        Args:
            status: Status identifier (e.g., "listening", "processing", "stopped")
            message: Optional custom message to display

        Supported Status Types:
            - "listening": Active speech recognition
            - "processing": Processing speech input
            - "stopped": Recognition stopped
            - "error": Error state
            - "success": Successful operation
            - "warning": Warning state

        Features:
            - Automatic message generation if not provided
            - Visual styling based on status type
            - Animation control for visual feedback
            - Consistent presentation across components
        """
        self.current_status = status

        # Generate appropriate message if not provided
        if message is None:
            message = self._generate_status_message(status)

        self.last_message = message

        logger.debug(f"Displaying status: {status} - {message}")

        # Update visual status indicator
        self._update_status_panel(status)

        # Update text label
        self._update_status_label(status, message)

        # Update status bar
        self._update_status_bar(message)

    def show_error(self, error_message: str) -> None:
        """
        Display error status with appropriate visual styling.

        Specialized method for error display that ensures consistent
        error presentation across all UI components.

        Args:
            error_message: The error message to display

        Features:
            - Red visual indicators
            - Error-specific animations
            - Persistent display until cleared
            - User-friendly error formatting
        """
        logger.warning(f"Displaying error: {error_message}")

        # Format error message for user display
        formatted_message = f"❌ {error_message}"

        self.show_status("error", formatted_message)

    def show_success(self, success_message: str) -> None:
        """
        Display success status with positive visual feedback.

        Args:
            success_message: The success message to display

        Features:
            - Green visual indicators
            - Positive feedback animations
            - Brief display duration
            - Encouraging visual styling
        """
        logger.info(f"Displaying success: {success_message}")

        # Format success message for user display
        formatted_message = f"✅ {success_message}"

        self.show_status("success", formatted_message)

    def show_warning(self, warning_message: str) -> None:
        """
        Display warning status with attention-getting styling.

        Args:
            warning_message: The warning message to display

        Features:
            - Yellow/orange visual indicators
            - Attention-getting animations
            - Medium persistence
            - Clear warning indicators
        """
        logger.warning(f"Displaying warning: {warning_message}")

        # Format warning message for user display
        formatted_message = f"⚠️ {warning_message}"

        self.show_status("warning", formatted_message)

    def clear_status(self) -> None:
        """
        Clear all status displays and return to idle state.

        Resets all status components to their default idle state,
        useful for cleanup or when no specific status needs to be shown.
        """
        logger.debug("Clearing status display")
        self.show_status("idle", "Ready")

    def get_current_status(self) -> tuple[str, str]:
        """
        Get the current status and message.

        Returns:
            tuple: (status_type, message) of current status

        Useful for:
            - Status persistence across UI updates
            - Debugging status state
            - External status queries
        """
        return self.current_status, self.last_message

    def _generate_status_message(self, status: str) -> str:
        """
        Generate appropriate default message for status type.

        Args:
            status: Status identifier

        Returns:
            str: Human-readable status message

        Default Messages:
            - Provides consistent messaging
            - User-friendly language
            - Emoji indicators for visual appeal
        """
        status_messages = {
            "listening": "🎧 Listening for speech...",
            "processing": "🔄 Processing speech...",
            "stopped": "⏹️ Recognition stopped",
            "error": "❌ An error occurred",
            "success": "✅ Operation successful",
            "warning": "⚠️ Warning",
            "idle": "🟢 Ready",
            "ready": "🟢 Ready to listen",
            "capturing": "🎤 Capturing speech...",
            "finishing": "⏳ Finishing recognition...",
        }

        return status_messages.get(status, f"Status: {status}")

    def _update_status_panel(self, status: str) -> None:
        """
        Update the visual status indicator panel.

        Args:
            status: Status type for visual styling

        Features:
            - Color-coded indicators
            - Animation control
            - Status-specific visual effects
        """
        if self.status_panel is None:
            return

        try:
            # Map status to visual styles
            if hasattr(self.status_panel, 'set_status'):
                self.status_panel.set_status(status)
            elif hasattr(self.status_panel, 'setStatus'):
                self.status_panel.setStatus(status)
            else:
                # Try to update style based on status
                self._apply_status_styling(self.status_panel, status)

        except Exception as e:
            logger.error(f"Failed to update status panel: {e}")

    def _update_status_label(self, status: str, message: str) -> None:
        """
        Update the status text label.

        Args:
            status: Status type for styling
            message: Message text to display

        Features:
            - Status-specific text styling
            - Color coding
            - Font weight adjustments
        """
        if self.status_label is None:
            return

        try:
            # Update text content
            if hasattr(self.status_label, 'setText'):
                self.status_label.setText(message)
            elif hasattr(self.status_label, 'setPlainText'):
                self.status_label.setPlainText(message)

            # Apply status-specific styling
            self._apply_status_styling(self.status_label, status)

        except Exception as e:
            logger.error(f"Failed to update status label: {e}")

    def _update_status_bar(self, message: str) -> None:
        """
        Update the application status bar.

        Args:
            message: Message to display in status bar

        Features:
            - Temporary message display
            - Automatic timeout handling
            - Integration with Qt status bar
        """
        if self.status_bar is None:
            return

        try:
            # Display message with timeout
            self.status_bar.showMessage(message, 5000)  # 5 second timeout

        except Exception as e:
            logger.error(f"Failed to update status bar: {e}")

    def _apply_status_styling(self, widget, status: str) -> None:
        """
        Apply status-specific styling to a widget.

        Args:
            widget: Widget to style
            status: Status type for styling

        Styling Features:
            - Color-coded status indicators
            - Status-specific fonts and effects
            - Consistent visual language
        """
        if not hasattr(widget, 'setStyleSheet'):
            return

        try:
            # Define status-specific styles
            status_styles = {
                "listening": """
                    color: #4CAF50;
                    font-weight: 600;
                    background-color: rgba(76, 175, 80, 0.1);
                """,
                "processing": """
                    color: #2196F3;
                    font-weight: 600;
                    background-color: rgba(33, 150, 243, 0.1);
                """,
                "error": """
                    color: #F44336;
                    font-weight: 700;
                    background-color: rgba(244, 67, 54, 0.1);
                """,
                "success": """
                    color: #4CAF50;
                    font-weight: 600;
                    background-color: rgba(76, 175, 80, 0.1);
                """,
                "warning": """
                    color: #FF9800;
                    font-weight: 600;
                    background-color: rgba(255, 152, 0, 0.1);
                """,
                "stopped": """
                    color: #9E9E9E;
                    font-weight: 500;
                    background-color: rgba(158, 158, 158, 0.1);
                """,
                "idle": """
                    color: white;
                    font-weight: 500;
                    background-color: transparent;
                """,
            }

            # Apply style if available
            style = status_styles.get(status, status_styles["idle"])
            current_style = widget.styleSheet()

            # Preserve existing styles and add status-specific ones
            widget.setStyleSheet(f"{current_style}\n{style}")

        except Exception as e:
            logger.error(f"Failed to apply status styling: {e}")

    def is_error_state(self) -> bool:
        """
        Check if currently displaying an error state.

        Returns:
            bool: True if in error state, False otherwise
        """
        return self.current_status == "error"

    def is_active_state(self) -> bool:
        """
        Check if in an active processing state.

        Returns:
            bool: True if actively processing, False otherwise
        """
        active_states = {"listening", "processing", "capturing", "finishing"}
        return self.current_status in active_states
