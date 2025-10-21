"""Application initialization and setup."""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from loguru import logger
from PyQt6.QtWidgets import QApplication

from logger.excel_logger import ExcelLogger
from logger.session_logger import SessionLogger
from app.services import (
    ExceptionHandlerService,
    SignalHandlerService,
    CleanupService,
    RecognizerCleanupService,
)

if TYPE_CHECKING:
    from speech import BaseSpeechRecognizer


class Application:
    """Main application class that handles initialization and lifecycle.

    Manages the application's dependencies, service initialization, and cleanup
    using the Service Layer pattern for separation of concerns.

    Attributes:
        recognizer: Speech recognition engine
        session: Session logger for tracking application events
        excel_logger: Excel output for fish catch data
        cleanup_service: Manages cleanup handlers
        exception_handler: Global exception handling
        signal_handler: POSIX signal handling for graceful shutdown
        recognizer_cleanup: Speech recognizer cleanup logic
    """

    def __init__(
        self,
        recognizer: BaseSpeechRecognizer,
        excel_logger: ExcelLogger = None,
        session_logger: SessionLogger = None,
    ):
        """Initialize application with dependencies.

        Args:
            recognizer: Speech recognizer instance
            excel_logger: Excel logger instance (created if not provided)
            session_logger: Session logger instance (uses singleton if not provided)
        """
        self.recognizer = recognizer
        self.session = session_logger or SessionLogger.get()
        self.excel_logger = excel_logger or ExcelLogger()

        # Initialize services
        self.cleanup_service = CleanupService()
        self.exception_handler = ExceptionHandlerService(self.session)
        self.signal_handler = SignalHandlerService(
            cleanup_callback=self.cleanup, quit_callback=None  # Set when Qt app is created
        )
        self.recognizer_cleanup = RecognizerCleanupService(self.recognizer)

        # Setup infrastructure
        self.exception_handler.install()
        self.cleanup_service.install_atexit()

        # Register cleanup handlers in order of execution
        self.cleanup_service.register(self.recognizer_cleanup.cleanup, "recognizer")
        self.cleanup_service.register(
            lambda: SessionLogger.get().log_end(), "session_logger"
        )

    def log_session_info(self, config: dict) -> None:
        """Log session configuration and recognizer info.

        Args:
            config: Configuration dictionary containing engine and settings
        """
        try:
            self.session.log_kv(
                "APP",
                {
                    "engine": config.get("engine"),
                    "numbers_only": config.get("numbers_only"),
                    "python": sys.version.split(" ")[0],
                },
            )

            # Log recognizer-specific configuration if available
            if hasattr(self.recognizer, "get_config"):
                self.session.log_kv("CONFIG", self.recognizer.get_config())
        except Exception:
            logger.warning("Failed to log session configuration")

    def create_qt_app(self, qt_args: list[str]) -> QApplication:
        """Create and configure the Qt application.

        Args:
            qt_args: Command-line arguments to pass to QApplication

        Returns:
            Configured QApplication instance
        """
        app = QApplication(qt_args)

        # Update signal handler with quit callback now that Qt is initialized
        self.signal_handler.quit_callback = app.quit
        self.signal_handler.install()

        # Connect cleanup to Qt shutdown
        app.aboutToQuit.connect(self.cleanup)

        return app

    def cleanup(self) -> None:
        """Execute cleanup through the cleanup service."""
        self.cleanup_service.cleanup()

    def create_main_window(self):
        """Create the main application window.

        Returns:
            MainWindow instance

        Note:
            Import is deferred to avoid PyQt dependency in tests
        """
        from gui.MainWindow import MainWindow  # local import to avoid PyQt in tests

        return MainWindow(self.recognizer, self.excel_logger)
