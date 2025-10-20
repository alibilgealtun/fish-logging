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

    Refactored to use service classes following Single Responsibility Principle.
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

        # Register cleanup handlers
        self.cleanup_service.register(self.recognizer_cleanup.cleanup, "recognizer")
        self.cleanup_service.register(
            lambda: SessionLogger.get().log_end(), "session_logger"
        )

    def log_session_info(self, config: dict) -> None:
        """Log session configuration and recognizer info."""
        try:
            self.session.log_kv(
                "APP",
                {
                    "engine": config.get("engine"),
                    "numbers_only": config.get("numbers_only"),
                    "python": sys.version.split(" ")[0],
                },
            )

            if hasattr(self.recognizer, "get_config"):
                self.session.log_kv("CONFIG", self.recognizer.get_config())
        except Exception:
            logger.warning("Failed to log session configuration")

    def create_qt_app(self, qt_args: list[str]) -> QApplication:
        """Create and configure the Qt application."""
        app = QApplication(qt_args)

        # Update signal handler with quit callback
        self.signal_handler.quit_callback = app.quit
        self.signal_handler.install()

        # Connect cleanup to Qt shutdown
        app.aboutToQuit.connect(self.cleanup)

        return app

    def cleanup(self) -> None:
        """Execute cleanup through the cleanup service."""
        self.cleanup_service.cleanup()

    def create_main_window(self):
        """Create the main application window."""
        from gui.MainWindow import MainWindow  # local import to avoid PyQt in tests

        return MainWindow(self.recognizer, self.excel_logger)
