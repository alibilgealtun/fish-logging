"""Application initialization and setup."""
from __future__ import annotations

import sys
import threading
from typing import TYPE_CHECKING

from loguru import logger
from PyQt6.QtWidgets import QApplication

from logger.excel_logger import ExcelLogger
from logger.session_logger import SessionLogger

if TYPE_CHECKING:
    from speech import BaseSpeechRecognizer


class Application:
    """Main application class that handles initialization and lifecycle."""

    def __init__(self, recognizer: BaseSpeechRecognizer):
        self.recognizer = recognizer
        self.session = SessionLogger.get()
        self.excel_logger = ExcelLogger()
        self._setup_exception_handling()

    def _setup_exception_handling(self) -> None:
        """Configure global exception handling for main and worker threads."""
        def excepthook(exc_type, exc_value, exc_traceback):
            try:
                import traceback
                tb = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                logger.error("Uncaught exception:\n{}", tb)
                self.session.log(tb)
            finally:
                sys.__excepthook__(exc_type, exc_value, exc_traceback)

        def thread_excepthook(args: threading.ExceptHookArgs) -> None:
            excepthook(args.exc_type, args.exc_value, args.exc_traceback)

        sys.excepthook = excepthook
        threading.excepthook = thread_excepthook

    def log_session_info(self, config: dict) -> None:
        """Log session configuration and recognizer info."""
        try:
            self.session.log_kv("APP", {
                "engine": config.get("engine"),
                "numbers_only": config.get("numbers_only"),
                "python": sys.version.split(" ")[0],
            })

            if hasattr(self.recognizer, "get_config"):
                self.session.log_kv("CONFIG", self.recognizer.get_config())
        except Exception:
            logger.warning("Failed to log session configuration")

    def create_qt_app(self, qt_args: list[str]) -> QApplication:
        """Create and configure the Qt application."""
        app = QApplication(qt_args)
        app.aboutToQuit.connect(lambda: SessionLogger.get().log_end())
        return app

    def create_main_window(self):
        """Create the main application window."""
        # Import here to avoid pulling Qt in tests
        from gui.MainWindow import MainWindow
        return MainWindow(self.recognizer, self.excel_logger)
