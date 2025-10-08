"""Application initialization and setup."""
from __future__ import annotations

import sys
import threading
import signal
import atexit
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
        # Ensure cleanup runs even if GUI not closed cleanly
        atexit.register(self.cleanup)

    def _setup_exception_handling(self) -> None:
        """Configure global exception handling for main and worker threads."""
        def excepthook(exc_type, exc_value, exc_traceback):
            try:
                import traceback
                tb = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                logger.error("Uncaught exception:\n{}", tb)
                self.session.log(tb)
            finally:
                try:
                    sys.__excepthook__(exc_type, exc_value, exc_traceback)
                except Exception:
                    pass

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
        # Replace previous simple aboutToQuit hook with full cleanup
        app.aboutToQuit.connect(self.cleanup)
        # Install signal handlers for SIGINT/SIGTERM to allow Ctrl+C graceful shutdown
        self._install_signal_handlers(app)
        return app

    def _install_signal_handlers(self, app: QApplication) -> None:
        """Install POSIX signal handlers to ensure graceful shutdown on Ctrl+C / kill."""
        def _handle(signum, frame):  # type: ignore[override]
            try:
                logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            except Exception:
                pass
            try:
                self.cleanup()
            finally:
                try:
                    app.quit()
                except Exception:
                    pass
        for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
            if sig is not None:
                try:
                    signal.signal(sig, _handle)
                except Exception:
                    # Some environments (like embedded interpreters) may not allow this
                    pass

    def cleanup(self) -> None:
        """Best-effort graceful shutdown of long-running resources / threads."""
        try:
            # Stop recognizer thread if still alive
            if self.recognizer is not None:
                try:
                    self.recognizer.stop()
                except Exception as e:
                    logger.debug(f"Recognizer stop() raised: {e}")
                # Wait briefly for thread termination
                try:
                    if getattr(self.recognizer, 'isRunning', lambda: False)():
                        self.recognizer.wait(5000)
                except Exception as e:
                    logger.debug(f"Recognizer wait() raised: {e}")
        except Exception:
            pass
        # Log session end (idempotent)
        try:
            SessionLogger.get().log_end()
        except Exception:
            pass

    def create_main_window(self):
        """Create the main application window."""
        from gui.MainWindow import MainWindow  # local import to avoid PyQt in tests
        return MainWindow(self.recognizer, self.excel_logger)
