"""Services for application infrastructure management.

This module implements the Service Layer pattern to separate infrastructure
concerns (exception handling, signal handling, cleanup) from application logic.
"""
from __future__ import annotations

import atexit
import signal
import sys
import threading
from typing import Optional

from loguru import logger

from core.error_handler import handle_exceptions


class ExceptionHandlerService:
    """Service for managing global exception handling.

    Captures uncaught exceptions from both the main thread and worker threads,
    logging them appropriately before allowing the default exception handling.

    Attributes:
        session_logger: Optional session logger for persistent error tracking
    """

    def __init__(self, session_logger=None):
        self.session_logger = session_logger
        self._original_excepthook = sys.excepthook
        self._original_thread_excepthook = threading.excepthook

    def install(self) -> None:
        """Install global exception handlers for main and worker threads."""
        def excepthook(exc_type, exc_value, exc_traceback):
            try:
                import traceback
                tb = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                logger.error("Uncaught exception:\n{}", tb)
                if self.session_logger:
                    self.session_logger.log(tb)
            finally:
                # Always call original handler to ensure proper termination
                try:
                    self._original_excepthook(exc_type, exc_value, exc_traceback)
                except Exception:
                    pass

        def thread_excepthook(args: threading.ExceptHookArgs) -> None:
            excepthook(args.exc_type, args.exc_value, args.exc_traceback)

        sys.excepthook = excepthook
        threading.excepthook = thread_excepthook

    def uninstall(self) -> None:
        """Restore original exception handlers."""
        sys.excepthook = self._original_excepthook
        threading.excepthook = self._original_thread_excepthook


class SignalHandlerService:
    """Service for managing POSIX signal handlers.

    Handles graceful shutdown on SIGINT, SIGTERM, SIGHUP, and SIGQUIT
    by executing cleanup callbacks before terminating.
    """

    def __init__(self, cleanup_callback: Optional[callable] = None, quit_callback: Optional[callable] = None):
        """Initialize signal handler service.

        Args:
            cleanup_callback: Function to call for cleanup (e.g., close resources)
            quit_callback: Function to call to exit application (e.g., QApplication.quit)
        """
        self.cleanup_callback = cleanup_callback
        self.quit_callback = quit_callback

    def install(self) -> None:
        """Install signal handlers for graceful shutdown.

        Registers handlers for common termination signals. Silently ignores
        signals that cannot be caught in the current environment.
        """
        def _handle(signum, frame):
            try:
                logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            except Exception:
                pass
            try:
                if self.cleanup_callback:
                    self.cleanup_callback()
            finally:
                try:
                    if self.quit_callback:
                        self.quit_callback()
                except Exception:
                    pass

        for sig_name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGQUIT"):
            sig = getattr(signal, sig_name, None)
            if sig is not None:
                try:
                    signal.signal(sig, _handle)
                except Exception:
                    # Some signals may not be catchable in certain environments
                    pass


class CleanupService:
    """Service for managing application cleanup.

    Maintains a registry of cleanup handlers and ensures they are executed
    exactly once in the correct order during application shutdown.
    """

    def __init__(self):
        self._cleanup_handlers = []
        self._cleaned_up = False

    def register(self, handler: callable, name: str = "") -> None:
        """Register a cleanup handler.

        Args:
            handler: Function to call during cleanup
            name: Optional name for the handler (for logging)
        """
        self._cleanup_handlers.append((handler, name))

    @handle_exceptions(message="Cleanup failed")
    def cleanup(self) -> None:
        """Execute all registered cleanup handlers.

        Handlers are executed in registration order. Failures in individual
        handlers are logged but don't prevent subsequent handlers from running.
        Cleanup is idempotent - calling multiple times has no additional effect.
        """
        if self._cleaned_up:
            return

        self._cleaned_up = True
        logger.info("Starting cleanup...")

        for handler, name in self._cleanup_handlers:
            try:
                logger.debug(f"Cleaning up: {name or handler.__name__}")
                handler()
            except Exception as e:
                logger.warning(f"Cleanup handler {name} failed: {e}")

        logger.info("Cleanup completed")

    def install_atexit(self) -> None:
        """Register cleanup to run at exit.

        Uses Python's atexit module to ensure cleanup runs even if
        the application terminates unexpectedly.
        """
        atexit.register(self.cleanup)


class RecognizerCleanupService:
    """Service specifically for speech recognizer cleanup.

    Handles the complex shutdown sequence required for speech recognizers,
    including graceful interruption and forced termination if needed.
    """

    def __init__(self, recognizer):
        self.recognizer = recognizer

    @handle_exceptions(message="Recognizer cleanup failed")
    def cleanup(self, timeout_ms: int = 8000, force_timeout_ms: int = 4000) -> None:
        """Clean up the speech recognizer.

        Args:
            timeout_ms: Timeout for graceful shutdown in milliseconds
            force_timeout_ms: Timeout for forced termination in milliseconds

        Note:
            Attempts graceful interruption first, then forced stop if necessary.
        """
        if self.recognizer is None:
            return

        try:
            # Request interruption for thread-based recognizers
            if hasattr(self.recognizer, "requestInterruption"):
                try:
                    self.recognizer.requestInterruption()
                except Exception:
                    pass

            # Stop the recognizer
            self.recognizer.stop()
        except Exception as e:
            logger.debug(f"Recognizer stop() raised: {e}")

        # Wait for thread termination
        try:
            if getattr(self.recognizer, 'isRunning', lambda: False)():
                finished = self.recognizer.wait(timeout_ms)
                if not finished and getattr(self.recognizer, 'isRunning', lambda: False)():
                    # Force termination
                    logger.warning("Recognizer thread did not exit in time; forcing terminate()")
                    try:
                        self.recognizer.terminate()
                    except Exception:
                        pass
                    try:
                        self.recognizer.wait(force_timeout_ms)
                    except Exception:
                        pass
        except Exception as e:
            logger.debug(f"Recognizer wait() raised: {e}")
