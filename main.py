from __future__ import annotations

# Note: Avoid importing PyQt6 modules at import-time to keep test environments headless-friendly.
# We'll import QtWebEngineWidgets and QApplication inside main().

import sys
import os
from typing import Any, cast

from logger.excel_logger  import ExcelLogger
from loguru import logger
from speech import BaseSpeechRecognizer, WhisperRecognizer, VoskRecognizer
# Allow optional Google engine without forcing import when unused
try:
    from speech import GoogleSpeechRecognizer  # type: ignore
except Exception:
    GoogleSpeechRecognizer = None  # type: ignore

from logger.session_logger import SessionLogger


def main() -> None:
    # Import WebEngine before QApplication is created to avoid context errors
    # and keep imports local to avoid issues during unit test collection.
    from PyQt6 import QtWebEngineWidgets  # noqa: F401
    from PyQt6.QtWidgets import QApplication
    # Import MainWindow lazily so importing main.py in tests doesn't pull PyQt GUI
    from gui.MainWindow import MainWindow

    try:
        logger.remove()
    except Exception:
        pass
    logger.add(sys.stderr, level="INFO")

    # Initialize process-wide session logger once, at app start
    session = SessionLogger.get()

    # Capture uncaught exceptions into the session log
    def _excepthook(exc_type, exc_value, exc_traceback):
        try:
            logger.error("Uncaught exception:")
            import traceback
            tb = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            session.log(tb)
        finally:
            # Delegate to default hook as well
            sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = _excepthook

    app = QApplication(sys.argv)

    # Ensure we close the session cleanly when the Qt app is quitting
    app.aboutToQuit.connect(lambda: SessionLogger.get().log_end())

    xlogger = ExcelLogger()
    # Choose engine via CLI arg --engine=whisper|vosk|google|whisperx or env SPEECH_ENGINE
    engine_arg = next((arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("--engine=") or arg.startswith("--model=")), None)
    engine = (engine_arg or os.getenv("SPEECH_ENGINE") or "whisper").lower()

    # Numbers-only flag (applies to Google recognizer)
    numbers_only = any(a == "--numbers-only" for a in sys.argv) or (os.getenv("SPEECH_NUMBERS_ONLY", "").lower() in {"1","true","yes"})

    try:
        recognizer = create_recognizer(engine, numbers_only=numbers_only)
    except Exception as e:
        logger.error(str(e))
        recognizer = create_recognizer("whisper", numbers_only=False)

    # Log session configuration once
    try:
        session.log_kv("APP", {
            "engine": engine,
            "numbers_only": numbers_only,
            "python": sys.version.split(" ")[0],
        })
        # If recognizer exposes config, log it
        if hasattr(recognizer, "get_config"):
            session.log_kv("CONFIG", getattr(recognizer, "get_config")())
    except Exception:
        pass

    win = MainWindow(recognizer, xlogger)
    win.show()

    sys.exit(app.exec())

def create_recognizer(engine: str, numbers_only: bool = False) -> BaseSpeechRecognizer:
    """Factory function to create a speech recognizer instance."""
    eng = engine.lower()
    if eng == "whisper":
        # Use the faster-whisper based recognizer
        return WhisperRecognizer()
    elif eng == "whisperx":
        # Lazy import to avoid heavy dependency unless requested
        from speech import WhisperXRecognizer  # type: ignore
        return WhisperXRecognizer()
    elif eng == "vosk":
        return VoskRecognizer()
    elif eng == "google":
        if GoogleSpeechRecognizer is None:
            raise RuntimeError("GoogleSpeechRecognizer not available. Install google-cloud-speech and retry.")
        # Credentials auto-detected via GOOGLE_APPLICATION_CREDENTIALS or google.json in project root
        return cast(Any, GoogleSpeechRecognizer)(numbers_only=numbers_only)
    else:
        raise ValueError(f"Unknown speech recognition engine: {engine}")

if __name__ == "__main__":
    main()
