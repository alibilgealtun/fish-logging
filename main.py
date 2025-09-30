from __future__ import annotations

# Import WebEngine before QApplication is created to avoid context errors
from PyQt6 import QtWebEngineWidgets

import sys
import os
from PyQt6.QtWidgets import QApplication

from gui.MainWindow import MainWindow
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
    # Choose engine via CLI arg --engine=whisper|vosk|google or env SPEECH_ENGINE
    engine_arg = next((arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("--model=")), None)
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
    if engine.lower() == "whisper":
        # Use the Whisper-based recognizer (compatible with whisper_recognizer.py)
        return WhisperRecognizer()
    elif engine.lower() == "vosk":
        return VoskRecognizer()
    elif engine.lower() == "google":
        if GoogleSpeechRecognizer is None:
            raise RuntimeError("GoogleSpeechRecognizer not available. Install google-cloud-speech and retry.")
        # Credentials auto-detected via GOOGLE_APPLICATION_CREDENTIALS or google.json in project root
        return GoogleSpeechRecognizer(numbers_only=numbers_only)
    else:
        raise ValueError(f"Unknown speech recognition engine: {engine}")

if __name__ == "__main__":
    main()
