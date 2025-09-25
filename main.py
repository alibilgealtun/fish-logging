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


def main() -> None:
    try:
        logger.remove()
    except Exception:
        pass
    logger.add(sys.stderr, level="INFO")
    app = QApplication(sys.argv)

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
