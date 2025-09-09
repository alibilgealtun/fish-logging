from __future__ import annotations

import sys
import os
from PyQt6.QtWidgets import QApplication

from gui.MainWindow import MainWindow
from logger.excel_logger  import ExcelLogger
from loguru import logger
from speech import BaseSpeechRecognizer, WhisperRecognizer, VoskRecognizer


def main() -> None:
    try:
        logger.remove()
    except Exception:
        pass
    logger.add(sys.stderr, level="INFO")
    app = QApplication(sys.argv)

    xlogger = ExcelLogger()
    # Choose engine via CLI arg --engine=whisper|vosk or env SPEECH_ENGINE
    engine_arg = next((arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("--model=")), None)
    engine = (engine_arg or os.getenv("SPEECH_ENGINE") or "whisper").lower()

    try:
        recognizer = create_recognizer(engine)
    except Exception as e:
        logger.error(str(e))
        recognizer = create_recognizer("whisper")

    win = MainWindow(recognizer, xlogger)
    win.show()

    sys.exit(app.exec())

def create_recognizer(engine: str) -> BaseSpeechRecognizer:
    """Factory function to create a speech recognizer instance."""
    if engine.lower() == "whisper":
        # Use the Whisper-based recognizer (compatible with whisper_recognizer.py)
        return WhisperRecognizer()
    elif engine.lower() == "vosk":
        return VoskRecognizer()
    else:
        raise ValueError(f"Unknown speech recognition engine: {engine}")

if __name__ == "__main__":
    main()
