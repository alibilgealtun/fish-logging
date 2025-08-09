from __future__ import annotations

import sys
from loguru import logger
from PyQt6.QtWidgets import QApplication

from gui import MainWindow
from logger import ExcelLogger
from speech import SpeechRecognizer


def main() -> None:
    # Ensure a single stderr sink to avoid duplicate logs
    try:
        logger.remove()
    except Exception:
        pass
    logger.add(sys.stderr, level="INFO")
    app = QApplication(sys.argv)

    # Use the testing_tt.py configuration (base.en CPU int8, VAD) via SpeechRecognizer
    speech = SpeechRecognizer()
    xlogger = ExcelLogger()

    win = MainWindow(speech, xlogger)
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
