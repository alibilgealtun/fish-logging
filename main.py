from __future__ import annotations

import sys
from loguru import logger
from PyQt6.QtWidgets import QApplication

from gui.MainWindow import MainWindow
from excel.logger import ExcelLogger
from speech import SpeechRecognizer


def main() -> None:
    try:
        logger.remove()
    except Exception:
        pass
    logger.add(sys.stderr, level="INFO")
    app = QApplication(sys.argv)

    speech = SpeechRecognizer()
    xlogger = ExcelLogger()

    win = MainWindow(speech, xlogger)
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
