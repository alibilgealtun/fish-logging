from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal

class BaseSpeechRecognizer(QThread):
    """
    Abstract Base Class (Interface) for all speech recognizer implementations.

    Defines the common controls, signals, and lifecycle methods that any
    speech recognizer engine must provide to be used by the application.
    """
    # Define all signals here so the main application can connect to them
    # regardless of the chosen engine.
    partial_text = pyqtSignal(str)
    final_text = pyqtSignal(str, float)
    error = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    specie_detected = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._stop_flag = False
        self._paused = False

    def run(self) -> None:
        """
        The main processing loop for the recognizer.
        This must be implemented by all subclasses.
        """
        raise NotImplementedError

    def begin(self) -> None:
        """Starts or restarts the recognizer thread."""
        self._stop_flag = False
        if not self.isRunning():
            self.start()

    def stop(self) -> None:
        """Requests the recognizer to stop."""
        self._stop_flag = True

    def is_stopped(self) -> bool:
        """Returns True if a stop has been requested."""
        return self._stop_flag

    def pause(self) -> None:
        """Pauses transcription."""
        self._paused = True
        self.status_changed.emit("paused")

    def resume(self) -> None:
        """Resumes transcription."""
        self._paused = False
        self.status_changed.emit("listening")