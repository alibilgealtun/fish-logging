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

    FISH_PROMPT = (
        "This is a continuous conversation about fish species and numbers (their measurements). "
        "The user typically speaks fish specie or number. "
        "Always prioritize fish species vocabulary over similar-sounding common words. "
        "If a word sounds like a fish name, bias towards the fish name. "
        "Common fish species include: trout, salmon, sea bass, tuna. "
        "Units are typically centimeters (cm) or millimeters (mm), 'cm' is preferred in the transcript. "
        "You might also hear 'cancel', 'wait' and 'start'."
    )

    def __init__(self, language: str = "en-US") -> None:
        super().__init__()
        self._stop_flag = False
        self._paused = False
        self._last_fish_specie = None
        self.language = language

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

    def __del__(self):  # Best-effort safeguard to avoid abort on interpreter shutdown
        try:
            if getattr(self, 'isRunning', lambda: False)():
                try:
                    self.stop()  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    self.wait(2000)
                except Exception:
                    pass
        except Exception:
            pass
