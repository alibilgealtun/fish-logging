from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal
from typing import Optional

class BaseSpeechRecognizer(QThread):
    """
    Abstract base class for all speech recognizer implementations.

    This class defines the common interface that all speech recognition engines
    must implement to be compatible with the fish logging application. It provides
    standardized signals, lifecycle methods, and control mechanisms that ensure
    consistent behavior across different recognition backends.

    Key Responsibilities:
    - Define PyQt signals for UI communication
    - Provide thread-safe start/stop/pause/resume controls
    - Establish common configuration patterns
    - Handle species context for number-only measurements
    - Manage noise profile settings
    - Ensure proper resource cleanup

    All concrete recognizer implementations must inherit from this class and
    implement the run() method with their specific recognition logic.
    """

    # PyQt signals that all recognizers must provide for UI integration
    # These signals enable decoupled communication between recognition engines and the GUI
    partial_text = pyqtSignal(str)          # Intermediate transcription updates
    final_text = pyqtSignal(str, float)     # Final results with confidence scores
    error = pyqtSignal(str)                 # Error messages for user notification
    status_changed = pyqtSignal(str)        # Status updates (listening, processing, etc.)
    specie_detected = pyqtSignal(str)       # Fish species detection events

    # Standard prompt for fish-focused recognition systems
    # This context helps guide recognition models toward fish-related vocabulary
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
        """Initialize the base recognizer with common state and configuration.

        Sets up thread infrastructure, control flags, and language settings
        that are common across all recognizer implementations.

        Args:
            language: Language code for recognition (e.g., "en-US", "es-ES")
                     Default is "en-US" for English (United States)
        """
        super().__init__()
        # Thread control flags - these coordinate between UI and recognition threads
        self._stop_flag = False      # Signals when recognition should stop
        self._paused = False         # Controls pause/resume functionality

        # Fish species context for number-only measurements
        # When users speak only numbers, this provides species context
        self._last_fish_specie = None

        # Language configuration for recognition engines
        self.language = language

    def run(self) -> None:
        """Main processing loop for the recognizer thread.

        This abstract method must be implemented by all subclasses to provide
        their specific recognition logic. The implementation should:

        1. Initialize recognition resources (models, APIs, etc.)
        2. Set up audio input streams
        3. Process audio in a continuous loop
        4. Emit appropriate signals for transcription results
        5. Handle pause/resume commands
        6. Clean up resources on exit

        The method should respect the _stop_flag and _paused state variables
        and handle exceptions gracefully to avoid crashing the application.

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement the run() method")

    def begin(self) -> None:
        """Start or restart the recognizer thread.

        This method provides a clean way to start the recognition process.
        It resets the stop flag and starts the QThread if it's not already running.
        Subclasses may override this to perform additional initialization.
        """
        self._stop_flag = False
        if not self.isRunning():
            self.start()

    def stop(self) -> None:
        """Request the recognizer to stop and interrupt the QThread event loop.

        This method provides a graceful shutdown mechanism that sets the stop flag
        and requests thread interruption. Subclasses should override this to
        perform additional cleanup (closing streams, releasing resources, etc.)
        """
        self._stop_flag = True
        try:
            # Notify QThread cooperatively to interrupt blocking operations
            self.requestInterruption()
        except Exception:
            # Ignore exceptions during shutdown to avoid cascading failures
            pass

    def is_stopped(self) -> bool:
        """Check if a stop request has been made.

        This method allows the recognition loop to check if it should exit.
        It's thread-safe and should be called regularly in recognition loops.

        Returns:
            bool: True if stop has been requested, False otherwise
        """
        return self._stop_flag

    def pause(self) -> None:
        """Pause transcription processing.

        When paused, the recognizer should continue listening but not process
        or emit transcription results. This is typically triggered by voice
        commands like "wait" or "pause".
        """
        self._paused = True
        self.status_changed.emit("paused")

    def resume(self) -> None:
        """Resume transcription processing.

        Resumes normal operation after being paused. This is typically
        triggered by voice commands like "start" or "continue".
        """
        self._paused = False
        self.status_changed.emit("listening")

    # ---- Helper methods for UI integration ----

    def set_last_species(self, name: Optional[str]) -> None:
        """Set the last detected/current species for number-only measurements.

        This method is called by the UI when users select a fish species or when
        a species is detected in transcription. It provides context for subsequent
        number-only measurements, allowing users to say just "25 cm" instead of
        "bass 25 cm" when the species is already established.

        Args:
            name: Name of the fish species to use as context, or None to clear
        """
        self._last_fish_specie = name or None

    def set_noise_profile(self, name: Optional[str]) -> None:
        """Set the noise profile hint for environmental optimization.

        Concrete recognizers may override this method to reconfigure their
        noise processing based on environmental conditions. The base implementation
        only stores the name if the subclass has defined the attribute.

        Common noise profiles:
        - "clean": Quiet indoor environments with minimal background noise
        - "human": Environments with background conversation/voices
        - "engine": Marine environments with engine noise
        - "mixed": General-purpose profile for varied conditions

        Args:
            name: Name of the noise profile to apply, or None for default
        """
        # Default implementation only stores the name if child class defined the attribute
        if hasattr(self, "_noise_profile_name"):
            setattr(self, "_noise_profile_name", (name or "mixed").lower())

    def __del__(self):
        """Destructor that provides best-effort cleanup to avoid crashes.

        This destructor attempts to gracefully shut down the recognizer thread
        to prevent Qt crashes during interpreter shutdown. It uses defensive
        programming techniques to handle cases where Qt objects may already
        be destroyed.
        """
        try:
            # Check if the thread is still running using safe attribute access
            if getattr(self, 'isRunning', lambda: False)():
                try:
                    # Attempt graceful shutdown
                    self.stop()
                except Exception:
                    # Ignore exceptions during cleanup
                    pass
                try:
                    # Wait for thread to finish with timeout
                    if not self.wait(2000):  # 2 second timeout
                        # Last resort: force termination to avoid Qt crashes
                        try:
                            self.terminate()
                        except Exception:
                            pass
                        try:
                            # Brief wait after termination
                            self.wait(1000)  # 1 second timeout
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass
