import logging
import os
import signal
import atexit
from datetime import datetime
from typing import Optional

from loguru import logger as loguru_logger


class SessionLogger:
    """Process-wide session logger.

    - Singleton: use SessionLogger.get() to retrieve the instance.
    - Creates logs/sessions/<timestamped>.log at first access.
    - Attaches a Loguru sink so all logger.info/warning/error are persisted.
    - Registers best-effort shutdown hooks (aboutToQuit should also call log_end()).
    """

    _instance: Optional["SessionLogger"] = None

    def __init__(self):
        # Private: use get()
        self.log_dir = "logs/sessions"
        os.makedirs(self.log_dir, exist_ok=True)
        # Filename uses minute_hour_day_month_year like existing convention
        timestamp = datetime.now().strftime("%M_%H_%d_%m_%Y")
        self.log_path = os.path.join(self.log_dir, f"whisper_session_{timestamp}.log")

        # Standard logging handler (for direct .log() calls)
        self._py_logger = logging.getLogger(f"WhisperSessionLogger_{timestamp}")
        self._py_logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
        formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        self._py_logger.handlers = []
        self._py_logger.addHandler(file_handler)

        # Attach a Loguru sink so application logs are mirrored to the session file
        self._sink_id: Optional[int] = None
        self.attach_loguru_sink()

        # Track state to avoid duplicate end entries
        self._ended: bool = False

        # Register best-effort shutdown handlers
        atexit.register(self._on_atexit)
        self._install_signal_handlers()

    # ---------- Singleton access ----------
    @classmethod
    def get(cls) -> "SessionLogger":
        if cls._instance is None:
            cls._instance = SessionLogger()
            cls._instance.log("=== SESSION START ===")
        return cls._instance

    # ---------- Wiring ----------
    def attach_loguru_sink(self) -> None:
        """Attach a loguru sink to mirror all loguru logs to the session file."""
        if self._sink_id is None:
            try:
                self._sink_id = loguru_logger.add(
                    self.log_path,
                    format="[{time:YYYY-MM-DD HH:mm:ss}] {level}: {message}",
                    level="INFO",
                )
            except Exception:
                self._sink_id = None

    def detach_loguru_sink(self) -> None:
        if self._sink_id is not None:
            try:
                loguru_logger.remove(self._sink_id)
            except Exception:
                pass
            finally:
                self._sink_id = None

    def _install_signal_handlers(self) -> None:
        def _handler(signum, frame):
            try:
                self.log(f"=== SESSION TERMINATING (signal {signum}) ===")
                self.log_end()
            finally:
                # Re-install default handler then re-raise signal to allow normal termination
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)

        for sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", None)):
            if sig is None:
                continue
            try:
                signal.signal(sig, _handler)
            except Exception:
                # Some environments disallow setting handlers; ignore
                pass

    def _on_atexit(self) -> None:
        # Ensure end marker is printed if not already
        self.log_end()

    # ---------- Public API ----------
    def log(self, message: str) -> None:
        self._py_logger.info(message)

    def log_kv(self, prefix: str, data: dict) -> None:
        for k, v in data.items():
            self.log(f"{prefix}: {k} = {v}")

    def log_start(self, config: dict) -> None:
        # Keep compatibility; called optionally by app at startup
        self.log("=== SESSION START ===")
        self.log_kv("CONFIG", config)

    def log_end(self) -> None:
        if self._ended:
            return
        self._ended = True
        self.log("=== SESSION END ===")
        # Flush and detach sink
        try:
            for h in list(self._py_logger.handlers):
                try:
                    h.flush()
                except Exception:
                    pass
        except Exception:
            pass
        self.detach_loguru_sink()

    # Convenience: per-segment timing
    def log_segment_timing(self, start_ts: float, end_ts: float, audio_samples: int, sample_rate: int, note: str = "") -> None:
        wall = max(0.0, end_ts - start_ts)
        audio_s = audio_samples / float(sample_rate) if sample_rate else 0.0
        self.log(
            f"SEGMENT timing: start={start_ts:.6f} end={end_ts:.6f} wall_s={wall:.3f} audio_s={audio_s:.3f} {note}".strip()
        )
