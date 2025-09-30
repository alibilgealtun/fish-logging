from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from typing import Optional, Any, List, Dict

import numpy as np
from loguru import logger
from PyQt6.QtCore import QThread, pyqtSignal

from noise.controller import NoiseController
from logger.session_logger import SessionLogger


@dataclass
class TranscriptionSegment:
    """Represents a transcription segment with text and confidence."""
    text: str
    confidence: float = 0.85


class BaseSpeechRecognizer(QThread):
    """
    Template base class for realtime speech recognizers.

    Centralizes the complete realtime pipeline:
    - Mic capture via sounddevice
    - NoiseController buffering + VAD segmentation
    - Optional number-prefix concatenation
    - Temporary wav creation for backends that need files
    - Backend transcription hook
    - Command handling (wait/start)
    - Text normalization + fish parsing
    - Signals + status + basic session logging

    Subclasses should only implement the backend-specific parts by
    overriding two hooks:
      - _load_backend_model(self) -> None
      - _backend_transcribe(self, segment: np.ndarray, wav_path: Optional[str]) -> List[TranscriptionSegment]

    They may optionally override:
      - _backend_post_init(self) -> None  # after model load
      - _get_backend_extra_config(self) -> Dict[str, Any]
    """

    # Signals (uniform across all recognizers)
    partial_text = pyqtSignal(str, name="partial_text")
    final_text = pyqtSignal(str, float, name="final_text")
    error = pyqtSignal(str, name="error")
    status_changed = pyqtSignal(str, name="status_changed")
    specie_detected = pyqtSignal(str, name="specie_detected")

    # ===== Common CONFIG defaults =====
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    CHUNK_S: float = 0.5

    # Noise control
    VAD_MODE: int = 3
    MIN_SPEECH_S: float = 0.2
    MAX_SEGMENT_S: float = 3.0
    PADDING_MS: int = 600

    # Optional decoding-related defaults (subclasses may override)
    MODEL_NAME: str = "base.en"
    DEVICE: str = "cpu"
    COMPUTE_TYPE: str = "int8"

    BEAM_SIZE: int = 3
    BEST_OF: int = 5
    TEMPERATURE: float = 0.0
    PATIENCE: float = 1.0
    LENGTH_PENALTY: float = 1.0
    REPETITION_PENALTY: float = 1.0
    WITHOUT_TIMESTAMPS: bool = True
    CONDITION_ON_PREVIOUS_TEXT: bool = True
    VAD_FILTER: bool = False
    VAD_PARAMETERS: Optional[dict] = None
    WORD_TIMESTAMPS: bool = False

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
        super().__init__(parent=None)
        self.language = language

        # Runtime state
        self._stop_flag: bool = False
        self._paused: bool = False
        self._last_fish_specie: Optional[str] = None
        self._last_status_msg: Optional[str] = None

        # Audio / backend
        self._chunk_frames: int = int(self.SAMPLE_RATE * self.CHUNK_S)
        self._stream: Optional[Any] = None  # sounddevice.InputStream at runtime
        self._model: Optional[Any] = None

        # Number prefix marker
        try:
            import soundfile as sf  # local import
            self._number_sound, _ = sf.read("number_prefix.wav", dtype='int16')
        except Exception:
            self._number_sound = (np.zeros(int(self.SAMPLE_RATE * 0.05))).astype(np.int16)

        # Noise controller
        self._noise_controller = NoiseController(
            sample_rate=self.SAMPLE_RATE,
            vad_mode=self.VAD_MODE,
            min_speech_s=self.MIN_SPEECH_S,
            max_segment_s=self.MAX_SEGMENT_S,
        )

    # ---------- Public controls ----------
    def begin(self) -> None:
        """Reset internal state and start the recognizer thread."""
        self._stop_flag = False
        self._last_status_msg = None

        # Recreate noise controller for a clean start
        self._noise_controller = NoiseController(
            sample_rate=self.SAMPLE_RATE,
            vad_mode=self.VAD_MODE,
            min_speech_s=self.MIN_SPEECH_S,
            max_segment_s=self.MAX_SEGMENT_S,
        )

        if not self.isRunning():
            try:
                self.start()
            except Exception as e:
                logger.error(f"Failed to (re)start recognizer: {e}")

    def stop(self) -> None:
        """Request the recognizer to stop and release resources."""
        self._stop_flag = True
        try:
            if self._stream is not None:
                # Import here to avoid module import overhead unless needed
                import sounddevice as sd  # noqa: F401
                self._stream.stop()
                self._stream.close()
                self._stream = None
        except Exception as e:
            logger.debug(f"Error stopping input stream: {e}")

        # Signal noise controller to stop
        try:
            self._noise_controller.stop()
        except Exception:
            pass

    def is_stopped(self) -> bool:
        return self._stop_flag

    def pause(self) -> None:
        self._paused = True
        self.status_changed.emit("paused")

    def resume(self) -> None:
        self._paused = False
        self.status_changed.emit("listening")

    def set_last_species(self, species: str) -> None:
        try:
            self._last_fish_specie = str(species) if species else None
        except Exception:
            self._last_fish_specie = None

    # ---------- Shared helpers ----------
    def get_config(self) -> dict:
        """Return relevant config parameters for logging/export."""
        cfg: Dict[str, Any] = {
            "SAMPLE_RATE": self.SAMPLE_RATE,
            "CHANNELS": self.CHANNELS,
            "CHUNK_S": self.CHUNK_S,
            "VAD_MODE": self.VAD_MODE,
            "MIN_SPEECH_S": self.MIN_SPEECH_S,
            "MAX_SEGMENT_S": self.MAX_SEGMENT_S,
            "PADDING_MS": self.PADDING_MS,
            "FISH_PROMPT": self.FISH_PROMPT,
        }
        # Common decoder/model fields if present
        for k in [
            "MODEL_NAME", "DEVICE", "COMPUTE_TYPE", "BEAM_SIZE", "BEST_OF", "TEMPERATURE",
            "PATIENCE", "LENGTH_PENALTY", "REPETITION_PENALTY", "WITHOUT_TIMESTAMPS",
            "CONDITION_ON_PREVIOUS_TEXT", "VAD_FILTER", "VAD_PARAMETERS", "WORD_TIMESTAMPS",
        ]:
            if hasattr(self, k):
                cfg[k] = getattr(self, k)
        # Backend extras
        try:
            extras = self._get_backend_extra_config() or {}
            cfg.update(extras)
        except Exception:
            pass
        return cfg

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            logger.debug(f"Audio status: {status}")
        try:
            pcm16 = (indata[:, 0] * 32767).astype(np.int16)
            self._noise_controller.push_audio(pcm16)
        except Exception as e:
            logger.debug(f"Audio callback error: {e}")

    @staticmethod
    def _write_wav_bytes(samples_int16: np.ndarray, samplerate: int) -> str:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            import soundfile as sf  # local import
            sf.write(path, samples_int16, samplerate, subtype="PCM_16")
        except Exception as e:
            # In case of failure, attempt cleanup and re-raise
            try:
                os.remove(path)
            except Exception:
                pass
            raise e
        return path

    def _emit_status_once(self, message: str) -> None:
        if message != self._last_status_msg:
            self._last_status_msg = message
            try:
                self.status_changed.emit(message)
            except Exception:
                logger.error(f"Failed to emit status_changed message: {message}")

    # ---------- Template hooks (to override) ----------
    def _load_backend_model(self) -> None:
        """Load backend model into self._model. Must be overridden by subclasses."""
        raise NotImplementedError

    def _backend_transcribe(self, segment: np.ndarray, wav_path: Optional[str]) -> List[TranscriptionSegment]:
        """Transcribe a speech segment and return list of segments (text/confidence)."""
        raise NotImplementedError

    def _backend_post_init(self) -> None:
        """Optional post-initialization step after model load (e.g., constraints)."""
        # default: no-op
        return None

    def _get_backend_extra_config(self) -> Optional[Dict[str, Any]]:
        """Optional extra config for logging provided by subclass."""
        return None

    # ---------- Thread main (shared) ----------
    def run(self) -> None:
        # Load backend
        try:
            self._emit_status_once("initializing")
            if self._model is None:
                self._load_backend_model()
                self._backend_post_init()
            else:
                logger.debug("Reusing already loaded backend model")
        except Exception as e:
            msg = f"Failed to load backend model: {e}"
            logger.error(msg)
            self.error.emit(msg)
            return

        # Open microphone stream
        try:
            import sounddevice as sd  # local import
            self._stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=self.CHANNELS,
                blocksize=self._chunk_frames,
                dtype="float32",
                callback=self._audio_callback,
            )
        except Exception as e:
            msg = f"Failed to open microphone stream: {e}"
            logger.error(msg)
            self.error.emit(msg)
            return

        # Start listening
        with self._stream:
            logger.info("Recording with noise control... Press Stop to end.")
            self.partial_text.emit("Listening…")
            self._emit_status_once("listening")

            segment_generator = self._noise_controller.collect_segments_with_timing(
                padding_ms=self.PADDING_MS
            )

            while not self.is_stopped():
                try:
                    item = next(segment_generator)
                    # Support (segment, start_ts, end_ts) tuple or raw segment
                    if isinstance(item, tuple) and len(item) == 3:
                        segment, start_ts, end_ts = item
                    else:
                        segment = item  # type: ignore
                        start_ts = end_ts = time.time()

                    if segment is None or getattr(segment, 'size', 0) == 0:
                        continue

                    # Duration check
                    segment_duration = segment.size / self.SAMPLE_RATE
                    if segment_duration < self.MIN_SPEECH_S:
                        continue

                    self._emit_status_once("processing")

                    # Log capture timing
                    try:
                        SessionLogger.get().log_segment_timing(
                            float(start_ts), float(end_ts), int(segment.size), self.SAMPLE_RATE, note="captured"
                        )
                    except Exception:
                        pass

                    # Prepend number marker and prepare wav for backends that need files
                    try:
                        combined_segment = np.concatenate((self._number_sound, segment))
                        wav_path = self._write_wav_bytes(combined_segment, self.SAMPLE_RATE)
                    except Exception as e:
                        logger.error(f"Failed to prepare audio for backend: {e}")
                        self._emit_status_once("listening")
                        continue

                    # Backend transcription
                    try:
                        segments = self._backend_transcribe(segment=combined_segment, wav_path=wav_path)
                    except Exception as e:
                        msg = f"Transcription error: {e}"
                        logger.error(msg)
                        self.error.emit(msg)
                        try:
                            os.remove(wav_path)
                        except Exception:
                            pass
                        self._emit_status_once("listening")
                        continue

                    # Normalize result into plain text
                    text_out = "".join((seg.text + " ") for seg in (segments or [])).strip()
                    if not text_out:
                        self._emit_status_once("listening")
                        try:
                            os.remove(wav_path)
                        except Exception:
                            pass
                        continue

                    logger.info(f"Raw transcription: {text_out}")
                    try:
                        SessionLogger.get().log(
                            f"TRANSCRIPT: '{text_out}' audio_s={segment_duration:.3f}"
                        )
                    except Exception:
                        pass

                    # Commands
                    lower = text_out.lower()
                    if "wait" in lower:
                        self.pause()
                        self.final_text.emit("Waiting until 'start' is said.", 0.85)
                        try:
                            os.remove(wav_path)
                        except Exception:
                            pass
                        continue
                    elif "start" in lower:
                        self.resume()
                        try:
                            os.remove(wav_path)
                        except Exception:
                            pass
                        continue

                    if self._paused:
                        logger.debug("Paused: ignoring transcription")
                        try:
                            os.remove(wav_path)
                        except Exception:
                            pass
                        continue

                    # Parse fish-specific output
                    try:
                        from parser import FishParser, TextNormalizer
                        fish_parser = FishParser()
                        text_normalizer = TextNormalizer()

                        corrected = text_normalizer.apply_fish_asr_corrections(text_out)
                        if corrected != text_out.lower():
                            logger.info(f"After ASR corrections: {corrected}")

                        result = fish_parser.parse_text(corrected)

                        if getattr(result, 'species', None) is not None:
                            self._last_fish_specie = result.species
                            try:
                                self.specie_detected.emit(result.species)
                            except Exception:
                                pass

                        if getattr(result, 'length_cm', None) is not None:
                            raw_val = float(result.length_cm)
                            num_str = (f"{raw_val:.1f}").rstrip("0").rstrip(".")
                            formatted = f"{self._last_fish_specie} {num_str} cm" if self._last_fish_specie else f"{num_str} cm"
                            logger.info(f">> {formatted}")
                            self.final_text.emit(formatted, 0.85)
                        else:
                            logger.info(f">> {corrected} (partial parse)")
                            self.final_text.emit(corrected, 0.85)

                    except Exception as e:
                        logger.error(f"Parser error: {e}")
                        logger.info(f">> {text_out}")
                        self.final_text.emit(text_out, 0.85)

                    # Cleanup
                    try:
                        os.remove(wav_path)
                    except Exception:
                        pass

                    self._emit_status_once("listening")

                except StopIteration:
                    break
                except Exception as e:
                    logger.error(f"Main loop error: {e}")
                    self.error.emit(f"Processing error: {e}")
                    time.sleep(0.1)
                    continue

        self._emit_status_once("stopped")
        logger.info("Speech recognizer stopped")
