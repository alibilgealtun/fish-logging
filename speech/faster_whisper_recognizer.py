from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
from loguru import logger
from PyQt6.QtCore import pyqtSignal
from .base_recognizer import BaseSpeechRecognizer
from noise.controller import NoiseController
from parser import ParserResult
import soundfile as sf
from services import get_audio_saver


@dataclass
class TranscriptionSegment:
    """Represents a final transcription segment with a text and confidence score.

    Attributes
    ----------
    text: str
        The recognized text.
    confidence: float
        Confidence score in range [0, 1].
    """

    text: str
    confidence: float


class WhisperRecognizer(BaseSpeechRecognizer):
    """
    Realtime CPU-only speech recognizer using NoiseController + faster-whisper.
    Optimized for high-noise environments with engine sounds and background speech.
    """

    # PyQt signals
    partial_text = pyqtSignal(str)
    final_text = pyqtSignal(str, float)
    error = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    specie_detected = pyqtSignal(str)

    # ===== CONFIG  =====
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    CHUNK_S: float = 0.5

    # Noise controller settings - optimized for engine noise + background speech
    VAD_MODE: int = 2 # More aggressive VAD for noisy environments
    MIN_SPEECH_S: float = 0.4
    MAX_SEGMENT_S: float = 3.0
    PADDING_MS: int = 600

    # === Model specific configs ===
    MODEL_NAME: str = "base.en"
    DEVICE: str = "cpu"
    COMPUTE_TYPE: str = "int8"

    # === Decoding parameters ===
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

    def set_last_species(self, species: str) -> None:
        # Public setter for last species (used by UI selector)
        try:
            self._last_fish_specie = str(species) if species else None
        except Exception:
            self._last_fish_specie = None

    def __init__(self, language: str = "en-US", noise_profile: Optional[str] = None) -> None:
        """Initialize recognizer state and resources.

        noise_profile: Optional profile name (clean|human|engine|mixed). If provided overrides
        VAD/segment parameters and suppressor config before constructing NoiseController.
        """
        super().__init__(language=language)
        self._stop_flag: bool = False
        self._paused: bool = False
        self._stream: Optional[Any] = None  # sounddevice.InputStream at runtime
        self._model: Optional[Any] = None  # faster_whisper.WhisperModel at runtime
        self._chunk_frames: int = int(self.SAMPLE_RATE * self.CHUNK_S)
        self._last_status_msg: Optional[str] = None
        self._noise_profile_name = (noise_profile or "mixed").lower()

        # Load number marker
        try:
            self._number_sound, _ = sf.read("number.wav", dtype='int16')
            logger.info("NUMBERR++++++++++++++++++")
        except Exception as e:
            logger.debug("NUMBER SOUND", e)
            self._number_sound = (np.zeros(int(self.SAMPLE_RATE * 0.05))).astype(np.int16)

        # Apply profile overrides
        from speech.noise_profiles import get_noise_profile, make_suppressor_config
        prof = get_noise_profile(self._noise_profile_name)
        for attr in ("VAD_MODE", "MIN_SPEECH_S", "MAX_SEGMENT_S", "PADDING_MS"):
            if attr in prof:
                setattr(self, attr, prof[attr])
        suppressor_cfg = make_suppressor_config(prof, self.SAMPLE_RATE)

        # Initialize noise controller
        self._noise_controller = NoiseController(
            sample_rate=self.SAMPLE_RATE,
            vad_mode=self.VAD_MODE,
            min_speech_s=self.MIN_SPEECH_S,
            max_segment_s=self.MAX_SEGMENT_S,
            suppressor_config=suppressor_cfg,
        )

    # ---------- Public control ----------
    def stop(self) -> None:
        """Request the recognizer to stop and release resources."""
        self._stop_flag = True
        try:
            # Cooperatively interrupt thread
            self.requestInterruption()
        except Exception:
            pass
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
        self._noise_controller.stop()

    def begin(self) -> None:
        """Reset internal state and start the recognizer thread."""
        # Reset flags/state for a clean restart
        self._stop_flag = False
        self._last_status_msg = None

        from speech.noise_profiles import get_noise_profile, make_suppressor_config
        prof = get_noise_profile(self._noise_profile_name)
        for attr in ("VAD_MODE", "MIN_SPEECH_S", "MAX_SEGMENT_S", "PADDING_MS"):
            if attr in prof:
                setattr(self, attr, prof[attr])
        suppressor_cfg = make_suppressor_config(prof, self.SAMPLE_RATE)
        self._noise_controller = NoiseController(
            sample_rate=self.SAMPLE_RATE,
            vad_mode=self.VAD_MODE,
            min_speech_s=self.MIN_SPEECH_S,
            max_segment_s=self.MAX_SEGMENT_S,
            suppressor_config=suppressor_cfg,
        )

        from logger.session_logger import SessionLogger
        self._session_logger = SessionLogger()
        self._session_logger.log_start(self.get_config())
        import loguru
        self._session_log_sink_id = loguru.logger.add(self._session_logger.log_path, format="[{time:YYYY-MM-DD HH:mm:ss}] {level}: {message}", level="INFO")

        if not self.isRunning():
            try:
                self.start()
            except Exception as e:
                logger.error(f"Failed to (re)start recognizer: {e}")

    def is_stopped(self) -> bool:
        """Return True if stop has been requested."""
        return self._stop_flag

    def pause(self) -> None:
        """Pause transcription (after hearing 'WAIT')."""
        self._paused = True
        self.status_changed.emit("paused")

    def resume(self) -> None:
        """Resume transcription (after hearing 'START')."""
        self._paused = False
        self.status_changed.emit("listening")

    def get_config(self) -> dict:
        """Return all relevant config parameters for logging/export."""
        base = {
            "SAMPLE_RATE": self.SAMPLE_RATE,
            "CHANNELS": self.CHANNELS,
            "CHUNK_S": self.CHUNK_S,
            "VAD_MODE": self.VAD_MODE,
            "MIN_SPEECH_S": self.MIN_SPEECH_S,
            "MAX_SEGMENT_S": self.MAX_SEGMENT_S,
            "PADDING_MS": self.PADDING_MS,
            "FISH_PROMPT": self.FISH_PROMPT,
            "MODEL_NAME": self.MODEL_NAME,
            "DEVICE": self.DEVICE,
            "COMPUTE_TYPE": self.COMPUTE_TYPE,
            "BEAM_SIZE": self.BEAM_SIZE,
            "BEST_OF": self.BEST_OF,
            "TEMPERATURE": self.TEMPERATURE,
            "PATIENCE": self.PATIENCE,
            "LENGTH_PENALTY": self.LENGTH_PENALTY,
            "REPETITION_PENALTY": self.REPETITION_PENALTY,
            "WITHOUT_TIMESTAMPS": self.WITHOUT_TIMESTAMPS,
            "CONDITION_ON_PREVIOUS_TEXT": self.CONDITION_ON_PREVIOUS_TEXT,
            "VAD_FILTER": self.VAD_FILTER,
            "VAD_PARAMETERS": self.VAD_PARAMETERS,
            "WORD_TIMESTAMPS": self.WORD_TIMESTAMPS,
            "NOISE_PROFILE": self._noise_profile_name,
        }
        return base

    # ---------- Internal helpers ----------
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Sounddevice callback: push mono PCM16 frames to noise controller."""
        if status:
            logger.debug(f"Audio status: {status}")
        try:
            # Convert to PCM16 and push to noise controller
            pcm16 = (indata[:, 0] * 32767).astype(np.int16)
            self._noise_controller.push_audio(pcm16)
        except Exception as e:
            logger.debug(f"Audio callback error: {e}")

    @staticmethod
    def _write_wav_bytes(samples_int16: np.ndarray, samplerate: int) -> str:
        """Write PCM16 samples to a temporary WAV file and return its path."""
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        # Local import to avoid module import overhead
        import soundfile as sf  # type: ignore
        sf.write(path, samples_int16, samplerate, subtype="PCM_16")
        return path

    def _emit_status_once(self, message: str) -> None:
        """Emit status_changed only when the message changes to avoid flooding UI."""
        if message != self._last_status_msg:
            self._last_status_msg = message
            try:
                self.status_changed.emit(message)
            except Exception:
                logger.error(f"Failed to emit status_changed message: {message}")

    # ---------- Thread main ----------
    def run(self) -> None:
        """Run the realtime STT loop with integrated noise control."""
        # Exit early if interruption/stop was requested before thread started
        if self.is_stopped() or self.isInterruptionRequested():
            return
        try:
            logger.info("Loading model... (first run will download it)")
            # Local import to avoid loading at module import time
            from faster_whisper import WhisperModel  # type: ignore
            self._model = WhisperModel(
                self.MODEL_NAME,
                device=self.DEVICE,
                compute_type=self.COMPUTE_TYPE,
                # Additional optimizations for speed
                download_root=None,
                local_files_only=False
            )
            # If app requested interruption while loading model, exit immediately
            if self.is_stopped() or self.isInterruptionRequested():
                return
        except Exception as e:
            msg = f"Failed to load Whisper model: {e}"
            logger.error(msg)
            self.error.emit(msg)
            if hasattr(self, '_session_logger'):
                self._session_logger.log(f"ERROR: {e}")
            return

        try:
            # Local import to avoid loading at module import time
            import sounddevice as sd  # type: ignore
            if self.is_stopped() or self.isInterruptionRequested():
                return
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

            # Use noise controller's segment collector
            segment_generator = self._noise_controller.collect_segments(
                padding_ms=self.PADDING_MS
            )

            while not self.is_stopped():
                # Honor external interruption requests
                if self.isInterruptionRequested():
                    break
                try:
                    # Get filtered and VAD-processed segment
                    segment = next(segment_generator)

                    if segment is None or segment.size == 0:
                        continue

                    # Additional length check (noise controller already filters, but double-check)
                    segment_duration = segment.size / self.SAMPLE_RATE
                    if segment_duration < self.MIN_SPEECH_S:
                        continue

                    # Update status
                    self._emit_status_once("processing")
                    logger.debug(f"Segment duration before concatting: {segment_duration}")
                    segment = np.concatenate((self._number_sound, segment))
                    logger.debug(f"Segment duration after concatting: {(segment.size / self.SAMPLE_RATE)}")
                    # Write segment to temporary file
                    wav_path = self._write_wav_bytes(segment, self.SAMPLE_RATE)

                    # Save the audio segment to file (for debugging/inspection)
                    audio_saver = get_audio_saver()
                    audio_saver.save_segment(segment, self.SAMPLE_RATE)

                    try:
                        assert self._model is not None
                        # Transcribe with optimized settings for speed
                        segments, info = self._model.transcribe(
                            wav_path,
                            beam_size=self.BEAM_SIZE,
                            best_of=self.BEST_OF,
                            temperature=self.TEMPERATURE,
                            patience=self.PATIENCE,
                            length_penalty=self.LENGTH_PENALTY,
                            repetition_penalty=self.REPETITION_PENALTY,
                            language="en",
                            condition_on_previous_text=True,
                            initial_prompt=self.FISH_PROMPT,
                            vad_filter=False,
                            vad_parameters=None,
                            without_timestamps=True,
                            word_timestamps=False
                        )
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

                    # Process transcription result
                    text_out = "".join(seg.text + " " for seg in segments).strip()
                    if not text_out:
                        self._emit_status_once("listening")
                        continue

                    logger.info(f"Raw transcription: {text_out}")

                    # --- Check for pause/resume commands ---
                    text_lower = text_out.lower()
                    if "wait" in text_lower:
                        self.pause()
                        self.final_text.emit("Waiting until 'start' is said.", 0.85)
                        try:
                            os.remove(wav_path)
                        except Exception as e:
                            logger.debug(e)
                        continue
                    elif "start" in text_lower:
                        self.resume()
                        try:
                            os.remove(wav_path)
                        except Exception as e:
                            logger.debug(e)
                        continue

                    # If paused, skip processing
                    if self._paused:
                        logger.debug("Paused: ignoring transcription")
                        try:
                            os.remove(wav_path)
                        except Exception as e:
                            logger.debug(e)
                        continue

                    # Apply ASR corrections and parsing
                    try:
                        from parser import FishParser, TextNormalizer
                        fish_parser = FishParser()
                        text_normalizer = TextNormalizer()

                        # Apply fish-specific ASR corrections
                        corrected_text = text_normalizer.apply_fish_asr_corrections(text_out)
                        if corrected_text != text_out.lower():
                            logger.info(f"After ASR corrections: {corrected_text}")

                        result: ParserResult = fish_parser.parse_text(corrected_text)

                        if result.species is not None:
                            self._last_fish_specie = result.species
                            self.specie_detected.emit(result.species)

                        if result.length_cm is not None:
                            # Format numeric output
                            raw_val = float(result.length_cm)
                            num_str = (f"{raw_val:.1f}").rstrip("0").rstrip(".")
                            formatted = f"{self._last_fish_specie} {num_str} cm"
                            logger.info(f">> {formatted}")
                            self.final_text.emit(formatted, 0.85)
                        else:
                            # Fallback to corrected text if parsing incomplete
                            logger.info(f">> {corrected_text} (partial parse)")
                            self.final_text.emit(corrected_text, 0.85)

                    except Exception as e:
                        logger.error(f"Parser error: {e}")
                        # Fallback to raw text
                        logger.info(f">> {text_out}")
                        self.final_text.emit(text_out, 0.85)

                    # Clean up temporary file
                    try:
                        os.remove(wav_path)
                    except Exception:
                        pass

                    # Return to listening state
                    self._emit_status_once("listening")

                except StopIteration:
                    # Generator exhausted (normal when stopping)
                    break
                except Exception as e:
                    logger.error(f"Main loop error: {e}")
                    self.error.emit(f"Processing error: {e}")
                    time.sleep(0.1)  # Brief pause before retrying
                    continue

        # On exit
        self._emit_status_once("stopped")
        logger.info("Speech recognizer stopped")
        if hasattr(self, '_session_logger'):
            self._session_logger.log_end()
        if hasattr(self, '_session_log_sink_id'):
            import loguru
            loguru.logger.remove(self._session_log_sink_id)
