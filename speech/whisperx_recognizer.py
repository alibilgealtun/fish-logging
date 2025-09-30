from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from loguru import logger
from PyQt6.QtCore import pyqtSignal

from .base_recognizer import BaseSpeechRecognizer
from noise.controller import NoiseController
from parser import ParserResult
from logger.session_logger import SessionLogger

# Attempt to import whisperx in a defensive way
_try_whisperx = True
_whisperx = None
try:
    import whisperx as _wx  # type: ignore
    _whisperx = _wx
except Exception as e:
    _try_whisperx = False
    _whisperx = None
    logger.debug(f"whisperx import failed: {e}")


@dataclass
class TranscriptionSegment:
    text: str
    confidence: float


class WhisperXRecognizer(BaseSpeechRecognizer):
    """
    Realtime CPU-only (or GPU if configured) speech recognizer using NoiseController + whisperx
    This mirrors the behavior of your faster_whisper-based recognizer as closely as possible.
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

    VAD_MODE: int = 2
    MIN_SPEECH_S: float = 0.4
    MAX_SEGMENT_S: float = 3.0
    PADDING_MS: int = 600

    FISH_PROMPT = (
        "This is a continuous conversation about fish species and numbers (their measurements). "
        "The user typically speaks fish specie or number. "
        "Always prioritize fish species vocabulary over similar-sounding common words. "
        "If a word sounds like a fish name, bias towards the fish name. "
        "Common fish species include: trout, salmon, sea bass, tuna. "
        "Units are typically centimeters (cm) or millimeters (mm), 'cm' is preferred in the transcript. "
        "You might also hear 'cancel', 'wait' and 'start'."
    )

    # === Model specific configs ===
    MODEL_NAME: str = "base.en"  # whisperx may accept the same model names depending on backend
    DEVICE: str = "cpu"
    # For whisperx the compute_type is typically handled by torch dtype/device; keep field for parity
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

    def __init__(self) -> None:
        super().__init__()
        self._stop_flag: bool = False
        self._paused: bool = False
        self._stream: Optional[sd.InputStream] = None
        self._model = None
        self._chunk_frames: int = int(self.SAMPLE_RATE * self.CHUNK_S)
        self._last_status_msg: Optional[str] = None
        self._last_fish_specie = None

        # Short "number" marker like faster_whisper used
        try:
            self._number_sound, _ = sf.read("tests/audio/number.wav", dtype='int16')
        except Exception:
            # fallback to silence of 0.05s if asset not found
            self._number_sound = (np.zeros(int(self.SAMPLE_RATE * 0.05))).astype(np.int16)

        # Initialize noise controller
        self._noise_controller = NoiseController(
            sample_rate=self.SAMPLE_RATE,
            vad_mode=self.VAD_MODE,
            min_speech_s=self.MIN_SPEECH_S,
            max_segment_s=self.MAX_SEGMENT_S
        )

    # ---------- Public control ----------
    def stop(self) -> None:
        self._stop_flag = True
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None
        except Exception as e:
            logger.debug(f"Error stopping input stream: {e}")

        self._noise_controller.stop()

    def begin(self) -> None:
        self._stop_flag = False
        self._last_status_msg = None

        # Recreate noise controller for clean state
        self._noise_controller = NoiseController(
            sample_rate=self.SAMPLE_RATE,
            vad_mode=self.VAD_MODE,
            min_speech_s=self.MIN_SPEECH_S,
            max_segment_s=self.MAX_SEGMENT_S
        )

        # Session logging is process-wide; nothing to initialize here.

        if not self.isRunning():
            try:
                self.start()
            except Exception as e:
                logger.error(f"Failed to (re)start WhisperX recognizer: {e}")

    def is_stopped(self) -> bool:
        return self._stop_flag

    def pause(self) -> None:
        self._paused = True
        self.status_changed.emit("paused")

    def resume(self) -> None:
        self._paused = False
        self.status_changed.emit("listening")

    def get_config(self) -> dict:
        return {
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
            "WORD_TIMESTAMPS": self.WORD_TIMESTAMPS
        }

    # ---------- Internal helpers ----------
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
        sf.write(path, samples_int16, samplerate, subtype="PCM_16")
        return path

    def _emit_status_once(self, message: str) -> None:
        if message != self._last_status_msg:
            self._last_status_msg = message
            try:
                self.status_changed.emit(message)
            except Exception:
                logger.error(f"Failed to emit status_changed message: {message}")

    # ---------- Thread main ----------
    def run(self) -> None:
        # Load whisperx model
        if not _try_whisperx or _whisperx is None:
            msg = "whisperx is not installed or failed to import. Please install whisperx and required deps."
            logger.error(msg)
            self.error.emit(msg)
            return

        # Try a few common ways to load whisperx models depending on the installed version:
        try:
            logger.info("Loading whisperx model... (this may download files on first run)")
            # Common pattern: model = whisperx.load_model(model_name, device=device)
            try:
                self._model = _whisperx.load_model(self.MODEL_NAME, device=self.DEVICE)
            except Exception:
                # Another pattern: whisperx.load_model(model, device, compute_type=...)
                try:
                    self._model = _whisperx.load_model(self.MODEL_NAME, device=self.DEVICE, compute_type=self.COMPUTE_TYPE)
                except Exception:
                    # Some versions expose WhisperXModel or require instantiating a wrapper
                    # Try to use attribute 'WhisperXModel' or 'WhisperModel' constructors
                    if hasattr(_whisperx, "WhisperXModel"):
                        self._model = _whisperx.WhisperXModel(self.MODEL_NAME, device=self.DEVICE)
                    elif hasattr(_whisperx, "WhisperModel"):
                        self._model = _whisperx.WhisperModel(self.MODEL_NAME, device=self.DEVICE)
                    else:
                        raise RuntimeError("Unsupported whisperx API in installed package")
        except Exception as e:
            msg = f"Failed to load whisperx model: {e}"
            logger.error(msg)
            self.error.emit(msg)
            if hasattr(self, '_session_logger'):
                self._session_logger.log(f"ERROR: {e}")
            return

        # Open microphone stream
        try:
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

        with self._stream:
            logger.info("Recording with noise control (whisperx)... Press Stop to end.")
            self.partial_text.emit("Listening…")
            self._emit_status_once("listening")

            segment_generator = self._noise_controller.collect_segments_with_timing(padding_ms=self.PADDING_MS)

            while not self.is_stopped():
                try:
                    item = next(segment_generator)
                    if isinstance(item, tuple) and len(item) == 3:
                        segment, start_ts, end_ts = item
                    else:
                        segment = item  # type: ignore
                        start_ts = end_ts = time.time()

                    if segment is None or segment.size == 0:
                        continue

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

                    # combine number sound with the current segment
                    combined_segment = np.concatenate((self._number_sound, segment))
                    wav_path = self._write_wav_bytes(combined_segment, self.SAMPLE_RATE)

                    # Transcription with whisperx - many versions return dict with "segments" key
                    try:
                        assert self._model is not None

                        # try a typical whisperx call pattern
                        transcription_result = None
                        try:
                            # Some versions offer model.transcribe(file, **kwargs)
                            transcription_result = self._model.transcribe(
                                wav_path,
                                beam_size=self.BEAM_SIZE,
                                best_of=self.BEST_OF,
                                temperature=self.TEMPERATURE,
                                patience=self.PATIENCE,
                                length_penalty=self.LENGTH_PENALTY,
                                repetition_penalty=self.REPETITION_PENALTY,
                                language="en",
                                condition_on_previous_text=self.CONDITION_ON_PREVIOUS_TEXT,
                                initial_prompt=self.FISH_PROMPT,
                                vad_filter=self.VAD_FILTER,
                                vad_parameters=self.VAD_PARAMETERS,
                                without_timestamps=self.WITHOUT_TIMESTAMPS,
                                word_timestamps=self.WORD_TIMESTAMPS
                            )
                        except Exception:
                            # fallback to a simpler signature
                            transcription_result = self._model.transcribe(wav_path)

                        # Normalize transcription_result into segments list similar to faster_whisper
                        segments = []
                        if isinstance(transcription_result, dict):
                            # Many implementations return {"segments": [...], "text": "..."}
                            raw_segments = transcription_result.get("segments", None)
                            if raw_segments is None:
                                # maybe transcription_result is a list of segments directly
                                raw_segments = transcription_result
                            if raw_segments is None:
                                # last-resort: try to use 'text'
                                text_out = transcription_result.get("text", "").strip()
                                if text_out:
                                    segments = [TranscriptionSegment(text=text_out, confidence=0.8)]
                                else:
                                    segments = []
                            else:
                                # map to TranscriptionSegment if objects/dicts
                                for seg in raw_segments:
                                    if isinstance(seg, dict):
                                        seg_text = seg.get("text", "") or seg.get("sentence", "") or ""
                                        seg_conf = seg.get("confidence", 0.85) or seg.get("score", 0.85)
                                        segments.append(TranscriptionSegment(text=seg_text.strip(), confidence=float(seg_conf)))
                                    else:
                                        # object with attributes
                                        try:
                                            seg_text = getattr(seg, "text", "")
                                            seg_conf = getattr(seg, "confidence", 0.85)
                                            segments.append(TranscriptionSegment(text=seg_text.strip(), confidence=float(seg_conf)))
                                        except Exception:
                                            # fallback - convert to string
                                            segments.append(TranscriptionSegment(text=str(seg), confidence=0.85))
                        elif isinstance(transcription_result, list):
                            for seg in transcription_result:
                                if isinstance(seg, dict):
                                    segments.append(TranscriptionSegment(text=seg.get("text", "").strip(), confidence=float(seg.get("confidence", 0.85))))
                                else:
                                    # maybe segment objects
                                    seg_text = getattr(seg, "text", str(seg))
                                    seg_conf = getattr(seg, "confidence", 0.85)
                                    segments.append(TranscriptionSegment(text=seg_text.strip(), confidence=float(seg_conf)))
                        else:
                            # Unknown type; make a best-effort string
                            text_out = str(transcription_result).strip()
                            if text_out:
                                segments = [TranscriptionSegment(text=text_out, confidence=0.8)]
                            else:
                                segments = []

                    except Exception as e:
                        msg = f"Transcription error (whisperx): {e}"
                        logger.error(msg)
                        self.error.emit(msg)
                        try:
                            os.remove(wav_path)
                        except Exception:
                            pass
                        self._emit_status_once("listening")
                        continue

                    # Build output text
                    text_out = "".join((seg.text + " ") for seg in segments).strip()
                    if not text_out:
                        self._emit_status_once("listening")
                        try:
                            os.remove(wav_path)
                        except Exception:
                            pass
                        continue

                    logger.info(f"Raw transcription (whisperx): {text_out}")
                    try:
                        SessionLogger.get().log(
                            f"TRANSCRIPT: '{text_out}' audio_s={segment_duration:.3f}"
                        )
                    except Exception:
                        pass

                    # --- Check for pause/resume commands ---
                    text_lower = text_out.lower()
                    if "wait" in text_lower:
                        self.pause()
                        self.final_text.emit("Waiting until 'start' is said.", 0.85)
                        try:
                            os.remove(wav_path)
                        except Exception:
                            pass
                        continue
                    elif "start" in text_lower:
                        self.resume()
                        try:
                            os.remove(wav_path)
                        except Exception:
                            pass
                        continue

                    if self._paused:
                        logger.debug("Paused: ignoring transcription (whisperx)")
                        try:
                            os.remove(wav_path)
                        except Exception:
                            pass
                        continue

                    # Apply ASR corrections and parsing (same as faster_whisper)
                    try:
                        from parser import FishParser, TextNormalizer
                        fish_parser = FishParser()
                        text_normalizer = TextNormalizer()

                        corrected_text = text_normalizer.apply_fish_asr_corrections(text_out)
                        if corrected_text != text_out.lower():
                            logger.info(f"After ASR corrections: {corrected_text}")

                        result: ParserResult = fish_parser.parse_text(corrected_text)

                        if result.species is not None:
                            self._last_fish_specie = result.species
                            self.specie_detected.emit(result.species)

                        if result.length_cm is not None:
                            raw_val = float(result.length_cm)
                            num_str = (f"{raw_val:.1f}").rstrip("0").rstrip(".")
                            formatted = f"{self._last_fish_specie} {num_str} cm"
                            logger.info(f">> {formatted}")
                            self.final_text.emit(formatted, 0.85)
                        else:
                            logger.info(f">> {corrected_text} (partial parse)")
                            self.final_text.emit(corrected_text, 0.85)

                    except Exception as e:
                        logger.error(f"Parser error (whisperx): {e}")
                        logger.info(f">> {text_out}")
                        self.final_text.emit(text_out, 0.85)

                    # Cleanup and return to listening
                    try:
                        os.remove(wav_path)
                    except Exception:
                        pass

                    self._emit_status_once("listening")

                except StopIteration:
                    break
                except Exception as e:
                    logger.error(f"Main loop error (whisperx): {e}")
                    self.error.emit(f"Processing error: {e}")
                    time.sleep(0.1)
                    continue

        # On exit
        self._emit_status_once("stopped")
        logger.info("WhisperX recognizer stopped")
