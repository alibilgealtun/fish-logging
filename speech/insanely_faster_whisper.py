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
from services import get_audio_saver

# Defensive import for insanely-fast-whisper
_try_ifw = True
_ifw = None
try:
    import insanely_fast_whisper as _ifw  # type: ignore
    _ifw = _ifw
except Exception as e:
    _try_ifw = False
    logger.debug(f"insanely-fast-whisper import failed: {e}")


@dataclass
class TranscriptionSegment:
    text: str
    confidence: float


class InsanelyFastWhisperRecognizer(BaseSpeechRecognizer):
    """
    Realtime speech recognizer using NoiseController + insanely-fast-whisper
    Mirrors faster_whisper.py structure closely.
    """

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
    MODEL_NAME: str = "tiny"  # IFW supports "tiny", "base", "small", "medium", "large"
    DEVICE: str = "cpu"
    COMPUTE_TYPE: str = "int8"  # not always used by IFW but kept for parity

    # Decoding parameters
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

    def __init__(self, noise_profile: Optional[str] = None) -> None:
        super().__init__()
        self._stop_flag: bool = False
        self._paused: bool = False
        self._stream: Optional[sd.InputStream] = None
        self._model = None
        self._chunk_frames: int = int(self.SAMPLE_RATE * self.CHUNK_S)
        self._last_status_msg: Optional[str] = None
        self._last_fish_specie = None
        self._noise_profile_name = (noise_profile or "mixed").lower()
        try:
            self._number_sound = self._load_number_prefix()
        except Exception as e:
            logger.debug(f"Failed to load number prefix: {e}")
            self._number_sound = (np.zeros(int(self.SAMPLE_RATE * 0.05))).astype(np.int16)
        from speech.noise_profiles import get_noise_profile, make_suppressor_config
        prof = get_noise_profile(self._noise_profile_name)
        for attr in ("VAD_MODE", "MIN_SPEECH_S", "MAX_SEGMENT_S", "PADDING_MS"):
            if attr in prof:
                setattr(self, attr, prof[attr])
        suppressor_cfg = make_suppressor_config(prof, self.SAMPLE_RATE)
        if self._noise_profile_name == "clean":
            from noise.simple_controller import SimpleNoiseController
            self._noise_controller = SimpleNoiseController(
                sample_rate=self.SAMPLE_RATE,
                vad_mode=self.VAD_MODE,
                min_speech_s=self.MIN_SPEECH_S,
                max_segment_s=self.MAX_SEGMENT_S,
            )
        else:
            self._noise_controller = NoiseController(
                sample_rate=self.SAMPLE_RATE,
                vad_mode=self.VAD_MODE,
                min_speech_s=self.MIN_SPEECH_S,
                max_segment_s=self.MAX_SEGMENT_S,
                suppressor_config=suppressor_cfg,
            )

    def _load_number_prefix(self) -> np.ndarray:
        """Load and prepare the number prefix audio as PCM16 mono at SAMPLE_RATE."""
        candidates = [
            os.path.join(os.getcwd(), "assets/audio/number.wav"),
        ]
        for p in candidates:
            try:
                if not os.path.exists(p):
                    continue
                data, sr = sf.read(p, dtype="int16")
                if data.ndim > 1:
                    data = data[:, 0]
                if sr != self.SAMPLE_RATE:
                    try:
                        from scipy.signal import resample_poly  # type: ignore
                        gcd = np.gcd(sr, self.SAMPLE_RATE)
                        up = self.SAMPLE_RATE // gcd
                        down = sr // gcd
                        data = resample_poly(data.astype(np.int16), up, down).astype(np.int16)
                    except Exception:
                        ratio = self.SAMPLE_RATE / float(sr)
                        idx = (np.arange(int(len(data) * ratio)) / ratio).astype(int)
                        idx = np.clip(idx, 0, len(data) - 1)
                        data = data[idx].astype(np.int16)
                    logger.info(f"Loaded number prefix '{os.path.basename(p)}' at {sr}Hz -> resampled to {self.SAMPLE_RATE}Hz")
                else:
                    logger.info(f"Loaded number prefix '{os.path.basename(p)}' at {sr}Hz (no resample)")
                return data.astype(np.int16)
            except Exception as e:
                logger.debug(f"Failed to load number prefix {p}: {e}")
        logger.warning("No number prefix audio found; using short silence")
        return (np.zeros(int(self.SAMPLE_RATE * 0.05))).astype(np.int16)

    def stop(self) -> None:
        self._stop_flag = True
        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
        except Exception as e:
            logger.debug(f"Error stopping stream: {e}")
        self._noise_controller.stop()

    def begin(self) -> None:
        self._stop_flag = False
        self._last_status_msg = None
        from speech.noise_profiles import get_noise_profile, make_suppressor_config
        prof = get_noise_profile(self._noise_profile_name)
        for attr in ("VAD_MODE", "MIN_SPEECH_S", "MAX_SEGMENT_S", "PADDING_MS"):
            if attr in prof:
                setattr(self, attr, prof[attr])
        suppressor_cfg = make_suppressor_config(prof, self.SAMPLE_RATE)
        if self._noise_profile_name == "clean":
            from noise.simple_controller import SimpleNoiseController
            self._noise_controller = SimpleNoiseController(
                sample_rate=self.SAMPLE_RATE,
                vad_mode=self.VAD_MODE,
                min_speech_s=self.MIN_SPEECH_S,
                max_segment_s=self.MAX_SEGMENT_S,
            )
        else:
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
        self._session_log_sink_id = loguru.logger.add(
            self._session_logger.log_path,
            format="[{time:YYYY-MM-DD HH:mm:ss}] {level}: {message}",
            level="INFO"
        )
        if not self.isRunning():
            try:
                self.start()
            except Exception as e:
                logger.error(f"Failed to (re)start InsanelyFastWhisper recognizer: {e}")

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
            self.status_changed.emit(message)

    def run(self) -> None:
        if not _try_ifw or _ifw is None:
            msg = "insanely-fast-whisper is not installed. Install it with `pip install insanely-fast-whisper`."
            logger.error(msg)
            self.error.emit(msg)
            return

        try:
            logger.info(f"Loading insanely-fast-whisper model '{self.MODEL_NAME}'...")
            # Typical usage: from insanely_fast_whisper import Transcriber
            if hasattr(_ifw, "Transcriber"):
                self._model = _ifw.Transcriber(model=self.MODEL_NAME, device=self.DEVICE)
            else:
                raise RuntimeError("insanely-fast-whisper Transcriber class not found")
        except Exception as e:
            msg = f"Failed to load insanely-fast-whisper model: {e}"
            logger.error(msg)
            self.error.emit(msg)
            return

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
            logger.info(f"Recording with noise control (IFW)... Press Stop to end. [profile={self._noise_profile_name}]")
            self.partial_text.emit("Listening…")
            self._emit_status_once("listening")
            segment_generator = self._noise_controller.collect_segments(padding_ms=self.PADDING_MS)

            while not self.is_stopped():
                try:
                    segment = next(segment_generator)

                    if segment is None or segment.size == 0:
                        continue

                    segment_duration = segment.size / self.SAMPLE_RATE
                    if segment_duration < self.MIN_SPEECH_S:
                        continue

                    self._emit_status_once("processing")

                    segment = np.concatenate((self._number_sound, segment))
                    wav_path = self._write_wav_bytes(segment, self.SAMPLE_RATE)

                    # Save the audio segment to file (for debugging/inspection)
                    audio_saver = get_audio_saver()
                    audio_saver.save_segment(segment, self.SAMPLE_RATE)

                    try:
                        assert self._model is not None
                        # Transcribe using IFW API
                        if hasattr(self._model, "transcribe"):
                            result = self._model.transcribe(wav_path)
                            if isinstance(result, dict) and "text" in result:
                                text_out = str(result.get("text", "")).strip()
                                segments = [TranscriptionSegment(text=text_out, confidence=0.85)] if text_out else []
                            elif isinstance(result, list):
                                segments = []
                                for seg in result:
                                    seg_text = seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", str(seg))
                                    if seg_text:
                                        segments.append(TranscriptionSegment(text=seg_text.strip(), confidence=0.85))
                            else:
                                text_out = str(result).strip()
                                segments = [TranscriptionSegment(text=text_out, confidence=0.85)] if text_out else []
                        else:
                            raise RuntimeError("IFW transcriber has no transcribe method")

                    except Exception as e:
                        msg = f"Transcription error (IFW): {e}"
                        logger.error(msg)
                        self.error.emit(msg)
                        try:
                            os.remove(wav_path)
                        except Exception:
                            pass
                        self._emit_status_once("listening")
                        continue

                    text_out = "".join(seg.text + " " for seg in segments).strip()
                    if not text_out:
                        self._emit_status_once("listening")
                        try:
                            os.remove(wav_path)
                        except Exception:
                            pass
                        continue

                    logger.info(f"Raw transcription (IFW): {text_out}")

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
                        logger.debug("Paused: ignoring transcription")
                        try:
                            os.remove(wav_path)
                        except Exception:
                            pass
                        continue

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
                        logger.error(f"Parser error (IFW): {e}")
                        logger.info(f">> {text_out}")
                        self.final_text.emit(text_out, 0.85)

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

        # On exit
        self._emit_status_once("stopped")
        logger.info("InsanelyFastWhisper recognizer stopped")
        if hasattr(self, '_session_logger'):
            self._session_logger.log_end()
        if hasattr(self, '_session_log_sink_id'):
            import loguru
            loguru.logger.remove(self._session_log_sink_id)
