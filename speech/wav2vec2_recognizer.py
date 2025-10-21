from __future__ import annotations

import os
import time
import tempfile
from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
from loguru import logger
from PyQt6.QtCore import pyqtSignal

from .base_recognizer import BaseSpeechRecognizer
from noise.controller import NoiseController
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


class Wav2Vec2Recognizer(BaseSpeechRecognizer):
    """
    Realtime CPU-only speech recognizer using NoiseController + Wav2Vec2 (CTC).
    Optimized for high-noise environments with engine sounds and background speech.
    """

    # PyQt signals (kept same as other recognizers)
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
    VAD_MODE: int = 2
    MIN_SPEECH_S: float = 0.4
    MAX_SEGMENT_S: float = 3.0
    PADDING_MS: int = 600

    # === Model specific configs ===
    MODEL_NAME: str = "facebook/wav2vec2-base-960h"  # English, 16kHz
    DEVICE: str = "cpu"  # Torch device
    FP16: bool = False

    def set_last_species(self, species: str) -> None:
        # Public setter for last species (used by UI selector)
        try:
            self._last_fish_specie = str(species) if species else None
        except Exception:
            self._last_fish_specie = None

    def __init__(self, language: str = "en-US", noise_profile: Optional[str] = None) -> None:
        super().__init__(language=language)
        self._stop_flag: bool = False
        self._paused: bool = False
        self._stream: Optional[Any] = None  # sounddevice.InputStream at runtime
        self._chunk_frames: int = int(self.SAMPLE_RATE * self.CHUNK_S)
        self._last_status_msg: Optional[str] = None
        self._noise_profile_name = (noise_profile or "mixed").lower()

        # Number prefix audio to bias initial decoding towards numbers
        try:
            self._number_sound = self._load_number_prefix()
        except Exception as e:
            logger.debug(f"Failed to load number prefix: {e}")
            self._number_sound = (np.zeros(int(self.SAMPLE_RATE * 0.05))).astype(np.int16)

        # Apply noise profile
        from speech.noise_profiles import get_noise_profile, make_suppressor_config
        prof = get_noise_profile(self._noise_profile_name)
        for attr in ("VAD_MODE", "MIN_SPEECH_S", "MAX_SEGMENT_S", "PADDING_MS"):
            if attr in prof:
                setattr(self, attr, prof[attr])
        suppressor_cfg = make_suppressor_config(prof, self.SAMPLE_RATE)

        # Initialize noise controller
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

        # Initialize session logger using singleton pattern (refactored)
        from logger.session_logger import SessionLogger
        self._session_logger = SessionLogger.get()
        self._session_logger.log_start(self.get_config())

        if not self.isRunning():
            try:
                self.start()
            except Exception as e:
                logger.error(f"Failed to (re)start recognizer: {e}")

    def is_stopped(self) -> bool:
        return self._stop_flag

    def pause(self) -> None:
        self._paused = True
        self.status_changed.emit("paused")

    def resume(self) -> None:
        self._paused = False
        self.status_changed.emit("listening")

    def get_config(self) -> dict:
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
            "ENGINE": "wav2vec2",
            "NOISE_PROFILE": self._noise_profile_name,
        }
        return base

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
        import soundfile as sf  # type: ignore
        sf.write(path, samples_int16, samplerate, subtype="PCM_16")
        return path

    def _emit_status_once(self, message: str) -> None:
        if message != self._last_status_msg:
            self._last_status_msg = message
            try:
                self.status_changed.emit(message)
            except Exception:
                logger.error(f"Failed to emit status_changed message: {message}")

    def _load_number_prefix(self) -> np.ndarray:
        candidates = [
            os.path.join(os.getcwd(), "number_prefix.wav"),
            os.path.join(os.getcwd(), "assets/audio/number.wav"),
            os.path.join(os.getcwd(), "tests", "audio", "number.wav"),
        ]
        import soundfile as sf
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
                    logger.info(
                        f"Loaded number prefix '{os.path.basename(p)}' at {sr}Hz -> resampled to {self.SAMPLE_RATE}Hz"
                    )
                else:
                    logger.info(f"Loaded number prefix '{os.path.basename(p)}' at {sr}Hz (no resample)")
                return data.astype(np.int16)
            except Exception as e:
                logger.debug(f"Failed to load number prefix {p}: {e}")
        logger.warning("No number prefix audio found; using short silence")
        return (np.zeros(int(self.SAMPLE_RATE * 0.05))).astype(np.int16)

    # ---------- Model helpers ----------
    def _lazy_load_model(self) -> None:
        if self._hf_model is not None and self._hf_processor is not None:
            return
        try:
            import torch  # type: ignore
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor  # type: ignore
            self._torch = torch
            self._hf_processor = Wav2Vec2Processor.from_pretrained(self.MODEL_NAME)
            self._hf_model = Wav2Vec2ForCTC.from_pretrained(self.MODEL_NAME)
            self._hf_model.to(self.DEVICE)
            self._hf_model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load Wav2Vec2 model '{self.MODEL_NAME}': {e}")

    def _infer(self, samples_int16: np.ndarray) -> TranscriptionSegment:
        """Run CTC inference and return decoded text and an approximate confidence.

        Confidence is approximated as the mean of max softmax probabilities across time.
        """
        assert self._hf_model is not None and self._hf_processor is not None and self._torch is not None
        torch = self._torch
        # Convert to float32 in range [-1, 1]
        audio_float = (samples_int16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
        inputs = self._hf_processor(audio_float, sampling_rate=self.SAMPLE_RATE, return_tensors="pt")
        input_values = inputs.input_values.to(self.DEVICE)
        attention_mask = None
        if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
            attention_mask = inputs.attention_mask.to(self.DEVICE)
        with torch.no_grad():
            logits = self._hf_model(input_values, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1)
            # Confidence estimation: average of max probability per timestep
            max_probs = probs.max(dim=-1).values
            conf = float(max_probs.mean().cpu().item()) if max_probs.numel() > 0 else 0.0
        text = self._hf_processor.batch_decode(pred_ids)[0].strip()
        # Normalize CTC output spacing/casing to match other engines
        text = " ".join(text.split())
        return TranscriptionSegment(text=text, confidence=conf)

    # ---------- Thread main ----------
    def run(self) -> None:
        if self.is_stopped() or self.isInterruptionRequested():
            return
        try:
            logger.info("Loading Wav2Vec2 model... (first run may download it)")
            self._lazy_load_model()
            if self.is_stopped() or self.isInterruptionRequested():
                return
        except Exception as e:
            msg = str(e)
            logger.error(msg)
            self.error.emit(msg)
            if hasattr(self, '_session_logger'):
                self._session_logger.log(f"ERROR: {e}")
            return

        try:
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

        with self._stream:
            logger.info(f"Recording with noise control... Press Stop to end. [profile={self._noise_profile_name}]")
            self.partial_text.emit("Listening…")
            self._emit_status_once("listening")

            segment_generator = self._noise_controller.collect_segments(padding_ms=self.PADDING_MS)

            while not self.is_stopped():
                if self.isInterruptionRequested():
                    break
                try:
                    segment = next(segment_generator)
                    if segment is None or segment.size == 0:
                        continue
                    segment_duration = segment.size / self.SAMPLE_RATE
                    if segment_duration < self.MIN_SPEECH_S:
                        continue

                    self._emit_status_once("processing")

                    segment = np.concatenate((self._number_sound, segment))

                    # Save the audio segment to file (for debugging/inspection)
                    try:
                        audio_saver = get_audio_saver()
                        audio_saver.save_segment(segment, self.SAMPLE_RATE)
                    except Exception as e:
                        logger.debug(f"Audio saver error: {e}")

                    try:
                        # Run model inference directly on the PCM data
                        result = self._infer(segment)
                    except Exception as e:
                        msg = f"Transcription error: {e}"
                        logger.error(msg)
                        self.error.emit(msg)
                        self._emit_status_once("listening")
                        continue

                    text_out = result.text
                    if not text_out:
                        self._emit_status_once("listening")
                        continue

                    logger.info(f"Raw transcription: {text_out}")

                    # --- Check for pause/resume commands ---
                    text_lower = text_out.lower()
                    if "wait" in text_lower:
                        self.pause()
                        self.final_text.emit("Waiting until 'start' is said.", result.confidence)
                        continue
                    elif "start" in text_lower:
                        self.resume()
                        continue

                    if self._paused:
                        logger.debug("Paused: ignoring transcription")
                        continue

                    # Apply ASR corrections and parsing
                    try:
                        from parser import FishParser, TextNormalizer, ParserResult
                        fish_parser = FishParser()
                        text_normalizer = TextNormalizer()

                        corrected_text = text_normalizer.apply_fish_asr_corrections(text_out)
                        if corrected_text != text_out.lower():
                            logger.info(f"After ASR corrections: {corrected_text}")

                        result_parse: ParserResult = fish_parser.parse_text(corrected_text)

                        if result_parse.species is not None:
                            self._last_fish_specie = result_parse.species
                            self.specie_detected.emit(result_parse.species)

                        if result_parse.length_cm is not None:
                            raw_val = float(result_parse.length_cm)
                            num_str = (f"{raw_val:.1f}").rstrip("0").rstrip(".")
                            formatted = f"{self._last_fish_specie} {num_str} cm"
                            logger.info(f">> {formatted}")
                            self.final_text.emit(formatted, max(0.0, min(1.0, result.confidence)))
                        else:
                            logger.info(f">> {corrected_text} (partial parse)")
                            self.final_text.emit(corrected_text, max(0.0, min(1.0, result.confidence)))
                    except Exception as e:
                        logger.error(f"Parser error: {e}")
                        logger.info(f">> {text_out}")
                        self.final_text.emit(text_out, max(0.0, min(1.0, result.confidence)))

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
        if hasattr(self, '_session_logger'):
            self._session_logger.log_end()
        if hasattr(self, '_session_log_sink_id'):
            import loguru
            loguru.logger.remove(self._session_log_sink_id)
