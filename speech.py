from __future__ import annotations

import os
import queue
import tempfile
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad
from faster_whisper import WhisperModel
from loguru import logger
from PyQt6.QtCore import QThread, pyqtSignal


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


class SpeechRecognizer(QThread):
    """
        Realtime CPU-only speech recognizer using webrtcvad + faster-whisper.

    """

    # PyQt signals
    partial_text = pyqtSignal(str)
    final_text = pyqtSignal(str, float)
    error = pyqtSignal(str)
    status_changed = pyqtSignal(str)

    # ===== CONFIG  =====
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    CHUNK_S: float = 1.0
    VAD_MODE: int = 3
    MIN_SPEECH_S: float = 0.5
    MAX_SEGMENT_S: float = 4.0
    FISH_PROMPT = (
        "This is a continuous conversation about fish species and their measurements. "
        "The user typically speaks in the format '<fish species> <number> <unit>'. "
        "Always prioritize fish species vocabulary over similar-sounding common words. "
        "If a word sounds like a fish name, bias towards the fish name. "
        "Common fish species include: trout, salmon, bass, sea bass, cod, mackerel, tuna, "
        "sardine, anchovy, snapper, grouper, bream, carp, pike, perch, haddock, halibut, "
        "flounder, mullet, herring. "
        "If you hear something like 'throughout' near a number or unit, transcribe it as 'trout'. "
        "Units are typically centimeters (cm) or millimeters (mm), 'cm' is preferred in the transcript. "
        "Examples: 'trout 15 centimeters' -> 'trout 15 cm', 'salmon 23 cm', 'bass 12 cm'."
    )


    MODEL_NAME: str = "base.en"
    DEVICE: str = "cpu"
    COMPUTE_TYPE: str = "int8"
    # =============================================

    def __init__(self) -> None:
        """Initialize recognizer state and resources."""
        super().__init__()
        self._stop_flag: bool = False
        self._audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self._model: Optional[WhisperModel] = None
        self._chunk_frames: int = int(self.SAMPLE_RATE * self.CHUNK_S)
        self._last_status_msg: Optional[str] = None

    # ---------- Public control ----------
    def stop(self) -> None:
        """Request the recognizer to stop and release resources."""
        self._stop_flag = True
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None
        except Exception as e:
            logger.debug(f"Error stopping input stream: {e}")
        # Unblock any pending queue get
        try:
            self._audio_queue.put_nowait(np.array([], dtype=np.int16))
        except Exception:
            pass

    def begin(self) -> None:
        """Reset internal state and start the recognizer thread.

        Safe to call after a previous stop; reinitializes the internal
        audio queue and status so the thread can run again cleanly.
        """
        # Reset flags/state for a clean restart
        self._stop_flag = False
        self._audio_queue = queue.Queue()
        self._last_status_msg = None
        if not self.isRunning():
            try:
                self.start()
            except Exception as e:
                logger.error(f"Failed to (re)start recognizer: {e}")

    def is_stopped(self) -> bool:
        """Return True if stop has been requested."""
        return self._stop_flag

    # ---------- Internal helpers ----------
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:  # type: ignore[override]
        """Sounddevice callback: push mono PCM16 frames into the queue."""
        if status:
            logger.debug(f"Audio status: {status}")
        try:
            pcm16 = (indata[:, 0] * 32767).astype(np.int16)
            self._audio_queue.put_nowait(pcm16)
        except Exception as e:
            logger.debug(f"Audio callback error: {e}")

    def _vad_collector(self, vad: webrtcvad.Vad, sample_rate: int, padding_ms: int = 400):
        """Yield voiced segments using webrtcvad with a small hangover padding.

        Parameters
        ----------
        vad: webrtcvad.Vad
            Initialized VAD instance.
        sample_rate: int
            Sample rate of the input stream.
        padding_ms: int
            Milliseconds of required trailing silence before yielding a segment.
        """
        frame_ms = 30
        frame_size = int(sample_rate * frame_ms / 1000)
        voiced: bool = False
        voiced_frames: list[np.ndarray] = []
        silence_frames: int = 0
        buf = np.array([], dtype=np.int16)

        while not self.is_stopped():
            data = self._audio_queue.get()
            if data is None:
                break
            if data.size == 0 and self.is_stopped():
                break
            buf = np.concatenate((buf, data)) if buf.size else data

            while buf.size >= frame_size:
                frame = buf[:frame_size]
                buf = buf[frame_size:]
                is_speech = vad.is_speech(frame.tobytes(), sample_rate)

                if is_speech:
                    voiced_frames.append(frame)
                    silence_frames = 0
                    voiced = True
                    self._emit_status_once("capturing")
                else:
                    if voiced:
                        silence_frames += 1
                        remaining_ms = max(0, padding_ms - silence_frames * frame_ms)
                        if remaining_ms > 0:
                            self._emit_status_once("finishing")
                        if (silence_frames * frame_ms) > padding_ms:
                            segment = np.concatenate(voiced_frames) if voiced_frames else np.array([], dtype=np.int16)
                            voiced_frames = []
                            voiced = False
                            self._emit_status_once("listening")
                            yield segment

            if voiced and voiced_frames:
                total_samples = sum(len(f) for f in voiced_frames)
                if total_samples / sample_rate >= self.MAX_SEGMENT_S:
                    segment = np.concatenate(voiced_frames)
                    voiced_frames = []
                    voiced = False
                    self._emit_status_once("finishing")
                    yield segment

    @staticmethod
    def _write_wav_bytes(samples_int16: np.ndarray, samplerate: int) -> str:
        """Write PCM16 samples to a temporary WAV file and return its path."""
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(path, samples_int16, samplerate, subtype="PCM_16")
        return path

    # ---------- Thread main ----------
    def run(self) -> None:  # type: ignore[override]
        """Run the realtime STT loop using testing_tt.py's configuration."""
        try:
            logger.info("Loading model... (first run will download it)")
            self._model = WhisperModel(self.MODEL_NAME, device=self.DEVICE, compute_type=self.COMPUTE_TYPE)
        except Exception as e:
            msg = f"Failed to load Whisper model: {e}"
            logger.error(msg)
            self.error.emit(msg)
            return

        vad = webrtcvad.Vad(self.VAD_MODE)
        collector = self._vad_collector(vad, self.SAMPLE_RATE)

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

        # Start listening
        with self._stream:
            logger.info("Recording... Press Stop to end.")
            self.partial_text.emit("Listening…")
            self._emit_status_once("listening")

            while not self.is_stopped():
                try:
                    segment = next(collector)
                except StopIteration:
                    break
                except Exception as e:
                    logger.debug(f"Collector error: {e}")
                    break

                if segment is None or segment.size == 0:
                    continue
                if (segment.size / self.SAMPLE_RATE) < self.MIN_SPEECH_S:
                    continue

                wav_path = self._write_wav_bytes(segment, self.SAMPLE_RATE)

                try:
                    assert self._model is not None
                    segments, info = self._model.transcribe(
                        wav_path,
                        beam_size=5,
                        language="en",
                        condition_on_previous_text=True,
                        initial_prompt=self.FISH_PROMPT,
                    )
                except Exception as e:
                    msg = f"Transcription error: {e}"
                    logger.error(msg)
                    self.error.emit(msg)
                    try:
                        os.remove(wav_path)
                    except Exception:
                        pass
                    continue


                text_out = "".join(seg.text + " " for seg in segments).strip()
                if text_out:
                    logger.info(f"Raw transcription: {text_out}")  # Debug log
                    
                    # Apply ASR corrections before parsing
                    try:
                        from parser import parse_text, apply_fish_asr_corrections 
                        
                        # First apply fish-specific ASR corrections
                        corrected_text = apply_fish_asr_corrections(text_out)
                        if corrected_text != text_out.lower():
                            logger.info(f"After ASR corrections: {corrected_text}")
                        
                        result = parse_text(corrected_text)
                        if result.species and result.length_cm is not None:
                            # Normalize numeric formatting: keep one decimal if needed
                            raw_val = float(result.length_cm)
                            num_str = (f"{raw_val:.1f}").rstrip("0").rstrip(".")
                            formatted = f"{result.species} {num_str} cm"
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

                try:
                    os.remove(wav_path)
                except Exception:
                    pass

            # Small sleep to yield
            time.sleep(0.05)
        # On exit
        self._emit_status_once("stopped")

    # ---------- Status helper ----------
    def _emit_status_once(self, message: str) -> None:
        """Emit status_changed only when the message changes to avoid flooding UI."""
        if message != self._last_status_msg:
            self._last_status_msg = message
            try:
                self.status_changed.emit(message)
            except Exception:
                pass
