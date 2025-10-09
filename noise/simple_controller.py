"""Simple noise controller without adaptive suppression.

This is the original noise controller that uses only high-pass filtering + VAD,
without spectral suppression. Some users find this works better for their voice
in normal conditions.
"""
from __future__ import annotations

import numpy as np
import queue
import webrtcvad
from scipy.signal import butter, lfilter


class SimpleNoiseController:
    """
    Handles noise suppression, high-pass filtering, and VAD-based speech
    segmentation for real-time speech recognition.

    This simpler controller uses only HPF + VAD without adaptive spectral suppression.
    """

    def __init__(self, sample_rate: int = 16000, vad_mode: int = 3,
                 min_speech_s: float = 0.5, max_segment_s: float = 4.0):
        """
        Parameters
        ----------
        sample_rate : int
            Input audio sample rate (Hz). Must be 8000, 16000, 32000, or 48000 for VAD.
        vad_mode : int
            Aggressiveness of VAD: 0 (least) → 3 (most aggressive).
        min_speech_s : float
            Minimum length of a valid speech segment (seconds).
        max_segment_s : float
            Maximum length before forcing segment cut (seconds).
        """
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(vad_mode)

        self.min_speech_s = min_speech_s
        self.max_segment_s = max_segment_s

        # High-pass filter (kill engine rumble <100 Hz)
        self.hp_b, self.hp_a = butter(
            N=6, Wn=100 / (sample_rate / 2), btype="highpass"
        )

        # Queue for streaming audio from sounddevice
        self._audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()

    # ---------------- Queue I/O ----------------
    def push_audio(self, pcm16: np.ndarray) -> None:
        """Push raw PCM16 audio into the controller's buffer."""
        if pcm16.size > 0:
            self._audio_queue.put_nowait(pcm16)

    def stop(self) -> None:
        """Unblock collector loop."""
        self._audio_queue.put_nowait(np.array([], dtype=np.int16))

    # ---------------- Processing ----------------
    def _highpass_filter(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return audio
        audio_float = audio.astype(np.float32) / 32767.0
        filtered = lfilter(self.hp_b, self.hp_a, audio_float)
        return (filtered * 32767).astype(np.int16)

    def collect_segments(self, padding_ms: int = 800):
        """
        Generator that yields speech segments after noise control.

        Parameters
        ----------
        padding_ms : int
            How long silence must last (ms) before cutting a segment.
        """
        frame_ms = 30
        frame_size = int(self.sample_rate * frame_ms / 1000)

        voiced = False
        voiced_frames: list[np.ndarray] = []
        silence_frames = 0
        buf = np.array([], dtype=np.int16)

        while True:
            try:
                data = self._audio_queue.get()
            except Exception:
                break

            if data.size == 0:
                break

            buf = np.concatenate((buf, data)) if buf.size else data

            while buf.size >= frame_size:
                frame = buf[:frame_size]
                buf = buf[frame_size:]

                # Step 1: filter engine rumble
                frame = self._highpass_filter(frame)

                # Step 2: check with VAD
                is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)

                if is_speech:
                    voiced_frames.append(frame)
                    silence_frames = 0
                    voiced = True
                else:
                    if voiced:
                        silence_frames += 1
                        if (silence_frames * frame_ms) > padding_ms:
                            segment = np.concatenate(voiced_frames) if voiced_frames else np.array([], dtype=np.int16)
                            voiced_frames = []
                            voiced = False
                            # Enforce min length
                            if segment.size / self.sample_rate >= self.min_speech_s:
                                yield segment

                # Enforce max length
                if voiced and voiced_frames:
                    total_samples = sum(len(f) for f in voiced_frames)
                    if total_samples / self.sample_rate >= self.max_segment_s:
                        segment = np.concatenate(voiced_frames)
                        voiced_frames = []
                        voiced = False
                        yield segment

