"""Real-time audio pipeline: HPF + VAD + adaptive suppression + segmentation.

This module provides the NoiseController class, which wires together:
- High-pass filtering to remove DC/rumble
- WebRTC VAD operating on 30 ms frames
- AdaptiveNoiseSuppressor (see .suppressor) applied per frame
- Segmentation that yields voiced PCM16 chunks with silence padding

Notes
- Frames are 30 ms to satisfy WebRTC VAD frame constraints (10/20/30 ms).
- All processing assumes 16 kHz mono PCM16 unless configured otherwise.
"""

from __future__ import annotations

from typing import Generator
import queue
import numpy as np

from .suppressor import AdaptiveNoiseSuppressor, SuppressorConfig


class NoiseController:
    """HPF + VAD + adaptive suppression + segmentation for real-time.

    Pipeline
    1) High-pass filter (remove DC/rumble)
    2) WebRTC VAD on 30 ms frames (10/20/30 ms supported by WebRTC)
    3) AdaptiveNoiseSuppressor per frame
    4) Buffer voiced frames and yield segments using padding and length limits
    """

    def __init__(self, sample_rate: int = 16000, vad_mode: int = 3,
                 min_speech_s: float = 0.5, max_segment_s: float = 4.0,
                 suppressor_config: SuppressorConfig | None = None):
        """Construct the controller and prepare DSP/VAD components.

        Parameters
        - sample_rate: Input/output sample rate (Hz)
        - vad_mode: WebRTC VAD aggressiveness (0..3, higher = more aggressive)
        - min_speech_s: Minimum segment duration to emit (seconds)
        - max_segment_s: Hard cap on a single segment length (seconds)
        - suppressor_config: Optional config for the suppressor (inherits sample_rate)
        """
        # Localize heavy deps to constructor to reduce import time for modules
        import webrtcvad  # type: ignore
        from scipy.signal import butter  # type: ignore

        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(vad_mode)
        self.min_speech_s = min_speech_s
        self.max_segment_s = max_segment_s

        # 6th-order HPF @ 100 Hz to remove low-frequency noise and DC
        _ba = butter(N=6, Wn=100 / (sample_rate / 2), btype="highpass")
        self.hp_b, self.hp_a = _ba[0], _ba[1]

        self._suppressor = AdaptiveNoiseSuppressor(
            suppressor_config or SuppressorConfig(sample_rate=sample_rate, frame_ms=30)
        )

        # Thread-safe queue of PCM16 chunks to be consumed by collect_segments()
        self._audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()

    def push_audio(self, pcm16: np.ndarray) -> None:
        """Enqueue a new PCM16 chunk for processing (non-blocking)."""
        if pcm16.size > 0:
            self._audio_queue.put_nowait(pcm16)

    def stop(self) -> None:
        """Signal end-of-stream to collect_segments() by pushing an empty array."""
        self._audio_queue.put_nowait(np.array([], dtype=np.int16))

    def _highpass_filter(self, audio: np.ndarray) -> np.ndarray:
        """Apply the configured high-pass filter in float domain, return PCM16."""
        if audio.size == 0:
            return audio
        from scipy.signal import lfilter  # type: ignore
        audio_float = audio.astype(np.float32) / 32767.0
        filtered = lfilter(self.hp_b, self.hp_a, audio_float)
        return np.asarray(filtered * 32767.0, dtype=np.int16)

    def collect_segments(self, padding_ms: int = 800) -> Generator[np.ndarray, None, None]:
        """Yield speech segments as PCM16 arrays.

        Segmentation rules
        - Frames are 30 ms to satisfy WebRTC VAD constraints.
        - While in a voiced region, keep appending frames. When VAD indicates
          sustained silence exceeding padding_ms, emit the segment if its
          duration >= min_speech_s.
        - If a voiced run grows beyond max_segment_s, emit early to bound latency.

        Parameters
        - padding_ms: Required trailing silence (ms) to close a segment
        """
        frame_ms = 30  # Keep aligned with VAD-supported frame sizes (10/20/30 ms)
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
                # End-of-stream sentinel
                break

            # Accumulate and process in fixed-size frames
            buf = np.concatenate((buf, data)) if buf.size else data
            while buf.size >= frame_size:
                frame = buf[:frame_size]
                buf = buf[frame_size:]

                # HPF before VAD improves robustness; use same frame size as VAD
                frame_hp = self._highpass_filter(frame)
                is_speech = self.vad.is_speech(frame_hp.tobytes(), self.sample_rate)

                # Apply spectral suppression guided by VAD decision
                frame_dn = self._suppressor.enhance_frame(frame_hp, is_speech=is_speech)

                if is_speech:
                    voiced_frames.append(frame_dn)
                    silence_frames = 0
                    voiced = True
                else:
                    if voiced:
                        silence_frames += 1
                        if (silence_frames * frame_ms) > padding_ms:
                            segment = np.concatenate(voiced_frames) if voiced_frames else np.array([], dtype=np.int16)
                            voiced_frames = []
                            voiced = False
                            if segment.size / self.sample_rate >= self.min_speech_s:
                                yield segment

                # Safety: cap very long segments to bound latency
                if voiced and voiced_frames:
                    total_samples = sum(len(f) for f in voiced_frames)
                    if total_samples / self.sample_rate >= self.max_segment_s:
                        segment = np.concatenate(voiced_frames)
                        voiced_frames = []
                        voiced = False
                        yield segment

