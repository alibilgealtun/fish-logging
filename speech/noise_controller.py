from __future__ import annotations

import numpy as np
import queue
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SuppressorConfig:
    sample_rate: int = 16000
    frame_ms: int = 30
    fft_size: Optional[int] = None
    noise_update_alpha: float = 0.98
    dd_beta: float = 0.98
    gain_floor: float = 0.05
    speech_band: Tuple[int, int] = (300, 3800)
    speech_band_boost: float = 0.10
    enable_loudness_gate: bool = True
    gate_threshold_db: float = 8.0
    gate_max_att_db: float = 35.0
    gate_softness: float = 0.6
    gate_attack: float = 0.3
    gate_release: float = 0.05
    peak_decay: float = 0.995


class AdaptiveNoiseSuppressor:
    """Frame-by-frame spectral noise suppressor with loudness gate."""
    def __init__(self, cfg: SuppressorConfig | None = None) -> None:
        self.cfg = cfg or SuppressorConfig()
        self._frame_len = int(self.cfg.sample_rate * self.cfg.frame_ms / 1000)
        if self.cfg.fft_size is None:
            n = 1
            while n < self._frame_len:
                n <<= 1
            self._fft_size = n
        else:
            self._fft_size = int(self.cfg.fft_size)
        self._noise_psd: Optional[np.ndarray] = None
        self._prev_gain: Optional[np.ndarray] = None
        self._eps = 1e-10
        self._speech_bin_lo = int(self.cfg.speech_band[0] * self._fft_size / self.cfg.sample_rate)
        self._speech_bin_hi = int(self.cfg.speech_band[1] * self._fft_size / self.cfg.sample_rate)
        self._speech_bin_hi = max(self._speech_bin_lo + 1, self._speech_bin_hi)
        self._peak_rms: float = 0.0
        self._have_speech_peak: bool = False
        self._extra_gain_smoothed: float = 1.0

    def reset(self) -> None:
        self._noise_psd = None
        self._prev_gain = None
        self._peak_rms = 0.0
        self._have_speech_peak = False
        self._extra_gain_smoothed = 1.0

    def _ensure_frame_length(self, frame: np.ndarray) -> np.ndarray:
        f = frame
        if f.dtype != np.int16:
            f = f.astype(np.int16)
        if f.ndim > 1:
            f = f[:, 0]
        if len(f) < self._frame_len:
            pad = self._frame_len - len(f)
            if pad > 0:
                f = np.pad(f, (0, pad), mode="constant")
        elif len(f) > self._frame_len:
            f = f[: self._frame_len]
        return f

    def _compute_frame_rms(self, x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(x * x) + 1e-12))

    def _update_peak_rms(self, frame_rms: float, is_speech: Optional[bool]) -> None:
        if not self.cfg.enable_loudness_gate:
            return
        if is_speech is True:
            if not self._have_speech_peak:
                self._peak_rms = frame_rms
                self._have_speech_peak = True
            else:
                decayed = self._peak_rms * self.cfg.peak_decay
                self._peak_rms = max(decayed, frame_rms)
        else:
            self._peak_rms *= self.cfg.peak_decay

    def _compute_extra_gain(self, frame_rms: float) -> float:
        if not self.cfg.enable_loudness_gate or not self._have_speech_peak:
            return 1.0
        peak = max(self._peak_rms, 1e-8)
        gap_db = 20.0 * np.log10(max(frame_rms, 1e-8) / peak)
        if gap_db >= -self.cfg.gate_threshold_db:
            target_gain = 1.0
        else:
            over = (-self.cfg.gate_threshold_db) - gap_db
            att_db = min(self.cfg.gate_max_att_db, over * (0.5 + 0.5 * self.cfg.gate_softness))
            target_gain = 10.0 ** (-att_db / 20.0)
        g_prev = self._extra_gain_smoothed
        alpha = self.cfg.gate_attack if target_gain < g_prev else self.cfg.gate_release
        g = (1.0 - alpha) * g_prev + alpha * target_gain
        self._extra_gain_smoothed = float(np.clip(g, 10.0 ** (-self.cfg.gate_max_att_db / 20.0), 1.0))
        return self._extra_gain_smoothed

    def enhance_frame(self, frame_pcm16: np.ndarray, is_speech: Optional[bool] = None) -> np.ndarray:
        f = self._ensure_frame_length(frame_pcm16)
        x = f.astype(np.float32) / 32767.0
        frame_rms = self._compute_frame_rms(x)
        self._update_peak_rms(frame_rms, is_speech)
        X = np.fft.rfft(x, n=self._fft_size)
        mag2 = (X.real * X.real + X.imag * X.imag)
        if self._noise_psd is None:
            self._noise_psd = mag2.copy()
        elif is_speech is False:
            a = self.cfg.noise_update_alpha
            self._noise_psd = a * self._noise_psd + (1.0 - a) * mag2
        noise_psd = self._noise_psd if self._noise_psd is not None else mag2
        gamma = mag2 / np.maximum(noise_psd, self._eps)
        if self._prev_gain is None:
            xi = np.maximum(gamma - 1.0, 0.0)
        else:
            beta = self.cfg.dd_beta
            xi_prev = (self._prev_gain**2) * mag2 / np.maximum(noise_psd, self._eps)
            xi = beta * xi_prev + (1.0 - beta) * np.maximum(gamma - 1.0, 0.0)
        G = xi / (1.0 + xi)
        lo, hi = self._speech_bin_lo, self._speech_bin_hi
        if hi > lo and hi <= len(G):
            G_speech = G[lo:hi]
            G[lo:hi] = np.minimum(1.0, G_speech + self.cfg.speech_band_boost * (1.0 - G_speech))
        G = np.maximum(G, self.cfg.gain_floor)
        self._prev_gain = G
        Y = G * X
        y = np.fft.irfft(Y, n=self._fft_size).real
        y = y[: self._frame_len]
        if self.cfg.enable_loudness_gate:
            extra_gain = self._compute_extra_gain(frame_rms)
            if extra_gain < 1.0:
                y *= extra_gain
        y = np.clip(y, -1.0, 1.0)
        y_i16 = np.asarray(y * 32767.0, dtype=np.int16)
        return y_i16


class NoiseController:
    """HPF + VAD + adaptive suppression + segmentation for real-time."""

    def __init__(self, sample_rate: int = 16000, vad_mode: int = 3,
                 min_speech_s: float = 0.5, max_segment_s: float = 4.0,
                 suppressor_config: SuppressorConfig | None = None):
        # Localize heavy deps to constructor
        import webrtcvad  # type: ignore
        from scipy.signal import butter  # type: ignore

        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(vad_mode)
        self.min_speech_s = min_speech_s
        self.max_segment_s = max_segment_s

        _ba = butter(N=6, Wn=100 / (sample_rate / 2), btype="highpass")
        self.hp_b, self.hp_a = _ba[0], _ba[1]

        self._suppressor = AdaptiveNoiseSuppressor(
            suppressor_config or SuppressorConfig(sample_rate=sample_rate, frame_ms=30)
        )

        self._audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()

    def push_audio(self, pcm16: np.ndarray) -> None:
        if pcm16.size > 0:
            self._audio_queue.put_nowait(pcm16)

    def stop(self) -> None:
        self._audio_queue.put_nowait(np.array([], dtype=np.int16))

    def _highpass_filter(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return audio
        from scipy.signal import lfilter  # type: ignore
        audio_float = audio.astype(np.float32) / 32767.0
        filtered = lfilter(self.hp_b, self.hp_a, audio_float)
        return np.asarray(filtered * 32767.0, dtype=np.int16)

    def collect_segments(self, padding_ms: int = 800):
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
                frame_hp = self._highpass_filter(frame)
                is_speech = self.vad.is_speech(frame_hp.tobytes(), self.sample_rate)
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
                if voiced and voiced_frames:
                    total_samples = sum(len(f) for f in voiced_frames)
                    if total_samples / self.sample_rate >= self.max_segment_s:
                        segment = np.concatenate(voiced_frames)
                        voiced_frames = []
                        voiced = False
                        yield segment
