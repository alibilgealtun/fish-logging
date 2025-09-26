"""Adaptive spectral noise suppressor and configuration.

This module exposes:
- SuppressorConfig: Tunable parameters for the spectral suppressor and loudness gate
- AdaptiveNoiseSuppressor: Lightweight frequency-domain suppressor (decision-directed)

Notes
- Frames are processed in 30 ms by default to satisfy WebRTC VAD constraints (10/20/30 ms).
- All processing assumes 16 kHz mono PCM16 unless configured otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class SuppressorConfig:
    """Configuration for AdaptiveNoiseSuppressor.

    Attributes
    - sample_rate: Audio sampling rate in Hz.
    - frame_ms: Frame size in milliseconds (commonly 10/20/30). Impacts latency.
    - fft_size: Optional FFT size; if None, next power-of-two >= frame length is used.
    - noise_update_alpha: EMA coefficient for noise PSD update during non-speech.
    - dd_beta: Decision-directed smoothing for a priori SNR estimation.
    - gain_floor: Minimum spectral gain to avoid musical noise and excessive attenuation.
    - speech_band: Frequency band (Hz) where speech is emphasized.
    - speech_band_boost: Linear blend factor to gently boost speech-band gains.
    - enable_loudness_gate: Enables a slow-acting loudness gate based on speech peak RMS.
    - gate_threshold_db: Level (dB below peak speech) below which gate attenuation is applied.
    - gate_max_att_db: Maximum attenuation (dB) applied by the gate.
    - gate_softness: Shapes how quickly attenuation increases once under threshold.
    - gate_attack: Smoothing factor when gain decreases (faster = more responsive).
    - gate_release: Smoothing factor when gain increases (higher = slower recovery).
    - peak_decay: Exponential decay for the tracked speech peak RMS.
    """

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
    """Frame-by-frame spectral noise suppressor with loudness gate.

    This component performs:
    - Short-time FFT per frame
    - Decision-directed a priori SNR estimation
    - Wiener-like gain computation with a configurable gain floor
    - Optional speech-band emphasis
    - Optional slow loudness gate to avoid amplifying background-only frames

    The algorithm is intentionally lightweight for real-time usage and avoids
    model-based approaches to reduce dependencies and latency.
    """

    def __init__(self, cfg: SuppressorConfig | None = None) -> None:
        """Initialize the suppressor with the provided config.

        If ``fft_size`` is not set, uses the next power-of-two >= frame length for
        efficient FFT computation.
        """
        self.cfg = cfg or SuppressorConfig()
        self._frame_len = int(self.cfg.sample_rate * self.cfg.frame_ms / 1000)
        if self.cfg.fft_size is None:
            # Next power of two for efficient FFTs
            n = 1
            while n < self._frame_len:
                n <<= 1
            self._fft_size = n
        else:
            self._fft_size = int(self.cfg.fft_size)

        # Adaptive state (initialized on first frames)
        self._noise_psd: Optional[np.ndarray] = None
        self._prev_gain: Optional[np.ndarray] = None
        self._eps = 1e-10

        # Map speech band (Hz) to rFFT bin indices (inclusive-exclusive [lo:hi])
        self._speech_bin_lo = int(self.cfg.speech_band[0] * self._fft_size / self.cfg.sample_rate)
        self._speech_bin_hi = int(self.cfg.speech_band[1] * self._fft_size / self.cfg.sample_rate)
        self._speech_bin_hi = max(self._speech_bin_lo + 1, self._speech_bin_hi)

        # Loudness gate tracking
        self._peak_rms: float = 0.0
        self._have_speech_peak: bool = False
        self._extra_gain_smoothed: float = 1.0

    def reset(self) -> None:
        """Reset adaptive state (noise PSD, gain history, loudness tracking)."""
        self._noise_psd = None
        self._prev_gain = None
        self._peak_rms = 0.0
        self._have_speech_peak = False
        self._extra_gain_smoothed = 1.0

    def _ensure_frame_length(self, frame: np.ndarray) -> np.ndarray:
        """Ensure input is mono PCM16 at configured frame length.

        - Converts dtype to int16 if needed.
        - Takes first channel if multi-channel.
        - Pads with zeros or truncates to exact frame length.
        """
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
        """Root-mean-square of a normalized float frame with numerical guard."""
        return float(np.sqrt(np.mean(x * x) + 1e-12))

    def _update_peak_rms(self, frame_rms: float, is_speech: Optional[bool]) -> None:
        """Update the tracked speech peak RMS used by the loudness gate.

        - During speech frames: track a decaying maximum.
        - During non-speech frames: decay toward zero.
        """
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
        """Compute slow-acting gain from the loudness gate.

        The gate attenuates frames that are far below the tracked speech peak to
        avoid lifting stationary background when no speech is present.
        """
        if not self.cfg.enable_loudness_gate or not self._have_speech_peak:
            return 1.0
        peak = max(self._peak_rms, 1e-8)
        gap_db = 20.0 * np.log10(max(frame_rms, 1e-8) / peak)
        if gap_db >= -self.cfg.gate_threshold_db:
            target_gain = 1.0
        else:
            # Distance below threshold determines attenuation up to gate_max_att_db
            over = (-self.cfg.gate_threshold_db) - gap_db
            att_db = min(self.cfg.gate_max_att_db, over * (0.5 + 0.5 * self.cfg.gate_softness))
            target_gain = 10.0 ** (-att_db / 20.0)
        g_prev = self._extra_gain_smoothed
        # Attack when reducing gain; release when increasing (smoother recovery)
        alpha = self.cfg.gate_attack if target_gain < g_prev else self.cfg.gate_release
        g = (1.0 - alpha) * g_prev + alpha * target_gain
        self._extra_gain_smoothed = float(np.clip(g, 10.0 ** (-self.cfg.gate_max_att_db / 20.0), 1.0))
        return self._extra_gain_smoothed

    def enhance_frame(self, frame_pcm16: np.ndarray, is_speech: Optional[bool] = None) -> np.ndarray:
        """Suppress noise on a single PCM16 frame.

        Parameters
        - frame_pcm16: 1-D int16 array of length ~= frame_len (will be fit exactly)
        - is_speech: Optional VAD decision for the frame to guide noise update

        Returns
        - Enhanced frame as int16 array with the same length as the configured frame
        """
        f = self._ensure_frame_length(frame_pcm16)
        x = f.astype(np.float32) / 32767.0

        frame_rms = self._compute_frame_rms(x)
        self._update_peak_rms(frame_rms, is_speech)

        X = np.fft.rfft(x, n=self._fft_size)
        mag2 = (X.real * X.real + X.imag * X.imag)

        # Initialize or update noise power spectral density (non-speech only)
        if self._noise_psd is None:
            self._noise_psd = mag2.copy()
        elif is_speech is False:
            a = self.cfg.noise_update_alpha
            self._noise_psd = a * self._noise_psd + (1.0 - a) * mag2

        noise_psd = self._noise_psd if self._noise_psd is not None else mag2
        gamma = mag2 / np.maximum(noise_psd, self._eps)  # a posteriori SNR

        # Decision-directed a priori SNR estimate
        if self._prev_gain is None:
            xi = np.maximum(gamma - 1.0, 0.0)
        else:
            beta = self.cfg.dd_beta
            xi_prev = (self._prev_gain**2) * mag2 / np.maximum(noise_psd, self._eps)
            xi = beta * xi_prev + (1.0 - beta) * np.maximum(gamma - 1.0, 0.0)

        # Wiener-like gain function with floor
        G = xi / (1.0 + xi)

        # Gentle emphasis in the typical speech band
        lo, hi = self._speech_bin_lo, self._speech_bin_hi
        if hi > lo and hi <= len(G):
            G_speech = G[lo:hi]
            G[lo:hi] = np.minimum(1.0, G_speech + self.cfg.speech_band_boost * (1.0 - G_speech))

        G = np.maximum(G, self.cfg.gain_floor)
        self._prev_gain = G

        Y = G * X
        y = np.fft.irfft(Y, n=self._fft_size).real
        y = y[: self._frame_len]

        # Apply loudness gate after IFFT (time-domain attenuation)
        if self.cfg.enable_loudness_gate:
            extra_gain = self._compute_extra_gain(frame_rms)
            if extra_gain < 1.0:
                y *= extra_gain

        # Back to int16 PCM
        y = np.clip(y, -1.0, 1.0)
        y_i16 = np.asarray(y * 32767.0, dtype=np.int16)
        return y_i16

