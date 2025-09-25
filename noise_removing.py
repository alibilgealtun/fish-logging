from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import soundfile as sf


@dataclass
class SuppressorConfig:
    sample_rate: int = 16000
    frame_ms: int = 30  # must match VAD frame in NoiseController for best results
    fft_size: Optional[int] = None  # if None, next pow2 of frame_len
    noise_update_alpha: float = 0.98  # EMA factor for noise PSD update
    dd_beta: float = 0.98             # decision-directed smoothing for a priori SNR
    gain_floor: float = 0.05          # avoid musical noise (-26 dB)
    speech_band: Tuple[int, int] = (300, 3800)  # lightly protect this band
    speech_band_boost: float = 0.10   # reduce suppression inside speech band


class AdaptiveNoiseSuppressor:
    """
    Lightweight spectral noise suppressor using Wiener filtering with
    decision-directed SNR estimation. Designed to be deterministic and
    dependency-light for real-time use.

    Use enhance_frame() on PCM16 frames. Optionally, pass is_speech flag (from VAD)
    so the internal noise estimator updates only on non-speech frames.
    """

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

        # Precompute index range for speech band protection
        self._speech_bin_lo = int(self.cfg.speech_band[0] * self._fft_size / self.cfg.sample_rate)
        self._speech_bin_hi = int(self.cfg.speech_band[1] * self._fft_size / self.cfg.sample_rate)
        self._speech_bin_hi = max(self._speech_bin_lo + 1, self._speech_bin_hi)

    def reset(self) -> None:
        self._noise_psd = None
        self._prev_gain = None

    def _ensure_frame_length(self, frame: np.ndarray) -> np.ndarray:
        # Ensure mono PCM16 length == frame_len (pad or trim)
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

    def enhance_frame(self, frame_pcm16: np.ndarray, is_speech: Optional[bool] = None) -> np.ndarray:
        """Enhance a single PCM16 frame and return PCM16 frame of same length."""
        f = self._ensure_frame_length(frame_pcm16)
        x = f.astype(np.float32) / 32767.0  # [-1,1]

        # Use rectangular window to avoid OLA complexity in streaming path
        # STFT
        X = np.fft.rfft(x, n=self._fft_size)
        mag2 = (X.real * X.real + X.imag * X.imag)  # |X|^2

        # Initialize / update noise PSD estimate
        if self._noise_psd is None:
            self._noise_psd = mag2.copy()
        elif is_speech is False:
            a = self.cfg.noise_update_alpha
            self._noise_psd = a * self._noise_psd + (1.0 - a) * mag2

        noise_psd = self._noise_psd if self._noise_psd is not None else mag2

        # A posteriori SNR
        gamma = mag2 / np.maximum(noise_psd, self._eps)
        # Decision-directed a priori SNR
        if self._prev_gain is None:
            xi = np.maximum(gamma - 1.0, 0.0)
        else:
            beta = self.cfg.dd_beta
            xi_prev = (self._prev_gain**2) * mag2 / np.maximum(noise_psd, self._eps)
            xi = beta * xi_prev + (1.0 - beta) * np.maximum(gamma - 1.0, 0.0)

        # Wiener filter gain
        G = xi / (1.0 + xi)

        # Protect typical speech band by reducing suppression slightly
        lo, hi = self._speech_bin_lo, self._speech_bin_hi
        if hi > lo and hi <= len(G):
            # Increase G towards 1 by speech_band_boost
            G_speech = G[lo:hi]
            G[lo:hi] = np.minimum(1.0, G_speech + self.cfg.speech_band_boost * (1.0 - G_speech))

        # Apply gain floor
        G = np.maximum(G, self.cfg.gain_floor)
        self._prev_gain = G

        # Apply gain to spectrum and reconstruct
        Y = G * X
        y = np.fft.irfft(Y, n=self._fft_size).real
        y = y[: self._frame_len]

        # Clip and convert back to int16
        y = np.clip(y, -1.0, 1.0)
        y_i16 = np.asarray(y * 32767.0, dtype=np.int16)
        return y_i16

    # ------------- WAV helpers -------------
    def process_wav(self, input_path: str, output_path: Optional[str] = None) -> Tuple[np.ndarray, int]:
        """
        Denoise a WAV file using the same frame-by-frame algorithm as realtime:
        - high-pass filter (<100Hz)
        - VAD-gated noise PSD updates
        - spectral suppression
        Returns (audio_int16, sample_rate). If output_path is provided, writes a WAV.
        """
        audio, sr = sf.read(input_path, dtype="int16")
        if audio.ndim > 1:
            audio = audio[:, 0]
        if sr != self.cfg.sample_rate:
            from scipy.signal import resample
            target_len = int(len(audio) * (self.cfg.sample_rate / sr))
            audio_f = audio.astype(np.float32) / 32767.0
            resampled = resample(audio_f, target_len)
            resampled = np.asarray(resampled, dtype=np.float32)
            audio = np.asarray(resampled * 32767.0, dtype=np.int16)
            sr = self.cfg.sample_rate

        # High-pass filter design (same as realtime)
        from scipy.signal import butter, lfilter
        hp_b, hp_a = butter(N=6, Wn=100 / (sr / 2), btype="highpass")

        # VAD for gating noise updates
        import webrtcvad
        vad = webrtcvad.Vad(2)

        out = np.empty_like(audio)
        self.reset()
        n = self._frame_len
        total = len(audio)

        # Warmup: treat first ~200ms as noise to seed the estimator
        warmup_frames = max(1, int(200 / self.cfg.frame_ms))
        idx = 0
        for _ in range(warmup_frames):
            if idx >= total:
                break
            frame = audio[idx : idx + n]
            frame = self._ensure_frame_length(frame)
            # Apply HPF before suppression
            frame_f = frame.astype(np.float32) / 32767.0
            frame_hp = lfilter(hp_b, hp_a, frame_f)
            frame_hp_i16 = np.asarray(frame_hp * 32767.0, dtype=np.int16)
            _ = self.enhance_frame(frame_hp_i16, is_speech=False)
            idx += n

        # Process frames with VAD gating
        idx = 0
        write_pos = 0
        while idx < total:
            frame = audio[idx : idx + n]
            is_last = len(frame) < n
            frame = self._ensure_frame_length(frame)

            # High-pass filter
            frame_f = frame.astype(np.float32) / 32767.0
            frame_hp = lfilter(hp_b, hp_a, frame_f)
            frame_hp_i16 = np.asarray(frame_hp * 32767.0, dtype=np.int16)

            # VAD expects 10/20/30 ms frames of PCM16 bytes
            is_speech = vad.is_speech(frame_hp_i16.tobytes(), sr)

            # Suppress with VAD hint
            y = self.enhance_frame(frame_hp_i16, is_speech=is_speech)

            end = write_pos + (len(frame) if not is_last else min(n, total - idx))
            out[write_pos:end] = y[: end - write_pos]
            write_pos = end
            idx += n

        if output_path:
            sf.write(output_path, out, sr, subtype="PCM_16")
        return out, sr


def process_wav(input_path: str, output_path: Optional[str] = None, sample_rate: int = 16000) -> str:
    """Convenience wrapper to denoise a WAV file and write output. Returns output path."""
    cfg = SuppressorConfig(sample_rate=sample_rate)
    sup = AdaptiveNoiseSuppressor(cfg)
    out_audio, sr = sup.process_wav(input_path, output_path)
    if output_path is None:
        import os
        root, ext = os.path.splitext(input_path)
        output_path = root + "_clean.wav"
        sf.write(output_path, out_audio, sr, subtype="PCM_16")
    return output_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Adaptive noise suppression for WAV or stdin")
    p.add_argument("wav", nargs="?", default="test.WAV", help="Input WAV file (default: test.WAV)")
    p.add_argument("-o", "--output", help="Output WAV path (default: <input>_clean.wav)")
    p.add_argument("-r", "--rate", type=int, default=16000, help="Target sample rate for processing")
    args = p.parse_args()
    out = process_wav(args.wav, args.output, sample_rate=args.rate)
    print(f"Wrote denoised file: {out}")
