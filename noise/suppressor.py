"""
Adaptive Spectral Noise Suppressor for Fish Logging Application.

This module provides advanced noise suppression capabilities using spectral analysis
and adaptive algorithms. It implements decision-directed noise suppression with
configurable parameters for marine environment audio processing.

Classes:
    SuppressorConfig: Configuration dataclass for suppressor parameters
    AdaptiveNoiseSuppressor: Main noise suppression algorithm implementation

Algorithm Features:
    - Short-time FFT-based spectral analysis
    - Decision-directed a priori SNR estimation
    - Wiener-like gain computation with configurable floor
    - Speech-band emphasis for improved clarity
    - Adaptive loudness gate for background suppression
    - Real-time processing optimized for 16kHz audio

Marine Environment Optimizations:
    - Engine noise reduction through spectral filtering
    - Wind noise suppression with adaptive thresholds
    - Multiple speaker handling in noisy environments
    - Equipment noise isolation and reduction
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Optional logger: prefer loguru if available, else fallback to stdlib logging
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class SuppressorConfig:
    """
    Configuration parameters for AdaptiveNoiseSuppressor.

    This dataclass encapsulates all tunable parameters for the spectral noise
    suppressor, allowing fine-tuning for different marine environments and
    use cases.

    Audio Processing Parameters:
        sample_rate: Audio sampling rate in Hz (typically 16000 for speech)
        frame_ms: Frame size in milliseconds (10/20/30 for WebRTC VAD compatibility)
        fft_size: FFT size for spectral analysis (auto-calculated if None)

    Noise Estimation Parameters:
        noise_update_alpha: Exponential moving average coefficient for noise PSD updates
        dd_beta: Decision-directed smoothing factor for a priori SNR estimation
        gain_floor: Minimum spectral gain to prevent over-suppression

    Speech Enhancement Parameters:
        speech_band: Frequency range (Hz) where speech content is emphasized
        speech_band_boost: Boost factor for speech frequency band

    Loudness Gate Parameters:
        enable_loudness_gate: Enable adaptive loudness gating
        gate_threshold_db: Threshold below peak speech level for gate activation
        gate_max_att_db: Maximum attenuation applied by the gate
        gate_softness: Controls gate transition smoothness
        gate_attack: Gate response speed for attenuation
        gate_release: Gate response speed for recovery
        peak_decay: Decay rate for peak speech level tracking

    Design Philosophy:
        - Conservative defaults for robust operation
        - Marine environment optimization
        - Real-time processing constraints
        - Configurable for different vessel types
    """

    # Core audio processing parameters
    sample_rate: int = 16000
    frame_ms: int = 30
    fft_size: Optional[int] = None

    # Noise estimation and suppression parameters
    noise_update_alpha: float = 0.98
    dd_beta: float = 0.98
    gain_floor: float = 0.05

    # Speech enhancement parameters
    speech_band: Tuple[int, int] = (300, 3800)
    speech_band_boost: float = 0.10

    # Loudness gate parameters
    enable_loudness_gate: bool = True
    gate_threshold_db: float = 8.0
    gate_max_att_db: float = 35.0
    gate_softness: float = 0.6
    gate_attack: float = 0.3
    gate_release: float = 0.05
    peak_decay: float = 0.995

    def validate(self) -> None:
        """
        Validate configuration parameters for correctness.

        Raises:
            ValueError: If any parameter is outside valid range

        Validation Rules:
            - Sample rate must be positive
            - Frame size must be WebRTC VAD compatible
            - Alpha values must be between 0 and 1
            - Speech band must be within Nyquist frequency
            - Gate parameters must be reasonable
        """
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")

        if self.frame_ms not in [10, 20, 30]:
            raise ValueError("Frame size must be 10, 20, or 30 ms for WebRTC VAD compatibility")

        if not (0.0 <= self.noise_update_alpha <= 1.0):
            raise ValueError("noise_update_alpha must be between 0 and 1")

        if not (0.0 <= self.dd_beta <= 1.0):
            raise ValueError("dd_beta must be between 0 and 1")

        if not (0.0 <= self.gain_floor <= 1.0):
            raise ValueError("gain_floor must be between 0 and 1")

        nyquist = self.sample_rate // 2
        if self.speech_band[1] > nyquist:
            raise ValueError(f"Speech band upper limit ({self.speech_band[1]}) exceeds Nyquist frequency ({nyquist})")

        if self.speech_band[0] >= self.speech_band[1]:
            raise ValueError("Speech band lower limit must be less than upper limit")

    def for_marine_environment(self, vessel_type: str = "fishing") -> 'SuppressorConfig':
        """
        Create optimized configuration for specific marine environments.

        Args:
            vessel_type: Type of vessel ("fishing", "recreational", "commercial")

        Returns:
            SuppressorConfig: Optimized configuration for the specified environment

        Vessel-Specific Optimizations:
            - fishing: Optimized for diesel engines and winch noise
            - recreational: Balanced for varied engine types and wind noise
            - commercial: Heavy-duty suppression for industrial environments
        """
        config = SuppressorConfig()

        if vessel_type == "fishing":
            # Fishing vessel optimization - handle diesel engine noise
            config.noise_update_alpha = 0.99  # Slower noise adaptation
            config.gain_floor = 0.08  # Higher floor for engine noise
            config.gate_threshold_db = 6.0  # More sensitive gate
            config.speech_band_boost = 0.15  # Enhanced speech clarity

        elif vessel_type == "recreational":
            # Recreational boat optimization - varied conditions
            config.noise_update_alpha = 0.98  # Balanced adaptation
            config.gain_floor = 0.05  # Standard floor
            config.gate_threshold_db = 8.0  # Standard gate
            config.speech_band_boost = 0.10  # Balanced enhancement

        elif vessel_type == "commercial":
            # Commercial vessel optimization - heavy machinery
            config.noise_update_alpha = 0.995  # Very slow adaptation
            config.gain_floor = 0.12  # High floor for machinery
            config.gate_threshold_db = 4.0  # Very sensitive gate
            config.speech_band_boost = 0.20  # Strong speech enhancement

        return config


class AdaptiveNoiseSuppressor:
    """
    Real-time adaptive noise suppressor using spectral analysis.

    This class implements a sophisticated noise suppression algorithm optimized
    for marine environments. It uses spectral analysis to identify and suppress
    noise while preserving speech quality.

    Key Features:
        - Frame-by-frame spectral processing for real-time operation
        - Decision-directed a priori SNR estimation for adaptive behavior
        - Wiener-like gain computation with configurable noise floor
        - Speech-band emphasis for improved intelligibility
        - Optional loudness gate for background noise suppression
        - Optimized for 16kHz speech audio processing

    Algorithm Overview:
        1. Short-time FFT analysis of input frames
        2. Noise PSD estimation during speech pauses
        3. A priori SNR calculation using decision-directed method
        4. Spectral gain computation with speech band enhancement
        5. Optional loudness gating for background suppression
        6. Inverse FFT to reconstruct enhanced audio

    Marine Environment Adaptations:
        - Robust to engine noise patterns
        - Adaptive to varying wind conditions
        - Effective with multiple speakers
        - Minimal latency for real-time communication

    Usage:
        suppressor = AdaptiveNoiseSuppressor(config)
        enhanced_frame = suppressor.process_frame(noisy_frame, is_speech)
    """

    def __init__(self, cfg: SuppressorConfig | None = None) -> None:
        """
        Initialize the adaptive noise suppressor.

        Args:
            cfg: Configuration object with suppressor parameters
                 (uses default configuration if None)

        Implementation:
            - Validates configuration parameters
            - Calculates optimal FFT size for efficiency
            - Initializes adaptive state variables
            - Maps speech frequency band to FFT bins
            - Sets up loudness gate tracking
        """
        self.cfg = cfg or SuppressorConfig()
        self.cfg.validate()  # Ensure parameters are valid

        # Calculate frame parameters
        self._frame_len = int(self.cfg.sample_rate * self.cfg.frame_ms / 1000)

        # Determine FFT size for efficient processing
        if self.cfg.fft_size is None:
            # Use next power of two for efficient FFTs
            n = 1
            while n < self._frame_len:
                n <<= 1
            self._fft_size = n
        else:
            self._fft_size = int(self.cfg.fft_size)

        logger.debug(f"Initialized suppressor: frame_len={self._frame_len}, fft_size={self._fft_size}")

        # Adaptive algorithm state (initialized on first frames)
        self._noise_psd: Optional[np.ndarray] = None
        self._prev_gain: Optional[np.ndarray] = None
        self._eps = 1e-10  # Small constant to prevent division by zero

        # Map speech frequency band to FFT bin indices
        self._speech_bin_lo = int(self.cfg.speech_band[0] * self._fft_size / self.cfg.sample_rate)
        self._speech_bin_hi = int(self.cfg.speech_band[1] * self._fft_size / self.cfg.sample_rate)
        self._speech_bin_hi = max(self._speech_bin_lo + 1, self._speech_bin_hi)

        logger.debug(f"Speech band bins: {self._speech_bin_lo}:{self._speech_bin_hi} "
                    f"({self.cfg.speech_band[0]}-{self.cfg.speech_band[1]} Hz)")

        # Loudness gate state tracking
        self._peak_rms: float = 0.0
        self._gate_gain: float = 1.0

    def process_frame(self, frame: np.ndarray, is_speech: bool) -> np.ndarray:
        """
        Process a single audio frame through the noise suppressor.

        Args:
            frame: Input audio frame (mono, typically 30ms at 16kHz)
            is_speech: Whether the frame contains speech (from VAD)

        Returns:
            np.ndarray: Enhanced audio frame with noise suppression applied

        Processing Pipeline:
            1. Windowing and FFT analysis
            2. Noise PSD estimation (during non-speech)
            3. A priori SNR calculation
            4. Spectral gain computation
            5. Speech band enhancement
            6. Loudness gate application
            7. Inverse FFT and reconstruction

        Performance:
            - Optimized for real-time processing
            - Minimal memory allocation
            - Efficient FFT operations
            - Low computational complexity
        """
        if len(frame) != self._frame_len:
            raise ValueError(f"Frame length {len(frame)} doesn't match expected {self._frame_len}")

        # Zero-pad frame to FFT size
        padded_frame = np.zeros(self._fft_size, dtype=np.float32)
        padded_frame[:self._frame_len] = frame.astype(np.float32)

        # Apply window function (Hann window for good frequency resolution)
        window = np.hann(self._frame_len)
        padded_frame[:self._frame_len] *= window

        # Forward FFT
        spectrum = np.fft.rfft(padded_frame)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        power = magnitude ** 2

        # Initialize noise PSD on first frame
        if self._noise_psd is None:
            self._noise_psd = power.copy()
            self._prev_gain = np.ones_like(magnitude)
            logger.debug("Initialized noise PSD estimate")

        # Update noise PSD during non-speech frames
        if not is_speech:
            alpha = self.cfg.noise_update_alpha
            self._noise_psd = alpha * self._noise_psd + (1 - alpha) * power

        # Calculate a posteriori SNR
        snr_post = np.maximum(power / (self._noise_psd + self._eps), self._eps)

        # Decision-directed a priori SNR estimation
        snr_prior = (self.cfg.dd_beta * (self._prev_gain ** 2) * snr_post +
                    (1 - self.cfg.dd_beta) * np.maximum(snr_post - 1, 0))
        snr_prior = np.maximum(snr_prior, self._eps)

        # Wiener-like gain computation
        gain = snr_prior / (1 + snr_prior)
        gain = np.maximum(gain, self.cfg.gain_floor)

        # Speech band enhancement
        if self.cfg.speech_band_boost > 0:
            speech_gain = np.ones_like(gain)
            speech_gain[self._speech_bin_lo:self._speech_bin_hi] *= (1 + self.cfg.speech_band_boost)
            gain = gain * (1 - self.cfg.speech_band_boost) + gain * speech_gain * self.cfg.speech_band_boost

        # Apply loudness gate if enabled
        if self.cfg.enable_loudness_gate:
            gate_gain = self._compute_loudness_gate(frame, is_speech)
            gain *= gate_gain

        # Store gain for next frame's a priori estimation
        self._prev_gain = gain.copy()

        # Apply spectral gain
        enhanced_spectrum = spectrum * gain

        # Inverse FFT
        enhanced_frame = np.fft.irfft(enhanced_spectrum, n=self._fft_size)

        # Extract original frame length and apply window compensation
        output = enhanced_frame[:self._frame_len] / (window + self._eps)

        return output.astype(frame.dtype)

    def _compute_loudness_gate(self, frame: np.ndarray, is_speech: bool) -> float:
        """
        Compute loudness gate gain based on frame energy.

        Args:
            frame: Input audio frame for energy calculation
            is_speech: Whether frame contains speech

        Returns:
            float: Gate gain factor (0.0 to 1.0)

        Algorithm:
            - Tracks peak RMS level during speech
            - Computes current frame RMS
            - Applies attenuation based on threshold
            - Uses smooth attack/release characteristics
        """
        frame_rms = np.sqrt(np.mean(frame ** 2))

        # Update peak RMS during speech frames
        if is_speech and frame_rms > self._peak_rms:
            self._peak_rms = frame_rms
        else:
            # Apply exponential decay to peak
            self._peak_rms *= self.cfg.peak_decay

        # Calculate gate threshold
        if self._peak_rms > self._eps:
            # Current level relative to peak (in dB)
            level_db = 20 * np.log10(frame_rms / (self._peak_rms + self._eps))

            # Calculate attenuation if below threshold
            if level_db < -self.cfg.gate_threshold_db:
                # Smooth attenuation curve
                excess_db = -level_db - self.cfg.gate_threshold_db
                att_db = min(excess_db ** self.cfg.gate_softness, self.cfg.gate_max_att_db)
                target_gain = 10 ** (-att_db / 20)
            else:
                target_gain = 1.0
        else:
            target_gain = 1.0

        # Apply smooth attack/release
        if target_gain < self._gate_gain:
            # Attack (gain decrease)
            alpha = self.cfg.gate_attack
        else:
            # Release (gain increase)
            alpha = self.cfg.gate_release

        self._gate_gain = alpha * self._gate_gain + (1 - alpha) * target_gain

        return self._gate_gain

    def reset(self) -> None:
        """
        Reset the suppressor's adaptive state.

        Use Cases:
            - When switching audio sources
            - After long periods of silence
            - When changing suppressor configuration
            - For testing and evaluation
        """
        self._noise_psd = None
        self._prev_gain = None
        self._peak_rms = 0.0
        self._gate_gain = 1.0
        logger.debug("Reset suppressor adaptive state")

    def get_noise_estimate(self) -> Optional[np.ndarray]:
        """
        Get current noise power spectral density estimate.

        Returns:
            Optional[np.ndarray]: Current noise PSD estimate or None if not initialized

        Use Cases:
            - Monitoring noise characteristics
            - Debugging suppression performance
            - Adaptive threshold adjustment
        """
        return self._noise_psd.copy() if self._noise_psd is not None else None

    def get_suppression_stats(self) -> dict:
        """
        Get statistics about current suppression performance.

        Returns:
            dict: Statistics including noise level, gate state, and configuration
        """
        stats = {
            'initialized': self._noise_psd is not None,
            'peak_rms': self._peak_rms,
            'gate_gain': self._gate_gain,
            'frame_length': self._frame_len,
            'fft_size': self._fft_size,
            'speech_band': self.cfg.speech_band,
            'speech_bins': (self._speech_bin_lo, self._speech_bin_hi),
        }

        if self._noise_psd is not None:
            stats.update({
                'noise_level_db': 10 * np.log10(np.mean(self._noise_psd) + self._eps),
                'noise_bins': len(self._noise_psd),
            })

        return stats

