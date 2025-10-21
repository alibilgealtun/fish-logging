"""Simple Noise Controller for Fish Logging Application.

This module provides a simplified noise processing pipeline that uses only
high-pass filtering and voice activity detection without adaptive spectral
suppression. It's designed for users who prefer lighter processing or have
good audio conditions.

Classes:
    SimpleNoiseController: Lightweight noise processing with HPF and VAD only

Features:
    - High-pass filtering to remove low-frequency noise
    - WebRTC VAD for speech detection
    - Real-time audio segmentation
    - Minimal computational overhead
    - Queue-based audio streaming support

Use Cases:
    - Clean audio environments with minimal noise
    - Resource-constrained systems
    - Users preferring minimal processing
    - Baseline comparison for adaptive algorithms
"""
from __future__ import annotations

import numpy as np
import queue
import webrtcvad
from scipy.signal import butter, lfilter

# Optional logger: prefer loguru if available, else fallback to stdlib logging
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class SimpleNoiseController:
    """
    Lightweight noise controller with basic filtering and voice activity detection.

    This controller provides essential noise processing without the computational
    overhead of adaptive spectral suppression. It's ideal for clean audio
    environments or resource-constrained systems.

    Key Features:
        - High-pass filtering to remove engine rumble and DC offset
        - WebRTC Voice Activity Detection for speech segmentation
        - Real-time audio processing with minimal latency
        - Queue-based streaming audio support
        - Configurable speech segment length limits

    Processing Pipeline:
        1. High-pass filtering (removes frequencies below 100Hz)
        2. WebRTC VAD for speech/non-speech classification
        3. Speech segmentation with configurable timing
        4. Real-time audio queue management

    Advantages:
        - Low CPU usage
        - Minimal memory footprint
        - Fast processing
        - Simple configuration
        - Reliable performance

    Limitations:
        - No adaptive noise suppression
        - Less effective in very noisy environments
        - No spectral enhancement
        - Basic noise handling
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        vad_mode: int = 3,
        min_speech_s: float = 0.5,
        max_segment_s: float = 4.0
    ) -> None:
        """
        Initialize the simple noise controller.

        Args:
            sample_rate: Audio sample rate in Hz (8000, 16000, 32000, or 48000)
            vad_mode: VAD aggressiveness level (0=least, 3=most aggressive)
            min_speech_s: Minimum valid speech segment duration (seconds)
            max_segment_s: Maximum segment length before forced cut (seconds)

        WebRTC VAD Requirements:
            - Sample rate must be 8000, 16000, 32000, or 48000 Hz
            - Audio must be 16-bit PCM format
            - Frame sizes must be 10, 20, or 30 milliseconds

        VAD Mode Guidelines:
            - 0: Quality mode (clean environments)
            - 1: Low bitrate mode (moderate noise)
            - 2: Aggressive mode (noisy environments)
            - 3: Very aggressive mode (very noisy, marine environments)
        """
        if sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError(f"Sample rate {sample_rate} not supported by WebRTC VAD")

        self.sample_rate = sample_rate
        self.vad_mode = vad_mode
        self.min_speech_s = min_speech_s
        self.max_segment_s = max_segment_s

        # Initialize WebRTC VAD
        self.vad = webrtcvad.Vad(vad_mode)

        # High-pass filter design (removes engine rumble below 100Hz)
        nyquist = sample_rate / 2
        cutoff = 100.0 / nyquist
        self.hp_b, self.hp_a = butter(
            N=6,  # 6th order for steep rolloff
            Wn=cutoff,
            btype="highpass"
        )

        # Filter state for continuous processing
        self.filter_state = None

        # Audio streaming queue for real-time processing
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()

        # Speech segmentation state
        self.speech_buffer = []
        self.silence_count = 0
        self.speech_count = 0

        # Processing statistics
        self.stats = {
            'frames_processed': 0,
            'speech_frames': 0,
            'silence_frames': 0,
            'segments_created': 0
        }

        logger.info(f"SimpleNoiseController initialized: {sample_rate}Hz, VAD mode {vad_mode}")

    def apply_highpass_filter(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply high-pass filter to remove low-frequency noise.

        Args:
            audio: Input audio signal (mono)

        Returns:
            np.ndarray: Filtered audio with low frequencies removed

        Filter Characteristics:
            - 6th order Butterworth high-pass filter
            - 100Hz cutoff frequency
            - Steep rolloff to remove engine noise
            - Maintains filter state for continuous processing
        """
        if self.filter_state is None:
            # Initialize filter state for continuous processing
            from scipy.signal import lfilter_zi
            self.filter_state = lfilter_zi(self.hp_b, self.hp_a) * audio[0]

        # Apply filter with state preservation
        filtered_audio, self.filter_state = lfilter(
            self.hp_b, self.hp_a, audio, zi=self.filter_state
        )

        return filtered_audio

    def detect_speech(self, audio_frame: np.ndarray) -> bool:
        """
        Detect speech in an audio frame using WebRTC VAD.

        Args:
            audio_frame: Audio frame (10ms, 20ms, or 30ms duration)

        Returns:
            bool: True if speech detected, False otherwise

        Requirements:
            - Frame must be appropriate length for VAD
            - Audio must be 16-bit PCM format
            - Sample rate must match controller configuration
        """
        # Convert to int16 format required by WebRTC VAD
        if audio_frame.dtype != np.int16:
            # Scale float audio to int16 range
            if audio_frame.dtype in [np.float32, np.float64]:
                audio_int16 = (audio_frame * 32767).astype(np.int16)
            else:
                audio_int16 = audio_frame.astype(np.int16)
        else:
            audio_int16 = audio_frame

        # Ensure frame length is compatible with VAD
        frame_duration_ms = len(audio_int16) * 1000 // self.sample_rate
        if frame_duration_ms not in [10, 20, 30]:
            logger.warning(f"Frame duration {frame_duration_ms}ms not optimal for VAD")

        try:
            # Perform voice activity detection
            is_speech = self.vad.is_speech(audio_int16.tobytes(), self.sample_rate)

            # Update statistics
            self.stats['frames_processed'] += 1
            if is_speech:
                self.stats['speech_frames'] += 1
            else:
                self.stats['silence_frames'] += 1

            return is_speech

        except Exception as e:
            logger.error(f"VAD detection failed: {e}")
            return False  # Assume silence on error

    def process_audio_chunk(self, audio_chunk: np.ndarray) -> list[np.ndarray]:
        """
        Process an audio chunk and return completed speech segments.

        Args:
            audio_chunk: Input audio data (any length)

        Returns:
            list[np.ndarray]: List of completed speech segments

        Processing Steps:
            1. Apply high-pass filtering
            2. Split into VAD-compatible frames
            3. Detect speech in each frame
            4. Accumulate speech segments
            5. Return completed segments based on timing rules
        """
        segments = []

        # Apply high-pass filtering
        filtered_audio = self.apply_highpass_filter(audio_chunk)

        # Process in 30ms frames (compatible with WebRTC VAD)
        frame_size = int(0.03 * self.sample_rate)  # 30ms frames
        num_frames = len(filtered_audio) // frame_size

        for i in range(num_frames):
            start_idx = i * frame_size
            end_idx = start_idx + frame_size
            frame = filtered_audio[start_idx:end_idx]

            # Detect speech in frame
            is_speech = self.detect_speech(frame)

            # Handle speech segmentation
            segment = self._handle_speech_frame(frame, is_speech)
            if segment is not None:
                segments.append(segment)

        return segments

    def _handle_speech_frame(self, frame: np.ndarray, is_speech: bool) -> np.ndarray | None:
        """
        Handle speech frame accumulation and segmentation logic.

        Args:
            frame: Audio frame to process
            is_speech: Whether frame contains speech

        Returns:
            Optional[np.ndarray]: Completed speech segment or None

        Segmentation Rules:
            - Accumulate consecutive speech frames
            - Add short silence padding between words
            - End segment after extended silence
            - Force segmentation at maximum length
            - Discard segments shorter than minimum length
        """
        if is_speech:
            self.speech_buffer.append(frame)
            self.speech_count += 1
            self.silence_count = 0

            # Check for maximum segment length
            segment_duration = len(self.speech_buffer) * len(frame) / self.sample_rate
            if segment_duration >= self.max_segment_s:
                return self._finalize_segment("max_length_reached")

        else:
            # Handle silence frame
            if len(self.speech_buffer) > 0:
                # Add some silence padding
                if self.silence_count < 3:  # Add up to 3 silence frames (90ms)
                    self.speech_buffer.append(frame)

                self.silence_count += 1

                # End segment after extended silence
                if self.silence_count >= 5:  # 150ms of silence
                    return self._finalize_segment("silence_detected")

        return None

    def _finalize_segment(self, reason: str) -> np.ndarray | None:
        """
        Finalize a speech segment and check minimum length requirements.

        Args:
            reason: Reason for segment finalization

        Returns:
            Optional[np.ndarray]: Finalized segment or None if too short
        """
        if not self.speech_buffer:
            return None

        # Calculate segment duration
        total_samples = sum(len(frame) for frame in self.speech_buffer)
        duration = total_samples / self.sample_rate

        # Check minimum length requirement
        if duration < self.min_speech_s:
            logger.debug(f"Discarding short segment: {duration:.2f}s < {self.min_speech_s}s")
            self._reset_segment_state()
            return None

        # Concatenate frames into segment
        segment = np.concatenate(self.speech_buffer)

        # Update statistics
        self.stats['segments_created'] += 1

        # Reset state for next segment
        self._reset_segment_state()

        logger.debug(f"Created segment: {duration:.2f}s, reason: {reason}")
        return segment

    def _reset_segment_state(self) -> None:
        """Reset speech segmentation state."""
        self.speech_buffer.clear()
        self.silence_count = 0
        self.speech_count = 0

    def queue_audio(self, audio_data: np.ndarray) -> None:
        """
        Queue audio data for streaming processing.

        Args:
            audio_data: Audio data to queue for processing

        Use Cases:
            - Real-time audio streaming
            - Buffered audio processing
            - Asynchronous audio handling
        """
        self._audio_queue.put(audio_data)

    def get_queued_audio(self, timeout: float = 0.1) -> np.ndarray | None:
        """
        Get audio data from the processing queue.

        Args:
            timeout: Maximum time to wait for audio data

        Returns:
            Optional[np.ndarray]: Audio data or None if timeout
        """
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear_queue(self) -> None:
        """Clear all queued audio data."""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    def reset(self) -> None:
        """
        Reset controller state for new audio source.

        Clears all internal state including filter state, speech buffer,
        and processing statistics.
        """
        self.filter_state = None
        self._reset_segment_state()
        self.clear_queue()

        # Reset statistics
        self.stats = {
            'frames_processed': 0,
            'speech_frames': 0,
            'silence_frames': 0,
            'segments_created': 0
        }

        logger.info("SimpleNoiseController reset completed")

    def get_processing_stats(self) -> dict:
        """
        Get processing statistics and performance metrics.

        Returns:
            dict: Statistics including frame counts, speech ratio, and configuration
        """
        stats = self.stats.copy()

        # Calculate derived metrics
        if stats['frames_processed'] > 0:
            stats['speech_ratio'] = stats['speech_frames'] / stats['frames_processed']
        else:
            stats['speech_ratio'] = 0.0

        # Add configuration information
        stats.update({
            'sample_rate': self.sample_rate,
            'vad_mode': self.vad_mode,
            'min_speech_s': self.min_speech_s,
            'max_segment_s': self.max_segment_s,
            'queue_size': self._audio_queue.qsize(),
        })

        return stats
