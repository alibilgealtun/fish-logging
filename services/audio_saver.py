"""Service for saving audio segments to disk before transcription."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from loguru import logger


class AudioSaver:
    """
    Service for saving audio segments to a configured directory.

    This service handles:
    - Creating the output directory if it doesn't exist
    - Generating unique filenames with timestamps and sequence numbers
    - Saving audio segments as WAV files
    - Optional cleanup of old segments
    """

    def __init__(self, segments_dir: str = "audio/segments", enabled: bool = False):
        """
        Initialize the audio saver service.

        Args:
            segments_dir: Directory where audio segments will be saved
            enabled: Whether to actually save segments (can be disabled for performance)
        """
        self.segments_dir = Path(segments_dir)
        self.enabled = enabled
        self._sequence_counter = 0
        self._current_session_prefix = None

        if self.enabled:
            self._ensure_directory_exists()

    def _ensure_directory_exists(self) -> None:
        """Create the segments directory if it doesn't exist."""
        try:
            self.segments_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Audio segments directory ready: {self.segments_dir}")
        except Exception as e:
            logger.error(f"Failed to create audio segments directory: {e}")
            self.enabled = False

    def _get_session_prefix(self) -> str:
        """
        Get or create a session prefix for filename generation.

        Returns a timestamp-based prefix that's consistent for the session.
        """
        if self._current_session_prefix is None:
            now = datetime.now()
            self._current_session_prefix = now.strftime("%Y%m%d_%H%M")
        return self._current_session_prefix

    def _generate_filename(self) -> str:
        """
        Generate a unique filename for the audio segment.

        Format: segment_YYYYMMDD_HHMM_NNNN.wav
        where NNNN is a zero-padded sequence number.
        """
        self._sequence_counter += 1
        prefix = self._get_session_prefix()
        return f"segment_{prefix}_{self._sequence_counter:04d}.wav"

    def save_segment(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        filename: Optional[str] = None
    ) -> Optional[str]:
        """
        Save an audio segment to disk.

        Args:
            audio_data: Audio samples as numpy array (typically int16 PCM)
            sample_rate: Sample rate in Hz
            filename: Optional custom filename (defaults to auto-generated)

        Returns:
            Path to the saved file, or None if saving is disabled or failed
        """
        if not self.enabled:
            return None

        if audio_data is None or audio_data.size == 0:
            logger.debug("Skipping empty audio segment")
            return None

        try:
            # Generate filename if not provided
            if filename is None:
                filename = self._generate_filename()

            # Ensure filename has .wav extension
            if not filename.endswith('.wav'):
                filename += '.wav'

            # Full path
            filepath = self.segments_dir / filename

            # Save the audio file
            sf.write(filepath, audio_data, sample_rate, subtype="PCM_16")

            logger.debug(f"Saved audio segment: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to save audio segment: {e}")
            return None

    def reset_session(self) -> None:
        """
        Reset the session counter for a new recording session.

        This will generate a new timestamp prefix for subsequent segments.
        """
        self._current_session_prefix = None
        self._sequence_counter = 0
        logger.debug("Audio saver session reset")

    def cleanup_old_segments(self, max_age_days: int = 7) -> int:
        """
        Remove audio segments older than the specified number of days.

        Args:
            max_age_days: Maximum age in days for segments to keep

        Returns:
            Number of files deleted
        """
        if not self.enabled or not self.segments_dir.exists():
            return 0

        try:
            from datetime import timedelta
            cutoff_time = datetime.now() - timedelta(days=max_age_days)
            deleted_count = 0

            for filepath in self.segments_dir.glob("segment_*.wav"):
                try:
                    # Get file modification time
                    mtime = datetime.fromtimestamp(filepath.stat().st_mtime)

                    if mtime < cutoff_time:
                        filepath.unlink()
                        deleted_count += 1
                        logger.debug(f"Deleted old segment: {filepath}")

                except Exception as e:
                    logger.warning(f"Failed to delete {filepath}: {e}")

            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old audio segments")

            return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup old segments: {e}")
            return 0


# Singleton instance for global access
_global_audio_saver: Optional[AudioSaver] = None


def get_audio_saver() -> AudioSaver:
    """
    Get the global AudioSaver instance.

    This creates a singleton instance on first call.
    Call initialize_audio_saver() first to configure it properly.
    """
    global _global_audio_saver
    if _global_audio_saver is None:
        _global_audio_saver = AudioSaver()
    return _global_audio_saver


def initialize_audio_saver(segments_dir: str, enabled: bool) -> AudioSaver:
    """
    Initialize the global AudioSaver instance with configuration.

    Args:
        segments_dir: Directory where segments will be saved
        enabled: Whether saving is enabled

    Returns:
        The initialized AudioSaver instance
    """
    global _global_audio_saver
    _global_audio_saver = AudioSaver(segments_dir=segments_dir, enabled=enabled)
    return _global_audio_saver


__all__ = ["AudioSaver", "get_audio_saver", "initialize_audio_saver"]

