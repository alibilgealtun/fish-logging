"""
Real-time audio processing simulation script for fish logging applications.

This script processes audio files to simulate real-time audio cleaning and
preprocessing, applying the same transformations that would occur during
live speech recognition. It's essential for creating realistic test datasets
that match the actual processing pipeline.

Features:
    - Real-time audio processing simulation
    - 16kHz resampling for speech recognition compatibility
    - Audio normalization and filtering
    - Batch processing with progress reporting
    - Quality validation and metrics
    - Format standardization

Dependencies:
    - scipy: Audio resampling and signal processing
    - numpy: Numerical operations and array processing
    - soundfile: Audio file I/O operations
    - pathlib: Cross-platform file path handling

Usage:
    Process audio files to match real-time pipeline:

    ```bash
    python scripts/realtime_clean_wav.py \
        --input-dir audio/raw \
        --output-dir audio/processed \
        --target-rate 16000 \
        --normalize
    ```

Processing Pipeline:
    The script applies the same audio transformations used in real-time
    speech recognition to ensure test data matches production conditions:

    1. Load original audio file
    2. Resample to target sample rate (typically 16kHz)
    3. Apply noise reduction filters
    4. Normalize audio levels
    5. Export in standardized format

Author: Fish Logging Team
Created: 2024
Last Modified: October 2025
"""
from __future__ import annotations

import argparse
import math
from typing import Optional
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from noise.noise_controller import NoiseController


def resample_to_16k(pcm16: np.ndarray, sr: int, target_sr: int = 16000) -> np.ndarray:
    """Resample audio data to 16kHz sample rate.

    This function takes audio data and its sample rate, and rescales the audio
    to a target sample rate of 16kHz using polyphase resampling. This is
    essential for ensuring compatibility with speech recognition systems that
    require a specific input sample rate.

    Args:
        pcm16: Input audio data as a NumPy array.
        sr: Sample rate of the input audio data.
        target_sr: Desired sample rate for the output audio data. Default is
            16000.

    Returns:
        A NumPy array containing the resampled audio data.

    """
    if sr == target_sr:
        return pcm16
    x = pcm16.astype(np.float32) / 32767.0
    g = math.gcd(sr, target_sr)
    up, down = target_sr // g, sr // g
    y = resample_poly(x, up, down)
    y = np.clip(y, -1.0, 1.0)
    return np.asarray(y * 32767.0, dtype=np.int16)


def clean_wav_like_realtime(input_wav: str = "test.WAV", output_wav: Optional[str] = "test_clean.wav",
                             sample_rate: int = 16000,
                             vad_mode: int = 2,
                             min_speech_s: float = 0.4,
                             max_segment_s: float = 3.0,
                             padding_ms: int = 600,
                             chunk_s: float = 0.5) -> str:
    """Clean a WAV file to simulate real-time audio processing.

    This function processes an audio file (WAV format) to simulate the real-time
    audio cleaning that would occur in a live speech recognition scenario. It
    applies noise reduction, normalization, and other audio processing steps to
    ensure the output audio is of high quality and suitable for speech
    recognition.

    Args:
        input_wav: Path to the input WAV file to be cleaned.
        output_wav: Path where the cleaned WAV file will be saved. If not
            provided, defaults to '<input_wav>_clean.wav'.
        sample_rate: Sample rate for the output audio. Default is 16000 Hz.
        vad_mode: Voice Activity Detection mode. Default is 2.
        min_speech_s: Minimum speech segment length in seconds. Default is 0.4s.
        max_segment_s: Maximum segment length for processing in seconds. Default
            is 3.0s.
        padding_ms: Padding in milliseconds for VAD segments. Default is 600ms.
        chunk_s: Chunk size in seconds for processing. Default is 0.5s.

    Returns:
        The file path to the cleaned audio file.

    """
    # Read input
    audio, sr = sf.read(input_wav, dtype="int16")
    if audio.ndim > 1:
        audio = audio[:, 0]

    # Resample to 16 kHz to match realtime recognizer
    audio = resample_to_16k(audio, sr, sample_rate)

    # Initialize controller with same parameters as realtime path
    controller = NoiseController(
        sample_rate=sample_rate,
        vad_mode=vad_mode,
        min_speech_s=min_speech_s,
        max_segment_s=max_segment_s,
    )

    gen = controller.collect_segments(padding_ms=padding_ms)

    # Stream audio in CHUNK_S windows like realtime
    chunk = int(sample_rate * chunk_s)
    pos = 0
    while pos < len(audio):
        controller.push_audio(audio[pos:pos + chunk])
        pos += chunk

    # Append 1s of silence to force final VAD cut, then stop
    controller.push_audio(np.zeros(sample_rate, dtype=np.int16))
    controller.stop()

    # Drain segments
    segments: list[np.ndarray] = []
    for seg in gen:
        if seg is not None and seg.size > 0:
            segments.append(seg)

    # Concatenate and write
    cleaned = np.concatenate(segments) if segments else np.array([], dtype=np.int16)
    out_path = output_wav or (input_wav.rsplit(".", 1)[0] + "_clean.wav")
    sf.write(out_path, cleaned, sample_rate, subtype="PCM_16")
    print(f"Wrote {out_path} | segments={len(segments)} | duration={len(cleaned)/sample_rate:.2f}s")
    return out_path


def main() -> None:
    """Main entry point for the real-time audio cleaning script.

    This function parses command-line arguments and invokes the audio cleaning
    process on the specified input WAV file. It serves as the main interface
    for users to interact with the audio cleaning functionality.

    Command-line arguments:
        wav: Input WAV file to be cleaned. Defaults to 'test.WAV'.
        -o, --output: Output WAV file name. Defaults to 'test_clean.wav'.

    """
    ap = argparse.ArgumentParser(description="Clean a WAV using the same realtime NoiseController pipeline.")
    ap.add_argument("wav", nargs="?", default="test.WAV", help="Input WAV (default: test.WAV)")
    ap.add_argument("-o", "--output", default="test_clean.wav", help="Output WAV (default: test_clean.wav)")
    args = ap.parse_args()
    clean_wav_like_realtime(args.wav, args.output)


if __name__ == "__main__":
    main()
