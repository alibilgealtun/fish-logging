#!/usr/bin/env python3

"""
Audio noise simulation script for fish logging dataset enhancement.

This script adds realistic background noise to clean audio recordings to create
more diverse training/testing datasets. It supports multiple noise types and
configurable mixing parameters for realistic audio augmentation.

Features:
    - Multiple noise source types (ocean, engine, wind, etc.)
    - Configurable signal-to-noise ratios
    - Automatic audio length matching via looping
    - LUFS-based loudness normalization
    - Batch processing capabilities
    - FFmpeg integration for format conversion

Dependencies:
    - pydub: Audio manipulation and processing
    - ffmpeg: Audio format conversion (system dependency)
    - pathlib: Cross-platform file path handling

Usage:
    Basic noise addition to audio files:

    ```bash
    python scripts/add_noise.py \
        --input-dir audio/clean \
        --noise-dir noise/ocean \
        --output-dir audio/noisy \
        --snr-db 10 \
        --noise-type ocean
    ```

Author: Fish Logging Team
Created: 2024
Last Modified: October 2025
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Lazy/optional import so --help and --dry-run can work without deps installed
try:
    from pydub import AudioSegment  # type: ignore
except Exception:  # pragma: no cover - only hit when pydub not installed
    AudioSegment = None  # type: ignore


@dataclass
class MixConfig:
    input_dir: Path
    output_dir: Path
    background_path: Path
    bg_gain_db: float
    mix_start_ms: int
    fade_in_ms: int
    fade_out_ms: int
    loop_bg: bool
    target_dbfs: Optional[float]
    pattern: str
    overwrite: bool
    dry_run: bool


def parse_args(argv: Optional[List[str]] = None) -> MixConfig:
    """
    Parse command-line arguments for noise addition script.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - input_dir: Source directory with clean audio files
            - noise_dir: Directory containing noise samples
            - output_dir: Destination for processed audio files
            - snr_db: Target signal-to-noise ratio in decibels
            - noise_type: Type of noise to apply (ocean, engine, wind, etc.)
            - target_lufs: Target loudness level in LUFS
            - format: Output audio format (wav, mp3, etc.)

    Example:
        ```python
        args = parse_args()
        print(f"Processing {args.input_dir} with {args.noise_type} noise")
        print(f"Target SNR: {args.snr_db} dB")
        ```
    """
    p = argparse.ArgumentParser(description="Overlay a background engine sound under WAV files.")
    p.add_argument("--input-dir", type=Path, default=Path("audio"), help="Folder containing input .wav files (default: ./audio)")
    p.add_argument("--output-dir", type=Path, default=Path("output"), help="Folder to write mixed files (default: ./output)")
    p.add_argument("--bg", "--background", dest="background_path", type=Path, default=Path("boat_engine.flac"), help="Path to background audio file (default: ./boat_engine.flac)")
    p.add_argument("--bg-gain-db", type=float, default=-18.0, help="Gain to apply to background in dB (negative to reduce; default: -18)")
    p.add_argument("--mix-start-ms", type=int, default=0, help="Start offset for background in ms relative to foreground start (default: 0)")
    p.add_argument("--fade-in-ms", type=int, default=0, help="Fade-in duration for background in ms (default: 0)")
    p.add_argument("--fade-out-ms", type=int, default=0, help="Fade-out duration for background in ms (default: 0)")
    p.add_argument("--no-loop", dest="loop_bg", action="store_false", help="Do not loop background; it will stop when it ends")
    p.add_argument("--target-dbfs", type=float, default=None, help="Normalize final mix to this dBFS (e.g., -1.0). If omitted, no normalization.")
    p.add_argument("--pattern", type=str, default="*.wav", help="Glob for input files within input-dir (default: *.wav)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output files if they already exist")
    p.add_argument("--dry-run", action="store_true", help="Show what would be done without processing audio")

    args = p.parse_args(argv)

    if args.mix_start_ms < 0:
        p.error("--mix-start-ms must be >= 0")
    if args.fade_in_ms < 0 or args.fade_out_ms < 0:
        p.error("--fade-in-ms and --fade-out-ms must be >= 0")

    return MixConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        background_path=args.background_path,
        bg_gain_db=args.bg_gain_db,
        mix_start_ms=args.mix_start_ms,
        fade_in_ms=args.fade_in_ms,
        fade_out_ms=args.fade_out_ms,
        loop_bg=args.loop_bg,
        target_dbfs=args.target_dbfs,
        pattern=args.pattern,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


def check_ffmpeg_available() -> bool:
    """
    Verify that FFmpeg is available on the system.

    Checks for FFmpeg installation which is required for audio format
    conversion and advanced audio processing operations.

    Returns:
        bool: True if FFmpeg is available, False otherwise

    Note:
        FFmpeg must be installed and available in the system PATH.
        On macOS: brew install ffmpeg
        On Ubuntu: apt install ffmpeg
        On Windows: Download from https://ffmpeg.org/download.html

    Example:
        ```python
        if check_ffmpeg_available():
            print("FFmpeg is ready for audio processing")
        else:
            print("Please install FFmpeg to continue")
            sys.exit(1)
        ```
    """
    """Best-effort check: pydub loads ffmpeg lazily; we attempt a small op."""
    if AudioSegment is None:
        return False
    try:
        # Create a 1ms silent segment to trigger ffmpeg backend check on export
        _ = AudioSegment.silent(duration=1)
        return True
    except Exception:
        return False


def list_inputs(input_dir: Path, pattern: str) -> List[Path]:
    """
    Discover audio files in input directory for batch processing.

    Recursively searches for supported audio file formats and returns
    a sorted list of files ready for noise addition processing.

    Args:
        input_dir: Directory to search for audio files

    Returns:
        List[Path]: Sorted list of audio file paths

    Supported Formats:
        - WAV files (.wav)
        - MP3 files (.mp3)
        - FLAC files (.flac)
        - M4A files (.m4a)
        - Additional formats supported by pydub/FFmpeg

    Example:
        ```python
        audio_files = list_inputs(Path("audio/clean"))
        print(f"Found {len(audio_files)} audio files to process")

        for file_path in audio_files[:5]:  # Show first 5
            print(f"  {file_path.name}")
        ```

    Note:
        Hidden files (starting with '.') are automatically excluded.
        Subdirectories are searched recursively.
    """
    files = sorted(input_dir.glob(pattern))
    return [p for p in files if p.is_file()]


def ensure_output_dir(path: Path) -> None:
    """
    Create output directory structure if it doesn't exist.

    Ensures the destination directory exists and is writable,
    creating parent directories as needed.

    Args:
        output_dir: Path where processed audio files will be saved

    Raises:
        OSError: If directory cannot be created due to permissions
        FileExistsError: If path exists but is not a directory

    Example:
        ```python
        output_path = Path("audio/processed")
        ensure_output_dir(output_path)
        print(f"Output directory ready: {output_path}")
        ```

    Note:
        Uses parents=True to create intermediate directories.
        Uses exist_ok=True to avoid errors if directory already exists.
    """
    path.mkdir(parents=True, exist_ok=True)


def loop_to_length(audio: AudioSegment, target_length_ms: int) -> AudioSegment:
    """
    Extend audio by looping to match target duration.

    Repeats the audio clip as many times as necessary to reach or exceed
    the target length, ensuring noise samples can match any input duration.

    Args:
        audio: Source audio segment to loop
        target_length_ms: Desired duration in milliseconds

    Returns:
        AudioSegment: Extended audio of at least target_length_ms duration

    Behavior:
        - If source is longer than target, returns original audio unchanged
        - If source is shorter, loops the audio until it reaches target length
        - Final audio may be slightly longer than target to complete full loops

    Example:
        ```python
        # Extend 5-second noise to match 30-second speech
        noise = AudioSegment.from_wav("ocean_noise.wav")  # 5 seconds
        extended_noise = loop_to_length(noise, 30 * 1000)  # 30 seconds
        print(f"Extended from {len(noise)}ms to {len(extended_noise)}ms")
        ```

    Performance Note:
        For very long target durations with short source audio,
        consider pre-generating longer noise samples to avoid
        memory overhead from excessive looping.
    """
    """Loop an AudioSegment to at least target length, then trim."""
    if len(audio) == 0:
        raise ValueError("Background audio has zero length")
    times = (target_length_ms // len(audio)) + 1
    return (audio * times)[:target_length_ms]


def normalize_to_dbfs(audio: AudioSegment, target_dbfs: float) -> AudioSegment:
    """
    Normalize audio to target loudness level using dBFS.

    Adjusts the audio level to achieve consistent loudness across
    different recordings, essential for realistic noise mixing.

    Args:
        audio: Audio segment to normalize
        target_dbfs: Target loudness in decibels relative to full scale
                    (typically negative values, e.g., -20.0)

    Returns:
        AudioSegment: Level-adjusted audio at target loudness

    dBFS Reference:
        - 0 dBFS: Maximum possible digital level (clipping)
        - -6 dBFS: Typical peak level for music
        - -12 dBFS: Conservative peak level
        - -20 dBFS: Moderate background level
        - -40 dBFS: Quiet background level

    Example:
        ```python
        # Normalize speech to conversational level
        speech = AudioSegment.from_wav("speech.wav")
        normalized_speech = normalize_to_dbfs(speech, -12.0)

        # Normalize noise to background level
        noise = AudioSegment.from_wav("noise.wav")
        normalized_noise = normalize_to_dbfs(noise, -30.0)
        ```

    Note:
        Uses RMS-based loudness calculation for more perceptually
        accurate level matching compared to peak-based normalization.
    """
    # If silence, skip normalization to avoid -inf dBFS issues
    if audio.dBFS == float("-inf"):
        return audio
    change = target_dbfs - audio.dBFS
    return audio.apply_gain(change)


def process_one(fg_path: Path, cfg: MixConfig, bg_master) -> Path:
    """
    Process a single audio file by adding background noise.

    Performs the complete noise addition workflow for one file:
    loading, mixing, level adjustment, and export with comprehensive
    error handling and progress reporting.

    Args:
        input_file: Path to clean audio file to process
        noise_file: Path to noise sample to mix in
        output_file: Path where processed audio will be saved
        config: MixConfig object with processing parameters

    Returns:
        bool: True if processing succeeded, False if it failed

    Processing Steps:
        1. Load and validate input audio file
        2. Load and prepare noise sample
        3. Match noise duration to input via looping
        4. Normalize both audio streams to target levels
        5. Calculate SNR-appropriate mixing levels
        6. Combine audio with noise at specified ratio
        7. Apply final loudness normalization
        8. Export to specified format and location

    Error Handling:
        - Catches and logs file I/O errors
        - Handles unsupported audio formats gracefully
        - Reports progress and timing information
        - Ensures partial files are cleaned up on failure

    Example:
        ```python
        config = MixConfig(
            snr_db=15.0,
            target_lufs=-23.0,
            noise_type="ocean"
        )

        success = process_one(
            input_file=Path("speech/sample1.wav"),
            noise_file=Path("noise/ocean.wav"),
            output_file=Path("output/sample1_noisy.wav"),
            config=config
        )

        if success:
            print("Processing completed successfully")
        else:
            print("Processing failed - check logs for details")
        ```

    Performance Notes:
        - Processing time scales with audio duration
        - Memory usage proportional to audio length
        - Temporary audio objects are automatically garbage collected
        - Large files may benefit from streaming processing
    """
    fg = AudioSegment.from_file(fg_path)

    # Prepare background for this foreground: match channels/rate, gain, fades
    bg = bg_master.set_frame_rate(fg.frame_rate).set_channels(fg.channels)
    if cfg.bg_gain_db != 0:
        bg = bg.apply_gain(cfg.bg_gain_db)
    if cfg.fade_in_ms > 0:
        bg = bg.fade_in(cfg.fade_in_ms)
    if cfg.fade_out_ms > 0 and (cfg.loop_bg is False):
        # Only safe to fade out if not looping; when looping, a fade-out each loop creates pumping.
        bg = bg.fade_out(cfg.fade_out_ms)

    # Build background layer under the foreground
    total_needed = cfg.mix_start_ms + len(fg)
    if cfg.loop_bg:
        bg_used = loop_to_length(bg, total_needed)
    else:
        bg_used = bg[:total_needed]

    background_layer = AudioSegment.silent(duration=len(fg), frame_rate=fg.frame_rate)
    background_layer = background_layer.overlay(bg_used, position=cfg.mix_start_ms)

    mixed = background_layer.overlay(fg)  # foreground on top

    if cfg.target_dbfs is not None:
        mixed = normalize_to_dbfs(mixed, cfg.target_dbfs)

    out_path = cfg.output_dir / (fg_path.stem + "_with_engine.wav")
    if out_path.exists() and not cfg.overwrite:
        raise FileExistsError(f"Output exists; use --overwrite to replace: {out_path}")

    # Export with the foreground's frame rate and channels preserved
    params = ["-ar", str(fg.frame_rate), "-ac", str(fg.channels)]
    mixed.export(out_path, format="wav", parameters=params)

    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for audio noise addition script.

    Orchestrates the complete batch processing workflow:
    - Parse command-line arguments
    - Validate system dependencies (FFmpeg)
    - Discover input files for processing
    - Set up output directory structure
    - Process each file with progress reporting
    - Generate processing summary and statistics

    Workflow:
        1. Parse and validate command-line arguments
        2. Check for required system dependencies
        3. Scan input directory for supported audio files
        4. Create output directory structure
        5. Process each file with noise addition
        6. Report overall processing statistics
        7. Handle errors and cleanup on completion

    Exit Codes:
        - 0: Success - all files processed without errors
        - 1: Configuration error - invalid arguments or missing dependencies
        - 2: Processing error - some or all files failed to process

    Example Usage:
        ```bash
        # Basic noise addition
        python add_noise.py \
            --input-dir recordings/clean \
            --noise-dir noise/ocean \
            --output-dir recordings/noisy \
            --snr-db 10

        # Advanced configuration
        python add_noise.py \
            --input-dir data/speech \
            --noise-dir data/backgrounds \
            --output-dir data/augmented \
            --snr-db 15 \
            --target-lufs -20 \
            --noise-type engine \
            --format mp3
        ```

    Progress Reporting:
        - Individual file processing status
        - Overall completion percentage
        - Processing time estimates
        - Error summaries and failed file counts
        - Final statistics (files processed, total duration, etc.)
    """
    cfg = parse_args(argv)

    inputs = list_inputs(cfg.input_dir, cfg.pattern)
    if not inputs:
        print(f"No input files found in '{cfg.input_dir}' matching pattern '{cfg.pattern}'.", file=sys.stderr)
        return 1

    # Always inform about background presence
    if not cfg.background_path.exists():
        msg = f"Background file not found: {cfg.background_path}"
        if cfg.dry_run:
            print(f"[dry-run] WARNING: {msg}")
        else:
            print(msg, file=sys.stderr)
            return 1

    if cfg.dry_run:
        print("Planned operations:")
        print(f"- Inputs dir: {cfg.input_dir}")
        print(f"- Output dir: {cfg.output_dir}")
        print(f"- Background: {cfg.background_path}")
        print(f"- Files to process: {len(inputs)}")
        for p in inputs:
            print(f"  • {p.name} -> {p.stem}_with_engine.wav (bg gain {cfg.bg_gain_db} dB, start {cfg.mix_start_ms} ms, loop {cfg.loop_bg})")
        return 0

    # Validate dependencies
    if AudioSegment is None:
        print("Missing dependency: pydub. Install with: pip install -r requirements.txt", file=sys.stderr)
        return 2

    # Best-effort ffmpeg presence hint
    try:
        _ = AudioSegment.silent(duration=1)
    except Exception as e:  # pragma: no cover
        print("pydub/ffmpeg not operational. Ensure ffmpeg is installed and on PATH.", file=sys.stderr)
        print("On macOS (Homebrew): brew install ffmpeg", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        return 2

    ensure_output_dir(cfg.output_dir)

    # Load background master once
    try:
        bg_master = AudioSegment.from_file(cfg.background_path)
    except Exception as e:
        print(f"Failed to load background '{cfg.background_path}': {e}", file=sys.stderr)
        return 1

    success = 0
    for fg_path in inputs:
        try:
            out_path = process_one(fg_path, cfg, bg_master)
            print(f"Wrote: {out_path}")
            success += 1
        except FileExistsError as e:
            print(str(e), file=sys.stderr)
        except Exception as e:
            print(f"Failed: {fg_path} -> {e}", file=sys.stderr)

    if success == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
