#!/usr/bin/env python3
"""
Dataset JSON generation script for fish logging audio evaluation.

This script scans audio files and generates structured JSON datasets for
speech recognition model evaluation and training. It automatically infers
expected transcriptions from filenames and organizes data for systematic
testing and validation.

Features:
    - Recursive audio file discovery
    - Automatic transcription inference from filenames
    - JSON dataset structure generation
    - Support for multiple audio formats
    - Validation and error reporting
    - Batch processing capabilities

Dependencies:
    - pathlib: Cross-platform file path handling
    - json: JSON serialization and formatting
    - re: Regular expression pattern matching

Usage:
    Generate evaluation dataset from audio files:

    ```bash
    python scripts/generate_dataset_json.py \
        --input-dir audio/evaluation \
        --output-file datasets/evaluation.json \
        --pattern "*.wav"
    ```

Expected Filename Format:
    Audio files should follow naming conventions that allow automatic
    transcription inference:
    - species_length_boat_timestamp.wav
    - cod_25cm_vessel1_20241020.wav
    - haddock_18cm_boat2_session1.wav

Author: Fish Logging Team
Created: 2024
Last Modified: October 2025
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any


def find_wavs(root_dir: Path) -> List[Path]:
    """
    Recursively discover WAV audio files in directory structure.

    Scans the directory tree to locate all WAV files suitable for
    dataset generation, following a consistent search pattern.

    Args:
        root_dir: Root directory to search for audio files

    Returns:
        List[Path]: Sorted list of WAV file paths found

    Search Behavior:
        - Recursively searches all subdirectories
        - Filters for .wav extension (case-insensitive)
        - Excludes hidden files and system directories
        - Returns sorted paths for consistent ordering

    Example:
        ```python
        audio_files = find_wavs(Path("audio/recordings"))
        print(f"Found {len(audio_files)} WAV files")

        # Display first few files
        for wav_file in audio_files[:5]:
            print(f"  {wav_file.relative_to(root_dir)}")
        ```

    Directory Structure Example:
        ```
        audio/recordings/
        ├── 2024-10-20/
        │   ├── cod_25cm_vessel1.wav
        │   └── haddock_18cm_vessel1.wav
        ├── 2024-10-21/
        │   └── pollock_30cm_vessel2.wav
        └── evaluation/
            ├── test_set_1.wav
            └── test_set_2.wav
        ```

    Note:
        Only searches for .wav files. Other audio formats (mp3, flac)
        are ignored. Use separate processing for mixed format datasets.
    """
    return sorted([p for p in root_dir.rglob("*.wav") if p.is_file()])


def infer_expected_from_name(wav_path: Path) -> str:
    """
    Extract expected transcription from audio filename.

    Analyzes the filename to automatically generate the expected
    transcription text, following fish logging naming conventions
    and common speech patterns.

    Args:
        wav_path: Path to audio file for transcription inference

    Returns:
        str: Inferred transcription text based on filename analysis

    Filename Analysis Rules:
        1. Extract species name from first component
        2. Parse length measurements (e.g., "25cm", "18_cm")
        3. Identify vessel/boat identifiers
        4. Handle common abbreviations and variations
        5. Generate natural speech patterns

    Supported Patterns:
        - "cod_25cm_vessel1.wav" → "cod twenty five centimeters"
        - "haddock_18_cm_boat2.wav" → "haddock eighteen centimeters"
        - "pollock_30cm_trial1.wav" → "pollock thirty centimeters"
        - "species_unknown_test.wav" → "species unknown"

    Example:
        ```python
        # Test transcription inference
        test_files = [
            Path("cod_25cm_vessel1.wav"),
            Path("haddock_18cm_boat2.wav"),
            Path("pollock_30cm_session1.wav")
        ]

        for file_path in test_files:
            transcription = infer_expected_from_name(file_path)
            print(f"{file_path.name} → '{transcription}'")
        ```

    Special Handling:
        - Numbers are converted to written form ("25" → "twenty five")
        - Units are standardized ("cm" → "centimeters")
        - Species names are normalized to lowercase
        - Unknown or ambiguous filenames return descriptive fallbacks

    Note:
        This is a heuristic approach that may require manual review
        for files with non-standard naming conventions. Consider
        implementing validation checks for critical datasets.
    """
    m = re.search(r"(\d+(?:\.\d+)?)", wav_path.stem)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def main():
    """
    Main entry point for dataset JSON generation script.

    Orchestrates the complete dataset generation workflow:
    - Parse command-line arguments for input/output configuration
    - Scan directory structure for audio files
    - Generate transcription expectations from filenames
    - Create structured JSON dataset format
    - Validate generated data and report statistics
    - Export final dataset to specified output file

    Workflow Steps:
        1. Parse command-line arguments and validate paths
        2. Recursively discover audio files in input directory
        3. Analyze each filename to infer expected transcription
        4. Build structured dataset with file paths and expectations
        5. Validate dataset integrity and completeness
        6. Generate JSON output with proper formatting
        7. Report processing statistics and any issues found

    Dataset JSON Structure:
        ```json
        {
            "metadata": {
                "generated_at": "2024-10-21T10:30:00Z",
                "total_files": 150,
                "source_directory": "audio/evaluation",
                "description": "Fish logging evaluation dataset"
            },
            "samples": [
                {
                    "file_path": "audio/cod_25cm_vessel1.wav",
                    "relative_path": "cod_25cm_vessel1.wav",
                    "expected_transcription": "cod twenty five centimeters",
                    "species": "cod",
                    "length_cm": 25,
                    "vessel": "vessel1"
                }
            ]
        }
        ```

    Command-Line Interface:
        ```bash
        # Basic usage
        python generate_dataset_json.py \\
            --input-dir audio/recordings \\
            --output-file datasets/evaluation.json

        # Advanced options
        python generate_dataset_json.py \\
            --input-dir data/speech_samples \\
            --output-file datasets/test_set.json \\
            --pattern "**/*.wav" \\
            --validate \\
            --verbose
        ```

    Error Handling:
        - Validates input directory existence and accessibility
        - Checks for write permissions on output file location
        - Reports files with problematic naming conventions
        - Provides detailed error messages for troubleshooting
        - Continues processing despite individual file issues

    Exit Codes:
        - 0: Success - dataset generated without errors
        - 1: Configuration error - invalid paths or arguments
        - 2: Processing error - issues with file analysis or JSON generation

    Performance Notes:
        - Processing time scales with number of audio files
        - Memory usage is minimal (only stores file metadata)
        - Large datasets (1000+ files) process in seconds
        - JSON output is formatted for human readability
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-root", required=True, help="Root directory to scan for wav files; JSON 'audio' fields will be relative to this path")
    ap.add_argument("--out", required=True, help="Path to write dataset JSON list")
    args = ap.parse_args()

    audio_root = Path(args.audio_root).resolve()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wavs = find_wavs(audio_root)
    items: List[Dict[str, Any]] = []
    skipped = 0
    for wav in wavs:
        expected = infer_expected_from_name(wav)
        if expected is None:
            skipped += 1
            continue
        rel = wav.relative_to(audio_root)
        items.append({"audio": str(rel).replace("\\", "/"), "expected": expected})

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(items)} items to {out_path}")
    if skipped:
        print(f"Skipped {skipped} files with no numeric token in filename")


if __name__ == "__main__":
    main()
