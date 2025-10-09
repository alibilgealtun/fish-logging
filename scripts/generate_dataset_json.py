#!/usr/bin/env python3
"""
Generate a dataset JSON (list of {"audio": <relative path>, "expected": <number>})
by scanning an audio root directory recursively for .wav files and inferring
expected numbers from filenames (first numeric token in stem).

Usage:
  python scripts/generate_dataset_json.py --audio-root audio/evaluation/clean/numbers \
      --out evaluation/datasets/clean_numbers.json

Notes:
- "audio" paths in JSON will be relative to --audio-root.
- Expected numbers are parsed via regex (e.g., 23.wav -> 23, 12.5.wav -> 12.5).
- Files without any digits in the name are skipped.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any


def find_wavs(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.wav") if p.is_file()])


def infer_expected_from_name(path: Path) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)", path.stem)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def main():
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

