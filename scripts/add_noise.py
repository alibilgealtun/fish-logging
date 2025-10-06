#!/usr/bin/env python3

"""
Mix a background engine sound under all WAV files in a folder.

- Input folder (default: ./audio) is scanned for .wav files.
- Background audio (default: ./boat_engine.flac) is looped and mixed under each input.
- Outputs are written to ./output with the same base filename.

Dependencies:
- pydub (pip install pydub)
- ffmpeg installed and on PATH (required by pydub)

Run `python add_noise.py --help` for options.
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
    files = sorted(input_dir.glob(pattern))
    return [p for p in files if p.is_file()]


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def loop_to_length(seg, target_len_ms: int):
    """Loop an AudioSegment to at least target length, then trim."""
    if len(seg) == 0:
        raise ValueError("Background audio has zero length")
    times = (target_len_ms // len(seg)) + 1
    return (seg * times)[:target_len_ms]


def normalize_to_dbfs(seg, target_dbfs: float):
    # If silence, skip normalization to avoid -inf dBFS issues
    if seg.dBFS == float("-inf"):
        return seg
    change = target_dbfs - seg.dBFS
    return seg.apply_gain(change)


def process_one(fg_path: Path, cfg: MixConfig, bg_master) -> Path:
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

