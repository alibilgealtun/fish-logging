from __future__ import annotations
"""Command-line interface to execute ASR numeric evaluation grids.

Now supports:
- --dataset-json for explicit JSON list of items
- --audio-root for audio directory referenced by JSON
- --concat-number to prepend number.wav automatically
- --number-audio custom path to number.wav
- --grid config file (JSON/YAML) defining parameter lists so you don't type 50 args

Default (no args): uses tests/data/numbers.json + tests/audio with concat-number enabled if present.
"""
import argparse
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

from loguru import logger

from .config import expand_parameter_grid
from .pipeline import ASREvaluator

try:  # optional YAML support
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore


def _load_grid_file(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Grid file not found: {p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML not installed; cannot read YAML grid file")
        return yaml.safe_load(text) or {}
    return json.loads(text)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run numeric ASR evaluation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", help="Directory dataset root (fallback discovery)")
    p.add_argument("--dataset-json", dest="dataset_json", help="Explicit JSON list file (audio, expected)")
    p.add_argument("--audio-root", dest="audio_root", help="Base directory for audio files referenced by dataset JSON")
    p.add_argument("--concat-number", action="store_true", help="Prepend number.wav prefix during decoding")
    p.add_argument("--number-audio", dest="number_audio", help="Explicit path to number.wav for concatenation")
    p.add_argument("--grid", help="Grid config file (JSON/YAML) specifying parameter lists")

    p.add_argument("--models", nargs="+", help="Model backends")
    p.add_argument("--sizes", nargs="+", help="Model sizes per backend")
    p.add_argument("--compute-types", nargs="+", help="Compute types (int8,float16,float32)")
    p.add_argument("--devices", nargs="+", help="Devices (cpu,cuda)")
    p.add_argument("--beams", nargs="+", type=int, help="Beam sizes")
    p.add_argument("--chunks", nargs="+", type=int, help="Streaming chunk sizes (ms)")
    p.add_argument("--vad-modes", nargs="+", type=int, help="VAD modes 0-3")
    p.add_argument("--languages", nargs="+", help="Language codes")

    p.add_argument("--real-time", action="store_true", help="Simulate real-time pacing")
    p.add_argument("--max-samples", type=int, help="Limit number of samples for quick test")
    p.add_argument("--output-dir", default=os.environ.get("EVAL_OUTPUT_DIR", "evaluation_outputs"))
    p.add_argument("--plots", action="store_true", help="Generate basic plots")
    return p.parse_args(argv)


def _defaults_if_missing(ns: argparse.Namespace) -> None:
    """Fill in defaults if user provided minimal / no args."""
    # If no model params supplied & no grid, assign sensible defaults
    if ns.grid:
        return
    if not ns.models:
        ns.models = ["faster-whisper"]
    if not ns.sizes:
        ns.sizes = ["base.en"]
    if not ns.compute_types:
        ns.compute_types = ["int8"]
    if not ns.devices:
        ns.devices = ["cpu"]
    if not ns.beams:
        ns.beams = [5]
    if not ns.chunks:
        ns.chunks = [500]
    if not ns.vad_modes:
        ns.vad_modes = [2]
    if not ns.languages:
        ns.languages = ["en"]

    # Dataset JSON auto default if none specified
    if not ns.dataset_json:
        default_json = Path("tests/data/numbers.json")
        if default_json.exists():
            ns.dataset_json = str(default_json)
            if not ns.audio_root:
                ar = Path("tests/audio")
                if ar.exists():
                    ns.audio_root = str(ar)
            if ns.audio_root:
                ns.concat_number = True  # enable automatically for numbers dataset
    # If still nothing, fallback dataset dir
    if not ns.dataset and not ns.dataset_json:
        ns.dataset = "."


def _apply_grid(ns: argparse.Namespace) -> argparse.Namespace:
    if not ns.grid:
        return ns
    cfg = _load_grid_file(ns.grid)
    # CLI args override grid if both present
    def _maybe(attr, key):
        return getattr(ns, attr) if getattr(ns, attr) else cfg.get(key)
    ns.models = _maybe("models", "models") or ["faster-whisper"]
    ns.sizes = _maybe("sizes", "sizes") or ["base.en"]
    ns.compute_types = _maybe("compute_types", "compute_types") or ["int8"]
    ns.devices = _maybe("devices", "devices") or ["cpu"]
    ns.beams = _maybe("beams", "beams") or [5]
    ns.chunks = _maybe("chunks", "chunks") or [500]
    ns.vad_modes = _maybe("vad_modes", "vad_modes") or [2]
    ns.languages = _maybe("languages", "languages") or ["en"]
    ns.dataset_json = ns.dataset_json or cfg.get("dataset_json")
    ns.audio_root = ns.audio_root or cfg.get("audio_root")
    if cfg.get("concat_number") and not ns.concat_number:
        ns.concat_number = True
    if cfg.get("number_audio") and not ns.number_audio:
        ns.number_audio = cfg.get("number_audio")
    return ns


def _generate_configs(ns: argparse.Namespace):
    return expand_parameter_grid(
        models=ns.models,
        sizes=ns.sizes,
        compute_types=ns.compute_types,
        devices=ns.devices,
        beams=ns.beams,
        chunk_sizes_ms=ns.chunks,
        environments=["clean"],
        vad_modes=ns.vad_modes,
        streaming_modes=[True],
        languages=ns.languages,
        dataset_json=ns.dataset_json,
        concat_number=ns.concat_number,
        number_audio_path=ns.number_audio,
        audio_root=ns.audio_root,
    )


def _maybe_plot(df, output_dir: Path):  # pragma: no cover - plotting
    if df.empty:
        return
    try:
        import seaborn as sns  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        logger.warning(f"Plotting skipped (missing libs): {e}")
        return
    try:
        plt.figure(figsize=(6, 4))
        import pandas as pd  # ensure available
        df_plot = df.copy()
        plt.title("Numeric Exact Match by Model Size")
        sns.barplot(data=df_plot, x="model_size", y="numeric_exact_match", hue="model_name")
        plt.tight_layout()
        plt.savefig(output_dir / "nem_by_model_size.png", dpi=150)
        plt.close()

        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df_plot, x="model_size", y="RTF", hue="model_name")
        plt.title("RTF Distribution")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(output_dir / "rtf_distribution.png", dpi=150)
        plt.close()
    except Exception as e:
        logger.warning(f"Plotting failed: {e}")


def main(argv: List[str] | None = None):
    ns = parse_args(argv)
    ns = _apply_grid(ns)
    _defaults_if_missing(ns)

    logger.info("Resolved configuration:")
    logger.info(f" models={ns.models} sizes={ns.sizes} beams={ns.beams} chunks={ns.chunks}")
    logger.info(f" dataset_json={ns.dataset_json} audio_root={ns.audio_root} concat_number={ns.concat_number}")

    configs = _generate_configs(ns)
    if not configs:
        logger.error("No configurations generated.")
        return 1
    logger.info(f"Generated {len(configs)} configurations")

    # Determine dataset root used only for directory discovery fallback
    dataset_root = ns.dataset or (ns.audio_root if ns.audio_root else ".")
    evaluator = ASREvaluator(dataset_root=dataset_root, output_dir=ns.output_dir, real_time=ns.real_time)
    df = evaluator.evaluate_configs(configs, max_samples=ns.max_samples)
    if ns.plots:
        _maybe_plot(df, Path(ns.output_dir))
    logger.info("Evaluation complete")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
