from __future__ import annotations
"""Command-line interface to execute ASR numeric evaluation grids.

Now supports:
- --dataset-json for explicit JSON list of items
- --audio-root for audio directory referenced by JSON
- --concat-number to disable prepending number.wav automatically (default True)
- --number-audio custom path to number.wav
- --grid config file (JSON/YAML) defining parameter lists so you don't type 50 args

Default (no args): uses tests/data/numbers.json + tests/audio with concat-number enabled if present.
"""
import argparse
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any

from loguru import logger

from .config import expand_parameter_grid, expand_model_spec_grid
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
    # Inverted semantics: default True, passing --concat-number DISABLES concatenation
    p.add_argument("--concat-number", dest="concat_number", action="store_false", default=True, help="Disable prepending number.wav prefix during decoding")
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
    p.add_argument("--preset", help="Named preset (quick, full) located in evaluation/presets")
    p.add_argument("--production-replay", action="store_true", help="Replay audio through production NoiseController segmentation (closer to main.py)")
    p.add_argument("--model-specs", dest="model_specs_path", help="JSON/YAML file with per-model specs list (alternative to flat lists)")
    ns = p.parse_args(argv)
    # Track whether user explicitly provided the concat flag (to disable)
    raw_argv = argv if argv is not None else sys.argv[1:]
    setattr(ns, "concat_number_cli_provided", "--concat-number" in (raw_argv or []))
    return ns


def _load_preset(name: str) -> Dict[str, Any]:
    preset_path = Path(__file__).parent / "presets" / f"{name}.json"
    if not preset_path.exists():
        raise FileNotFoundError(f"Preset not found: {preset_path}")
    return json.loads(preset_path.read_text(encoding="utf-8"))


def _load_central_specs() -> Dict[str, Any]:
    """Load central unified config (evaluation/presets/model_specs.json) if present.

    Expected structure:
    {
      "dataset_json": str,
      "audio_root": str,
      "concat_number": bool,
      "production_replay": bool,
      "model_specs": [ {...}, {...} ]
    }
    """
    central_path = Path(__file__).parent / "presets" / "model_specs.json"
    if not central_path.exists():
        return {}
    try:
        return json.loads(central_path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        logger.error(f"Failed loading central model_specs.json: {e}")
        return {}


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
            # Enable automatically for numbers dataset unless user disabled via CLI
            if ns.audio_root and not getattr(ns, "concat_number_cli_provided", False):
                ns.concat_number = True
    # If still nothing, fallback dataset dir
    if not ns.dataset and not ns.dataset_json:
        ns.dataset = "."


def _apply_grid(ns: argparse.Namespace) -> argparse.Namespace:
    if not ns.grid:
        return ns
    cfg = _load_grid_file(ns.grid)
    # If grid provides model_specs, stash it (only used later if no explicit --model-specs passed)
    if 'model_specs' in cfg and not getattr(ns, 'model_specs_path', None):
        ns.model_specs_inline = cfg['model_specs']  # dynamically attach
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
    # Respect CLI override; otherwise, honor grid value if provided
    if "concat_number" in cfg and not getattr(ns, "concat_number_cli_provided", False):
        ns.concat_number = bool(cfg.get("concat_number"))
    if cfg.get("number_audio") and not ns.number_audio:
        ns.number_audio = cfg.get("number_audio")
    return ns


def _generate_configs(ns: argparse.Namespace):
    # Priority: explicit --model-specs file > inline model_specs from grid > flat list expansion
    model_specs: list | None = None
    if getattr(ns, 'model_specs_path', None):
        try:
            model_specs = _load_grid_file(ns.model_specs_path).get('model_specs')
            if model_specs is None:  # allow raw list in file
                # If the file itself is a list, re-load raw
                p = Path(ns.model_specs_path)
                text = p.read_text(encoding='utf-8')
                if p.suffix.lower() in {'.yaml', '.yml'} and yaml:
                    data = yaml.safe_load(text)
                else:
                    data = json.loads(text)
                if isinstance(data, list):
                    model_specs = data
        except Exception as e:
            logger.error(f"Failed loading model-specs file: {e}")
    elif hasattr(ns, 'model_specs_inline'):
        model_specs = ns.model_specs_inline
    if model_specs:
        setattr(ns, '_used_model_specs_count', len(model_specs))
        return expand_model_spec_grid(
            model_specs=model_specs,
            dataset_json=ns.dataset_json,
            concat_number=ns.concat_number,
            number_audio_path=ns.number_audio,
            audio_root=ns.audio_root,
            production_replay=ns.production_replay,
        )
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
        production_replay=ns.production_replay,
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
    # 1) Load central model_specs.json (base layer)
    central = _load_central_specs()
    if central:
        # Attach central model_specs inline if user did NOT pass --model-specs
        if not getattr(ns, 'model_specs_path', None) and 'model_specs' in central:
            ns.model_specs_inline = central['model_specs']
        # Base-level keys (only set if not already provided via CLI args)
        if not getattr(ns, 'dataset_json', None) and central.get('dataset_json'):
            ns.dataset_json = central.get('dataset_json')
        if not getattr(ns, 'audio_root', None) and central.get('audio_root'):
            ns.audio_root = central.get('audio_root')
        # Flags (apply only if user did not explicitly override via CLI)
        if not getattr(ns, "concat_number_cli_provided", False) and "concat_number" in central:
            ns.concat_number = bool(central.get('concat_number'))
        if not ns.production_replay and central.get('production_replay'):
            ns.production_replay = bool(central.get('production_replay'))
    # 2) Apply grid (overrides central where specified)
    ns = _apply_grid(ns)
    # 3) Apply defaults (skip flat defaults if we already have model_specs so we don't pollute)
    if not (hasattr(ns, 'model_specs_inline') or getattr(ns, 'model_specs_path', None)):
        _defaults_if_missing(ns)
    else:
        # Ensure minimal essentials if central/preset lacked them
        if not getattr(ns, 'dataset_json', None):
            # fallback to central or test numbers
            default_json = Path('tests/data/numbers.json')
            if default_json.exists():
                ns.dataset_json = str(default_json)
                if not getattr(ns, 'audio_root', None):
                    ar = Path('tests/audio')
                    if ar.exists():
                        ns.audio_root = str(ar)
                # Only enable if user did not explicitly disable via CLI
                if ns.audio_root and not getattr(ns, "concat_number_cli_provided", False):
                    ns.concat_number = True

    logger.info("Resolved configuration:")
    logger.info(f" models={ns.models} sizes={ns.sizes} beams={ns.beams} chunks={ns.chunks}")
    logger.info(f" dataset_json={ns.dataset_json} audio_root={ns.audio_root} concat_number={ns.concat_number}")
    logger.info(f" production_replay={ns.production_replay}")
    if hasattr(ns, 'model_specs_inline') or getattr(ns, 'model_specs_path', None):
        src = 'cli_file' if getattr(ns, 'model_specs_path', None) else 'central_default'
        logger.info(f" model_specs_source={src}")

    # Create a unique subdirectory for this invocation so previous runs are preserved
    ts_base = time.strftime('run_%Y_%m_%d_%H%M%S')
    base_out = Path(ns.output_dir)
    attempt = 0
    while True:
        run_invocation_id = ts_base if attempt == 0 else f"{ts_base}_{attempt}"
        run_out_dir = base_out / run_invocation_id
        if not run_out_dir.exists():
            run_out_dir.mkdir(parents=True, exist_ok=True)
            break
        attempt += 1
    logger.info(f" run_output_dir={run_out_dir}")

    if ns.preset:
        try:
            preset_cfg = _load_preset(ns.preset)
            # Only fill missing cli args / grid-injected values
            for k, v in preset_cfg.items():
                if getattr(ns, k, None) in (None, [], False):
                    setattr(ns, k, v)
        except Exception as e:
            logger.error(f"Failed loading preset '{ns.preset}': {e}")

    logger.info(f" preset={ns.preset} grid={ns.grid}")

    configs = _generate_configs(ns)
    if hasattr(ns, '_used_model_specs_count'):
        logger.info(f" using model_specs entries: {ns._used_model_specs_count}")
    if not configs:
        logger.error("No configurations generated.")
        return 1
    logger.info(f"Generated {len(configs)} configurations")

    # Persist the resolved model specifications / parameter sources for traceability
    try:
        specs_dump_path = run_out_dir / "model_specs_used.json"
        payload = {}
        if hasattr(ns, 'model_specs_inline'):
            payload = {"model_specs": getattr(ns, 'model_specs_inline')}
        elif getattr(ns, 'model_specs_path', None):
            try:
                raw_text = Path(ns.model_specs_path).read_text(encoding='utf-8')
                # attempt to parse; if fail store raw
                try:
                    payload = json.loads(raw_text)
                except Exception:
                    payload = {"raw": raw_text}
            except Exception as e:
                payload = {"error": f"failed to read original model_specs file: {e}"}
        else:
            # Flat parameter mode -> synthesize a spec summary
            payload = {
                "flat_parameters": {
                    "models": ns.models,
                    "sizes": ns.sizes,
                    "compute_types": ns.compute_types,
                    "devices": ns.devices,
                    "beams": ns.beams,
                    "chunks": ns.chunks,
                    "vad_modes": ns.vad_modes,
                    "languages": ns.languages,
                }
            }
        # Attach top-level run context
        payload["context"] = {
            "dataset_json": ns.dataset_json,
            "audio_root": ns.audio_root,
            "concat_number": ns.concat_number,
            "production_replay": ns.production_replay,
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ')
        }
        specs_dump_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding='utf-8')
        logger.info(f" wrote model specs snapshot -> {specs_dump_path}")
    except Exception as e:
        logger.debug(f"Failed writing model_specs_used.json: {e}")

    # Determine dataset root used only for directory discovery fallback
    dataset_root = ns.dataset or (ns.audio_root if ns.audio_root else ".")
    evaluator = ASREvaluator(dataset_root=dataset_root, output_dir=run_out_dir, real_time=ns.real_time)
    df = evaluator.evaluate_configs(configs, max_samples=ns.max_samples)
    if ns.plots:
        _maybe_plot(df, run_out_dir)
    logger.info("Evaluation complete")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
