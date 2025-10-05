from __future__ import annotations
"""Evaluation pipeline orchestrating ASR model tests.

This module wires together:
- Dataset discovery (metadata.json files with expected numeric labels)
- Model loading (faster-whisper, whisperx; extensible registry)
- Streaming simulation (chunked playback time accounting)
- Normalization + metric computations
- Resource usage sampling (CPU, RAM, optional GPU)
- Aggregation + parquet / Excel export

Design goals
------------
1. Non-intrusive: does not modify core app code.
2. Deterministic and reproducible: every row carries full config.
3. Extensible: new model backends registered via simple function.
4. Safe failures: missing files, model errors captured as failure rows.
"""
import os
import json
import time
import math
import traceback
import random
import re
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Iterable, Tuple, Optional, Any

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
import psutil

from .config import EvaluationConfig
from .normalization import ASRNormalizer
from .metrics import (
    word_error_rate,
    char_error_rate,
    digit_error_rate,
    numeric_exact_match,
    mean_absolute_error_numbers,
    compute_percentiles,
)
from parser import FishParser  # type: ignore

# Optional GPU monitoring
try:  # pragma: no cover - environment specific
    import pynvml  # type: ignore
    pynvml.nvmlInit()
    _GPU_AVAILABLE = True
except Exception:  # pragma: no cover
    _GPU_AVAILABLE = False

# Audio loading / resampling
try:
    import soundfile as sf  # type: ignore
except Exception as e:  # pragma: no cover
    sf = None  # type: ignore

try:
    from scipy.signal import resample  # type: ignore
except Exception:  # pragma: no cover
    resample = None  # type: ignore

# ---------------------- Model backend registry ----------------------
BackendFn = Any

_MODEL_CACHE: Dict[str, Any] = {}
_MODEL_LOADERS: Dict[str, BackendFn] = {}


def register_backend(name: str):
    def _decorator(fn: BackendFn) -> BackendFn:
        _MODEL_LOADERS[name.lower()] = fn
        return fn
    return _decorator


@register_backend("faster-whisper")
def _load_faster_whisper(model_size: str, device: str, compute_type: str, **_) -> Any:
    from faster_whisper import WhisperModel  # type: ignore
    return WhisperModel(model_size, device=device, compute_type=compute_type)


@register_backend("whisperx")
def _load_whisperx(model_size: str, device: str, compute_type: str, **_) -> Any:
    import whisperx  # type: ignore
    try:
        return whisperx.load_model(model_size, device=device, compute_type=compute_type)
    except Exception:
        return whisperx.load_model(model_size, device=device)


@register_backend("dummy")
def _load_dummy(model_size: str, device: str, compute_type: str, **_) -> Any:
    class _Dummy:
        def __init__(self, token: str = "one"):
            self.token = token
    return _Dummy()


# Placeholder for future model integration (e.g., qwen)
# @register_backend("qwen")
# def _load_qwen(...): pass


# ---------------------- Helper functions ----------------------

def _load_audio(path: Path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    if sf is None:
        raise RuntimeError("soundfile not available; install soundfile to run evaluation")
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != target_sr:
        if resample is None:
            raise RuntimeError("scipy not available for resampling")
        n_samples = int(len(audio) * target_sr / sr)
        audio = resample(audio, n_samples)
        sr = target_sr
    return audio.astype(np.float32), sr


def _format_exception(ex: Exception) -> str:
    return " | ".join([str(ex), traceback.format_exc(limit=1)])


# ---------------------- Dataset Handling ----------------------
class DatasetSample:
    __slots__ = ("audio_path", "expected_number", "duration", "meta", "speaker_id", "accent", "gender", "age", "recording_device", "noise_type", "snr_db", "raw_entry")
    def __init__(self, audio_path: Path, expected_number: Optional[float], duration: float, meta: Dict[str, Any],
                 speaker_id: str, accent: str, gender: str, age: str, recording_device: str, noise_type: str, snr_db: Optional[float], raw_entry: Dict[str, Any] | None = None):
        self.audio_path = audio_path
        self.expected_number = expected_number
        self.duration = duration
        self.meta = meta
        self.speaker_id = speaker_id
        self.accent = accent
        self.gender = gender
        self.age = age
        self.recording_device = recording_device
        self.noise_type = noise_type
        self.snr_db = snr_db
        self.raw_entry = raw_entry or {}


def discover_dataset(root: Path) -> List[DatasetSample]:
    samples: List[DatasetSample] = []
    for meta_path in root.rglob("metadata.json"):
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            logger.warning(f"Failed reading metadata {meta_path}: {e}")
            continue
        speaker_id = meta.get("speaker_id", "unknown")
        sp = meta.get("speaker_profile", {})
        accent = sp.get("ethnicity") or sp.get("accent", "unknown")
        gender = sp.get("gender", "unk")
        age = sp.get("age", "unk")
        rec_cond = meta.get("recording_conditions", {})
        recording_device = rec_cond.get("microphone", "unk")
        noise_type = rec_cond.get("environment", "clean")
        snr_db = rec_cond.get("snr_db")
        contents = meta.get("contents", {})
        audios = contents.get("audios", [])
        for entry in audios:
            audio_rel = entry.get("audio")
            expected = entry.get("expected")
            dur = float(entry.get("duration", 0.0))
            if audio_rel is None:
                continue
            audio_path = meta_path.parent / audio_rel
            if not audio_path.exists():
                logger.warning(f"Missing audio file {audio_path}")
                continue
            try:
                samples.append(
                    DatasetSample(audio_path, float(expected) if expected is not None else None, dur, meta,
                                  speaker_id, accent, gender, age, recording_device, noise_type, snr_db)
                )
            except Exception:
                continue
    # Fallback: no metadata.json files discovered -> infer from wav filenames
    if not samples:
        wav_files = list(root.rglob('*.wav'))
        for wav in wav_files:
            try:
                match = re.search(r'(\d+(?:\.\d+)?)', wav.stem)
                if not match:
                    continue
                expected_val = float(match.group(1))
                # Get duration (frames / samplerate) lightweight
                if sf is not None:
                    try:
                        info = sf.info(str(wav))
                        duration = info.frames / info.samplerate if info.samplerate else 0.0
                    except Exception:
                        duration = 0.0
                else:
                    duration = 0.0
                speaker_id = root.name  # use folder name as speaker id
                samples.append(
                    DatasetSample(
                        audio_path=wav,
                        expected_number=expected_val,
                        duration=duration,
                        meta={},
                        speaker_id=speaker_id,
                        accent="unknown",
                        gender="unk",
                        age="unk",
                        recording_device="unk",
                        noise_type="clean",
                        snr_db=None,
                    )
                )
            except Exception:
                continue
    return samples


def _load_json_dataset(json_path: Path, audio_root: Path | None) -> List[DatasetSample]:
    samples: List[DatasetSample] = []
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed loading dataset json {json_path}: {e}")
        return samples
    if not isinstance(data, list):
        logger.error(f"Dataset json must be a list of objects: {json_path}")
        return samples
    for entry in data:
        if not isinstance(entry, dict):
            continue
        audio_name = entry.get("audio")
        expected = entry.get("expected")
        if not audio_name:
            continue
        audio_path = (audio_root / audio_name) if audio_root else (json_path.parent / audio_name)
        if not audio_path.exists():
            logger.warning(f"JSON dataset audio missing: {audio_path}")
            continue
        try:
            dur = 0.0
            if sf is not None:
                try:
                    info = sf.info(str(audio_path))
                    dur = info.frames / info.samplerate if info.samplerate else 0.0
                except Exception:
                    dur = 0.0
            samples.append(
                DatasetSample(
                    audio_path=audio_path,
                    expected_number=float(expected) if expected is not None else None,
                    duration=dur,
                    meta={},
                    speaker_id=audio_root.name if audio_root else json_path.stem,
                    accent="unknown",
                    gender="unk",
                    age="unk",
                    recording_device="unk",
                    noise_type="clean",
                    snr_db=None,
                    raw_entry=entry,
                )
            )
        except Exception:
            continue
    return samples


# ---------------------- Streaming Simulation ----------------------
class StreamingSimulator:
    def __init__(self, chunk_ms: int = 500, real_time: bool = False):
        self.chunk_ms = chunk_ms
        self.real_time = real_time

    def run(self, audio: np.ndarray, sample_rate: int, decode_fn) -> Dict[str, Any]:
        """Simulate feeding audio in fixed chunks then perform single decode.

        Limitations: This does not perform true incremental decoding (would
        require specialized streaming APIs). First token latency is approximated
        as the time at which full decode finishes for now.
        """
        total_samples = len(audio)
        chunk_size = int(sample_rate * self.chunk_ms / 1000.0)
        start_wall = time.time()
        fed_samples = 0
        while fed_samples < total_samples:
            start_idx = fed_samples
            end = min(start_idx + chunk_size, total_samples)
            if self.real_time:
                chunk_duration = (end - start_idx) / sample_rate
                time.sleep(chunk_duration)
            fed_samples = end
        decode_start = time.time()
        text = decode_fn(audio, sample_rate)
        decode_end = time.time()
        processing_time = decode_end - decode_start
        total_elapsed = decode_end - start_wall
        audio_duration = total_samples / sample_rate
        rtf = processing_time / audio_duration if audio_duration > 0 else float("inf")
        return {
            "recognized_text_raw": text,
            "audio_start_time": start_wall,
            "first_token_time": decode_end,  # placeholder approximation
            "final_result_time": decode_end,
            "processing_time_s": processing_time,
            "total_elapsed_s": total_elapsed,
            "audio_duration_s": audio_duration,
            "rtf": rtf,
            "latency_ms": (decode_end - start_wall) * 1000.0,
        }


# --- Number prefix handling ---
def _maybe_load_number_prefix(cfg: EvaluationConfig) -> tuple[Optional[np.ndarray], int]:
    if not cfg.concat_number:
        return None, 0
    candidate_paths: List[Path] = []
    if cfg.number_audio_path:
        candidate_paths.append(Path(cfg.number_audio_path))
    else:
        # default search order
        if cfg.audio_root:
            candidate_paths.append(Path(cfg.audio_root) / "number.wav")
        candidate_paths.append(Path("number_prefix.wav"))
        candidate_paths.append(Path("tests/audio/number.wav"))
    for p in candidate_paths:
        if p.exists():
            try:
                n_audio, sr = sf.read(str(p), dtype='float32') if sf else (None, 0)
                if n_audio is None:
                    continue
                if sr != 16000 and resample is not None:
                    n_audio = resample(n_audio, int(len(n_audio) * 16000 / sr))
                    sr = 16000
                if n_audio.ndim > 1:
                    n_audio = n_audio[:, 0]
                logger.info(f"Using number prefix audio: {p}")
                return n_audio.astype(np.float32), sr
            except Exception as e:
                logger.warning(f"Failed loading number prefix {p}: {e}")
    logger.warning("concat_number=True but no number audio found; continuing without prefix")
    return None, 0


# ---------------------- Evaluator ----------------------
class ASREvaluator:
    def __init__(self, dataset_root: str | Path, output_dir: str | Path = "evaluation_outputs", real_time: bool = False):
        self.dataset_root = Path(dataset_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.real_time = real_time
        self.normalizer = ASRNormalizer()
        self.fish_parser = FishParser()

    # ---- Model management ----
    def _get_model(self, cfg: EvaluationConfig):
        key = f"{cfg.model.name}|{cfg.model.size}|{cfg.model.compute_type}|{cfg.model.device}"
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        loader = _MODEL_LOADERS.get(cfg.model.name.lower())
        if not loader:
            raise ValueError(f"No backend loader registered for {cfg.model.name}")
        logger.info(f"Loading model {key}")
        model = loader(model_size=cfg.model.size, device=cfg.model.device, compute_type=cfg.model.compute_type)
        _MODEL_CACHE[key] = model
        return model

    def _decode(self, model, cfg: EvaluationConfig, audio: np.ndarray, sample_rate: int) -> str:
        name = cfg.model.name.lower()
        if name == "dummy":
            # Always return a fixed token; useful for lightweight tests
            return getattr(model, "token", "one")
        if name == "faster-whisper":
            segments, _info = model.transcribe(
                audio,
                beam_size=cfg.inference_config.beam_size,
                language=cfg.model.language,
                condition_on_previous_text=cfg.inference_config.condition_on_previous_text,
                vad_filter=cfg.inference_config.vad_filter,
                vad_parameters=cfg.inference_config.vad_parameters,
                without_timestamps=True,
            )
            text = " ".join(getattr(s, "text", "") for s in segments if getattr(s, "text", ""))
            return text.strip()
        elif name == "whisperx":
            try:
                result = model.transcribe(audio, language=cfg.model.language)
            except TypeError:
                result = model.transcribe(audio)
            if isinstance(result, dict) and "segments" in result:
                return " ".join(seg.get("text", "") for seg in result.get("segments", [])).strip()
            if isinstance(result, dict) and "text" in result:
                return str(result.get("text", "")).strip()
            if isinstance(result, list):
                return " ".join(getattr(s, "text", str(s)) for s in result).strip()
            return str(result).strip()
        else:
            raise ValueError(f"Decoding not implemented for backend {name}")

    def _get_dataset_samples(self, cfg: EvaluationConfig) -> List[DatasetSample]:
        # Priority 1: JSON dataset if provided
        if cfg.dataset_json:
            json_path = Path(cfg.dataset_json)
            audio_root = Path(cfg.audio_root) if cfg.audio_root else json_path.parent
            samples = _load_json_dataset(json_path, audio_root)
            if not samples:
                logger.warning(f"No samples loaded from dataset_json={json_path}")
            return samples
        # Else fallback to directory discovery logic
        return discover_dataset(self.dataset_root)

    def evaluate_configs(self, configs: Iterable[EvaluationConfig], max_samples: int | None = None) -> pd.DataFrame:
        dataset_dir = self.dataset_root
        rows: List[Dict[str, Any]] = []

        for cfg in configs:
            model = self._get_model(cfg)
            simulator = StreamingSimulator(chunk_ms=cfg.inference_config.chunk_size_ms, real_time=self.real_time)
            number_prefix, _ = _maybe_load_number_prefix(cfg)
            samples = self._get_dataset_samples(cfg)
            if max_samples is not None and samples:
                samples = samples[:max_samples]
            logger.info(f"Evaluating config {cfg.test_run_id} on {len(samples)} samples (json_mode={bool(cfg.dataset_json)})")

            for sample in tqdm(samples, desc=f"{cfg.model.name}-{cfg.model.size}"):
                try:
                    audio, sr = _load_audio(sample.audio_path)
                except Exception as e:
                    rows.append({
                        "test_run_id": cfg.test_run_id,
                        "wav_id": sample.audio_path.name,
                        "error": _format_exception(e),
                        "model_name": cfg.model.name,
                        "model_size": cfg.model.size,
                        "config_json": cfg.to_json(),
                        "speaker_id": sample.speaker_id,
                        "accent": sample.accent,
                        "gender": sample.gender,
                        "age": sample.age,
                        "recording_device": sample.recording_device,
                        "noise_type": sample.noise_type,
                        "snr_db": sample.snr_db,
                    })
                    continue

                # Resource snapshot before decode
                proc = psutil.Process(os.getpid())
                cpu_before = psutil.cpu_percent(interval=0.0)
                mem_info = proc.memory_info()
                gpu_util = None
                gpu_mem = None
                if _GPU_AVAILABLE:  # pragma: no cover
                    try:
                        h = pynvml.nvmlDeviceGetHandleByIndex(0)
                        util = pynvml.nvmlDeviceGetUtilizationRates(h)
                        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                        gpu_util = util.gpu
                        gpu_mem = mem.used / (1024 ** 2)
                    except Exception:
                        pass

                def _decode_fn(aud, sr_):
                    # Prepend number prefix if requested
                    if number_prefix is not None:
                        try:
                            pref = number_prefix
                            if pref.dtype != aud.dtype:
                                pref = pref.astype(aud.dtype)
                            aud = np.concatenate([pref, aud])
                        except Exception as e:
                            logger.debug(f"Prefix concat failed: {e}")
                    return self._decode(model, cfg, aud, sr_)

                timing = simulator.run(audio, sr, _decode_fn)
                recognized_raw = timing["recognized_text_raw"]
                cleaned_text = recognized_raw.rstrip('.,?!').strip()
                # Parser pass
                parser_species = None
                parser_length_cm = None
                parser_exact_match = None
                try:
                    parsed = self.fish_parser.parse_text(cleaned_text)
                    parser_species = getattr(parsed, 'species', None)
                    parser_length_cm = getattr(parsed, 'length_cm', None)
                    if parser_length_cm is not None and sample.expected_number is not None:
                        parser_exact_match = 1 if float(parser_length_cm) == float(sample.expected_number) else 0
                except Exception as e:
                    logger.debug(f"Parser error on {sample.audio_path.name}: {e}")

                norm = self.normalizer.normalize(recognized_raw)

                expected_number = sample.expected_number
                predicted_number = norm.predicted_number
                expected_text = "" if expected_number is None else str(expected_number)
                wer = word_error_rate(norm.normalized_text, expected_text)
                cer = char_error_rate(norm.normalized_text, expected_text)
                der = digit_error_rate(predicted_number, expected_number)
                nem = numeric_exact_match(predicted_number, expected_number)
                error_type = self.normalizer.classify_error(predicted_number, expected_number, recognized_raw, expected_text)

                absolute_error = None
                if predicted_number is not None and expected_number is not None:
                    absolute_error = abs(predicted_number - expected_number)

                row = {
                    "test_run_id": cfg.test_run_id,
                    "wav_id": sample.audio_path.name,
                    "recognized_text_raw": recognized_raw,
                    "recognized_text_normalized": norm.normalized_text,
                    "predicted_number": predicted_number,
                    "expected_number": expected_number,
                    "speaker_id": sample.speaker_id,
                    "gender": sample.gender,
                    "age": sample.age,
                    "accent": sample.accent,
                    "recording_device": sample.recording_device,
                    "noise_type": sample.noise_type,
                    "snr_db": sample.snr_db,
                    "duration_s": timing["audio_duration_s"],
                    "model_name": cfg.model.name,
                    "model_size": cfg.model.size,
                    "config_json": cfg.to_json(),
                    "vad_setting": cfg.vad_mode,
                    "processing_time_s": timing["processing_time_s"],
                    "RTF": timing["rtf"],
                    "latency_ms": timing["latency_ms"],
                    "cpu_percent": cpu_before,
                    "gpu_percent": gpu_util,
                    "memory_mb": mem_info.rss / (1024 ** 2),
                    "gpu_memory_mb": gpu_mem,
                    "numeric_exact_match": nem,
                    "parser_species": parser_species,
                    "parser_length_cm": parser_length_cm,
                    "parser_exact_match": parser_exact_match,
                    "WER": wer,
                    "CER": cer,
                    "DER": der,
                    "absolute_error": absolute_error,
                    "error_type": error_type,
                    "notes": None,
                }
                rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            self._write_outputs(df)
        return df

    # ---- Output generation ----
    def _write_outputs(self, df: pd.DataFrame) -> None:
        results_path = self.output_dir / "results.parquet"
        failures_path = self.output_dir / "failures.parquet"
        summary_path = self.output_dir / "summary.xlsx"

        df.to_parquet(results_path, index=False)
        failures = df[df["numeric_exact_match"] == 0]
        failures.to_parquet(failures_path, index=False)

        # Aggregations
        agg_funcs = {
            "numeric_exact_match": "mean",
            "parser_exact_match": "mean",
            "WER": "mean",
            "CER": "mean",
            "DER": "mean",
            "absolute_error": "mean",
            "RTF": ["mean"],
            "processing_time_s": ["mean"],
        }
        summary_model = df.groupby(["model_name", "model_size"]).agg(agg_funcs)
        summary_accent = df.groupby(["accent", "model_name", "model_size"]).agg(agg_funcs)
        summary_noise = df.groupby(["noise_type", "model_name", "model_size"]).agg(agg_funcs)

        # RTF percentiles overall
        rtf_percentiles = compute_percentiles(df["RTF"].dropna().tolist())
        rtf_meta = pd.DataFrame([
            {"p50": rtf_percentiles.p50, "p95": rtf_percentiles.p95, "p99": rtf_percentiles.p99}
        ])

        # RTF percentiles per model
        rtf_model_records = []
        for (mname, msize), sub in df.groupby(["model_name", "model_size"]):
            pct = compute_percentiles(sub["RTF"].dropna().tolist())
            rtf_model_records.append({
                "model_name": mname,
                "model_size": msize,
                "p50": pct.p50,
                "p95": pct.p95,
                "p99": pct.p99,
            })
        rtf_model_df = pd.DataFrame(rtf_model_records)

        with pd.ExcelWriter(summary_path, engine="openpyxl") as xl:
            df.head(200).to_excel(xl, sheet_name="sample", index=False)
            summary_model.to_excel(xl, sheet_name="by_model")
            summary_accent.to_excel(xl, sheet_name="by_accent")
            summary_noise.to_excel(xl, sheet_name="by_noise")
            rtf_meta.to_excel(xl, sheet_name="rtf_percentiles", index=False)
            rtf_model_df.to_excel(xl, sheet_name="rtf_by_model", index=False)

        logger.info(f"Wrote: {results_path}, {failures_path}, {summary_path}")
