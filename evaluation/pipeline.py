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
from parser.text_normalizer import TextNormalizer  # type: ignore
from speech.factory import create_recognizer  # type: ignore
from speech.base_recognizer import BaseSpeechRecognizer  # type: ignore

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
    __slots__ = ("audio_path", "expected_number", "duration", "meta", "speaker_id", "accent", "gender", "age", "recording_device", "noise_type", "snr_db", "raw_entry", "source")
    def __init__(self, audio_path: Path, expected_number: Optional[float], duration: float, meta: Dict[str, Any],
                 speaker_id: str, accent: str, gender: str, age: str, recording_device: str, noise_type: str, snr_db: Optional[float], raw_entry: Dict[str, Any] | None = None, source: str = "metadata"):
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
        self.source = source


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
                                  speaker_id, accent, gender, age, recording_device, noise_type, snr_db, source="metadata")
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
                        source="filename_fallback",
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
                    source="json",
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
            model = None if cfg.production_replay else self._get_model(cfg)
            simulator = None if cfg.production_replay else StreamingSimulator(chunk_ms=cfg.inference_config.chunk_size_ms, real_time=self.real_time)
            number_prefix, _ = (None, 0) if cfg.production_replay else _maybe_load_number_prefix(cfg)
            samples = self._get_dataset_samples(cfg)
            if max_samples is not None and samples:
                samples = samples[:max_samples]
            logger.info(f"Evaluating config {cfg.test_run_id} on {len(samples)} samples (json_mode={bool(cfg.dataset_json)}) prod_replay={cfg.production_replay}")

            # Production replay branch
            if cfg.production_replay:
                # Map model name to factory engine
                engine_map = {"faster-whisper": "whisper", "whisperx": "whisperx"}
                engine = engine_map.get(cfg.model.name.lower(), cfg.model.name.lower())
                try:
                    recognizer = create_recognizer(engine)
                except Exception as e:
                    logger.error(f"Failed to create recognizer for engine={engine}: {e}")
                    continue
                # Override basic decoding params (non-invasive)
                try:
                    if hasattr(recognizer, 'MODEL_NAME'):
                        recognizer.MODEL_NAME = cfg.model.size
                    if hasattr(recognizer, 'DEVICE'):
                        recognizer.DEVICE = cfg.model.device
                    if hasattr(recognizer, 'COMPUTE_TYPE'):
                        recognizer.COMPUTE_TYPE = cfg.model.compute_type
                    if hasattr(recognizer, 'BEAM_SIZE'):
                        recognizer.BEAM_SIZE = cfg.inference_config.beam_size
                    # Ensure VAD aggressiveness from evaluation config is reflected
                    if hasattr(recognizer, 'VAD_MODE'):
                        try:
                            recognizer.VAD_MODE = cfg.vad_mode
                        except Exception:
                            pass
                except Exception:
                    pass
                # Apply extra overrides from config.extra (advanced recognizer tuning)
                try:
                    overrides_applied = []
                    for attr in [
                        'SAMPLE_RATE','CHANNELS','CHUNK_S','VAD_MODE','MIN_SPEECH_S','MAX_SEGMENT_S','PADDING_MS',
                        'BEST_OF','TEMPERATURE','PATIENCE','LENGTH_PENALTY','REPETITION_PENALTY',
                        'WITHOUT_TIMESTAMPS','CONDITION_ON_PREVIOUS_TEXT','VAD_FILTER','VAD_PARAMETERS','WORD_TIMESTAMPS',
                        'FISH_PROMPT'
                    ]:
                        if attr.lower() in cfg.extra:
                            val = cfg.extra[attr.lower()]
                            try:
                                setattr(recognizer, attr, val)
                                overrides_applied.append(f"{attr}={val if isinstance(val,(int,float,str,bool)) else '<obj>'}")
                            except Exception:
                                pass
                    # Recompute chunk / noise controller if timing params changed
                    if any(k in cfg.extra for k in ['sample_rate','chunk_s','vad_mode','min_speech_s','max_segment_s']):
                        try:
                            recognizer._chunk_frames = int(recognizer.SAMPLE_RATE * recognizer.CHUNK_S)
                            from noise.controller import NoiseController as _NC
                            recognizer._noise_controller = _NC(sample_rate=recognizer.SAMPLE_RATE, vad_mode=recognizer.VAD_MODE,
                                                               min_speech_s=recognizer.MIN_SPEECH_S, max_segment_s=recognizer.MAX_SEGMENT_S)
                        except Exception:
                            pass
                    if overrides_applied:
                        logger.info(f"Applied recognizer overrides: {', '.join(overrides_applied)}")
                except Exception as e:
                    logger.debug(f"Recognizer override error: {e}")
                # Force model load
                try:
                    recognizer._load_backend_model()  # type: ignore
                    recognizer._backend_post_init()  # type: ignore
                except Exception as e:
                    logger.error(f"Recognizer backend load failed: {e}")
                    continue

                from noise.controller import NoiseController  # local import

                for sample in tqdm(samples, desc=f"{cfg.model.name}-{cfg.model.size}", bar_format="{desc} {n_fmt}/{total_fmt}", ncols=28):
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

                    # Build production-like NoiseController
                    nc = NoiseController(sample_rate=recognizer.SAMPLE_RATE, vad_mode=recognizer.VAD_MODE,
                                         min_speech_s=recognizer.MIN_SPEECH_S, max_segment_s=recognizer.MAX_SEGMENT_S)
                    # Convert audio to int16 stream chunks matching recognizer._chunk_frames
                    chunk_frames = int(recognizer.SAMPLE_RATE * recognizer.CHUNK_S)
                    if sr != recognizer.SAMPLE_RATE:
                        # simple resample
                        if resample is None:
                            logger.error("scipy required for resample in production_replay")
                            continue
                        audio = resample(audio, int(len(audio) * recognizer.SAMPLE_RATE / sr))
                        sr = recognizer.SAMPLE_RATE
                    pcm16_full = (audio * 32767).astype(np.int16)
                    start_feed_time = time.time()
                    fed = 0
                    while fed < pcm16_full.size:
                        end = min(fed + chunk_frames, pcm16_full.size)
                        chunk = pcm16_full[fed:end]
                        nc.push_audio(chunk)
                        fed = end
                        if self.real_time:
                            time.sleep(len(chunk)/recognizer.SAMPLE_RATE)
                    nc.stop()

                    segment_list = []
                    try:
                        for seg_arr, seg_start, seg_end in nc.collect_segments_with_timing(padding_ms=recognizer.PADDING_MS):
                            segment_list.append((seg_arr, seg_start, seg_end))
                    except Exception as e:
                        logger.debug(f"Segment collection error: {e}")

                    if not segment_list:
                        logger.debug(f"No segments for {sample.audio_path.name}; fallback single-pass decode")
                        # Fallback: direct full decode with existing model via recognizer backend
                        seg_audio = pcm16_full
                        seg_float = seg_audio.astype(np.float32) / 32767.0
                        combined = np.concatenate((getattr(recognizer, '_number_sound', np.zeros(0, dtype=np.int16)), seg_audio))
                        wav_path = BaseSpeechRecognizer._write_wav_bytes(combined, recognizer.SAMPLE_RATE)  # type: ignore
                        decode_start = time.time()
                        try:
                            segments = recognizer._backend_transcribe(segment=combined, wav_path=wav_path)  # type: ignore
                        except Exception as e:
                            logger.error(f"Transcribe failed: {e}")
                            segments = []
                        decode_end = time.time()
                        try:
                            os.remove(wav_path)
                        except Exception:
                            pass
                        raw_text = " ".join((s.text for s in segments)).strip()
                        avg_conf = None
                        try:
                            confs = [getattr(s,'confidence',None) for s in segments if getattr(s,'confidence',None) is not None]
                            if confs:
                                avg_conf = float(np.mean(confs))
                        except Exception:
                            pass
                        # Production-like parsing (as in BaseSpeechRecognizer)
                        final_display = raw_text
                        parser_species = None
                        parser_length_cm = None
                        try:
                            tn = TextNormalizer()
                            corrected = tn.apply_fish_asr_corrections(raw_text)
                            parsed = self.fish_parser.parse_text(corrected)
                            parser_species = getattr(parsed, 'species', None)
                            parser_length_cm = getattr(parsed, 'length_cm', None)
                            if parser_length_cm is not None:
                                val = float(parser_length_cm)
                                num_str = (f"{val:.1f}").rstrip("0").rstrip(".")
                                final_display = f"{parser_species} {num_str} cm" if parser_species else f"{num_str} cm"
                            else:
                                final_display = corrected
                        except Exception:
                            pass
                        norm = self.normalizer.normalize(raw_text)
                        expected_number = sample.expected_number
                        predicted_number = norm.predicted_number
                        expected_text = "" if expected_number is None else str(expected_number)
                        wer = word_error_rate(norm.normalized_text, expected_text)
                        cer = char_error_rate(norm.normalized_text, expected_text)
                        der = digit_error_rate(predicted_number, expected_number)
                        nem = numeric_exact_match(predicted_number, expected_number)
                        error_type = self.normalizer.classify_error(predicted_number, expected_number, raw_text, expected_text)
                        absolute_error = None if (predicted_number is None or expected_number is None) else abs(predicted_number - expected_number)
                        rows.append({
                            "test_run_id": cfg.test_run_id,
                            "wav_id": sample.audio_path.name,
                            "recognized_text_raw": raw_text,
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
                            "duration_s": len(pcm16_full)/recognizer.SAMPLE_RATE,
                            "model_name": cfg.model.name,
                            "model_size": cfg.model.size,
                            "config_json": cfg.to_json(),
                            "vad_setting": cfg.vad_mode,
                            "processing_time_s": decode_end - decode_start,
                            "RTF": (decode_end - decode_start) / (len(pcm16_full)/recognizer.SAMPLE_RATE),
                            "latency_ms": (decode_end - start_feed_time)*1000.0,
                            "cpu_percent": psutil.cpu_percent(interval=0.0),
                            "gpu_percent": None,
                            "memory_mb": psutil.Process(os.getpid()).memory_info().rss/(1024**2),
                            "gpu_memory_mb": None,
                            "numeric_exact_match": nem,
                            "parser_species": parser_species,
                            "parser_length_cm": parser_length_cm,
                            "parser_exact_match": 1 if (parser_length_cm is not None and expected_number is not None and float(parser_length_cm)==float(expected_number)) else None,
                            "final_display_text": final_display,
                            "WER": wer,
                            "CER": cer,
                            "DER": der,
                            "absolute_error": absolute_error,
                            "error_type": error_type,
                            "notes": "fallback_full_decode",
                            "sample_source": sample.source,
                            "used_prefix": 1,
                            "number_range": "unknown" if expected_number is None else ("0-9" if expected_number<10 else ("10-19" if expected_number<20 else ("20-29" if expected_number<30 else ("30-39" if expected_number<40 else ("40-49" if expected_number<50 else "50+"))))),
                            "segment_index": None,
                            "segment_count": 1,
                            "segment_duration_s": len(pcm16_full)/recognizer.SAMPLE_RATE,
                            "segment_processing_time_s": decode_end - decode_start,
                            "audio_start": start_feed_time,
                            "audio_end": start_feed_time + len(pcm16_full)/recognizer.SAMPLE_RATE,
                            "first_decoded_token_time": decode_start,
                            "final_result_time": decode_end,
                            "is_aggregate": 1,
                            "confidence": avg_conf,
                            "fish_prompt": getattr(recognizer, 'FISH_PROMPT', None),
                            "model_extra_json": json.dumps(cfg.extra, ensure_ascii=False, sort_keys=True),
                        })
                        _progress_log(sample.speaker_id, sample.audio_path.name, expected_number, predicted_number, nem, avg_conf)
                        _detailed_sample_log(sample.audio_path.name, raw_text, parser_species, parser_length_cm, expected_number, predicted_number)
                        continue

                    # After segment_list creation
                    if not segment_list:
                        rows.append({
                            "test_run_id": cfg.test_run_id,
                            "wav_id": sample.audio_path.name,
                            "recognized_text_raw": "",
                            "recognized_text_normalized": "",
                            "predicted_number": None,
                            "expected_number": sample.expected_number,
                            "speaker_id": sample.speaker_id,
                            "gender": sample.gender,
                            "age": sample.age,
                            "accent": sample.accent,
                            "recording_device": sample.recording_device,
                            "noise_type": sample.noise_type,
                            "snr_db": sample.snr_db,
                            "duration_s": 0.0,
                            "model_name": cfg.model.name,
                            "model_size": cfg.model.size,
                            "config_json": cfg.to_json(),
                            "vad_setting": cfg.vad_mode,
                            "processing_time_s": 0.0,
                            "RTF": 0.0,
                            "latency_ms": 0.0,
                            "cpu_percent": psutil.cpu_percent(interval=0.0),
                            "gpu_percent": None,
                            "memory_mb": psutil.Process(os.getpid()).memory_info().rss/(1024**2),
                            "gpu_memory_mb": None,
                            "numeric_exact_match": 0,
                            "parser_species": None,
                            "parser_length_cm": None,
                            "parser_exact_match": 0,
                            "final_display_text": "",
                            "WER": None,
                            "CER": None,
                            "DER": None,
                            "absolute_error": None,
                            "error_type": None,
                            "notes": "no_segments",
                            "sample_source": sample.source,
                            "used_prefix": 1,
                            "number_range": "unknown" if sample.expected_number is None else ("0-9" if sample.expected_number<10 else ("10-19" if sample.expected_number<20 else ("20-29" if sample.expected_number<30 else ("30-39" if sample.expected_number<40 else ("40-49" if sample.expected_number<50 else "50+"))))),
                            "segment_index": None,
                            "segment_count": 0,
                            "segment_duration_s": 0.0,
                            "segment_processing_time_s": 0.0,
                            "audio_start": None,
                            "audio_end": None,
                            "first_decoded_token_time": None,
                            "final_result_time": None,
                            "is_aggregate": 1,
                            "fish_prompt": getattr(recognizer, 'FISH_PROMPT', None),
                            "model_extra_json": json.dumps(cfg.extra, ensure_ascii=False, sort_keys=True),
                        })
                        continue

                    # Segment mode with production-like parsing & state
                    all_text_parts = []
                    segment_count = len(segment_list)
                    total_audio_dur = sum(seg[0].size for seg in segment_list)/recognizer.SAMPLE_RATE
                    aggregate_processing = 0.0
                    expected_number = sample.expected_number
                    last_species = None
                    measurement_pred = None

                    tn = TextNormalizer()

                    for idx,(seg_arr, seg_start, seg_end) in enumerate(segment_list):
                        combined = np.concatenate((getattr(recognizer, '_number_sound', np.zeros(0, dtype=np.int16)), seg_arr))
                        wav_path = BaseSpeechRecognizer._write_wav_bytes(combined, recognizer.SAMPLE_RATE)  # type: ignore
                        decode_start = time.time()
                        try:
                            segments = recognizer._backend_transcribe(segment=combined, wav_path=wav_path)  # type: ignore
                        except Exception as e:
                            logger.error(f"Segment transcribe failed: {e}")
                            segments = []
                        decode_end = time.time()
                        aggregate_processing += (decode_end - decode_start)
                        try:
                            os.remove(wav_path)
                        except Exception:
                            pass
                        raw_text = " ".join((s.text for s in segments)).strip()
                        seg_conf = None
                        try:
                            confs = [getattr(s,'confidence',None) for s in segments if getattr(s,'confidence',None) is not None]
                            if confs:
                                seg_conf = float(np.mean(confs))
                        except Exception:
                            pass
                        all_text_parts.append(raw_text)

                        # Production-style normalization & parsing
                        final_display = raw_text
                        parser_species = None
                        parser_length_cm = None
                        try:
                            corrected = tn.apply_fish_asr_corrections(raw_text)
                            parsed = self.fish_parser.parse_text(corrected)
                            parser_species = getattr(parsed, 'species', None)
                            if parser_species:
                                last_species = parser_species  # stateful like production
                            parser_length_cm = getattr(parsed, 'length_cm', None)
                            if parser_length_cm is not None:
                                val = float(parser_length_cm)
                                num_str = (f"{val:.1f}").rstrip("0").rstrip(".")
                                final_display = f"{last_species} {num_str} cm" if last_species else f"{num_str} cm"
                            else:
                                final_display = corrected
                        except Exception:
                            pass

                        # Numeric normalization fallback only if no parser length
                        predicted_number_seg = None
                        if parser_length_cm is not None:
                            predicted_number_seg = float(parser_length_cm)
                            if measurement_pred is None:
                                measurement_pred = predicted_number_seg
                        else:
                            # fallback normalization (no parse length)
                            norm_seg = self.normalizer.normalize(raw_text)
                            predicted_number_seg = norm_seg.predicted_number

                        # Determine range bucket
                        if expected_number is None:
                            number_range_val = "unknown"
                        else:
                            vexp = expected_number
                            if vexp < 10: number_range_val = "0-9"
                            elif vexp < 20: number_range_val = "10-19"
                            elif vexp < 30: number_range_val = "20-29"
                            elif vexp < 40: number_range_val = "30-39"
                            elif vexp < 50: number_range_val = "40-49"
                            else: number_range_val = "50+"

                        rows.append({
                            "test_run_id": cfg.test_run_id,
                            "wav_id": sample.audio_path.name,
                            "recognized_text_raw": raw_text,
                            "recognized_text_normalized": raw_text.lower(),
                            "predicted_number": predicted_number_seg,
                            "expected_number": expected_number,
                            "speaker_id": sample.speaker_id,
                            "gender": sample.gender,
                            "age": sample.age,
                            "accent": sample.accent,
                            "recording_device": sample.recording_device,
                            "noise_type": sample.noise_type,
                            "snr_db": sample.snr_db,
                            "duration_s": total_audio_dur,
                            "model_name": cfg.model.name,
                            "model_size": cfg.model.size,
                            "config_json": cfg.to_json(),
                            "vad_setting": cfg.vad_mode,
                            "processing_time_s": decode_end - decode_start,
                            "RTF": (decode_end - decode_start) / ((seg_arr.size)/recognizer.SAMPLE_RATE) if seg_arr.size>0 else None,
                            "latency_ms": (decode_start - seg_start)*1000.0,
                            "cpu_percent": psutil.cpu_percent(interval=0.0),
                            "gpu_percent": None,
                            "memory_mb": psutil.Process(os.getpid()).memory_info().rss/(1024**2),
                            "gpu_memory_mb": None,
                            "numeric_exact_match": None if expected_number is None or predicted_number_seg is None else int(abs(predicted_number_seg-expected_number) < 1e-6),
                            "parser_species": parser_species,
                            "parser_length_cm": parser_length_cm,
                            "parser_exact_match": None if (parser_length_cm is None or expected_number is None) else int(float(parser_length_cm)==float(expected_number)),
                            "final_display_text": final_display,
                            "WER": None,
                            "CER": None,
                            "DER": None,
                            "absolute_error": None if (expected_number is None or predicted_number_seg is None) else abs(predicted_number_seg-expected_number),
                            "error_type": None,
                            "notes": "segment_row",
                            "sample_source": sample.source,
                            "used_prefix": 1,
                            "number_range": number_range_val,
                            "segment_index": idx,
                            "segment_count": segment_count,
                            "segment_duration_s": seg_arr.size/recognizer.SAMPLE_RATE,
                            "segment_processing_time_s": decode_end - decode_start,
                            "audio_start": seg_start,
                            "audio_end": seg_end,
                            "first_decoded_token_time": decode_start,
                            "final_result_time": decode_end,
                            "is_aggregate": 0,
                            "is_measurement": 1 if parser_length_cm is not None else 0,
                            "fish_prompt": getattr(recognizer, 'FISH_PROMPT', None),
                            "model_extra_json": json.dumps(cfg.extra, ensure_ascii=False, sort_keys=True),
                        })
                        # Optional segment level logging (disabled by default for noise). Uncomment if needed.
                        # _progress_log(sample.speaker_id, f"{sample.audio_path.name}[seg{idx}]", expected_number, predicted_number_seg, rows[-1]["numeric_exact_match"], seg_conf)
                    # Aggregate row
                    aggregate_text = " ".join(t for t in all_text_parts if t).strip()
                    norm = self.normalizer.normalize(aggregate_text)
                    predicted_number = measurement_pred if measurement_pred is not None else norm.predicted_number
                    expected_text = "" if expected_number is None else str(expected_number)
                    wer = word_error_rate(norm.normalized_text, expected_text)
                    cer = char_error_rate(norm.normalized_text, expected_text)
                    der = digit_error_rate(predicted_number, expected_number)
                    nem = numeric_exact_match(predicted_number, expected_number)
                    error_type = self.normalizer.classify_error(predicted_number, expected_number, aggregate_text, expected_text)
                    absolute_error = None if (predicted_number is None or expected_number is None) else abs(predicted_number - expected_number)
                    # final display adopt last segment final_display_text if exists
                    last_display = None
                    for r in reversed(rows):
                        if r["wav_id"] == sample.audio_path.name and r.get("final_display_text"):
                            last_display = r["final_display_text"]
                            break
                    rows.append({
                        "test_run_id": cfg.test_run_id,
                        "wav_id": sample.audio_path.name,
                        "recognized_text_raw": aggregate_text,
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
                        "duration_s": total_audio_dur,
                        "model_name": cfg.model.name,
                        "model_size": cfg.model.size,
                        "config_json": cfg.to_json(),
                        "vad_setting": cfg.vad_mode,
                        "processing_time_s": aggregate_processing,
                        "RTF": aggregate_processing / total_audio_dur if total_audio_dur>0 else None,
                        "latency_ms": None,
                        "cpu_percent": psutil.cpu_percent(interval=0.0),
                        "gpu_percent": None,
                        "memory_mb": psutil.Process(os.getpid()).memory_info().rss/(1024**2),
                        "gpu_memory_mb": None,
                        "numeric_exact_match": nem,
                        "parser_species": None,
                        "parser_length_cm": measurement_pred,
                        "parser_exact_match": None if (measurement_pred is None or expected_number is None) else int(measurement_pred==expected_number),
                        "final_display_text": last_display,
                        "WER": wer,
                        "CER": cer,
                        "DER": der,
                        "absolute_error": absolute_error,
                        "error_type": error_type,
                        "notes": "aggregate",
                        "sample_source": sample.source,
                        "used_prefix": 1,
                        "number_range": "unknown" if expected_number is None else ("0-9" if expected_number<10 else ("10-19" if expected_number<20 else ("20-29" if expected_number<30 else ("30-39" if expected_number<40 else ("40-49" if expected_number<50 else "50+"))))),
                        "segment_index": None,
                        "segment_count": segment_count,
                        "segment_duration_s": None,
                        "segment_processing_time_s": aggregate_processing,
                        "audio_start": segment_list[0][1] if segment_list else None,
                        "audio_end": segment_list[-1][2] if segment_list else None,
                        "first_decoded_token_time": None,
                        "final_result_time": None,
                        "is_aggregate": 1,
                        "is_measurement": 1 if measurement_pred is not None else 0,
                        "fish_prompt": getattr(recognizer, 'FISH_PROMPT', None),
                        "model_extra_json": json.dumps(cfg.extra, ensure_ascii=False, sort_keys=True),
                    })
                    _progress_log(sample.speaker_id, sample.audio_path.name, expected_number, predicted_number, nem, rows[-1].get('confidence'))
                    _detailed_sample_log(sample.audio_path.name, aggregate_text, None, measurement_pred, expected_number, predicted_number)
                continue  # move to next config or sample set

            # === Original (non-production replay) path ===
            for sample in tqdm(samples, desc=f"{cfg.model.name}-{cfg.model.size}", bar_format="{desc} {n_fmt}/{total_fmt}", ncols=28):
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

                used_prefix_flag = 1 if number_prefix is not None else 0

                # Determine number range bucket
                if expected_number is None:
                    number_range = "unknown"
                else:
                    v = expected_number
                    if v < 10: number_range = "0-9"
                    elif v < 20: number_range = "10-19"
                    elif v < 30: number_range = "20-29"
                    elif v < 40: number_range = "30-39"
                    elif v < 50: number_range = "40-49"
                    else: number_range = "50+"
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
                    "sample_source": sample.source,
                    "used_prefix": used_prefix_flag,
                    "number_range": number_range,
                    "final_display_text": None,
                    "confidence": None,  # standard mode currently lacks per-segment confidence
                    "fish_prompt": cfg.extra.get('fish_prompt'),
                    "model_extra_json": json.dumps(cfg.extra, ensure_ascii=False, sort_keys=True),
                }
                rows.append(row)
                _progress_log(sample.speaker_id, sample.audio_path.name, expected_number, predicted_number, nem, None)
                _detailed_sample_log(sample.audio_path.name, recognized_raw, parser_species, parser_length_cm, expected_number, predicted_number)
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

        # Use only aggregate rows for summaries to avoid double counting
        if "is_aggregate" in df.columns:
            df_agg = df[df["is_aggregate"] == 1]
        else:
            df_agg = df  # backward compatibility
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
            # confidence intentionally not aggregated (can add later)
        }
        summary_model = df_agg.groupby(["model_name", "model_size"]).agg(agg_funcs)
        summary_accent = df_agg.groupby(["accent", "model_name", "model_size"]).agg(agg_funcs)
        summary_noise = df_agg.groupby(["noise_type", "model_name", "model_size"]).agg(agg_funcs)

        # RTF percentiles overall
        rtf_percentiles = compute_percentiles(df_agg["RTF"].dropna().tolist())
        rtf_meta = pd.DataFrame([
            {"p50": rtf_percentiles.p50, "p95": rtf_percentiles.p95, "p99": rtf_percentiles.p99}
        ])

        # RTF percentiles per model
        rtf_model_records = []
        for (mname, msize), sub in df_agg.groupby(["model_name", "model_size"]):
            pct = compute_percentiles(sub["RTF"].dropna().tolist())
            rtf_model_records.append({
                "model_name": mname,
                "model_size": msize,
                "p50": pct.p50,
                "p95": pct.p95,
                "p99": pct.p99,
            })
        rtf_model_df = pd.DataFrame(rtf_model_records)

        # Additional pivot: number range
        if "number_range" in df_agg.columns:
            try:
                summary_range = df_agg.groupby(["number_range", "model_name", "model_size"]).agg(agg_funcs)
            except Exception:
                summary_range = None
        else:
            summary_range = None
        with pd.ExcelWriter(summary_path, engine="openpyxl") as xl:
            df.head(200).to_excel(xl, sheet_name="sample", index=False)
            summary_model.to_excel(xl, sheet_name="by_model")
            summary_accent.to_excel(xl, sheet_name="by_accent")
            summary_noise.to_excel(xl, sheet_name="by_noise")
            rtf_meta.to_excel(xl, sheet_name="rtf_percentiles", index=False)
            rtf_model_df.to_excel(xl, sheet_name="rtf_by_model", index=False)
            if summary_range is not None:
                summary_range.to_excel(xl, sheet_name="by_range")

        logger.info(f"Wrote: {results_path}, {failures_path}, {summary_path}")

        # Write run summary metadata (json + markdown)
        try:
            import platform, sys, datetime
            summary_meta = {
                "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "total_rows": int(len(df)),
                "aggregate_rows": int(len(df_agg)),
                "models": sorted(df_agg.model_name.unique().tolist()),
                "model_sizes": sorted(df_agg.model_size.unique().tolist()),
                "numeric_exact_match_mean": float(df_agg.numeric_exact_match.dropna().mean()) if "numeric_exact_match" in df_agg else None,
                "parser_exact_match_mean": float(df_agg.parser_exact_match.dropna().mean()) if "parser_exact_match" in df_agg else None,
                "rtf_p50": rtf_percentiles.p50,
                "rtf_p95": rtf_percentiles.p95,
                "rtf_p99": rtf_percentiles.p99,
            }
            # Attempt to load model_specs_used.json (snapshot created at run start)
            try:
                ms_path = self.output_dir / "model_specs_used.json"
                if ms_path.exists():
                    ms_text = ms_path.read_text(encoding="utf-8")
                    try:
                        summary_meta["model_specs_full"] = json.loads(ms_text)
                    except Exception:
                        summary_meta["model_specs_full_raw"] = ms_text
            except Exception as e:  # non-fatal
                logger.debug(f"Could not attach model specs to summary: {e}")
            (self.output_dir / "run_summary.json").write_text(json.dumps(summary_meta, indent=2), encoding="utf-8")
            # Markdown
            md_lines = [
                "# Evaluation Summary", "", f"Created: {summary_meta['created_at']}",
                f"Models: {', '.join(summary_meta['models'])}",
                f"Model Sizes: {', '.join(summary_meta['model_sizes'])}",
                f"Rows: {summary_meta['total_rows']}",
                f"Numeric Exact Match Mean: {summary_meta['numeric_exact_match_mean']:.4f}" if summary_meta['numeric_exact_match_mean'] is not None else "Numeric Exact Match Mean: n/a",
                f"Parser Exact Match Mean: {summary_meta['parser_exact_match_mean']:.4f}" if summary_meta['parser_exact_match_mean'] is not None else "Parser Exact Match Mean: n/a",
                f"RTF p50/p95/p99: {rtf_percentiles.p50:.3f} / {rtf_percentiles.p95:.3f} / {rtf_percentiles.p99:.3f}",
            ]
            # Embed model specs if present
            if "model_specs_full" in summary_meta or "model_specs_full_raw" in summary_meta:
                md_lines.extend(["", "## Model Specs", ""])
                try:
                    if "model_specs_full" in summary_meta:
                        ms_pretty = json.dumps(summary_meta["model_specs_full"], indent=2, ensure_ascii=False, sort_keys=True)
                        md_lines.extend(["```json", ms_pretty, "```"])
                    else:
                        ms_raw = summary_meta["model_specs_full_raw"]
                        md_lines.extend(["```", ms_raw, "```"])
                except Exception:
                    pass
            md_lines.extend(["", "## Top 10 Failures", ""])
            fails_view = df[df.numeric_exact_match == 0].head(10)
            for _, fr in fails_view.iterrows():
                disp = fr.get('final_display_text') or ''
                md_lines.append(f"- {fr['wav_id']}: pred={fr.get('predicted_number')} exp={fr.get('expected_number')} raw='{fr.get('recognized_text_raw')}' final='{disp}'")
            (self.output_dir / "run_summary.md").write_text("\n".join(md_lines), encoding="utf-8")
        except Exception as e:
            logger.debug(f"Failed writing summary metadata: {e}")

# Progress logging helper (added to avoid NameError)
# (kept for single-line summary if needed)
def _progress_log(speaker: str, wav: str, expected, predicted, nem, confidence):
    try:
        status = "PASSED" if nem == 1 else "FAIL"
        logger.info(f"{speaker} -> {wav} - exp={expected} pred={predicted} {status} conf={confidence if confidence is not None else 'n/a'}")
    except Exception:
        pass

def _detailed_sample_log(wav_id: str, raw_text: str, parser_species, parser_length_cm, expected_number, predicted_number):
    """Pretty multi-line log similar to integration test output."""
    try:
        logger.info("────────────────────────────────────────")
        logger.info(f"🎤 Audio: {wav_id}")
        logger.info(f"📜 Raw transcript: '{raw_text}'")
        if parser_species is not None or parser_length_cm is not None:
            logger.info(f"📦 Parser result: species={parser_species} length_cm={parser_length_cm}")
        else:
            logger.info("📦 Parser result: <none>")
        logger.info(f"✅ Expected length: {expected_number}, 🧮 Got: {parser_length_cm if parser_length_cm is not None else predicted_number}")
    except Exception:
        pass
