from __future__ import annotations
"""Evaluation configuration utilities.

Defines dataclasses used to describe a single evaluation run as well as
helpers to expand a parameter grid.
"""
from dataclasses import dataclass, field, asdict
from itertools import product
from typing import Dict, List, Iterable, Any
import json
import hashlib
import time


DEFAULT_BASE_CONFIG = {
    "model": "faster-whisper",
    "size": "base.en",
    "vad": 2,
    "chunk": 500,  # ms streaming chunk size
    "beam_size": 5,
    "mode": "clean",
    "dataset": "numbers",  # subdirectory under test_data
}


@dataclass
class ModelConfig:
    name: str = "faster-whisper"
    size: str = "base.en"
    compute_type: str = "int8"
    device: str = "cpu"
    language: str = "en"


@dataclass
class InferenceConfig:
    beam_size: int = 5
    vad_filter: bool = False
    condition_on_previous_text: bool = False
    vad_parameters: Dict[str, Any] | None = None
    chunk_size_ms: int = 500
    streaming_mode: bool = True


@dataclass
class EvaluationConfig:
    test_run_id: str
    model: ModelConfig
    inference_config: InferenceConfig
    environment: str = "clean"  # noise condition label
    vad_mode: int = 2
    dataset_json: str | None = None
    concat_number: bool = True
    number_audio_path: str | None = None
    audio_root: str | None = None
    production_replay: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["model"] = asdict(self.model)
        d["inference_config"] = asdict(self.inference_config)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @property
    def cache_key(self) -> str:
        raw = f"{self.model.name}|{self.model.size}|{self.model.compute_type}|{self.model.device}"
        return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _timestamp_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{time.strftime('%Y_%m_%d_%H%M%S')}"


def make_config(
    model_name: str = "faster-whisper",
    size: str = "base.en",
    compute_type: str = "int8",
    device: str = "cpu",
    beam: int = 5,
    chunk_ms: int = 500,
    environment: str = "clean",
    vad_mode: int = 2,
    streaming: bool = True,
    language: str = "en",
    test_run_id: str | None = None,
    dataset_json: str | None = None,
    concat_number: bool = False,
    number_audio_path: str | None = None,
    audio_root: str | None = None,
    production_replay: bool = False,
    **extra: Any,
) -> EvaluationConfig:
    """Helper to build a single EvaluationConfig."""
    mc = ModelConfig(name=model_name, size=size, compute_type=compute_type, device=device, language=language)
    ic = InferenceConfig(beam_size=beam, chunk_size_ms=chunk_ms, streaming_mode=streaming)
    return EvaluationConfig(
        test_run_id=test_run_id or _timestamp_run_id(),
        model=mc,
        inference_config=ic,
        environment=environment,
        vad_mode=vad_mode,
        dataset_json=dataset_json,
        concat_number=concat_number,
        number_audio_path=number_audio_path,
        audio_root=audio_root,
        production_replay=production_replay,
        extra=extra,
    )


def expand_parameter_grid(
    models: Iterable[str],
    sizes: Iterable[str],
    compute_types: Iterable[str],
    devices: Iterable[str],
    beams: Iterable[int],
    chunk_sizes_ms: Iterable[int],
    environments: Iterable[str],
    vad_modes: Iterable[int],
    streaming_modes: Iterable[bool] = (True,),
    languages: Iterable[str] = ("en",),
    dataset_json: str | None = None,
    concat_number: bool = False,
    number_audio_path: str | None = None,
    audio_root: str | None = None,
    production_replay: bool = False,
    **extra_fixed: Any,
) -> List[EvaluationConfig]:
    """Expand a full Cartesian product of the provided parameter lists.

    Large grids can explode quickly; caller is responsible for filtering.
    """
    configs: List[EvaluationConfig] = []
    for (m, s, ct, dev, beam, chunk, env, vad, sm, lang) in product(
        models, sizes, compute_types, devices, beams, chunk_sizes_ms, environments, vad_modes, streaming_modes, languages
    ):
        cfg = make_config(
            model_name=m,
            size=s,
            compute_type=ct,
            device=dev,
            beam=beam,
            chunk_ms=chunk,
            environment=env,
            vad_mode=vad,
            streaming=sm,
            language=lang,
            dataset_json=dataset_json,
            concat_number=concat_number,
            number_audio_path=number_audio_path,
            audio_root=audio_root,
            production_replay=production_replay,
            **extra_fixed,
        )
        configs.append(cfg)
    return configs
