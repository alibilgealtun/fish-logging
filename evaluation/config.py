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

# Recognizer attribute names (lowercase) that can be overridden / grid-expanded via model_specs
RECOGNIZER_ATTRS = {
    'sample_rate', 'channels', 'chunk_s', 'min_speech_s', 'max_segment_s', 'padding_ms',
    'best_of', 'temperature', 'patience', 'length_penalty', 'repetition_penalty',
    'without_timestamps', 'condition_on_previous_text', 'vad_filter', 'vad_parameters', 'word_timestamps',
    'fish_prompt', 'prompt'
}

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
    concat_number: bool = True,
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
    concat_number: bool = True,
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


def expand_model_spec_grid(
    model_specs: list[dict],
    dataset_json: str | None = None,
    concat_number: bool = True,
    number_audio_path: str | None = None,
    audio_root: str | None = None,
    production_replay: bool = False,
    **extra_fixed: Any,
) -> List[EvaluationConfig]:
    """Expand configs from a list of per-model spec dictionaries.

    Extended: In addition to the core dimension keys (sizes, compute_types, devices, beams,
    chunks, vad_modes, languages) any recognizer attribute listed in RECOGNIZER_ATTRS can now
    also be provided either as a single scalar (applied uniformly) or a list to be included
    as its own Cartesian dimension. These extra attributes are injected into EvaluationConfig.extra
    so they become available as overrides during production_replay (and for logging).

    Example spec entry:
      {
        "name": "faster-whisper",
        "sizes": ["base.en", "small.en"],
        "compute_types": ["int8"],
        "devices": ["cpu"],
        "beams": [5, 10],
        "chunks": [500],            # streaming chunk size (ms) for evaluation StreamingSimulator
        "vad_modes": [2,3],
        "min_speech_s": [0.15, 0.25],
        "max_segment_s": 3.5,
        "padding_ms": 600,
        "chunk_s": 0.5,  # recognizer internal CHUNK_S seconds (distinct from streaming chunk ms)
        "sample_rate": 16000
      }
    """
    core_defaults = {
        'sizes': ['base.en'],
        'compute_types': ['int8'],
        'devices': ['cpu'],
        'beams': [5],
        'chunks': [500],            # streaming chunk size (ms) for evaluation StreamingSimulator
        'vad_modes': [2],
        'languages': ['en'],
    }
    all_cfgs: List[EvaluationConfig] = []
    for spec in model_specs:
        if not isinstance(spec, dict):
            continue
        name = spec.get('name') or spec.get('model')
        if not name:
            continue
        # Gather core lists
        sizes = spec.get('sizes', core_defaults['sizes'])
        compute_types = spec.get('compute_types', core_defaults['compute_types'])
        devices = spec.get('devices', core_defaults['devices'])
        beams = spec.get('beams', core_defaults['beams'])
        chunks = spec.get('chunks', core_defaults['chunks'])
        vad_modes = spec.get('vad_modes', core_defaults['vad_modes'])
        languages = spec.get('languages', core_defaults['languages'])

        # Build dict of extra recognizer attr name -> list of values (ensure list)
        recognizer_dim_values: dict[str, list[Any]] = {}
        for key, value in spec.items():
            k_l = key.lower()
            if k_l in RECOGNIZER_ATTRS:
                if isinstance(value, list):
                    recognizer_dim_values[k_l] = value
                else:
                    recognizer_dim_values[k_l] = [value]
        # Normalize alias 'prompt' -> 'fish_prompt' to avoid duplication
        if 'prompt' in recognizer_dim_values and 'fish_prompt' not in recognizer_dim_values:
            recognizer_dim_values['fish_prompt'] = recognizer_dim_values.pop('prompt')
        # Keys to exclude from extra (core + recognizer attr dims)
        exclude_keys = {'name','model','sizes','compute_types','devices','beams','chunks','vad_modes','languages'} | set(recognizer_dim_values.keys()) | {'prompt'}
        # Remaining spec items become fixed extras for every combination
        spec_fixed_extras = {k: v for k, v in spec.items() if k not in exclude_keys}

        # Create Cartesian product across core + dynamic recognizer dimensions
        recognizer_dim_names = sorted(recognizer_dim_values.keys())  # stable order
        recognizer_dim_lists = [recognizer_dim_values[n] for n in recognizer_dim_names]
        # If no recognizer dims, still iterate once
        if not recognizer_dim_names:
            recognizer_dim_names = []
            recognizer_dim_lists = [[]]

        # For each combination of core and recognizer dims
        for (s, ct, dev, beam, chunk, vad, lang) in product(sizes, compute_types, devices, beams, chunks, vad_modes, languages):
            # Iterate recognizer attribute combinations (if any)
            if recognizer_dim_lists == [[]]:  # sentinel for no extra dims
                recog_products = [()]
            else:
                recog_products = product(*recognizer_dim_lists)
            for recog_vals in recog_products:
                extra: Dict[str, Any] = {}
                # Insert recognizer dynamic attrs
                for idx, val in enumerate(recog_vals):
                    extra[recognizer_dim_names[idx]] = val
                # Insert fixed recognizer attrs (spec fixed extras) and any external fixed extras
                extra.update(spec_fixed_extras)
                extra.update(extra_fixed)  # global extras passed into expand_model_spec_grid
                cfg = make_config(
                    model_name=name,
                    size=s,
                    compute_type=ct,
                    device=dev,
                    beam=beam,
                    chunk_ms=chunk,
                    vad_mode=vad,
                    language=lang,
                    dataset_json=dataset_json,
                    concat_number=concat_number,
                    number_audio_path=number_audio_path,
                    audio_root=audio_root,
                    production_replay=production_replay,
                    **extra,
                )
                all_cfgs.append(cfg)
    return all_cfgs
