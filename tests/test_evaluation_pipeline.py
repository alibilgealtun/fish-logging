import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from evaluation import EvaluationConfig, ModelConfig, InferenceConfig
from evaluation.pipeline import ASREvaluator


def _make_sine(sr: int, dur_s: float):
    t = np.linspace(0, dur_s, int(sr * dur_s), endpoint=False)
    return 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)


def test_dummy_backend_pipeline_basic(tmp_path: Path):
    # Build synthetic dataset structure
    dataset_root = tmp_path / "numbers"
    speaker_dir = dataset_root / "speaker_1_test"
    speaker_dir.mkdir(parents=True, exist_ok=True)

    # Two audio files: expected 1 and 2
    sr = 16000
    audio1 = _make_sine(sr, 0.5)
    audio2 = _make_sine(sr, 0.6)
    sf.write(speaker_dir / "1.wav", audio1, sr)
    sf.write(speaker_dir / "2.wav", audio2, sr)

    metadata = {
        "speaker_id": "speaker_1",
        "speaker_profile": {"ethnicity": "turkish", "age": "30", "gender": "male"},
        "recording_conditions": {"microphone": "test_mic", "environment": "quiet_room"},
        "contents": {"type": "number", "audios": [
            {"audio": "1.wav", "duration": 0.5, "expected": 1},
            {"audio": "2.wav", "duration": 0.6, "expected": 2},
        ]},
    }
    with (speaker_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f)

    # Dummy backend config (always returns token 'one')
    model_cfg = ModelConfig(name="dummy", size="na", compute_type="na", device="cpu", language="en")
    infer_cfg = InferenceConfig(beam_size=1, chunk_size_ms=200, streaming_mode=True)
    eval_cfg = EvaluationConfig(test_run_id="test_dummy", model=model_cfg, inference_config=infer_cfg, environment="clean", vad_mode=2)

    evaluator = ASREvaluator(dataset_root=dataset_root, output_dir=tmp_path / "out", real_time=False)
    df = evaluator.evaluate_configs([eval_cfg])

    # Basic assertions
    assert not df.empty, "DataFrame should not be empty"
    assert {"recognized_text_raw", "predicted_number", "expected_number"}.issubset(df.columns)
    # First sample should match expected 1 (numeric exact match ==1), second should fail
    first_row = df[df["wav_id"] == "1.wav"].iloc[0]
    second_row = df[df["wav_id"] == "2.wav"].iloc[0]
    assert first_row["numeric_exact_match"] == 1
    assert second_row["numeric_exact_match"] == 0

    # Output artifacts
    out_dir = tmp_path / "out"
    assert (out_dir / "results.parquet").exists()
    assert (out_dir / "failures.parquet").exists()
    assert (out_dir / "summary.xlsx").exists()

    # Failures parquet should include the misprediction (2.wav)
    failures = pd.read_parquet(out_dir / "failures.parquet")
    assert "2.wav" in failures["wav_id"].values

