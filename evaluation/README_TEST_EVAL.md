# Evaluation & Testing Guide

Comprehensive reference for automated numeric ASR evaluation and integration tests. This file covers:
- How the evaluation pipeline works
- Central configuration (single JSON) model specification
- Standard vs Production Replay modes
- Output artifacts & columns
- Typical workflows, troubleshooting, extension hooks

---
## 1. Quick Start
Run a full numeric evaluation using only the central config file:
```bash
python -m evaluation.run_evaluation
```
This reads `evaluation/presets/model_specs.json`, expands model configurations, loads the dataset, runs all models, and writes results into a timestamped folder under `evaluation_outputs/`.

Run only 10 samples (smoke check):
```bash
python -m evaluation.run_evaluation --max-samples 10
```
Enable production replay (segmenting audio like the live app):
```bash
python -m evaluation.run_evaluation --production-replay
```

Use an alternative specs file:
```bash
python -m evaluation.run_evaluation --model-specs custom_specs.json
```

---
## 2. Central Configuration File
Location:
```
evaluation/presets/model_specs.json
```
Purpose: single source of truth (dataset path, audio root, flags, model grids). **You edit this file to change what gets evaluated.**

Example (default):
```json
{
  "dataset_json": "tests/data/numbers.json",
  "audio_root": "tests/audio",
  "concat_number": true,
  "production_replay": false,
  "model_specs": [
    {
      "name": "faster-whisper",
      "sizes": ["base.en"],
      "compute_types": ["int8"],
      "devices": ["cpu"],
      "beams": [5],
      "chunks": [500],
      "vad_modes": [2],
      "languages": ["en"],
      "fish_prompt": "Custom domain instruction (optional)"
    }
  ]
}
```
New (2025-10) Supported Recognizer Override Keys (lowercase in JSON):
- sample_rate, channels, chunk_s
- min_speech_s, max_segment_s, padding_ms
- best_of, temperature, patience, length_penalty, repetition_penalty
- without_timestamps, condition_on_previous_text, vad_filter, vad_parameters, word_timestamps
- fish_prompt (or alias prompt)

If any of these are lists they are expanded as additional grid dimensions per model spec. Scalars apply uniformly.

`fish_prompt` allows you to inject a custom domain prompt used by the recognizer (overrides the built‑in FISH_PROMPT). Alias `prompt` is accepted and normalized to `fish_prompt`.

Each `model_specs` entry (per model) may include:
- name (required)
- sizes, compute_types, devices, beams, chunks, vad_modes, languages (optional lists)
- any extra custom fields (propagated to `EvaluationConfig.extra`)

Only the **Cartesian product inside each single model spec** is expanded. There is **no cross‑product across different models**, preventing invalid combos and config explosion.

### Adding Another Model
```json
{
  "name": "whisperx",
  "sizes": ["small.en"],
  "compute_types": ["float16"],
  "devices": ["cuda"],
  "beams": [3],
  "chunks": [500],
  "vad_modes": [2]
}
```
Insert into `model_specs` array and re‑run the evaluation command.

### Custom Prompt Override Example
```json
{
  "model_specs": [
    {
      "name": "faster-whisper",
      "sizes": ["base.en"],
      "beams": [5,10],
      "min_speech_s": [0.15, 0.25],
      "fish_prompt": "Focus on fish species and length numbers (cm). Ignore unrelated chatter."
    }
  ]
}
```
The resulting rows in `results.parquet` will include columns `fish_prompt` and `model_extra_json` so you can audit which prompt and overrides were applied.

---
## 3. Override Precedence
Highest → lowest:
1. CLI options (e.g. `--production-replay`, `--max-samples`, `--model-specs`)
2. Grid file (`--grid`) values
3. Central config (`evaluation/presets/model_specs.json`)
4. Internal defaults

If `--model-specs` is provided, it replaces the `model_specs` in central config (other top-level values like dataset_json still merge if not overridden).

---
## 4. Data Source Priority (Samples)
1. dataset_json (from central config or CLI)
2. metadata.json recursive discovery
3. Filename digit inference (e.g. 23.wav => expected 23)

Each row is tagged with `sample_source`: json | metadata | filename_fallback.

---
## 5. Modes: Standard vs Production Replay
| Aspect | Standard | Production Replay |
|--------|----------|-------------------|
| Audio handling | Whole file decoded once | Streamed into NoiseController, segmented |
| Prefix usage | Applied once if concat_number=true | Applied per segment (mirrors live) |
| Rows produced | Single aggregate row | Segment rows (is_aggregate=0) + final aggregate (is_aggregate=1) |
| Latency approximation | Single pass timing | Segment timing from feed + decode |
| Parser display | Directly on aggregate | Segment raws + parser on aggregate/fallback |

`production_replay` is the closest offline reproduction of the runtime path in the application.

---
## 6. Output Artifacts (Per Run Folder)
Created under: `evaluation_outputs/run_YYYY_MM_DD_HHMMSS[_n]/`

| File | Description |
|------|-------------|
| results.parquet | All rows (segment + aggregate) |
| failures.parquet | Subset of aggregate rows with numeric mismatch |
| summary.xlsx | Aggregated performance sheets |
| run_summary.json | Machine‑readable summary snapshot |
| run_summary.md | Human summary + top failures |
| nem_by_model_size.png* | (Optional plots) Numeric exact match bar chart |
| rtf_distribution.png* | (Optional plots) RTF distribution (log) |
| model_specs_used.json | Snapshot of the model specs / flat parameters + context |

(* only if `--plots` passed)

Added columns (2025-10):
- fish_prompt: The effective recognizer prompt for that row (per segment & aggregate in production replay, or from spec in standard mode).
- model_extra_json: JSON dump of all extra override attributes applied (including prompt and timing overrides).

---
## 7. Key Columns (Condensed)
| Column | Meaning |
|--------|---------|
| is_aggregate | 1 aggregate transcript; 0 per segment |
| wav_id | Audio filename |
| recognized_text_raw | Raw decoded text |
| recognized_text_normalized | Normalized for numeric extraction |
| predicted_number | Extracted number (normalizer or parser) |
| expected_number | Ground truth label |
| numeric_exact_match | 1 if predicted == expected (within tolerance) |
| parser_length_cm / parser_species | Parser outputs (if found) |
| absolute_error | |predicted - expected| |
| WER / CER / DER | Error metrics (aggregate) |
| RTF | Real-time factor (processing_time / audio_duration) |
| processing_time_s | Decode time (segment or aggregate sum) |
| latency_ms | End‑to‑end latency (standard) or segment latency |
| used_prefix | 1 if number prefix applied |
| segment_index / segment_count | Segment structure |
| notes | segment_row / aggregate / fallback_full_decode / no_segments |
| fish_prompt | Effective prompt string (if overridden) |
| model_extra_json | Serialized JSON of extra overrides from spec |

---
## 8. Failure Analysis
Primary failure filter:
```python
import pandas as pd
fails = pd.read_parquet('evaluation_outputs/run_.../failures.parquet')
print(fails[['wav_id','predicted_number','expected_number','recognized_text_raw']].head())
```
`run_summary.md` lists the first 10 failing examples.

Numeric mismatches can differ from integration tests because integration tests rely strictly on parser length whereas evaluation may fall back to text normalization when the parser does not yield a measurement.

---
## 9. Typical Workflows
| Goal | Action |
|------|--------|
| Add new model variant | Append new spec block in central config |
| Compare segment behavior | Set `production_replay: true` and re‑run |
| Disable number prefix (standard) | Set `concat_number: false` |
| Quick smoke test | `--max-samples 5` |
| GPU run | Set model spec devices to `["cuda"]` and adjust compute_types |
| Diff two runs | Compare two `results.parquet` (e.g. by wav_id, numeric_exact_match) |

---
## 10. Troubleshooting
| Symptom | Likely Cause | Resolution |
|---------|--------------|-----------|
| 0 configs generated | Malformed model_specs / missing name | Validate JSON & ensure `name` present |
| All predictions wrong | Prefix missing / wrong dataset_json | Check `concat_number`, verify dataset paths |
| Unexpected extra rows | Production replay enabled | Filter `is_aggregate == 1` for summary |
| Slow run | Too many spec combinations | Reduce sizes / beams / chunks lists |
| GPU unused | devices set to cpu | Set devices to ["cuda"] + correct drivers |
| Value off by 1 | Parser missed number, fallback normalization | Inspect raw + normalized text |

---
## 11. Extension Points
| Need | Where |
|------|------|
| New backend loader | Register with `@register_backend` in `evaluation/pipeline.py` |
| Extra per‑row feature | Append field to row dict before DataFrame creation |
| New metric | Implement in `evaluation/metrics.py` then compute & add to rows |
| Custom aggregation | Modify `_write_outputs` in `evaluation/pipeline.py` |
| Alternate preset sets | Add json under `evaluation/presets/` and load with `--preset` |
| Add recognizer override | include key in model spec (e.g. `"min_speech_s": 0.2`). It will appear in `model_extra_json` and applied to production replay recognizers. |

---
## 12. Standard vs Integration Tests
Integration test `tests/test_numbers_integration.py` calls the recognizer directly (no segmentation caching pipeline) and asserts parser length equality. Evaluation pipeline may succeed via numeric normalization even if parser length is absent → expect small mismatch in failure counts.

To reconcile differences:
1. Identify a wav_id that differs
2. Check its aggregate row in `results.parquet`
3. Compare `recognized_text_raw` vs integration CSV log
4. Inspect whether parser_length_cm was None but predicted_number extracted.

---
## 13. Future Improvements (Planned Ideas)
- Prompt version hashing for reproducible comparisons
- Validation schema for model_specs.json to flag unknown keys
- Optional `--no-segment-prefix` flag for production replay
- Initial species state injection for deterministic parser behavior across segments
- True streaming token timestamps for accurate first token latency
- Unified diff tool for comparing two run folders automatically
- Confidence calibration (probability vs numeric accuracy)

---
## 14. Minimal Custom Specs Example
`custom_specs.json`:
```json
{
  "model_specs": [
    {"name": "faster-whisper", "sizes": ["tiny.en","base.en"], "beams": [1,5], "chunks": [200,500]},
    {"name": "whisperx", "sizes": ["small.en"], "compute_types": ["float16"], "devices": ["cuda"], "beams": [3] }
  ]
}
```
Run:
```bash
python -m evaluation.run_evaluation --model-specs custom_specs.json
```

---
## 15. Quick Command Reference
```bash
# Default full evaluation via central config
python -m evaluation.run_evaluation

# Production replay + limited sample count
python -m evaluation.run_evaluation --production-replay --max-samples 8

# Alternate model specs
python -m evaluation.run_evaluation --model-specs custom_specs.json

# Smoke test (5 samples)
python -m evaluation.run_evaluation --max-samples 5

# Generate plots
python -m evaluation.run_evaluation --plots

# Disable number prefix (default is enabled)
python -m evaluation.run_evaluation --concat-number
```

---
## 16. Glossary
| Term | Meaning |
|------|---------|
| Aggregate row | Final combined transcript for a file (is_aggregate=1) |
| Segment row | Single segmented decode slice (production replay) |
| Numeric exact match | predicted_number == expected_number |
| RTF | Real-time factor (processing_time / audio_duration) |
| DER | Digit error rate |
| Prefix audio | number.wav concatenated before content |

---
For application usage, architecture, and non‑evaluation details see the project root `README.md`.
