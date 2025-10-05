## Voice2FishLog

A Python + PyQt6 desktop application that uses the `faster-whisper` large-v2 model for real-time speech-to-text, extracts fish species and lengths, and logs them into Excel using `openpyxl`. It can run offline once the model is cached.

### Features
- Real-time mic transcription (offline-capable if model is downloaded)
- Robust parsing with fuzzy species matching and unit normalization
- Spoken-number to float conversion
- Excel logging with cancel/undo of last entry
- Noise evaluation script with WER measurement

### Setup
1. Install Python 3.10+
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Pre-download the Whisper `large-v2` model to run fully offline. It will be cached on first run in `~/.cache`.

### Run
```bash
python main.py
```

### Excel Log
- File: `logs.xlsx`
- Columns: Date | Time | Species | Length (cm) | Confidence

### Noise Evaluation
- Put noisy `.wav` files and references in `samples/`.
- Create `samples/refs.txt` with lines in the form: `filename.wav|reference text`.
- Run:
  ```bash
  python noise_eval.py
  ```

### Notes
- Windows 10/11 primary target; Linux supported
- If your microphone is not the default input, set it in Windows Sound settings or modify `speech.py` to choose a device.

# Fish Logging

This application supports multiple speech-to-text engines optimized for fish species and measurements.

## Engines
- Whisper (default)
- Vosk
- Google Cloud Speech-to-Text (new)

## Google Cloud Speech-to-Text

1) Enable API and create credentials
- In Google Cloud Console, enable “Cloud Speech-to-Text API”.
- Create a Service Account with role: Speech-to-Text User.
- Create a JSON key and download it.

2) Provide credentials
- Option A: Set the environment variable
  - macOS/Linux:
    export GOOGLE_APPLICATION_CREDENTIALS="/absolute/path/to/key.json"
  - Windows (PowerShell):
    $env:GOOGLE_APPLICATION_CREDENTIALS="C:\\path\\to\\key.json"
- Option B: Place the key file at the project root named google.json (auto-detected).

3) Install dependencies
pip install -r requirements.txt

4) Run with Google engine
python main.py --model=google

Notes
- The recognizer uses phrase hints from config/species.json, config/numbers.json, and config/units.json to bias recognition.
- Real-time mode uses a noise controller and voice activity detection to segment audio and sends short segments to Google for recognition.
- Use commands “wait” and “start” to pause/resume parsing.

### ASR Evaluation Pipeline (Numeric Focus)

The `evaluation/` package provides a reproducible pipeline to benchmark multiple ASR backends on numeric transcription accuracy and real‑time performance.

Key metrics: WER, CER, Digit Error Rate (DER), Numeric Exact Match %, Mean Absolute Error (MAE), RTF (p50/p95/p99). Breakdowns by accent, noise type, model size.

Dataset structure (example):
```
 test_data/
   numbers/
     speaker_1_turkish/
       1.wav
       2.wav
       metadata.json
```
Metadata example:
```
{
  "speaker_id": "speaker_1",
  "speaker_profile": {"ethnicity": "turkish", "age": "25", "gender": "male"},
  "recording_conditions": {"microphone": "DJI Mic 2", "environment": "quiet_room"},
  "contents": {"type": "number", "audios": [
    {"audio": "1.wav", "duration": 3.4, "expected": 1},
    {"audio": "2.wav", "duration": 2.3, "expected": 2}
  ]}
}
```

Run a quick single configuration:
```bash
python -m evaluation.run_evaluation \
  --dataset test_data/numbers \
  --models faster-whisper \
  --sizes base.en \
  --compute-types int8 \
  --devices cpu \
  --beams 5 \
  --chunks 500 \
  --vad-modes 2
```

Multiple combinations (grid):
```bash
python -m evaluation.run_evaluation \
  --dataset test_data/numbers \
  --models faster-whisper whisperx \
  --sizes tiny.en base.en \
  --compute-types int8 \
  --devices cpu \
  --beams 1 5 \
  --chunks 200 500 \
  --vad-modes 1 2 3
```

Outputs (written to `evaluation_outputs/`):
- `results.parquet` – per‑sample detailed logs
- `failures.parquet` – only mispredicted samples (numeric mismatch)
- `summary.xlsx` – aggregated breakdown (model, accent, noise) + RTF percentiles

Add `--plots` to generate basic visualization PNGs (WER vs model size, RTF distribution).

Optional real‑time simulation (adds artificial time pacing):
```bash
python -m evaluation.run_evaluation --dataset test_data/numbers --real-time
```

Use `--max-samples N` to subsample for quick smoke tests.

Environment variables:
- `EVAL_OUTPUT_DIR` to override output directory path.

## ASR Evaluation Detailed Guide (Extended Documentation)

This section documents every evaluation feature added, all CLI flags, modes, outputs, and internal columns so you can reproduce or extend tests without reading the code.

### 1. Core Concepts
- **Standard Mode**: Single-pass transcript of an audio file (no segmentation). (Previous `StreamingSimulator` approximation.)
- **Production Replay Mode** (`--production-replay`): Replays audio exactly like the running app would:
  - Feeds audio in CHUNK_S-sized frames (recognizer.CHUNK_S, default 0.5s) as int16 into a fresh `NoiseController`.
  - Uses recognizer's VAD settings (VAD_MODE, MIN_SPEECH_S, MAX_SEGMENT_S, PADDING_MS).
  - Segments produced by WebRTC VAD + padding rules.
  - Each segment is transcribed individually with the recognizer backend (faster-whisper / whisperx) using the same `_backend_transcribe` method as the live app.
  - A number prefix audio (recognizer._number_sound) is prepended for each segment (mirrors live pipeline) — independent of `--concat-number`.
  - Segment rows (is_aggregate=0) + a merged aggregate row (is_aggregate=1) are logged.
  - Aggregate row undergoes parser normalization (TextNormalizer + FishParser) generating `final_display_text` just like UI.

### 2. CLI Flags (Evaluation)
| Flag | Purpose | Notes |
|------|---------|-------|
| `--dataset` | Directory root for fallback discovery (metadata.json or filename inference) | Optional if `--dataset-json` used |
| `--dataset-json` | Explicit JSON list of objects `{"audio": "1.wav", "expected": 1}` | Highest priority data source |
| `--audio-root` | Directory where JSON-listed audio files reside | Defaults to JSON directory if omitted |
| `--models` | Backend names (e.g. `faster-whisper`, `whisperx`) | Grid-enabled |
| `--sizes` | Model size variants (e.g. `tiny.en`, `base.en`) | Whisper / whisperx formats |
| `--compute-types` | Precision (`int8`, `float16`, `float32`) | Passed to backend loaders when supported |
| `--devices` | Device list (`cpu`, `cuda`) | One or many |
| `--beams` | Beam sizes for decoding | Maps to recognizer or backend parameter |
| `--chunks` | Streaming chunk size (ms) for approximation mode | Not used in production replay segmentation (CHUNK_S is used), but still logged |
| `--vad-modes` | WebRTC VAD aggressiveness (0–3) | Applied in standard-mode config metadata only; production replay uses recognizer's own attributes |
| `--languages` | Language(s) passed to backend | e.g. `en` |
| `--concat-number` | Prepend number prefix audio ONCE in non-production mode | In production replay prefix always applies per segment (live behavior) |
| `--number-audio` | Override path to a prefix audio (number.wav) | Search order override |
| `--grid` | Path to JSON/YAML defining parameter lists | CLI overrides grid values |
| `--preset` | Load preset from `evaluation/presets/<name>.json` | Populates missing options |
| `--production-replay` | Enable full NoiseController + segmentation + per-segment decode | Closest to `main.py` |
| `--max-samples` | Subsample dataset for quick runs | Applies after dataset load |
| `--real-time` | Sleep during feed (frame pacing) | For latency-like replay |
| `--plots` | Generate simple PNG charts | WER vs size, RTF distribution |
| `--output-dir` | Output folder (defaults `evaluation_outputs/`) | All artifacts stored here |

### 3. Data Sources Priority
1. `--dataset-json` (+ optional `--audio-root`)
2. `metadata.json` discovery (recursive)
3. Filename inference (`*.wav` name digits -> expected value)

Each sample annotated with `sample_source`: `json`, `metadata`, `filename_fallback`.

### 4. Number Prefix Behavior
| Mode | Behavior |
|------|----------|
| Standard | Only if `--concat-number` provided (single prepend of chosen number audio) |
| Production Replay | Always uses recognizer’s `_number_sound` per segment (matches live recorder) |

Field `used_prefix` indicates whether a prefix was applied (1/0).

### 5. Output Artifacts
| File | Description |
|------|-------------|
| `results.parquet` | All sample/segment + aggregate rows (wide schema) |
| `failures.parquet` | Rows (aggregate only) where `numeric_exact_match == 0` |
| `summary.xlsx` | Multiple sheets (sample preview, pivots, RTF stats) |
| `run_summary.json` | High-level metrics, environment info |
| `run_summary.md` | Human-readable summary + top failures |
| `nem_by_model_size.png` | (If `--plots`) Bar plot of numeric_exact_match |
| `rtf_distribution.png` | (If `--plots`) Boxplot of RTF |

### 6. `summary.xlsx` Sheets
| Sheet | Contents |
|-------|----------|
| `sample` | First 200 rows (mixed segment & aggregate) |
| `by_model` | Aggregated metrics by (model_name, model_size) |
| `by_accent` | Aggregated metrics by accent, model size |
| `by_noise` | Aggregated metrics by noise_type |
| `by_range` | Number range buckets (0-9,10-19,...) if available |
| `rtf_percentiles` | Global RTF p50/p95/p99 |
| `rtf_by_model` | RTF percentiles per model+size |

Note: Aggregations use only `is_aggregate == 1` rows to avoid double counting segments.

### 7. Row Types
| Field | Meaning |
|-------|---------|
| `is_aggregate` | 1 => final merged transcript for file, 0 => per-segment transcription row |
| `segment_index` | 0-based index (segment rows) or null (aggregate) |
| `segment_count` | Total segments for that file (present on all related rows) |

### 8. Key Metrics / Columns
| Column | Description |
|--------|-------------|
| `recognized_text_raw` | Raw concatenated text (segment or aggregate) |
| `recognized_text_normalized` | Lowercased & punctuation-stripped form used for numeric extraction |
| `predicted_number` | Parsed numeric value (spoken or digit) or null |
| `expected_number` | From dataset (JSON or inferred) |
| `numeric_exact_match` | 1 if predicted == expected (within tolerance), else 0 |
| `WER`, `CER`, `DER` | Word / char / digit error metrics (aggregate rows) |
| `absolute_error` | |predicted - expected| (numeric rows only) |
| `parser_species` | Species recognized by FishParser (aggregate prod replay & fallback) |
| `parser_length_cm` | Length in cm parsed by FishParser |
| `parser_exact_match` | 1 if parser_length_cm == expected_number |
| `final_display_text` | What UI would show (e.g. `salmon 23 cm`) |
| `error_type` | Heuristic classification: substitution, deletion, insertion, ordering, formatting, etc. |
| `used_prefix` | Number prefix used (1/0) |
| `sample_source` | json / metadata / filename_fallback |
| `number_range` | Bucket (0-9,10-19,20-29,30-39,40-49,50+) |
| `RTF` | Real-time factor = processing_time / audio_duration |
| `processing_time_s` | Decode time (sum of segments for aggregate) |
| `latency_ms` | In standard mode: end-start; in segment rows approximate first segment latency; may be null aggregate in prod replay |
| `audio_start`, `audio_end` | Segment timestamps from NoiseController in replay mode |
| `segment_duration_s` | Duration of individual segment |
| `segment_processing_time_s` | Time spent decoding that segment |

### 9. Production Replay Notes
- Parser is applied only on aggregate & fallback rows (segments keep raw text for forensic debugging). 
- Prefix is always added per segment (mirrors live). 
- If no segment produced (short audio), pipeline falls back to a single full decode row (`notes=fallback_full_decode`).
- `first_decoded_token_time` is approximated as decode start time (framework doesn’t emit incremental tokens here).

### 10. Example Commands
**Quick single config (standard mode):**
```bash
python -m evaluation.run_evaluation \
  --dataset-json tests/data/numbers.json \
  --audio-root tests/audio \
  --models faster-whisper \
  --sizes base.en \
  --compute-types int8 \
  --devices cpu \
  --beams 5 --chunks 500 --vad-modes 2 \
  --max-samples 10
```

**Production replay (NoiseController segmentation):**
```bash
python -m evaluation.run_evaluation \
  --dataset-json tests/data/numbers.json \
  --audio-root tests/audio \
  --models faster-whisper \
  --sizes base.en \
  --compute-types int8 \
  --devices cpu \
  --beams 3 \
  --vad-modes 2 \
  --production-replay \
  --max-samples 5
```

**Grid file (grid.json):**
```json
{
  "models": ["faster-whisper", "whisperx"],
  "sizes": ["tiny.en", "base.en"],
  "compute_types": ["int8"],
  "devices": ["cpu"],
  "beams": [1,5],
  "chunks": [200,500],
  "vad_modes": [1,2],
  "dataset_json": "tests/data/numbers.json",
  "audio_root": "tests/audio",
  "concat_number": true
}
```
Run:
```bash
python -m evaluation.run_evaluation --grid grid.json --max-samples 8
```

**Preset usage (quick preset example):**
```
python -m evaluation.run_evaluation --preset quick
```
(Requires `evaluation/presets/quick.json` to exist.)

### 11. Failure Analysis
- Failures are stored twice: full row in `results.parquet`, filtered subset in `failures.parquet` (aggregate rows only). 
- `error_type` gives coarse reason; refining alignment can be implemented later if needed.
- `final_display_text` helps correlate UI-facing output vs numeric normalization.

### 12. Data Dictionary (Consolidated)
| Name | Type | Level | Description |
|------|------|-------|-------------|
| test_run_id | str | All | Unique run id |
| wav_id | str | All | Audio filename |
| is_aggregate | int (0/1) | All | 1 = merged text, 0 = per-segment |
| segment_index | int / null | Segment | Segment order |
| segment_count | int | All | Total segments for file |
| sample_source | str | All | json / metadata / filename_fallback |
| used_prefix | int | All | Prefix applied (1/0) |
| recognized_text_raw | str | All | Raw decoded text |
| recognized_text_normalized | str | All | Normalized text (lower/punct stripped) |
| predicted_number | float / null | Aggregate | Parsed numeric hypothesis |
| expected_number | float / null | All | Ground truth numeric |
| numeric_exact_match | int / null | Aggregate | 1 if predicted == expected |
| parser_species | str / null | Aggregate (prod) | Fish species parsed |
| parser_length_cm | float / null | Aggregate (prod) | Parsed length cm |
| parser_exact_match | int / null | Aggregate (prod) | Parser length == expected |
| final_display_text | str / null | Aggregate (prod) | UI-style final line |
| WER / CER / DER | float / null | Aggregate | Error metrics |
| absolute_error | float / null | Aggregate | |predicted - expected| |
| error_type | str / null | Aggregate | substitution / deletion / ... |
| number_range | str | All | 0-9 / 10-19 / ... |
| processing_time_s | float | All | Decode time (segment or combined) |
| RTF | float / null | All | Real-time factor |
| latency_ms | float / null | Aggregate (standard) / segment | Approximate latency |
| audio_start / audio_end | float / null | Segment | Wallclock timestamps from NoiseController |
| segment_duration_s | float / null | Segment | Duration seconds |
| segment_processing_time_s | float / null | Segment | Decode time for that segment |
| model_name / model_size | str | All | Backend + size |
| config_json | str | All | Serialized EvaluationConfig |
| vad_setting | int | All | VAD mode integer |
| cpu_percent | float | All | Snapshot CPU usage |
| memory_mb | float | All | Process RSS in MB |
| gpu_percent / gpu_memory_mb | float / null | When NVML present | GPU utilization / memory |
| notes | str / null | All | Extra marker (segment_row, aggregate, fallback_full_decode) |

### 13. Known Limitations
- first_decoded_token_time approximated (no incremental token streaming API used). 
- WER/CER simplified tokenization (not full alignment S/I/D breakdown). 
- Parser not run per segment to save time (only aggregate + fallback). 
- Production replay still uses full-segment decode, not incremental hypothesis emission. 
- Confidence calibration (log-prob vs accuracy) not yet implemented.

### 14. Extending
- Add presets in `evaluation/presets/` to reduce CLI verbosity.
- Add synthetic noise augmentation ahead of evaluation for environment profiling.
- Integrate real incremental decoding for first token latency accuracy.

---
If anything here is unclear, search for the corresponding column name in `evaluation/pipeline.py` for ground-truth implementation details.
