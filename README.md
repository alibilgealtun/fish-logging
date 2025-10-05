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
