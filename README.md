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
