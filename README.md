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
