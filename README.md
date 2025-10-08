# Voice2FishLog

A desktop application (PyQt6) for real‑time speech → structured fish log entries (species + length in cm). Supports multiple ASR backends and a reproducible numeric evaluation pipeline.

---
## 1. Features
- Multiple ASR engines: faster‑whisper (default), whisperx, vosk, (optional) Google Cloud
- Real‑time segmentation (NoiseController + WebRTC VAD)
- Robust normalization (numbers, units, species spelling variants)
- FishParser: extract species + length (cm) from free speech
- Optional number prefix (number.wav) to prime numeric recognition
- Undo / remove last entry, session & Excel logging
- Offline capable once models cached
- Selectable noise profiles (clean / human / engine / mixed) to adapt suppression & VAD

---
## 2. Installation
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
(First run will download required ASR model weights into user cache.)

---
## 3. Quick Start
```bash
python main.py
```
Speak e.g. “salmon twenty three point five” → UI shows a normalized entry (species + length cm) → save / undo.

---
## 4. Configuration Files (config/)
| File | Purpose |
|------|---------|
| species.json | Species lexicon + fuzzy aliases |
| numbers.json | Spoken number variants / mapping |
| units.json | Unit variants (cm, centimeter, santim, etc.) |
| asr_corrections.json | Post‑ASR correction pairs |
| google_sheets.json | Optional Sheets backup credentials/IDs |

---
## 5. ASR Engines
Engine selection is handled by `speech/factory.py`.

### faster‑whisper
Default engine. Model size / precision defined in recognizer class constants.

### whisperx / vosk / google
Additional recognizers live under `speech/`. For Google Cloud:
```bash
export GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/key.json
```
(Ensure the Speech‑to‑Text API is enabled and service account has permission.)

---
## 6. Processing Pipeline (Live)
1. Microphone audio buffered in CHUNK_S frames
2. NoiseController + WebRTC VAD produce segments (with padding)
3. (Optional) number prefix prepended
4. Segment sent to selected ASR backend
5. Text normalized + corrections applied
6. FishParser extracts species + length
7. Entry rendered & logged

Recognizer constants (e.g. VAD_MODE, MIN_SPEECH_S, MAX_SEGMENT_S, PADDING_MS) tune segmentation behavior.

---
## 6.1 Noise Profiles (Adaptive Acoustic Modes)
The app supports selectable noise profiles to optimize speech segmentation and suppression in different environments. You can change the active profile any time in the Settings tab (⚙️) via the “Noise Profile” dropdown. Switching restarts the recognizer with new parameters.

| Profile | Intended Environment | Key Changes |
|---------|----------------------|-------------|
| Mixed (default) | General use (both human chatter + some engine) | Balanced VAD (mode 2), moderate suppression |
| Human Voices | Background conversations / crew talk | Slightly lower min segment, higher speech band boost |
| Engine Noise | Steady engine / mechanical hum | More aggressive VAD (mode 3), slower noise adaptation, stronger gate |
| Clean / Quiet | Calm cabin / near‑silent room | Less aggressive VAD (mode 1), gentler suppression, faster segment close |

Internals per profile (see `speech/noise_profiles.py`):
- Adjusts: VAD_MODE, MIN_SPEECH_S, MAX_SEGMENT_S, PADDING_MS
- Builds a tailored `SuppressorConfig` (gain_floor, noise_update_alpha, speech_band_boost, gate settings).

CLI / environment usage:
```bash
python main.py --noise-profile engine
# or
export SPEECH_NOISE_PROFILE=human
python main.py
```

If unsure, keep “Mixed”. Use “Engine Noise” only when low‑frequency drone dominates; use “Human Voices” when overlapping crew speech causes false merges. “Clean” minimizes latency in quiet spaces.

---
## 7. Logging
- Session / Excel logs under `logs/`
- Undo removes last accepted measurement
- (Optionally) integrate Google Sheets via service credentials

---
## 8. Parsing & Normalization
- Corrections (asr_corrections.json) → canonical text
- Number + unit normalization (numbers.json, units.json)
- FishParser resolves final (species, length_cm)
- Display formatting: `<species> <value> cm`

---
## 9. Performance Tips
| Goal | Tip |
|------|-----|
| Faster CPU inference | Use smaller model (tiny/tiny.en/base.en) & int8 compute |
| Better accuracy | Increase beam size or choose larger model |
| Lower latency | Reduce CHUNK_S & aggressive VAD (trade segmentation) |
| GPU boost | Use whisperx or faster‑whisper with CUDA device |

---
## 10. Evaluation & Testing
A dedicated, reproducible evaluation pipeline (numeric accuracy, latency, segmentation realism) lives in `evaluation/`.

👉 See: `evaluation/README_TEST_EVAL.md` (test & evaluation guide).

Key points:
- Central config: `evaluation/presets/model_specs.json` (single source of models, dataset, flags)
- Per‑run artifacts in timestamped subfolders under `evaluation_outputs/`
- Production replay mode emulates live segmentation path

---
## 11. Extending
| Task | Steps |
|------|-------|
| Add new ASR engine | Create recognizer (inherits BaseSpeechRecognizer) + register in `speech/factory.py` |
| Add metric / column (eval) | Insert into row dict in `evaluation/pipeline.py` and extend summary aggregation if needed |
| New species / aliases | Update species.json + corrections if required |
| Additional language | Supply language code + extend numbers/units mapping |

---
## 12. Troubleshooting
| Symptom | Cause | Fix |
|---------|-------|-----|
| GUI fails to start | PyQt not installed | Reinstall requirements |
| Slow decoding | Large model on CPU | Switch to base/tiny + int8 |
| Species missing | Not in species.json | Add alias / base name |
| Wrong number | Prefix missing / segmentation | Provide number.wav or adjust VAD_MODE |
| No segments | VAD too strict | Lower VAD aggressiveness or MIN_SPEECH_S |

---
For deep dive into metrics & automated benchmarks: open `evaluation/README_TEST_EVAL.md`.
