import json
import csv
from pathlib import Path
from datetime import datetime
import pytest
from loguru import logger
import time
from scipy.signal import resample_poly

from speech.insanely_faster_whisper import InsanelyFastWhisperRecognizer
from parser import FishParser, ParserResult

import soundfile as sf
import numpy as np
import tempfile
import os

# Availability check
try:
    import insanely_fast_whisper as _ifw  # type: ignore
    _IFW_AVAILABLE = True
except Exception as e:  # pragma: no cover
    logger.warning(f"insanely-fast-whisper not available for tests: {e}")
    _IFW_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _IFW_AVAILABLE, reason="insanely-fast-whisper not installed")


class WhisperTestHarness:  # match baseline name, only model differs
    def __init__(self, use_concat: bool = False):
        self.recognizer = InsanelyFastWhisperRecognizer()
        self.parser = FishParser()
        self.recognizer._last_fish_specie = "salmon"
        self.use_concat = use_concat

        if self.recognizer._model is None:
            # Try multiple API patterns before skipping
            model_loaded = False
            load_errors = []
            # Pattern 1: Transcriber class
            try:
                if hasattr(_ifw, "Transcriber"):
                    logger.info("Loading InsanelyFastWhisper model via Transcriber for tests...")
                    self.recognizer._model = _ifw.Transcriber(model=self.recognizer.MODEL_NAME, device=self.recognizer.DEVICE)  # type: ignore[attr-defined]
                    model_loaded = True
            except Exception as e:
                load_errors.append(f"Transcriber: {e}")
            # Pattern 2: load_model function
            if not model_loaded:
                try:
                    if hasattr(_ifw, "load_model"):
                        logger.info("Loading InsanelyFastWhisper model via load_model() for tests...")
                        self.recognizer._model = _ifw.load_model(self.recognizer.MODEL_NAME, device=self.recognizer.DEVICE)  # type: ignore[attr-defined]
                        model_loaded = True
                except Exception as e:
                    load_errors.append(f"load_model: {e}")
            # Pattern 3: fall back to faster_whisper if exposed (some forks re-export)
            if not model_loaded:
                try:
                    from faster_whisper import WhisperModel  # type: ignore
                    logger.info("Falling back to faster_whisper WhisperModel for insanely-fast-whisper tests...")
                    self.recognizer._model = WhisperModel(
                        self.recognizer.MODEL_NAME,
                        device=self.recognizer.DEVICE,
                        compute_type=self.recognizer.COMPUTE_TYPE
                    )
                    model_loaded = True
                except Exception as e:
                    load_errors.append(f"faster_whisper fallback: {e}")
            if not model_loaded:
                pytest.skip("insanely_fast_whisper API unsupported for these tests: " + "; ".join(load_errors))

        if self.use_concat:
            try:
                self._number_sound, _ = sf.read("assets/audio/number.wav", dtype='int16')
            except Exception:
                self._number_sound = (np.zeros(int(self.recognizer.SAMPLE_RATE * 0.05))).astype(np.int16)

    def _write_wav_bytes(self, samples_int16: np.ndarray, samplerate: int) -> str:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(path, samples_int16, samplerate, subtype="PCM_16")
        return path

    def transcribe_file(self, wav_path: str) -> tuple[str, ParserResult]:
        if self.use_concat:
            audio, sr = sf.read(wav_path, dtype='int16')
            target_sr = self.recognizer.SAMPLE_RATE
            if sr != target_sr:
                gcd = np.gcd(sr, target_sr)
                up = target_sr // gcd
                down = sr // gcd
                audio = resample_poly(audio, up, down).astype(np.int16)
                sr = target_sr
            combined = np.concatenate((self._number_sound, audio))
            wav_path = self._write_wav_bytes(combined, sr)

        model = self.recognizer._model
        # Try rich signature first, then minimal
        try:
            result = model.transcribe(  # type: ignore[attr-defined]
                wav_path,
                beam_size=getattr(self.recognizer, 'BEAM_SIZE', 3),
                best_of=getattr(self.recognizer, 'BEST_OF', 5),
                temperature=getattr(self.recognizer, 'TEMPERATURE', 0.0),
                initial_prompt=getattr(self.recognizer, 'FISH_PROMPT', None),
                without_timestamps=getattr(self.recognizer, 'WITHOUT_TIMESTAMPS', True)
            )
        except Exception:
            result = model.transcribe(wav_path)  # type: ignore[attr-defined]

        # Normalize to segments list
        segments_text: list[str] = []
        if isinstance(result, dict):
            if isinstance(result.get("segments"), list):
                for seg in result["segments"]:
                    if isinstance(seg, dict):
                        segments_text.append(seg.get("text", "").strip())
                    else:
                        segments_text.append(getattr(seg, "text", str(seg)).strip())
            elif "text" in result:
                segments_text.append(str(result.get("text", "")).strip())
        elif isinstance(result, list):
            for seg in result:
                if isinstance(seg, dict):
                    segments_text.append(seg.get("text", "").strip())
                else:
                    segments_text.append(getattr(seg, "text", str(seg)).strip())
        else:
            segments_text.append(str(result).strip())

        raw_text = "".join(t + " " for t in segments_text).strip()
        cleaned_text = raw_text.rstrip('.,?!').strip()
        parsed = self.parser.parse_text(cleaned_text)
        return raw_text, parsed

# --- Load dataset ---
DATASET_PATH = Path(__file__).parent / "data/numbers.json"
with open(DATASET_PATH) as f:
    TEST_DATA = json.load(f)

# --- Results folder setup ---
RESULTS_FOLDER = Path(__file__).parent / "results"
RESULTS_FOLDER.mkdir(exist_ok=True)

# Timestamp for unique filenames
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

# File paths (match baseline naming)
CSV_FILE = RESULTS_FOLDER / f"test_results_{timestamp_str}.csv"
FAIL_CSV_FILE = RESULTS_FOLDER / f"test_failures_{timestamp_str}.csv"

# CSV header identical to baseline
CSV_FIELDS = [
    "audio", "raw_text", "species", "length_cm", "expected", "passed", "timestamp",
    "SAMPLE_RATE", "CHANNELS", "CHUNK_S", "VAD_MODE", "MIN_SPEECH_S", "MAX_SEGMENT_S", "PADDING_MS",
    "FISH_PROMPT", "MODEL_NAME", "DEVICE", "COMPUTE_TYPE",
    "BEAM_SIZE", "BEST_OF", "TEMPERATURE", "PATIENCE", "LENGTH_PENALTY", "REPETITION_PENALTY",
    "WITHOUT_TIMESTAMPS", "CONDITION_ON_PREVIOUS_TEXT", "VAD_FILTER", "VAD_PARAMETERS", "WORD_TIMESTAMPS"
]


def get_whisper_config(recognizer):
    return recognizer.get_config()


def log_result(audio, raw_text, parsed, expected, passed, config):
    record = {
        "audio": audio,
        "raw_text": raw_text,
        "species": parsed.species,
        "length_cm": parsed.length_cm,
        "expected": expected,
        "passed": passed,
        "timestamp": datetime.now().isoformat(),
        **config
    }
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(record)
    if not passed:
        with open(FAIL_CSV_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(record)


@pytest.mark.parametrize("case", TEST_DATA)
def test_numbers(case):
    harness = WhisperTestHarness(use_concat=False)
    audio_path = Path(__file__).parent / "audio" / case["audio"]
    expected = case["expected"]

    raw_text, result = harness.transcribe_file(str(audio_path))
    config = get_whisper_config(harness.recognizer)

    logger.info("────────────────────────────────────────")
    logger.info(f"🎤 Audio: {case['audio']}")
    logger.info(f"📜 Raw transcript: '{raw_text}'")
    logger.info(f"📦 Parser result: {result}")
    logger.info(f"✅ Expected length: {expected}, 🧮 Got: {result.length_cm}")

    passed = result.length_cm == expected
    log_result(case["audio"], raw_text, result, expected, passed, config)

    assert passed, (
        f"\nAudio: {case['audio']}\n"\
        f"Raw transcript: '{raw_text}'\n"\
        f"Parser result: {result}\n"\
        f"Expected: {expected}, Got: {result.length_cm}"
    )


def run_all_tests():
    start_time = time.time()
    for case in TEST_DATA:
        test_numbers(case)  # type: ignore[arg-type]
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"⏱️ All tests completed in {duration:.2f} seconds.")
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Total test duration (seconds):", f"{duration:.2f}"])


if __name__ == "__main__":
    logger.info("🟢 Starting new InsanelyFastWhisper test session")
    logger.info(f"Results CSV: {CSV_FILE}")
    logger.info(f"Fail CSV: {FAIL_CSV_FILE}")
    run_all_tests()
