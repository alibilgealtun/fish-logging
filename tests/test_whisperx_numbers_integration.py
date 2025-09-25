import json
import csv
from pathlib import Path
from datetime import datetime
import pytest
from loguru import logger
import time
from scipy.signal import resample_poly

from speech.whisperx_recognizer import WhisperXRecognizer
from parser import FishParser, ParserResult

import soundfile as sf
import numpy as np
import tempfile
import os

# Availability check (skip module if missing whisperx)
try:
    import whisperx  # type: ignore
    _WHISPERX_AVAILABLE = True
except Exception as e:  # pragma: no cover
    logger.warning(f"whisperx not available for tests: {e}")
    _WHISPERX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _WHISPERX_AVAILABLE, reason="whisperx not installed")


class WhisperTestHarness:  # same class name as baseline; only model differs
    def __init__(self, use_concat: bool = False):
        self.recognizer = WhisperXRecognizer()
        self.parser = FishParser()
        self.recognizer._last_fish_specie = "salmon"
        self.use_concat = use_concat

        if self.recognizer._model is None:
            import whisperx as _wx  # type: ignore
            logger.info("Loading WhisperX model for tests...")
            model = None
            try:
                model = _wx.load_model(self.recognizer.MODEL_NAME, device=self.recognizer.DEVICE)
            except Exception:
                try:
                    model = _wx.load_model(self.recognizer.MODEL_NAME, device=self.recognizer.DEVICE, compute_type=self.recognizer.COMPUTE_TYPE)
                except Exception:
                    if hasattr(_wx, "WhisperXModel"):
                        model = _wx.WhisperXModel(self.recognizer.MODEL_NAME, device=self.recognizer.DEVICE)
                    elif hasattr(_wx, "WhisperModel"):
                        model = _wx.WhisperModel(self.recognizer.MODEL_NAME, device=self.recognizer.DEVICE)
                    else:
                        raise RuntimeError("Unable to load whisperx model with available APIs")
            self.recognizer._model = model

        if self.use_concat:
            try:
                self._number_sound, _ = sf.read("tests/audio/number.wav", dtype='int16')
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
        # Mirror baseline call; fallback if signature mismatch
        try:
            segments_result = model.transcribe(  # type: ignore[attr-defined]
                wav_path,
                language="en",
                beam_size=self.recognizer.BEAM_SIZE,
                best_of=self.recognizer.BEST_OF,
                temperature=self.recognizer.TEMPERATURE,
                patience=self.recognizer.PATIENCE,
                length_penalty=self.recognizer.LENGTH_PENALTY,
                repetition_penalty=self.recognizer.REPETITION_PENALTY,
                without_timestamps=self.recognizer.WITHOUT_TIMESTAMPS,
                condition_on_previous_text=self.recognizer.CONDITION_ON_PREVIOUS_TEXT,
                initial_prompt=self.recognizer.FISH_PROMPT,
                vad_filter=self.recognizer.VAD_FILTER,
                vad_parameters=self.recognizer.VAD_PARAMETERS,
                word_timestamps=self.recognizer.WORD_TIMESTAMPS
            )
        except Exception:
            segments_result = model.transcribe(wav_path)  # type: ignore[attr-defined]

        # Normalize like baseline to a list of segment objects with .text attr
        segments_text: list[str] = []
        if isinstance(segments_result, dict):
            raw_segments = segments_result.get("segments")
            if raw_segments is None and "text" in segments_result:
                val = segments_result.get("text", "")
                if val:
                    segments_text.append(str(val).strip())
            else:
                for seg in raw_segments or []:
                    if isinstance(seg, dict):
                        segments_text.append(seg.get("text", "").strip())
                    else:
                        segments_text.append(getattr(seg, "text", str(seg)).strip())
        elif isinstance(segments_result, list):
            for seg in segments_result:
                if isinstance(seg, dict):
                    segments_text.append(seg.get("text", "").strip())
                else:
                    segments_text.append(getattr(seg, "text", str(seg)).strip())
        else:
            segments_text.append(str(segments_result).strip())

        raw_text = "".join(t + " " for t in segments_text).strip()
        cleaned_text = raw_text.rstrip('.,?!').strip()
        result = self.parser.parse_text(cleaned_text)
        return raw_text, result

# --- Load dataset ---
DATASET_PATH = Path(__file__).parent / "data/numbers.json"
with open(DATASET_PATH) as f:
    TEST_DATA = json.load(f)

# --- Results folder setup ---
RESULTS_FOLDER = Path(__file__).parent / "results"
RESULTS_FOLDER.mkdir(exist_ok=True)

# Timestamp for unique filenames
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

# File paths (same naming pattern as baseline)
CSV_FILE = RESULTS_FOLDER / f"test_results_{timestamp_str}.csv"
FAIL_CSV_FILE = RESULTS_FOLDER / f"test_failures_{timestamp_str}.csv"

# CSV header (identical to baseline)
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
    logger.info("🟢 Starting new WhisperX test session")
    logger.info(f"Results CSV: {CSV_FILE}")
    logger.info(f"Fail CSV: {FAIL_CSV_FILE}")
    run_all_tests()
