import json
import csv
from pathlib import Path
from datetime import datetime
import pytest
from loguru import logger
import time

from speech.whisper_recognizer import WhisperRecognizer
from parser import FishParser, ParserResult

# --- Test harness ---
class WhisperTestHarness:
    def __init__(self):
        self.recognizer = WhisperRecognizer()
        self.parser = FishParser()
        self.recognizer._last_fish_specie = "salmon"  # default species

        # Load whisper model once
        if self.recognizer._model is None:
            from faster_whisper import WhisperModel
            logger.info("Loading Whisper model for tests...")
            self.recognizer._model = WhisperModel(
                self.recognizer.MODEL_NAME,
                device=self.recognizer.DEVICE,
                compute_type=self.recognizer.COMPUTE_TYPE
            )

    def transcribe_file(self, wav_path: str) -> tuple[str, ParserResult]:
        """
        Run recognizer pipeline on a single wav file (batch mode).
        Returns (raw_transcript, ParserResult).
        """
        segments, _ = self.recognizer._model.transcribe(
            wav_path,
            language="en",
            beam_size=self.recognizer.BEAM_SIZE,
            best_of=self.recognizer.BEST_OF,
            temperature=self.recognizer.TEMPERATURE,
            patience=self.recognizer.PATIENCE,
            length_penalty=self.recognizer.LENGTH_PENALTY,
            repetition_penalty=self.recognizer.REPETITION_PENALTY,
            without_timestamps=self.recognizer.WITHOUT_TIMESTAMPS if hasattr(self.recognizer, 'WITHOUT_TIMESTAMPS') else True,
            condition_on_previous_text=self.recognizer.CONDITION_ON_PREVIOUS_TEXT if hasattr(self.recognizer, 'CONDITION_ON_PREVIOUS_TEXT') else True,
            initial_prompt=self.recognizer.FISH_PROMPT,
            vad_filter=self.recognizer.VAD_FILTER if hasattr(self.recognizer, 'VAD_FILTER') else False,
            vad_parameters=getattr(self.recognizer, 'VAD_PARAMETERS', None),
            word_timestamps=getattr(self.recognizer, 'WORD_TIMESTAMPS', False)
        )

        raw_text = "".join(seg.text + " " for seg in segments).strip()
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

# File paths
CSV_FILE = RESULTS_FOLDER / f"test_results_{timestamp_str}.csv"
FAIL_CSV_FILE = RESULTS_FOLDER / f"test_failures_{timestamp_str}.csv"

# CSV header
CSV_FIELDS = [
    "audio", "raw_text", "species", "length_cm", "expected", "passed", "timestamp",
    "SAMPLE_RATE", "CHANNELS", "CHUNK_S", "VAD_MODE", "MIN_SPEECH_S", "MAX_SEGMENT_S", "PADDING_MS",
    "FISH_PROMPT", "MODEL_NAME", "DEVICE", "COMPUTE_TYPE",
    "BEAM_SIZE", "BEST_OF", "TEMPERATURE", "PATIENCE", "LENGTH_PENALTY", "REPETITION_PENALTY",
    "WITHOUT_TIMESTAMPS", "CONDITION_ON_PREVIOUS_TEXT", "VAD_FILTER", "VAD_PARAMETERS", "WORD_TIMESTAMPS"
]


# Helper to extract all config values from WhisperRecognizer
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

    # --- CSV append (all tests) ---
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(record)

    # --- CSV append (fail only) ---
    if not passed:
        with open(FAIL_CSV_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(record)


# --- Pytest tests ---
@pytest.mark.parametrize("case", TEST_DATA)
def test_numbers(case):
    harness = WhisperTestHarness()
    audio_path = Path(__file__).parent / "audio" / case["audio"]
    expected = case["expected"]

    raw_text, result = harness.transcribe_file(str(audio_path))
    config = get_whisper_config(harness.recognizer)

    # --- Logging for console ---
    logger.info("────────────────────────────────────────")
    logger.info(f"🎤 Audio: {case['audio']}")
    logger.info(f"📜 Raw transcript: '{raw_text}'")
    logger.info(f"📦 Parser result: {result}")
    logger.info(f"✅ Expected length: {expected}, 🧮 Got: {result.length_cm}")

    # --- Assert with logging ---
    passed = result.length_cm == expected
    log_result(case["audio"], raw_text, result, expected, passed, config)

    assert passed, (
        f"\nAudio: {case['audio']}\n"
        f"Raw transcript: '{raw_text}'\n"
        f"Parser result: {result}\n"
        f"Expected: {expected}, Got: {result.length_cm}"
    )


def run_all_tests():
    start_time = time.time()
    # Run all test cases
    for case in TEST_DATA:
        test_numbers(case)
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"⏱️ All tests completed in {duration:.2f} seconds.")
    # Optionally, write duration to a file or CSV
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"Total test duration (seconds):", f"{duration:.2f}"])


# --- Session start logging ---
if __name__ == "__main__":
    logger.info("🟢 Starting new test session")
    logger.info(f"Results CSV: {CSV_FILE}")
    logger.info(f"Fail CSV: {FAIL_CSV_FILE}")
    run_all_tests()
