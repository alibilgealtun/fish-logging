import pytest
import tempfile
import os
from gtts import gTTS
from pathlib import Path
from datetime import datetime
from loguru import logger

import soundfile as sf
import numpy as np

from speech.faster_whisper_recognizer import WhisperRecognizer
from parser import FishParser, ParserResult
from tests.test_numbers_integration import WhisperTestHarness, log_result, get_whisper_config, CSV_FILE, CSV_FIELDS

# ----------------------
# Synthetic test dataset
# ----------------------

SYNTH_CASES = [
    {"text": "fifty one", "expected": 51},
    {"text": "ninety five", "expected": 95},
    {"text": "one hundred twenty three centimeters", "expected": 123},
    {"text": "seventy three", "expected": 73},
    {"text": "forty two centimeters", "expected": 42},
    {"text": "three hundred and five", "expected": 305},
]

def synthesize_audio(text: str) -> str:
    """
    Generate temporary WAV from TTS for a given text.
    Returns the path to the wav file.
    """
    tts = gTTS(text=text, lang="en")
    mp3_fd, mp3_path = tempfile.mkstemp(suffix=".mp3")
    os.close(mp3_fd)
    tts.save(mp3_path)

    # Convert MP3 to WAV (gTTS outputs MP3)
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(wav_fd)

    data, sr = sf.read(mp3_path, dtype="int16")
    sf.write(wav_path, data, sr, subtype="PCM_16")

    os.remove(mp3_path)
    return wav_path


@pytest.mark.parametrize("case", SYNTH_CASES)
def test_synthetic_numbers(case):
    """
    Generate audio with TTS, run through WhisperTestHarness,
    and check if the parser extracts the expected number.
    """
    harness = WhisperTestHarness(use_concat=False)

    # 1. Generate audio
    wav_path = synthesize_audio(case["text"])

    # 2. Run recognizer
    raw_text, result = harness.transcribe_file(wav_path)
    config = get_whisper_config(harness.recognizer)

    # 3. Logging
    logger.info("──────────────────────────────")
    logger.info(f"📝 Synthetic phrase: '{case['text']}'")
    logger.info(f"📜 Raw transcript: '{raw_text}'")
    logger.info(f"📦 Parser result: {result}")
    logger.info(f"✅ Expected length: {case['expected']}, 🧮 Got: {result.length_cm}")

    # 4. Log to CSV (like your real tests)
    passed = result.length_cm == case["expected"]
    log_result(case["text"], raw_text, result, case["expected"], passed, config)

    # 5. Assert
    assert passed, (
        f"\nSynthetic text: {case['text']}\n"
        f"Raw transcript: '{raw_text}'\n"
        f"Parser result: {result}\n"
        f"Expected: {case['expected']}, Got: {result.length_cm}"
    )

    # Cleanup temp file
    try:
        os.remove(wav_path)
    except:
        pass
