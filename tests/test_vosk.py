from speech.vosk_recognizer import VoskRecognizer
from parser import FishParser, ParserResult
import soundfile as sf
import numpy as np
import json
import pytest
from pathlib import Path
import os

class VoskTestHarness:
    def __init__(self):
        self.recognizer = VoskRecognizer()
        self.parser = FishParser()
        self.recognizer._last_fish_specie = "salmon"

        # Determine model path (small first, then fallback)
        small = Path(self.recognizer.MODEL_PATH)
        large = Path(self.recognizer.FALLBACK_MODEL_PATH)
        if small.is_dir():
            self._model_path = small
        elif large.is_dir():
            self._model_path = large
        else:
            pytest.skip(
                f"Skipping Vosk tests: no model directory found at '{small}' or '{large}'."
            )

        # Load vosk model lazily (import only if present)
        try:
            from vosk import Model  # type: ignore
            self.recognizer._model = Model(model_path=str(self._model_path))
        except Exception as e:
            pytest.skip(f"Skipping Vosk tests: failed to load vosk Model: {e}")

    def _new_recognizer(self):
        """Create a fresh KaldiRecognizer with constraints for each file."""
        try:
            from vosk import KaldiRecognizer  # type: ignore
            self.recognizer._recognizer = KaldiRecognizer(self.recognizer._model, self.recognizer.SAMPLE_RATE)  # type: ignore[arg-type]
            # Apply vocabulary constraints (private helper is fine for tests)
            try:
                self.recognizer._setup_vosk_constraints()
            except Exception:
                pass
        except Exception as e:
            pytest.skip(f"Skipping Vosk tests: failed to create KaldiRecognizer: {e}")

    def transcribe_file(self, wav_path: str) -> tuple[str, ParserResult]:
        # (Re)create recognizer per file for clean state
        self._new_recognizer()
        assert self.recognizer._recognizer is not None

        audio, sr = sf.read(wav_path, dtype='int16')
        if audio.ndim > 1:
            audio = audio[:, 0]

        # Resample to target if needed
        if sr != self.recognizer.SAMPLE_RATE:
            from scipy.signal import resample_poly
            gcd = np.gcd(sr, self.recognizer.SAMPLE_RATE)
            up = self.recognizer.SAMPLE_RATE // gcd
            down = sr // gcd
            audio = resample_poly(audio, up, down).astype(np.int16)
            sr = self.recognizer.SAMPLE_RATE

        # Feed entire audio (small test files) as bytes
        pcm_bytes = audio.tobytes()
        try:
            self.recognizer._recognizer.AcceptWaveform(pcm_bytes)
            result_json = self.recognizer._recognizer.FinalResult()
        except Exception as e:
            pytest.fail(f"Vosk AcceptWaveform failed for {wav_path}: {e}")

        try:
            result_dict = json.loads(result_json)
        except Exception:
            result_dict = {"text": ""}

        raw_text = (result_dict.get("text") or "").strip()
        cleaned_text = raw_text.rstrip('.,?!').strip()
        parsed = self.parser.parse_text(cleaned_text)
        return raw_text, parsed

# --- Pytest tests ---
DATASET_PATH = Path(__file__).parent / "data/numbers.json"
with open(DATASET_PATH) as f:
    TEST_DATA = json.load(f)

@pytest.mark.parametrize("case", TEST_DATA)
def test_numbers(case):
    harness = VoskTestHarness()
    audio_path = Path(__file__).parent / "audio" / case["audio"]
    expected = case["expected"]

    raw_text, result = harness.transcribe_file(str(audio_path))
    assert result.length_cm == expected, (
        f"\nAudio: {case['audio']}\n"
        f"Raw transcript: '{raw_text}'\n"
        f"Parser result: {result}\n"
        f"Expected: {expected}, Got: {result.length_cm}"
    )
