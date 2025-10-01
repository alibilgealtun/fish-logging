from __future__ import annotations

import os
import json
from typing import Optional, List, Any

import numpy as np
from loguru import logger

from .base_recognizer import BaseSpeechRecognizer, TranscriptionSegment


class VoskRecognizer(BaseSpeechRecognizer):
    """
    Vosk-based recognizer using the shared realtime pipeline from BaseSpeechRecognizer.
    Implements backend-specific model loading, optional constraints, and transcription.
    """

    _BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Try small model first - it's faster and often better for limited vocab
    MODEL_PATH = os.path.join(_BASE_DIR, "models", "vosk-model-small-en-us-0.15")
    # Fallback to larger model if small one doesn't exist
    FALLBACK_MODEL_PATH = os.path.join(_BASE_DIR, "models", "vosk-model-en-us-0.22-lgraph")

    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    CHUNK_S: float = 0.3  # Smaller chunks for responsiveness

    # Aggressive noise settings for clear speech detection
    VAD_MODE: int = 3  # Most aggressive
    MIN_SPEECH_S: float = 0.5
    MAX_SEGMENT_S: float = 4.0
    PADDING_MS: int = 300

    def __init__(self) -> None:
        super().__init__()
        self._vosk_model: Optional[Any] = None
        self._recognizer: Optional[Any] = None

    # ---------- Backend hooks ----------
    def _load_backend_model(self) -> None:
        # Choose model path
        model_path = None
        if os.path.isdir(self.MODEL_PATH):
            model_path = self.MODEL_PATH
            logger.info(f"Using Vosk small model: {self.MODEL_PATH}")
        elif os.path.isdir(self.FALLBACK_MODEL_PATH):
            model_path = self.FALLBACK_MODEL_PATH
            logger.info(f"Using Vosk fallback model: {self.FALLBACK_MODEL_PATH}")
        else:
            raise RuntimeError(
                f"No Vosk model found in {self.MODEL_PATH} or {self.FALLBACK_MODEL_PATH}"
            )

        try:
            from vosk import Model, KaldiRecognizer  # type: ignore
            self._vosk_model = Model(model_path=model_path)
            self._recognizer = KaldiRecognizer(self._vosk_model, self.SAMPLE_RATE)
        except Exception as e:
            raise RuntimeError(f"Failed to load Vosk model: {e}")

    def _backend_post_init(self) -> None:
        # Apply simple constraints to improve accuracy
        try:
            self._setup_vosk_constraints()
        except Exception as e:
            logger.warning(f"Vosk constraints setup failed: {e}")

    def _backend_transcribe(self, segment: np.ndarray, wav_path: Optional[str]) -> List[TranscriptionSegment]:
        if self._recognizer is None:
            raise RuntimeError("Vosk recognizer not initialized")

        try:
            self._recognizer.Reset()
            # Ensure int16 mono
            if segment.dtype != np.int16:
                segment = segment.astype(np.int16)
            self._recognizer.AcceptWaveform(segment.tobytes())
            result_json = self._recognizer.FinalResult()
            data = json.loads(result_json or "{}")
            text = (data.get("text") or "").strip()
        except Exception as e:
            raise RuntimeError(f"Vosk processing error: {e}")

        if not text:
            return []
        return [TranscriptionSegment(text=text, confidence=0.9)]

    # ---------- Vosk-specific helpers ----------
    def _create_simple_word_list(self) -> List[str]:
        """Create a simple, effective word list for fish recognition."""
        try:
            # Get data from centralized config
            from config.config import ConfigLoader
            loader = ConfigLoader()
            config, _ = loader.load([])
            
            species_cfg = config.species_data
            numbers_cfg = config.numbers_data
            units_cfg = config.units_data

            words = set()

            # Add fish species - break them into individual words
            species_list = species_cfg.get("species", [])
            for species in species_list:
                if species and species.strip():
                    for word in species.lower().split():
                        if word:
                            words.add(word)

            # Add species normalizations
            normalization = species_cfg.get("normalization", {})
            for original, normalized in normalization.items():
                if original:
                    for word in original.lower().split():
                        if word:
                            words.add(word)
                if normalized:
                    for word in normalized.lower().split():
                        if word:
                            words.add(word)

            # Add numbers 0-100
            for i in range(101):
                words.add(str(i))

            # Add number words
            number_words = numbers_cfg.get("number_words", {})
            for word in number_words.keys():
                if word:
                    words.add(word.lower())

            # Add decimal words
            decimal_tokens = numbers_cfg.get("decimal_tokens", [])
            for token in decimal_tokens:
                if token:
                    words.add(token.lower())

            # Add misheard tokens
            misheard = numbers_cfg.get("misheard_tokens", {})
            for word in misheard.keys():
                if word:
                    words.add(word.lower())

            # Add units
            unit_synonyms = units_cfg.get("synonyms", {})
            for unit, synonyms in unit_synonyms.items():
                if unit:
                    words.add(unit.lower())
                if isinstance(synonyms, list):
                    for syn in synonyms:
                        if syn:
                            words.add(syn.lower())
                elif synonyms:
                    words.add(synonyms.lower())

            # Essential words
            essential = [
                "centimeter", "centimeters", "cm", "inch", "inches", "in",
                "point", "dot", "and", "wait", "start", "pause", "stop",
                "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
                "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred"
            ]
            words.update(essential)

            word_list = sorted([w for w in words if w and len(w) > 0])
            logger.info(f"Created word list with {len(word_list)} words")
            return word_list

        except Exception as e:
            logger.error(f"Failed to create word list: {e}")
            return ["fish", "bass", "trout", "cm", "inch", "point", "wait", "start"] + [str(i) for i in range(100)]

    def _setup_vosk_constraints(self) -> None:
        """Set up Vosk with simple word constraints."""
        try:
            assert self._recognizer is not None
            word_list = self._create_simple_word_list()

            # Try SetWords first
            try:
                words_json = json.dumps(word_list)
                self._recognizer.SetWords(words_json)
                logger.info(f"Applied word list constraint with {len(word_list)} words")
                return
            except Exception as e:
                logger.warning(f"SetWords failed: {e}, trying SetGrammar")

            # Fallback to SetGrammar
            try:
                grammar_phrases = [" ".join(word_list)]
                grammar_json = json.dumps(grammar_phrases)
                self._recognizer.SetGrammar(grammar_json)
                logger.info("Applied simple grammar constraint")
                return
            except Exception as e:
                logger.warning(f"SetGrammar also failed: {e}, using unrestricted model")
        except Exception as e:
            logger.error(f"Failed to setup Vosk constraints: {e}")
