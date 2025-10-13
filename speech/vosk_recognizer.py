from __future__ import annotations

import os
import json
import time
from typing import Optional, Set, List, Any

import numpy as np
from loguru import logger
from PyQt6.QtCore import pyqtSignal

from .base_recognizer import BaseSpeechRecognizer
from noise.controller import NoiseController
from parser import ParserResult


class VoskRecognizer(BaseSpeechRecognizer):
    """
    Simple, effective Vosk recognizer for fish species and measurements.
    No bullshit, just works.
    """

    partial_text = pyqtSignal(str)
    final_text = pyqtSignal(str, float)
    error = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    specie_detected = pyqtSignal(str)

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

    def __init__(self, noise_profile: Optional[str] = None) -> None:
        super().__init__()
        self._stream: Optional[Any] = None  # sounddevice.InputStream at runtime
        self._chunk_frames: int = int(self.SAMPLE_RATE * self.CHUNK_S)
        self._last_status_msg: Optional[str] = None
        self._last_fish_specie = None

        self._model: Optional[Any] = None  # vosk.Model at runtime
        self._recognizer: Optional[Any] = None  # vosk.KaldiRecognizer at runtime

        self._noise_profile_name = (noise_profile or "mixed").lower()

        # Apply noise profile overrides
        from speech.noise_profiles import get_noise_profile, make_suppressor_config
        prof = get_noise_profile(self._noise_profile_name)
        for attr in ("VAD_MODE", "MIN_SPEECH_S", "MAX_SEGMENT_S", "PADDING_MS"):
            if attr in prof:
                setattr(self, attr, prof[attr])
        suppressor_cfg = make_suppressor_config(prof, self.SAMPLE_RATE)
        if self._noise_profile_name == "clean":
            from noise.simple_controller import SimpleNoiseController
            self._noise_controller = SimpleNoiseController(
                sample_rate=self.SAMPLE_RATE,
                vad_mode=self.VAD_MODE,
                min_speech_s=self.MIN_SPEECH_S,
                max_segment_s=self.MAX_SEGMENT_S,
            )
        else:
            self._noise_controller = NoiseController(
                sample_rate=self.SAMPLE_RATE,
                vad_mode=self.VAD_MODE,
                min_speech_s=self.MIN_SPEECH_S,
                max_segment_s=self.MAX_SEGMENT_S,
                suppressor_config=suppressor_cfg,
            )

    def stop(self) -> None:
        self._stop_flag = True
        try:
            if self._stream is not None:
                # Local import to ensure module loads only if used
                import sounddevice as sd  # noqa: F401
                self._stream.stop()
                self._stream.close()
                self._stream = None
        except Exception as e:
            logger.debug(f"Error stopping stream: {e}")
        self._noise_controller.stop()

    def begin(self) -> None:
        self._stop_flag = False
        self._last_status_msg = None
        from speech.noise_profiles import get_noise_profile, make_suppressor_config
        prof = get_noise_profile(self._noise_profile_name)
        for attr in ("VAD_MODE", "MIN_SPEECH_S", "MAX_SEGMENT_S", "PADDING_MS"):
            if attr in prof:
                setattr(self, attr, prof[attr])
        suppressor_cfg = make_suppressor_config(prof, self.SAMPLE_RATE)
        if self._noise_profile_name == "clean":
            from noise.simple_controller import SimpleNoiseController
            self._noise_controller = SimpleNoiseController(
                sample_rate=self.SAMPLE_RATE,
                vad_mode=self.VAD_MODE,
                min_speech_s=self.MIN_SPEECH_S,
                max_segment_s=self.MAX_SEGMENT_S,
            )
        else:
            self._noise_controller = NoiseController(
                sample_rate=self.SAMPLE_RATE,
                vad_mode=self.VAD_MODE,
                min_speech_s=self.MIN_SPEECH_S,
                max_segment_s=self.MAX_SEGMENT_S,
                suppressor_config=suppressor_cfg,
            )
        if not self.isRunning():
            try:
                self.start()
            except Exception as e:
                logger.error(f"Failed to start recognizer: {e}")

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            logger.debug(f"Audio status: {status}")
        try:
            pcm16 = (indata[:, 0] * 32767).astype(np.int16)
            self._noise_controller.push_audio(pcm16)
        except Exception as e:
            logger.debug(f"Audio callback error: {e}")

    def _emit_status_once(self, message: str) -> None:
        if message != self._last_status_msg:
            self._last_status_msg = message
            try:
                self.status_changed.emit(message)
            except Exception:
                logger.error(f"Failed to emit status: {message}")

    def _create_simple_word_list(self) -> List[str]:
        """Create a simple, effective word list for fish recognition."""
        try:
            config_dir = os.path.join(self._BASE_DIR, "config")
            
            # Load configs
            with open(os.path.join(config_dir, "species.json"), "r", encoding="utf-8") as f:
                species_cfg = json.load(f)
            with open(os.path.join(config_dir, "numbers.json"), "r", encoding="utf-8") as f:
                numbers_cfg = json.load(f)
            with open(os.path.join(config_dir, "units.json"), "r", encoding="utf-8") as f:
                units_cfg = json.load(f)

            words = set()

            # Add fish species - break them into individual words
            species_list = species_cfg.get("species", [])
            for species in species_list:
                if species and species.strip():
                    # Add each word in the species name
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

            # Add essential words
            essential = [
                "centimeter", "centimeters", "cm", "inch", "inches", "in",
                "point", "dot", "and", "wait", "start", "pause", "stop",
                "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
                "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred"
            ]
            words.update(essential)

            # Clean up and return
            word_list = sorted([w for w in words if w and len(w) > 0])
            logger.info(f"Created word list with {len(word_list)} words")
            return word_list

        except Exception as e:
            logger.error(f"Failed to create word list: {e}")
            # Fallback word list
            return ["fish", "bass", "trout", "cm", "inch", "point", "wait", "start"] + [str(i) for i in range(100)]

    def _setup_vosk_constraints(self) -> None:
        """Set up Vosk with simple word constraints."""
        try:
            assert self._recognizer is not None
            
            # Create simple word list
            word_list = self._create_simple_word_list()
            
            # Try SetWords first (simpler and often more reliable)
            try:
                words_json = json.dumps(word_list)
                self._recognizer.SetWords(words_json)
                logger.info(f"Applied word list constraint with {len(word_list)} words")
                return
            except Exception as e:
                logger.warning(f"SetWords failed: {e}, trying SetGrammar")

            # Fallback to SetGrammar with simple phrases
            try:
                # Create simple grammar with just the word list
                grammar_phrases = [" ".join(word_list)]
                grammar_json = json.dumps(grammar_phrases)
                self._recognizer.SetGrammar(grammar_json)
                logger.info("Applied simple grammar constraint")
                return
            except Exception as e:
                logger.warning(f"SetGrammar also failed: {e}, using unrestricted model")

        except Exception as e:
            logger.error(f"Failed to setup Vosk constraints: {e}")

    def run(self) -> None:
        """Main recognition loop - keep it simple and effective."""
        
        # Try to load model (small first, then fallback)
        model_path = None
        if os.path.isdir(self.MODEL_PATH):
            model_path = self.MODEL_PATH
            logger.info(f"Using small model: {self.MODEL_PATH}")
        elif os.path.isdir(self.FALLBACK_MODEL_PATH):
            model_path = self.FALLBACK_MODEL_PATH
            logger.info(f"Using fallback model: {self.FALLBACK_MODEL_PATH}")
        else:
            msg = f"No Vosk model found in {self.MODEL_PATH} or {self.FALLBACK_MODEL_PATH}"
            logger.error(msg)
            self.error.emit(msg)
            return
            
        try:
            # Local import of vosk to avoid heavy import during test collection
            from vosk import Model, KaldiRecognizer  # type: ignore
            self._model = Model(model_path=model_path)
            self._recognizer = KaldiRecognizer(self._model, self.SAMPLE_RATE)
            
            # Setup word constraints
            self._setup_vosk_constraints()
            
        except Exception as e:
            msg = f"Failed to load Vosk model: {e}"
            logger.error(msg)
            self.error.emit(msg)
            return

        # Open microphone
        try:
            import sounddevice as sd  # type: ignore
            self._stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=self.CHANNELS,
                blocksize=self._chunk_frames,
                dtype="float32",
                callback=self._audio_callback,
            )
        except Exception as e:
            msg = f"Failed to open microphone: {e}"
            logger.error(msg)
            self.error.emit(msg)
            return

        # Main loop
        with self._stream:
            logger.info("Fish species recognition active")
            self.partial_text.emit("Ready for fish species...")
            self._emit_status_once("listening")

            segment_generator = self._noise_controller.collect_segments(padding_ms=self.PADDING_MS)

            while not self.is_stopped():
                try:
                    segment = next(segment_generator)
                    if segment is None or segment.size == 0:
                        continue

                    segment_duration = segment.size / self.SAMPLE_RATE
                    if segment_duration < self.MIN_SPEECH_S:
                        continue

                    self._emit_status_once("processing")

                    # Simple Vosk processing
                    try:
                        self._recognizer.Reset()
                        
                        # Convert to int16
                        if segment.dtype != np.int16:
                            segment = segment.astype(np.int16)
                        
                        # Feed audio to recognizer
                        self._recognizer.AcceptWaveform(segment.tobytes())
                        result_json = self._recognizer.FinalResult()
                        result = json.loads(result_json)
                        text_out = (result.get("text") or "").strip()

                    except Exception as e:
                        logger.error(f"Vosk processing error: {e}")
                        self._emit_status_once("listening")
                        continue

                    if not text_out:
                        self._emit_status_once("listening")
                        continue

                    logger.info(f"Recognized: '{text_out}'")

                    # Handle commands
                    text_lower = text_out.lower()
                    if any(cmd in text_lower for cmd in ["wait", "pause"]):
                        self.pause()
                        self.final_text.emit("Waiting until 'start' is said.", 0.85)
                        continue
                    elif "start" in text_lower:
                        self.resume()
                        continue

                    if self._paused:
                        continue

                    # Parse fish data (mirror faster-whisper behavior)
                    try:
                        from parser import FishParser, TextNormalizer

                        parser = FishParser()
                        normalizer = TextNormalizer()

                        # Apply corrections and parse
                        corrected = normalizer.apply_fish_asr_corrections(text_out)
                        if corrected != text_out.lower():
                            logger.info(f"After ASR corrections: {corrected}")
                        parsed: ParserResult = parser.parse_text(corrected)

                        # Update species
                        if parsed.species is not None:
                            self._last_fish_specie = parsed.species
                            self.specie_detected.emit(parsed.species)

                        if parsed.length_cm is not None:
                            length = float(parsed.length_cm)
                            length_str = f"{length:.1f}".rstrip('0').rstrip('.')
                            formatted = f"{self._last_fish_specie} {length_str} cm"
                            logger.info(f">> {formatted}")
                            self.final_text.emit(formatted, 0.85)
                        else:
                            logger.info(f">> {corrected} (partial parse)")
                            self.final_text.emit(corrected, 0.85)

                    except Exception as e:
                        logger.error(f"Parsing error: {e}")
                        logger.info(f">> {text_out}")
                        self.final_text.emit(text_out, 0.85)

                    self._emit_status_once("listening")

                except StopIteration:
                    break
                except Exception as e:
                    logger.error(f"Loop error: {e}")
                    time.sleep(0.1)

        self._emit_status_once("stopped")
        logger.info("Fish recognition stopped")