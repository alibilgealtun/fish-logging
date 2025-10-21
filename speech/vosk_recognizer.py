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
    Offline speech recognizer using Vosk for fish species and measurements.

    This implementation provides completely offline speech recognition using
    pre-trained Vosk models. It's designed to be simple, effective, and privacy-focused
    since all processing happens locally without sending audio to external services.

    Key Features:
    - Completely offline operation (no internet required)
    - Optimized vocabulary constraints for fish species and measurements
    - Fast small model with fallback to larger model
    - Aggressive noise filtering for marine environments
    - Custom word lists and grammar constraints
    - Real-time processing with minimal latency

    Model Strategy:
    - Prefers small model (vosk-model-small-en-us-0.15) for speed
    - Falls back to larger model (vosk-model-en-us-0.22-lgraph) if small unavailable
    - Uses vocabulary constraints to improve accuracy for domain-specific terms
    """

    # PyQt signals for UI communication
    partial_text = pyqtSignal(str)  # Intermediate transcription results
    final_text = pyqtSignal(str, float)  # Final text with confidence score
    error = pyqtSignal(str)  # Error messages
    status_changed = pyqtSignal(str)  # Status updates (listening, processing, etc.)
    specie_detected = pyqtSignal(str)  # Fish species detection events

    # Base directory for finding models and configuration files
    _BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Model paths - prefer small model for speed, fallback to larger for accuracy
    MODEL_PATH = os.path.join(_BASE_DIR, "models", "vosk-model-small-en-us-0.15")
    FALLBACK_MODEL_PATH = os.path.join(_BASE_DIR, "models", "vosk-model-en-us-0.22-lgraph")

    # ===== AUDIO CONFIGURATION =====
    SAMPLE_RATE: int = 16000  # Standard sample rate for Vosk models
    CHANNELS: int = 1  # Mono audio input
    CHUNK_S: float = 0.3  # Smaller chunks for better responsiveness

    # ===== NOISE CONTROLLER SETTINGS =====
    # Aggressive settings for clear speech detection in noisy environments
    VAD_MODE: int = 3  # Most aggressive Voice Activity Detection
    MIN_SPEECH_S: float = 0.5  # Minimum speech segment duration
    MAX_SEGMENT_S: float = 4.0  # Maximum speech segment duration
    PADDING_MS: int = 300  # Audio padding around speech segments

    def __init__(self, noise_profile: Optional[str] = None) -> None:
        """Initialize the Vosk recognizer with specified noise profile.

        Sets up the recognizer with audio processing components and noise control.
        Models are loaded lazily in the run() method to avoid import overhead.

        Args:
            noise_profile: Optional profile name for noise optimization
                         Valid values: "clean", "human", "engine", "mixed"
                         If None, defaults to "mixed"
        """
        super().__init__()

        # Audio processing components (initialized later)
        self._stream: Optional[Any] = None  # sounddevice.InputStream at runtime
        self._chunk_frames: int = int(self.SAMPLE_RATE * self.CHUNK_S)
        self._last_status_msg: Optional[str] = None
        self._last_fish_specie = None

        # Vosk model components (loaded in run method)
        self._model: Optional[Any] = None  # vosk.Model at runtime
        self._recognizer: Optional[Any] = None  # vosk.KaldiRecognizer at runtime

        # Apply noise profile configuration
        self._noise_profile_name = (noise_profile or "mixed").lower()

        # Configure noise processing based on profile
        from speech.noise_profiles import get_noise_profile, make_suppressor_config
        prof = get_noise_profile(self._noise_profile_name)

        # Override default parameters with profile-specific settings
        for attr in ("VAD_MODE", "MIN_SPEECH_S", "MAX_SEGMENT_S", "PADDING_MS"):
            if attr in prof:
                setattr(self, attr, prof[attr])

        # Create suppressor configuration
        suppressor_cfg = make_suppressor_config(prof, self.SAMPLE_RATE)

        # Initialize appropriate noise controller
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
        """Request the recognizer to stop and release all resources.

        Safely shuts down the audio stream and stops the noise controller.
        This method is designed to be called from the main thread.
        """
        self._stop_flag = True
        try:
            # Stop and close audio stream
            if self._stream is not None:
                import sounddevice as sd  # noqa: F401
                self._stream.stop()
                self._stream.close()
                self._stream = None
        except Exception as e:
            logger.debug(f"Error stopping stream: {e}")

        # Stop noise controller processing
        self._noise_controller.stop()

    def begin(self) -> None:
        """Reset internal state and start/restart the recognizer thread.

        Performs complete reinitialization with current noise profile settings.
        Safe to call multiple times for restarting the recognizer.
        """
        # Reset control flags
        self._stop_flag = False
        self._last_status_msg = None

        # Reapply noise profile settings (may have changed)
        from speech.noise_profiles import get_noise_profile, make_suppressor_config
        prof = get_noise_profile(self._noise_profile_name)
        for attr in ("VAD_MODE", "MIN_SPEECH_S", "MAX_SEGMENT_S", "PADDING_MS"):
            if attr in prof:
                setattr(self, attr, prof[attr])

        # Recreate noise controller with updated settings
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

        # Start the recognition thread if not already running
        if not self.isRunning():
            try:
                self.start()
            except Exception as e:
                logger.error(f"Failed to start recognizer: {e}")

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Sounddevice callback function for real-time audio processing.

        Converts input audio to the appropriate format and feeds it to the
        noise controller for voice activity detection and filtering.

        Args:
            indata: Input audio data as float32 numpy array, shape (frames, channels)
            frames: Number of audio frames in this chunk
            time_info: Timing information from sounddevice (unused)
            status: Status flags from sounddevice for error detection
        """
        if status:
            logger.debug(f"Audio status: {status}")

        try:
            # Convert float32 stereo/mono to PCM16 mono for noise controller
            pcm16 = (indata[:, 0] * 32767).astype(np.int16)
            self._noise_controller.push_audio(pcm16)
        except Exception as e:
            logger.debug(f"Audio callback error: {e}")

    def _emit_status_once(self, message: str) -> None:
        """Emit status_changed signal only when the message changes.

        Prevents flooding the UI with identical status messages by tracking
        the last emitted message and only emitting when it changes.

        Args:
            message: Status message to emit ("listening", "processing", "stopped", etc.)
        """
        if message != self._last_status_msg:
            self._last_status_msg = message
            try:
                self.status_changed.emit(message)
            except Exception:
                logger.error(f"Failed to emit status: {message}")

    def _create_simple_word_list(self) -> List[str]:
        """Create an optimized word list for fish species recognition.

        Builds a comprehensive vocabulary from configuration files including:
        - Fish species names (broken into individual words)
        - Numbers and number words
        - Measurement units and synonyms
        - Essential command words

        This vocabulary constraint significantly improves recognition accuracy
        by limiting Vosk to domain-relevant terms.

        Returns:
            List[str]: Sorted list of vocabulary words for recognition
        """
        try:
            config_dir = os.path.join(self._BASE_DIR, "config")
            
            # Load configuration files
            with open(os.path.join(config_dir, "species.json"), "r", encoding="utf-8") as f:
                species_cfg = json.load(f)
            with open(os.path.join(config_dir, "numbers.json"), "r", encoding="utf-8") as f:
                numbers_cfg = json.load(f)
            with open(os.path.join(config_dir, "units.json"), "r", encoding="utf-8") as f:
                units_cfg = json.load(f)

            words = set()

            # Add fish species names - break multi-word species into individual words
            species_list = species_cfg.get("species", [])
            for species in species_list:
                if species and species.strip():
                    # Split species names and add each word separately
                    for word in species.lower().split():
                        if word:
                            words.add(word)

            # Add species name normalizations and variants
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

            # Add numeric digits 0-100 for measurements
            for i in range(101):
                words.add(str(i))

            # Add spoken number words from configuration
            number_words = numbers_cfg.get("number_words", {})
            for word in number_words.keys():
                if word:
                    words.add(word.lower())

            # Add decimal-related tokens
            decimal_tokens = numbers_cfg.get("decimal_tokens", [])
            for token in decimal_tokens:
                if token:
                    words.add(token.lower())

            # Add commonly misheard number variations
            misheard = numbers_cfg.get("misheard_tokens", {})
            for word in misheard.keys():
                if word:
                    words.add(word.lower())

            # Add measurement units and their synonyms
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

            # Add essential words for commands and common speech
            essential = [
                # Units
                "centimeter", "centimeters", "cm", "inch", "inches", "in",
                # Decimal indicators
                "point", "dot", "and",
                # Commands
                "wait", "start", "pause", "stop",
                # Basic numbers
                "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty",
                "seventy", "eighty", "ninety", "hundred"
            ]
            words.update(essential)

            # Clean up and sort the final word list
            word_list = sorted([w for w in words if w and len(w) > 0])
            logger.info(f"Created word list with {len(word_list)} words")
            return word_list

        except Exception as e:
            logger.error(f"Failed to create word list: {e}")
            # Fallback to minimal word list if configuration loading fails
            return [
                "fish", "bass", "trout", "salmon", "cm", "inch", "point", "wait", "start"
            ] + [str(i) for i in range(100)]

    def _setup_vosk_constraints(self) -> None:
        """Configure Vosk with vocabulary constraints for improved accuracy.

        Applies word list constraints to the Vosk recognizer to limit recognition
        to domain-relevant vocabulary. This significantly improves accuracy for
        fish species and measurement recognition by reducing false matches.

        Tries multiple constraint methods in order of preference:
        1. SetWords() - Simple word list constraint (preferred)
        2. SetGrammar() - Grammar-based constraint (fallback)
        3. Unrestricted - If both fail, continue without constraints
        """
        try:
            assert self._recognizer is not None
            
            # Create optimized word list for fish domain
            word_list = self._create_simple_word_list()
            
            # Method 1: Try SetWords (simpler and often more reliable)
            try:
                words_json = json.dumps(word_list)
                self._recognizer.SetWords(words_json)
                logger.info(f"Applied word list constraint with {len(word_list)} words")
                return
            except Exception as e:
                logger.warning(f"SetWords failed: {e}, trying SetGrammar")

            # Method 2: Fallback to SetGrammar with simple phrases
            try:
                # Create simple grammar from word list
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
        """Main recognition thread execution method.

        Implements the complete recognition pipeline:
        1. Loads appropriate Vosk model (small preferred, large fallback)
        2. Sets up vocabulary constraints for fish domain
        3. Initializes audio input stream
        4. Processes audio segments through noise controller
        5. Recognizes speech using Vosk
        6. Applies fish-specific parsing and corrections
        7. Emits structured results via PyQt signals

        The method continues until stop is requested and handles errors gracefully.
        """

        # Determine which model to use (prefer small for speed)
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
            
        # Load Vosk model and create recognizer
        try:
            from vosk import Model, KaldiRecognizer  # type: ignore
            self._model = Model(model_path=model_path)
            self._recognizer = KaldiRecognizer(self._model, self.SAMPLE_RATE)
            
            # Apply vocabulary constraints for improved accuracy
            self._setup_vosk_constraints()
            
        except Exception as e:
            msg = f"Failed to load Vosk model: {e}"
            logger.error(msg)
            self.error.emit(msg)
            return

        # Initialize audio input stream
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

        # Main recognition loop
        with self._stream:
            logger.info("Fish species recognition active")
            self.partial_text.emit("Ready for fish species...")
            self._emit_status_once("listening")

            # Get segment generator from noise controller
            segment_generator = self._noise_controller.collect_segments(padding_ms=self.PADDING_MS)

            while not self.is_stopped():
                try:
                    # Get next filtered audio segment
                    segment = next(segment_generator)
                    if segment is None or segment.size == 0:
                        continue

                    # Check segment duration meets minimum requirements
                    segment_duration = segment.size / self.SAMPLE_RATE
                    if segment_duration < self.MIN_SPEECH_S:
                        continue

                    # Update UI status during processing
                    self._emit_status_once("processing")

                    # Process audio segment with Vosk
                    try:
                        self._recognizer.Reset()  # Reset recognizer state

                        # Ensure audio is in correct format (PCM16)
                        if segment.dtype != np.int16:
                            segment = segment.astype(np.int16)
                        
                        # Feed audio to Vosk recognizer and get result
                        self._recognizer.AcceptWaveform(segment.tobytes())
                        result_json = self._recognizer.FinalResult()
                        result = json.loads(result_json)
                        text_out = (result.get("text") or "").strip()

                    except Exception as e:
                        logger.error(f"Vosk processing error: {e}")
                        self._emit_status_once("listening")
                        continue

                    # Skip empty results
                    if not text_out:
                        self._emit_status_once("listening")
                        continue

                    logger.info(f"Recognized: '{text_out}'")

                    # Handle voice commands for pause/resume
                    text_lower = text_out.lower()
                    if any(cmd in text_lower for cmd in ["wait", "pause"]):
                        self.pause()
                        self.final_text.emit("Waiting until 'start' is said.", 0.85)
                        continue
                    elif "start" in text_lower:
                        self.resume()
                        continue

                    # Skip processing if currently paused
                    if self._paused:
                        continue

                    # Apply fish-specific parsing and corrections
                    try:
                        from parser import FishParser, TextNormalizer

                        parser = FishParser()
                        normalizer = TextNormalizer()

                        # Apply domain-specific ASR corrections
                        corrected = normalizer.apply_fish_asr_corrections(text_out)
                        if corrected != text_out.lower():
                            logger.info(f"After ASR corrections: {corrected}")

                        # Parse for fish species and measurements
                        parsed: ParserResult = parser.parse_text(corrected)

                        # Update species context if detected
                        if parsed.species is not None:
                            self._last_fish_specie = parsed.species
                            self.specie_detected.emit(parsed.species)

                        # Format and emit results
                        if parsed.length_cm is not None:
                            # Format numeric measurement with species context
                            length = float(parsed.length_cm)
                            length_str = f"{length:.1f}".rstrip('0').rstrip('.')
                            formatted = f"{self._last_fish_specie} {length_str} cm"
                            logger.info(f">> {formatted}")
                            self.final_text.emit(formatted, 0.85)
                        else:
                            # Fallback to corrected text if parsing incomplete
                            logger.info(f">> {corrected} (partial parse)")
                            self.final_text.emit(corrected, 0.85)

                    except Exception as e:
                        logger.error(f"Parsing error: {e}")
                        # Final fallback to raw recognition result
                        logger.info(f">> {text_out}")
                        self.final_text.emit(text_out, 0.85)

                    # Return to listening state
                    self._emit_status_once("listening")

                except StopIteration:
                    # Generator exhausted (normal when stopping)
                    break
                except Exception as e:
                    logger.error(f"Loop error: {e}")
                    time.sleep(0.1)  # Brief pause before retrying

        # Cleanup on exit
        self._emit_status_once("stopped")
        logger.info("Fish recognition stopped")