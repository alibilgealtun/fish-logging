"""Google Cloud Speech-to-Text Integration for Fish Logging.

This module provides real-time and batch speech recognition using Google Cloud
Speech-to-Text API. It's optimized for fish species and measurement recognition
with phrase hints and optional numbers-only mode for measurement capture.
"""
import os
from typing import Optional, Dict, Any, List
from .base_recognizer import BaseSpeechRecognizer

import numpy as np
from loguru import logger
import soundfile as sf
from PyQt6.QtCore import pyqtSignal
from noise.controller import NoiseController
import json


class GoogleSpeechRecognizer(BaseSpeechRecognizer):
    """
    Google Cloud Speech-to-Text recognizer for fish species and measurements.

    This implementation provides both real-time and batch recognition capabilities
    using Google's cloud-based speech recognition service. It features phrase hints
    for improved accuracy on fish-related vocabulary and supports a numbers-only
    mode for efficient measurement capture.

    Key Features:
    - Real-time streaming recognition with noise control
    - Batch file transcription capabilities
    - Phrase hints for fish species and measurement terms
    - Numbers-only mode for measurement-focused sessions
    - Automatic credential detection and management
    - Strong boosting for domain-specific vocabulary
    - Integration with local noise controller for preprocessing

    Authentication:
    - Uses GOOGLE_APPLICATION_CREDENTIALS environment variable
    - Auto-detects google.json in project root
    - Supports various Google Cloud credential types
    """

    # PyQt signals for UI communication (explicit redefinition for consistency)
    partial_text = pyqtSignal(str)  # Intermediate transcription results
    final_text = pyqtSignal(str, float)  # Final text with confidence score
    error = pyqtSignal(str)  # Error messages
    status_changed = pyqtSignal(str)  # Status updates (listening, processing, etc.)
    specie_detected = pyqtSignal(str)  # Fish species detection events

    # ===== AUDIO CONFIGURATION =====
    # Aligned with other recognizers for consistency
    SAMPLE_RATE: int = 16000  # Google Cloud Speech standard sample rate
    CHANNELS: int = 1  # Mono audio input
    CHUNK_S: float = 0.5  # Audio chunk duration for processing

    # ===== NOISE CONTROLLER SETTINGS =====
    VAD_MODE: int = 2  # Voice Activity Detection aggressiveness
    MIN_SPEECH_S: float = 0.4  # Minimum speech segment duration
    MAX_SEGMENT_S: float = 3.0  # Maximum speech segment duration
    PADDING_MS: int = 600  # Audio padding around speech segments

    def __init__(
        self,
        language: str = "en-US",
        credentials_path: Optional[str] = None,
        numbers_only: bool = False,
        noise_profile: Optional[str] = None
    ):
        """Initialize Google Cloud Speech-to-Text recognizer.

        Sets up the recognizer with Google Cloud authentication, phrase hints,
        and noise processing configuration. Handles credential auto-detection
        and builds domain-specific vocabulary for improved accuracy.

        Args:
            language: Language code for recognition (e.g., "en-US", "es-ES")
            credentials_path: Path to Google Cloud service account JSON file.
                            If None, uses GOOGLE_APPLICATION_CREDENTIALS env var
                            or auto-detects google.json in project root
            numbers_only: If True, optimizes for number/measurement recognition only
            noise_profile: Noise processing profile ("clean", "human", "engine", "mixed")
        """
        super().__init__(language=language)

        # Configuration
        self.numbers_only = bool(numbers_only)
        self._noise_profile_name = (noise_profile or "mixed").lower()

        # Handle Google Cloud authentication
        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        elif not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            # Auto-detect credentials in project root
            candidate = os.path.join(os.getcwd(), "google.json")
            if os.path.exists(candidate) and self._is_valid_gcp_credentials_file(candidate):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = candidate
                logger.info("Using credentials at project root: google.json")

        # Initialize components
        self._client = None  # Google Speech client (lazy loaded)
        self._stream = None  # Audio input stream
        self._chunk_frames: int = int(self.SAMPLE_RATE * self.CHUNK_S)
        self._last_status_msg: Optional[str] = None

        # Load number prefix audio for improved number recognition
        self._number_sound = self._load_number_prefix()

        # Apply noise profile configuration
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

        # Build phrase hints for improved recognition accuracy
        self._phrase_hints = self._build_phrase_hints(numbers_only=self.numbers_only)

    @staticmethod
    def _is_valid_gcp_credentials_file(path: str) -> bool:
        """Validate that a file contains valid Google Cloud credentials.

        Checks if the JSON file contains a valid credential type that can be
        used with Google Cloud services.

        Args:
            path: Path to the potential credentials file

        Returns:
            bool: True if file contains valid GCP credentials, False otherwise
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check for valid Google Cloud credential types
            cred_type = data.get("type")
            return cred_type in {
                "authorized_user",               # OAuth2 user credentials
                "service_account",               # Service account key
                "external_account",              # Workload identity federation
                "external_account_authorized_user",  # External account user
                "impersonated_service_account",  # Impersonated service account
                "gdch_service_account",         # Google Distributed Cloud
            }
        except Exception:
            return False

    def _get_client(self):
        """Get or create Google Cloud Speech client with lazy initialization.

        Creates the Speech client only when needed to avoid import overhead
        and authentication checks during initialization.

        Returns:
            google.cloud.speech.SpeechClient: Configured Speech client

        Raises:
            Exception: If Google Cloud Speech is not available or authentication fails
        """
        if self._client is not None:
            return self._client

        try:
            from google.cloud import speech as speech
            self._client = speech.SpeechClient()
            return self._client
        except Exception as e:
            msg = (
                "Google Cloud Speech not available: "
                f"{e}. Provide a service account key JSON or run 'gcloud auth application-default login'."
            )
            logger.error(msg)
            self.error.emit(msg)
            raise

    def _build_phrase_hints(self, numbers_only: bool = False) -> List[str]:
        """Build phrase hints for improved recognition accuracy.

        Creates a vocabulary list from configuration files to boost recognition
        of domain-specific terms. In numbers-only mode, focuses on numeric
        vocabulary and units. Otherwise includes fish species names as well.

        Args:
            numbers_only: If True, restrict hints to numbers, units, and control words

        Returns:
            List[str]: List of phrase hints to boost in recognition
        """
        hints: List[str] = []

        # Load number words from configuration
        try:
            with open(os.path.join("config", "numbers.json"), "r", encoding="utf-8") as f:
                numbers_cfg = json.load(f)
            number_words = list(numbers_cfg.get("number_words", {}).keys())
        except Exception:
            # Fallback number vocabulary if config loading fails
            number_words = [
                "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
                "eighty", "ninety", "hundred", "point", "dot", "comma"
            ]

        # Load measurement units from configuration
        try:
            with open(os.path.join("config", "units.json"), "r", encoding="utf-8") as f:
                units_cfg = json.load(f)
            unit_words = list(units_cfg.get("synonyms", {}).keys())
        except Exception:
            # Fallback unit vocabulary
            unit_words = ["cm", "centimeter", "centimeters", "mm", "millimeter", "millimeters"]

        if numbers_only:
            # Numbers-only mode: focus on numeric vocabulary
            hints.extend(number_words)
            hints.extend(unit_words)
            hints.extend(["point", "dot", "comma"])  # Decimal indicators
            hints.extend(["wait", "start"])  # Basic control commands
        else:
            # General mode: include fish species for comprehensive recognition
            try:
                with open(os.path.join("config", "species.json"), "r", encoding="utf-8") as f:
                    species_cfg = json.load(f)
                for item in species_cfg.get("items", []):
                    name = item.get("name")
                    if name:
                        hints.append(str(name))
            except Exception:
                pass

            # Add numeric and unit vocabulary
            hints.extend(number_words)
            hints.extend(unit_words)
            hints.extend(["wait", "start", "cancel"])  # Full command set

        # Deduplicate and limit size to stay within Google Cloud limits
        seen = set()
        uniq: List[str] = []
        for h in hints:
            h2 = h.strip()
            if not h2:
                continue
            k = h2.lower()
            if k in seen:
                continue
            seen.add(k)
            uniq.append(h2)
            if len(uniq) >= 500:  # Google Cloud Speech limit
                break

        return uniq

    # -------- Batch/File API Methods --------

    def transcribe_file(self, file_path: str) -> Dict[str, Any]:
        """Transcribe a single audio file using Google Cloud Speech-to-Text.

        Processes an audio file through Google's batch recognition API with
        domain-specific phrase hints and configuration optimizations.

        Args:
            file_path: Path to the audio file to transcribe

        Returns:
            Dict[str, Any]: Dictionary containing:
                - text: Transcribed text (processed for numbers-only mode if enabled)
                - raw_response: Original Google Cloud Speech response
        """
        from google.cloud import speech as speech

        # Read audio file content
        with open(file_path, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)

        # Build speech contexts with phrase hints and strong boosting
        contexts = []
        if self._phrase_hints:
            try:
                # Use strong boosting (20.0) for domain-specific terms
                contexts = [speech.SpeechContext(phrases=self._phrase_hints, boost=20.0)]
            except Exception:
                # Fallback without boost parameter if not supported
                contexts = [speech.SpeechContext(phrases=self._phrase_hints)]

        # Configure recognition parameters
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.SAMPLE_RATE,
            language_code=self.language,
            enable_automatic_punctuation=not self.numbers_only,  # Disable for numbers-only
            model="latest_short",  # Optimized for short commands/phrases
            speech_contexts=contexts,
        )

        # Perform recognition
        response = self._get_client().recognize(config=config, audio=audio)

        # Extract transcription text
        transcripts = [result.alternatives[0].transcript for result in response.results]
        full_text = " ".join(transcripts).strip()

        # Post-process for numbers-only mode
        if self.numbers_only:
            import re
            # Extract only numeric patterns (digits with optional decimals)
            matches = re.findall(r"\d+(?:[.,]\d+)?", full_text)
            if matches:
                # Join multiple numbers if found
                full_text = " ".join(matches)
            # If no numbers found, keep original text as fallback

        return {
            "text": full_text,
            "raw_response": response,
        }

    def transcribe_batch(self, files: List[str]) -> List[Dict[str, Any]]:
        """Transcribe multiple audio files in batch.

        Processes a list of audio files and returns results for each file.

        Args:
            files: List of file paths to transcribe

        Returns:
            List[Dict[str, Any]]: List of dictionaries, each containing:
                - file: Original file path
                - text: Transcribed text
                - raw_response: Google Cloud Speech response
        """
        results = []
        for file_path in files:
            results.append({
                "file": file_path,
                **self.transcribe_file(file_path)
            })
        return results

    # -------- Real-time Recognition Methods --------

    def set_last_species(self, species: str) -> None:
        """Set the last detected fish species for context in number-only transcriptions.

        Args:
            species: Name of the fish species to use as context, or None to clear
        """
        try:
            self._last_fish_specie = str(species) if species else None
        except Exception:
            self._last_fish_specie = None

    def _load_number_prefix(self) -> np.ndarray:
        """Load number prefix audio for improved number recognition.

        Loads a short audio clip to prepend to speech segments. This helps
        Google's recognition better identify numeric content by providing
        acoustic context that signals number recognition mode.

        Returns:
            np.ndarray: PCM16 mono audio samples for the number prefix
        """
        candidates = [
            os.path.join(os.getcwd(), "assets/audio/number.wav"),
        ]

        for path in candidates:
            try:
                if os.path.exists(path):
                    data, sr = sf.read(path, dtype='int16')

                    # Resample if necessary using simple nearest-neighbor
                    if sr != self.SAMPLE_RATE:
                        ratio = self.SAMPLE_RATE / sr
                        idx = (np.arange(int(len(data) * ratio)) / ratio).astype(int)
                        data = data[idx]

                    # Convert to mono if stereo
                    if data.ndim > 1:
                        data = data[:, 0]

                    return data.astype(np.int16)
            except Exception as e:
                logger.debug(f"Failed to load number prefix {path}: {e}")

        # Fallback to short silence
        return (np.zeros(int(self.SAMPLE_RATE * 0.05))).astype(np.int16)

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Sounddevice callback for real-time audio capture.

        Args:
            indata: Input audio data as float32 numpy array
            frames: Number of audio frames in this chunk
            time_info: Timing information from sounddevice (unused)
            status: Status flags from sounddevice for error detection
        """
        if status:
            logger.debug(f"Audio status: {status}")

        try:
            # Convert float32 to PCM16 and feed to noise controller
            pcm16 = (indata[:, 0] * 32767).astype(np.int16)
            self._noise_controller.push_audio(pcm16)
        except Exception as e:
            logger.debug(f"Audio callback error: {e}")

    def _emit_status_once(self, message: str) -> None:
        """Emit status_changed signal only when the message changes.

        Args:
            message: Status message to emit
        """
        if message != self._last_status_msg:
            self._last_status_msg = message
            try:
                self.status_changed.emit(message)
            except Exception:
                logger.error(f"Failed to emit status_changed message: {message}")

    def begin(self) -> None:
        """Reset internal state and start/restart the recognizer thread."""
        self._stop_flag = False
        self._last_status_msg = None

        # Rebuild noise controller with current profile settings
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
                logger.error(f"Failed to (re)start Google recognizer: {e}")

    def stop(self) -> None:
        """Request the recognizer to stop and release all resources."""
        self._stop_flag = True
        try:
            if self._stream is not None:
                import sounddevice as sd  # noqa: F401
                self._stream.stop()
                self._stream.close()
                self._stream = None
        except Exception as e:
            logger.debug(f"Error stopping input stream: {e}")

        self._noise_controller.stop()

    def run(self) -> None:
        """Main real-time recognition loop.

        Implements the complete real-time recognition pipeline:
        1. Initializes audio input stream
        2. Configures Google Cloud Speech recognition
        3. Processes audio segments from noise controller
        4. Sends segments to Google Cloud Speech API
        5. Applies fish-specific parsing and corrections
        6. Emits structured results via PyQt signals
        """
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
            msg = f"Failed to open microphone stream: {e}"
            logger.error(msg)
            self.error.emit(msg)
            return

        # Configure Google Cloud Speech recognition
        from google.cloud import speech as speech
        contexts = []
        if self._phrase_hints:
            try:
                contexts = [speech.SpeechContext(phrases=self._phrase_hints, boost=20.0)]
            except Exception:
                contexts = [speech.SpeechContext(phrases=self._phrase_hints)]

        base_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.SAMPLE_RATE,
            language_code=self.language,
            enable_automatic_punctuation=not self.numbers_only,
            model="latest_short",
            speech_contexts=contexts,
        )

        # Main recognition loop
        with self._stream:
            logger.info(f"Recording with noise control (Google STT)... Press Stop to end. [profile={self._noise_profile_name}]")
            try:
                self.partial_text.emit("Listening…")
            except Exception:
                pass
            self._emit_status_once("listening")

            segment_generator = self._noise_controller.collect_segments(
                padding_ms=self.PADDING_MS
            )

            while not self.is_stopped():
                try:
                    # Get next audio segment from noise controller
                    segment = next(segment_generator)
                    if segment is None or segment.size == 0:
                        continue

                    # Check minimum duration requirement
                    if segment.size / self.SAMPLE_RATE < self.MIN_SPEECH_S:
                        continue

                    self._emit_status_once("processing")

                    # Send audio segment to Google Cloud Speech
                    audio = speech.RecognitionAudio(content=segment.tobytes())

                    try:
                        response = self._get_client().recognize(config=base_config, audio=audio)
                    except Exception as e:
                        msg = f"Google STT error: {e}"
                        logger.error(msg)
                        self.error.emit(msg)
                        self._emit_status_once("listening")
                        continue

                    # Extract recognition results
                    if not response.results:
                        self._emit_status_once("listening")
                        continue

                    # Get the most likely transcription
                    first_alt = response.results[0].alternatives[0]
                    text_out = first_alt.transcript.strip()
                    confidence = float(getattr(first_alt, "confidence", 0.85) or 0.85)

                    if not text_out:
                        self._emit_status_once("listening")
                        continue

                    logger.info(f"Raw transcription: {text_out}")

                    # Handle voice commands for pause/resume
                    text_lower = text_out.lower()
                    if "wait" in text_lower:
                        self.pause()
                        self.final_text.emit("Waiting until 'start' is said.", confidence)
                        self._emit_status_once("paused")
                        continue
                    elif "start" in text_lower:
                        self.resume()
                        self._emit_status_once("listening")
                        continue

                    # Skip processing if currently paused
                    if self._paused:
                        logger.debug("Paused: ignoring transcription")
                        self._emit_status_once("paused")
                        continue

                    # Apply fish-specific parsing and formatting
                    try:
                        from parser import FishParser, TextNormalizer
                        fish_parser = FishParser()
                        text_normalizer = TextNormalizer()
                        corrected_text = text_normalizer.apply_fish_asr_corrections(text_out)
                        result = fish_parser.parse_text(corrected_text)

                        if self.numbers_only:
                            # Numbers-only mode: emit only numeric measurements
                            if result.length_cm is None:
                                self._emit_status_once("listening")
                                continue
                            raw_val = float(result.length_cm)
                            num_str = (f"{raw_val:.1f}").rstrip("0").rstrip(".")
                            self.final_text.emit(num_str, confidence)
                        else:
                            # General mode: handle species and measurements
                            if result.species is not None:
                                self._last_fish_specie = result.species
                                try:
                                    self.specie_detected.emit(result.species)
                                except Exception:
                                    pass

                            if result.length_cm is not None:
                                raw_val = float(result.length_cm)
                                num_str = (f"{raw_val:.1f}").rstrip("0").rstrip(".")
                                formatted = f"{self._last_fish_specie} {num_str} cm"
                                self.final_text.emit(formatted, confidence)
                            else:
                                self.final_text.emit(corrected_text, confidence)
                    except Exception:
                        # Fallback processing for parsing errors
                        if self.numbers_only:
                            # Extract any digits as last resort
                            import re
                            m = re.search(r"\d+(?:[.,]\d+)?", text_out)
                            if m:
                                val = m.group(0).replace(",", ".")
                                self.final_text.emit(val, confidence)
                        else:
                            self.final_text.emit(text_out, confidence)

                    self._emit_status_once("listening")

                except StopIteration:
                    # Generator exhausted (normal when stopping)
                    break
                except Exception as e:
                    logger.error(f"Main loop error: {e}")
                    self.error.emit(f"Processing error: {e}")
                    continue

        # Cleanup on exit
        self._emit_status_once("stopped")
        logger.info("Google speech recognizer stopped")

    def get_config(self) -> Dict[str, Any]:
        """Return comprehensive configuration for logging and debugging.

        Returns:
            Dict[str, Any]: Complete configuration including audio settings,
                           language, mode, and noise processing parameters
        """
        return {
            "SAMPLE_RATE": self.SAMPLE_RATE,
            "CHANNELS": self.CHANNELS,
            "CHUNK_S": self.CHUNK_S,
            "VAD_MODE": self.VAD_MODE,
            "MIN_SPEECH_S": self.MIN_SPEECH_S,
            "MAX_SEGMENT_S": self.MAX_SEGMENT_S,
            "PADDING_MS": self.PADDING_MS,
            "LANGUAGE": self.language,
            "NUMBERS_ONLY": self.numbers_only,
            "NOISE_PROFILE": self._noise_profile_name,
        }
