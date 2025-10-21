from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
from loguru import logger
from PyQt6.QtCore import pyqtSignal
from .base_recognizer import BaseSpeechRecognizer
from noise.controller import NoiseController
from parser import ParserResult
import soundfile as sf
from services import get_audio_saver


@dataclass
class TranscriptionSegment:
    """Represents a final transcription segment with a text and confidence score.

    This data class encapsulates the result of a speech recognition operation,
    containing both the recognized text and an associated confidence metric.

    Attributes
    ----------
    text: str
        The recognized text content from the audio segment.
    confidence: float
        Confidence score in range [0, 1], where 1.0 indicates highest confidence
        and 0.0 indicates lowest confidence in the transcription accuracy.
    """

    text: str
    confidence: float


class WhisperRecognizer(BaseSpeechRecognizer):
    """
    Real-time CPU-only speech recognizer using NoiseController + faster-whisper.

    This implementation is specifically optimized for high-noise marine environments
    with engine sounds and background speech. It combines advanced noise suppression
    with OpenAI's Whisper model for accurate fish species and measurement recognition.

    The recognizer operates in real-time by:
    1. Capturing audio through sounddevice
    2. Processing audio through noise controller for VAD and suppression
    3. Transcribing speech segments using faster-whisper
    4. Parsing transcribed text for fish species and measurements
    5. Emitting structured results via PyQt signals

    Key Features:
    - Real-time audio processing with configurable chunk sizes
    - Advanced noise suppression for marine environments
    - Fish-specific vocabulary and parsing
    - Pause/resume voice commands ("wait"/"start")
    - Session logging and audio segment saving
    - Configurable noise profiles for different environments
    """

    # PyQt signals for communicating with the UI
    partial_text = pyqtSignal(str)  # Intermediate transcription results
    final_text = pyqtSignal(str, float)  # Final text with confidence score
    error = pyqtSignal(str)  # Error messages
    status_changed = pyqtSignal(str)  # Status updates (listening, processing, etc.)
    specie_detected = pyqtSignal(str)  # Fish species detection events

    # ===== AUDIO CONFIGURATION =====
    SAMPLE_RATE: int = 16000  # Audio sampling rate in Hz (Whisper's preferred rate)
    CHANNELS: int = 1  # Mono audio input
    CHUNK_S: float = 0.5  # Audio chunk duration in seconds for processing

    # ===== NOISE CONTROLLER SETTINGS =====
    # Optimized for marine environments with engine noise and background speech
    VAD_MODE: int = 2  # Voice Activity Detection mode (0=least aggressive, 3=most aggressive)
    MIN_SPEECH_S: float = 0.4  # Minimum speech segment duration in seconds
    MAX_SEGMENT_S: float = 3.0  # Maximum speech segment duration in seconds
    PADDING_MS: int = 600  # Audio padding around speech segments in milliseconds

    # ===== WHISPER MODEL CONFIGURATION =====
    MODEL_NAME: str = "base.en"  # Whisper model size (tiny, base, small, medium, large)
    DEVICE: str = "cpu"  # Processing device (cpu/cuda)
    COMPUTE_TYPE: str = "int8"  # Quantization type for faster inference

    # ===== WHISPER DECODING PARAMETERS =====
    # These parameters control the quality vs speed trade-off of transcription
    BEAM_SIZE: int = 3  # Beam search width (higher = better quality, slower)
    BEST_OF: int = 5  # Number of candidates to consider (higher = better quality, slower)
    TEMPERATURE: float = 0.0  # Sampling temperature (0.0 = deterministic)
    PATIENCE: float = 1.0  # Beam search patience factor
    LENGTH_PENALTY: float = 1.0  # Length penalty for beam search
    REPETITION_PENALTY: float = 1.0  # Penalty for repeated tokens
    WITHOUT_TIMESTAMPS: bool = True  # Skip timestamp generation for speed
    CONDITION_ON_PREVIOUS_TEXT: bool = True  # Use previous text as context
    VAD_FILTER: bool = False  # Use Whisper's internal VAD (we use external)
    VAD_PARAMETERS: Optional[dict] = None  # Parameters for Whisper's VAD
    WORD_TIMESTAMPS: bool = False  # Generate word-level timestamps

    def set_last_species(self, species: str) -> None:
        """Set the last detected fish species for context in number-only transcriptions.

        This method is used by the UI to provide context when users speak only
        measurements without species names. The recognizer will prepend this
        species to standalone numeric measurements.

        Args:
            species: Name of the fish species to use as context, or None to clear
        """
        try:
            self._last_fish_specie = str(species) if species else None
        except Exception:
            self._last_fish_specie = None

    def __init__(self, language: str = "en-US", noise_profile: Optional[str] = None) -> None:
        """Initialize the WhisperRecognizer with specified configuration.

        Sets up the recognizer with audio processing components, noise control,
        and model parameters. Loads the number prefix audio and applies noise
        profile optimizations.

        Args:
            language: Language code for recognition (default: "en-US")
            noise_profile: Optional profile name for noise optimization.
                         Valid values: "clean", "human", "engine", "mixed"
                         If None, defaults to "mixed"

        Raises:
            Exception: If critical components fail to initialize
        """
        super().__init__(language=language)
        # Thread control flags
        self._stop_flag: bool = False
        self._paused: bool = False

        # Audio processing components (initialized later)
        self._stream: Optional[Any] = None  # sounddevice.InputStream at runtime
        self._model: Optional[Any] = None  # faster_whisper.WhisperModel at runtime

        # Audio configuration
        self._chunk_frames: int = int(self.SAMPLE_RATE * self.CHUNK_S)
        self._last_status_msg: Optional[str] = None
        self._noise_profile_name = (noise_profile or "mixed").lower()

        # Load number marker audio (prepended to segments for better number recognition)
        try:
            self._number_sound = self._load_number_prefix()
        except Exception as e:
            logger.debug(f"Failed to load number prefix: {e}")
            # Fallback to short silence if number prefix audio is unavailable
            self._number_sound = (np.zeros(int(self.SAMPLE_RATE * 0.05))).astype(np.int16)

        # Apply noise profile optimizations
        from speech.noise_profiles import get_noise_profile, make_suppressor_config
        prof = get_noise_profile(self._noise_profile_name)

        # Override default parameters with profile-specific settings
        for attr in ("VAD_MODE", "MIN_SPEECH_S", "MAX_SEGMENT_S", "PADDING_MS"):
            if attr in prof:
                setattr(self, attr, prof[attr])

        # Create suppressor configuration for the noise controller
        suppressor_cfg = make_suppressor_config(prof, self.SAMPLE_RATE)

        # Initialize appropriate noise controller based on profile
        if self._noise_profile_name == "clean":
            # Use simplified controller for clean environments
            from noise.simple_controller import SimpleNoiseController
            self._noise_controller = SimpleNoiseController(
                sample_rate=self.SAMPLE_RATE,
                vad_mode=self.VAD_MODE,
                min_speech_s=self.MIN_SPEECH_S,
                max_segment_s=self.MAX_SEGMENT_S,
            )
        else:
            # Use full noise controller with suppression for noisy environments
            self._noise_controller = NoiseController(
                sample_rate=self.SAMPLE_RATE,
                vad_mode=self.VAD_MODE,
                min_speech_s=self.MIN_SPEECH_S,
                max_segment_s=self.MAX_SEGMENT_S,
                suppressor_config=suppressor_cfg,
            )

        # Initialize session logging for debugging and analysis
        from logger.session_logger import SessionLogger
        self._session_logger = SessionLogger.get()
        self._session_logger.log_start(self.get_config())

        # Start the recognizer thread if not already running
        if not self.isRunning():
            try:
                self.start()
            except Exception as e:
                logger.error(f"Failed to (re)start recognizer: {e}")

    def is_stopped(self) -> bool:
        """Check if a stop request has been made.

        Returns:
            bool: True if stop has been requested, False otherwise
        """
        return self._stop_flag

    def pause(self) -> None:
        """Pause transcription processing.

        Typically called when the "WAIT" voice command is detected.
        The recognizer will continue listening but won't process transcriptions
        until resume() is called.
        """
        self._paused = True
        self.status_changed.emit("paused")

    def resume(self) -> None:
        """Resume transcription processing.

        Typically called when the "START" voice command is detected.
        The recognizer will resume normal processing of audio segments.
        """
        self._paused = False
        self.status_changed.emit("listening")

    def get_config(self) -> dict:
        """Return comprehensive configuration for logging and debugging.

        Collects all relevant configuration parameters including audio settings,
        model parameters, and noise control settings for session logging.

        Returns:
            dict: Complete configuration dictionary with all settings
        """
        base = {
            # Audio configuration
            "SAMPLE_RATE": self.SAMPLE_RATE,
            "CHANNELS": self.CHANNELS,
            "CHUNK_S": self.CHUNK_S,

            # Noise control settings
            "VAD_MODE": self.VAD_MODE,
            "MIN_SPEECH_S": self.MIN_SPEECH_S,
            "MAX_SEGMENT_S": self.MAX_SEGMENT_S,
            "PADDING_MS": self.PADDING_MS,

            # Recognition settings
            "FISH_PROMPT": self.FISH_PROMPT,
            "MODEL_NAME": self.MODEL_NAME,
            "DEVICE": self.DEVICE,
            "COMPUTE_TYPE": self.COMPUTE_TYPE,

            # Decoding parameters
            "BEAM_SIZE": self.BEAM_SIZE,
            "BEST_OF": self.BEST_OF,
            "TEMPERATURE": self.TEMPERATURE,
            "PATIENCE": self.PATIENCE,
            "LENGTH_PENALTY": self.LENGTH_PENALTY,
            "REPETITION_PENALTY": self.REPETITION_PENALTY,
            "WITHOUT_TIMESTAMPS": self.WITHOUT_TIMESTAMPS,
            "CONDITION_ON_PREVIOUS_TEXT": self.CONDITION_ON_PREVIOUS_TEXT,
            "VAD_FILTER": self.VAD_FILTER,
            "VAD_PARAMETERS": self.VAD_PARAMETERS,
            "WORD_TIMESTAMPS": self.WORD_TIMESTAMPS,

            # Environment settings
            "NOISE_PROFILE": self._noise_profile_name,
        }
        return base

    # ---------- Public control methods ----------
    def stop(self) -> None:
        """Request the recognizer to stop and release all resources.

        This method safely shuts down the audio stream, stops the noise controller,
        and sets flags to terminate the recognition thread. It's designed to be
        called from the main thread to gracefully stop the recognizer.
        """
        self._stop_flag = True
        try:
            # Request cooperative thread interruption
            self.requestInterruption()
        except Exception:
            pass

        # Stop and close audio stream
        try:
            if self._stream is not None:
                import sounddevice as sd  # noqa: F401
                self._stream.stop()
                self._stream.close()
                self._stream = None
        except Exception as e:
            logger.debug(f"Error stopping input stream: {e}")

        # Signal noise controller to stop processing
        self._noise_controller.stop()

    def begin(self) -> None:
        """Reset internal state and start/restart the recognizer thread.

        This method performs a complete initialization of the recognizer,
        applying current noise profile settings and starting fresh with
        clean state. It's safe to call multiple times.
        """
        # Reset control flags for clean restart
        self._stop_flag = False
        self._last_status_msg = None

        # Reapply noise profile settings (may have changed)
        from speech.noise_profiles import get_noise_profile, make_suppressor_config
        prof = get_noise_profile(self._noise_profile_name)
        for attr in ("VAD_MODE", "MIN_SPEECH_S", "MAX_SEGMENT_S", "PADDING_MS"):
            if attr in prof:
                setattr(self, attr, prof[attr])

        # Recreate noise controller with current settings
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

        # Reinitialize session logging
        from logger.session_logger import SessionLogger
        self._session_logger = SessionLogger.get()
        self._session_logger.log_start(self.get_config())

        # Start the recognition thread if not already running
        if not self.isRunning():
            try:
                self.start()
            except Exception as e:
                logger.error(f"Failed to (re)start recognizer: {e}")


    # ---------- Internal helper methods ----------
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Sounddevice callback function for real-time audio processing.

        This callback is invoked by sounddevice for each audio chunk. It converts
        the input audio to the appropriate format and feeds it to the noise controller
        for further processing.

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

    @staticmethod
    def _write_wav_bytes(samples_int16: np.ndarray, samplerate: int) -> str:
        """Write PCM16 samples to a temporary WAV file for Whisper processing.

        Creates a temporary WAV file that can be passed to the Whisper model
        for transcription. The file is created with proper PCM16 encoding
        that Whisper expects.

        Args:
            samples_int16: Audio samples as 16-bit signed integers
            samplerate: Sample rate in Hz

        Returns:
            str: Path to the temporary WAV file
        """
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        # Use soundfile for reliable WAV writing
        sf.write(path, samples_int16, samplerate, subtype="PCM_16")
        return path

    def _emit_status_once(self, message: str) -> None:
        """Emit status_changed signal only when the message changes.

        This method prevents flooding the UI with identical status messages
        by tracking the last emitted message and only emitting when it changes.

        Args:
            message: Status message to emit ("listening", "processing", "stopped", etc.)
        """
        if message != self._last_status_msg:
            self._last_status_msg = message
            try:
                self.status_changed.emit(message)
            except Exception:
                logger.error(f"Failed to emit status_changed message: {message}")

    def _load_number_prefix(self) -> np.ndarray:
        """Load and prepare the number prefix audio for improved number recognition.

        The number prefix is a short audio clip that is prepended to each speech
        segment before transcription. This helps Whisper better recognize numeric
        content by providing acoustic context that signals number recognition mode.

        The method searches common paths for the audio file, ensures mono channel,
        resamples to match the recognizer's sample rate, and returns PCM16 data.

        Returns:
            np.ndarray: Prepared number prefix audio as PCM16 samples

        Raises:
            Exception: If no valid number prefix audio can be loaded (falls back to silence)
        """
        # Search common locations for number prefix audio
        candidates = [
            os.path.join(os.getcwd(), "number_prefix.wav"),
            os.path.join(os.getcwd(), "assets/audio/number.wav"),
            os.path.join(os.getcwd(), "tests", "audio", "number.wav"),
        ]

        for p in candidates:
            try:
                if not os.path.exists(p):
                    continue

                # Load audio file as PCM16
                data, sr = sf.read(p, dtype="int16")

                # Convert to mono if stereo (use first channel to preserve amplitude)
                if data.ndim > 1:
                    data = data[:, 0]

                # Resample if necessary to match recognizer sample rate
                if sr != self.SAMPLE_RATE:
                    try:
                        # Use high-quality polyphase resampling if available
                        from scipy.signal import resample_poly  # type: ignore
                        # Compute integer up/down factors for optimal quality
                        gcd = np.gcd(sr, self.SAMPLE_RATE)
                        up = self.SAMPLE_RATE // gcd
                        down = sr // gcd
                        data = resample_poly(data.astype(np.int16), up, down).astype(np.int16)
                    except Exception:
                        # Fallback to simple nearest-neighbor resampling
                        ratio = self.SAMPLE_RATE / float(sr)
                        idx = (np.arange(int(len(data) * ratio)) / ratio).astype(int)
                        idx = np.clip(idx, 0, len(data) - 1)
                        data = data[idx].astype(np.int16)

                    logger.info(f"Loaded number prefix '{os.path.basename(p)}' at {sr}Hz -> resampled to {self.SAMPLE_RATE}Hz")
                else:
                    logger.info(f"Loaded number prefix '{os.path.basename(p)}' at {sr}Hz (no resample)")

                return data.astype(np.int16)

            except Exception as e:
                logger.debug(f"Failed to load number prefix {p}: {e}")

        # Fallback to 50ms of silence if no prefix audio found
        logger.warning("No number prefix audio found; using short silence")
        return (np.zeros(int(self.SAMPLE_RATE * 0.05))).astype(np.int16)

    # ---------- Main recognition loop ----------
    def run(self) -> None:
        """Main thread execution method for real-time speech recognition.

        This method implements the core recognition loop that:
        1. Loads the Whisper model
        2. Initializes the audio input stream
        3. Continuously processes audio segments from the noise controller
        4. Transcribes speech using Whisper
        5. Parses results for fish species and measurements
        6. Emits structured results via PyQt signals

        The loop continues until stop is requested or an unrecoverable error occurs.
        It handles voice commands for pause/resume and applies fish-specific
        ASR corrections and parsing to improve accuracy for marine data collection.
        """
        # Early exit if stop was requested before thread started
        if self.is_stopped() or self.isInterruptionRequested():
            return

        # Initialize Whisper model
        try:
            logger.info("Loading model... (first run will download it)")
            from faster_whisper import WhisperModel  # type: ignore
            self._model = WhisperModel(
                self.MODEL_NAME,
                device=self.DEVICE,
                compute_type=self.COMPUTE_TYPE,
                download_root=None,  # Use default download location
                local_files_only=False  # Allow downloading if needed
            )

            # Check for interruption after potentially long model loading
            if self.is_stopped() or self.isInterruptionRequested():
                return
        except Exception as e:
            msg = f"Failed to load Whisper model: {e}"
            logger.error(msg)
            self.error.emit(msg)
            if hasattr(self, '_session_logger'):
                self._session_logger.log(f"ERROR: {e}")
            return

        # Initialize audio input stream
        try:
            import sounddevice as sd  # type: ignore
            if self.is_stopped() or self.isInterruptionRequested():
                return

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

        # Main recognition loop
        with self._stream:
            logger.info(f"Recording with noise control... Press Stop to end. [profile={self._noise_profile_name}]")
            self.partial_text.emit("Listening…")
            self._emit_status_once("listening")

            # Get segment generator from noise controller
            segment_generator = self._noise_controller.collect_segments(
                padding_ms=self.PADDING_MS
            )

            while not self.is_stopped():
                # Honor external interruption requests
                if self.isInterruptionRequested():
                    break

                try:
                    # Get next filtered and VAD-processed audio segment
                    segment = next(segment_generator)

                    # Skip empty or invalid segments
                    if segment is None or segment.size == 0:
                        continue

                    # Double-check segment duration (noise controller pre-filters, but be safe)
                    segment_duration = segment.size / self.SAMPLE_RATE
                    if segment_duration < self.MIN_SPEECH_S:
                        continue

                    # Update UI status during processing
                    self._emit_status_once("processing")

                    # Prepend number prefix to improve number recognition
                    segment = np.concatenate((self._number_sound, segment))

                    # Create temporary WAV file for Whisper
                    wav_path = self._write_wav_bytes(segment, self.SAMPLE_RATE)

                    # Save audio segment for debugging/analysis
                    audio_saver = get_audio_saver()
                    audio_saver.save_segment(segment, self.SAMPLE_RATE)

                    # Transcribe audio segment using Whisper
                    try:
                        assert self._model is not None
                        segments, info = self._model.transcribe(
                            wav_path,
                            beam_size=self.BEAM_SIZE,
                            best_of=self.BEST_OF,
                            temperature=self.TEMPERATURE,
                            patience=self.PATIENCE,
                            length_penalty=self.LENGTH_PENALTY,
                            repetition_penalty=self.REPETITION_PENALTY,
                            language="en",
                            condition_on_previous_text=True,
                            initial_prompt=self.FISH_PROMPT,  # Fish-specific context
                            vad_filter=False,  # We handle VAD externally
                            vad_parameters=None,
                            without_timestamps=True,  # Skip timestamps for speed
                            word_timestamps=False
                        )
                    except Exception as e:
                        msg = f"Transcription error: {e}"
                        logger.error(msg)
                        self.error.emit(msg)
                        # Clean up temp file and continue
                        try:
                            os.remove(wav_path)
                        except Exception:
                            pass
                        self._emit_status_once("listening")
                        continue

                    # Extract transcribed text from segments
                    text_out = "".join(seg.text + " " for seg in segments).strip()
                    if not text_out:
                        self._emit_status_once("listening")
                        continue

                    logger.info(f"Raw transcription: {text_out}")

                    # --- Handle voice commands for pause/resume ---
                    text_lower = text_out.lower()
                    if "wait" in text_lower:
                        self.pause()
                        self.final_text.emit("Waiting until 'start' is said.", 0.85)
                        try:
                            os.remove(wav_path)
                        except Exception as e:
                            logger.debug(e)
                        continue
                    elif "start" in text_lower:
                        self.resume()
                        try:
                            os.remove(wav_path)
                        except Exception as e:
                            logger.debug(e)
                        continue

                    # Skip processing if currently paused
                    if self._paused:
                        logger.debug("Paused: ignoring transcription")
                        try:
                            os.remove(wav_path)
                        except Exception as e:
                            logger.debug(e)
                        continue

                    # Apply fish-specific ASR corrections and parsing
                    try:
                        from parser import FishParser, TextNormalizer
                        fish_parser = FishParser()
                        text_normalizer = TextNormalizer()

                        # Apply domain-specific ASR corrections for fish names
                        corrected_text = text_normalizer.apply_fish_asr_corrections(text_out)
                        if corrected_text != text_out.lower():
                            logger.info(f"After ASR corrections: {corrected_text}")

                        # Parse for fish species and measurements
                        result: ParserResult = fish_parser.parse_text(corrected_text)

                        # Update current species context if detected
                        if result.species is not None:
                            self._last_fish_specie = result.species
                            self.specie_detected.emit(result.species)

                        # Format and emit final result
                        if result.length_cm is not None:
                            # Format numeric measurement with species context
                            raw_val = float(result.length_cm)
                            num_str = (f"{raw_val:.1f}").rstrip("0").rstrip(".")
                            formatted = f"{self._last_fish_specie} {num_str} cm"
                            logger.info(f">> {formatted}")
                            self.final_text.emit(formatted, 0.85)
                        else:
                            # Fallback to corrected text if parsing incomplete
                            logger.info(f">> {corrected_text} (partial parse)")
                            self.final_text.emit(corrected_text, 0.85)

                    except Exception as e:
                        logger.error(f"Parser error: {e}")
                        # Final fallback to raw transcription
                        logger.info(f">> {text_out}")
                        self.final_text.emit(text_out, 0.85)

                    # Clean up temporary WAV file
                    try:
                        os.remove(wav_path)
                    except Exception:
                        pass

                    # Return to listening state
                    self._emit_status_once("listening")

                except StopIteration:
                    # Generator exhausted (normal when stopping)
                    break
                except Exception as e:
                    logger.error(f"Main loop error: {e}")
                    self.error.emit(f"Processing error: {e}")
                    time.sleep(0.1)  # Brief pause before retrying
                    continue

        # Cleanup on exit
        self._emit_status_once("stopped")
        logger.info("Speech recognizer stopped")
        if hasattr(self, '_session_logger'):
            self._session_logger.log_end()
        if hasattr(self, '_session_log_sink_id'):
            import loguru
            loguru.logger.remove(self._session_log_sink_id)
