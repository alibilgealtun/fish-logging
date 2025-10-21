"""AssemblyAI Streaming Speech Recognition Integration.

This module provides real-time speech recognition using AssemblyAI's WebSocket streaming API.
It's optimized for high-noise marine environments and integrates with the application's
noise control and fish species parsing systems.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Optional, List
from queue import Queue

import numpy as np
import sounddevice as sd
from loguru import logger
from PyQt6.QtCore import pyqtSignal

from .base_recognizer import BaseSpeechRecognizer
from noise.controller import NoiseController
from services import get_audio_saver


class AssemblyAIRecognizer(BaseSpeechRecognizer):
    """
    Real-time speech recognizer using AssemblyAI's WebSocket streaming API.

    This implementation provides cloud-based speech recognition with built-in noise
    handling and word boosting capabilities. It's particularly effective for marine
    environments due to AssemblyAI's robust noise suppression and the ability to
    boost recognition of fish species names and measurement terms.

    Key Features:
    - WebSocket-based streaming for low-latency recognition
    - Word boosting for improved fish species recognition
    - Built-in noise handling and voice activity detection
    - Real-time partial and final transcript processing
    - Integration with local noise controller for additional filtering
    - Session logging and audio segment saving

    Architecture:
    - Main thread: Manages audio capture and coordinates other threads
    - WebSocket thread: Handles real-time communication with AssemblyAI
    - Audio callback: Captures and queues audio data for streaming
    """

    # PyQt signals for UI communication
    partial_text = pyqtSignal(str)  # Intermediate transcription results
    final_text = pyqtSignal(str, float)  # Final text with confidence score
    error = pyqtSignal(str)  # Error messages
    status_changed = pyqtSignal(str)  # Status updates (initializing, listening, etc.)
    specie_detected = pyqtSignal(str)  # Fish species detection events

    # ===== AUDIO CONFIGURATION =====
    SAMPLE_RATE: int = 16000  # Audio sampling rate (AssemblyAI requirement)
    CHANNELS: int = 1  # Mono audio input
    CHUNK_S: float = 0.05  # 50ms chunks for low-latency streaming

    # ===== NOISE CONTROLLER SETTINGS =====
    # Local noise processing before sending to AssemblyAI
    VAD_MODE: int = 2  # Voice Activity Detection aggressiveness
    MIN_SPEECH_S: float = 0.4  # Minimum speech segment duration
    MAX_SEGMENT_S: float = 3.0  # Maximum speech segment duration
    PADDING_MS: int = 600  # Audio padding around speech segments

    def __init__(
        self,
        language: str = "en",
        api_key: Optional[str] = None,
        noise_profile: Optional[str] = None
    ) -> None:
        """Initialize AssemblyAI recognizer with API credentials and configuration.

        Sets up the recognizer with WebSocket communication, local noise processing,
        and phrase hints for improved accuracy on fish-related vocabulary.

        Args:
            language: Language code for recognition (e.g., "en", "es")
            api_key: AssemblyAI API key. If None, reads from ASSEMBLYAI_API_KEY env var
            noise_profile: Noise profile name for local processing
                         Valid values: "clean", "human", "engine", "mixed"

        Raises:
            ValueError: If API key is not provided via parameter or environment variable
        """
        super().__init__(language=f"{language}-US" if language == "en" else language)

        # Configuration and state initialization
        self._noise_profile_name = (noise_profile or "mixed").lower()
        self._api_key = api_key or os.getenv("ASSEMBLYAI_API_KEY")
        self._last_fish_specie: Optional[str] = None  # Track last detected species for context

        # Validate API key availability
        if not self._api_key:
            raise ValueError(
                "AssemblyAI API key not provided. Set ASSEMBLYAI_API_KEY environment "
                "variable or pass api_key parameter."
            )

        # WebSocket and threading components
        self._ws = None  # WebSocket connection (initialized in thread)
        self._ws_thread: Optional[threading.Thread] = None  # WebSocket communication thread
        self._audio_queue: Queue = Queue()  # Queue for streaming audio data
        self._chunk_frames = int(self.SAMPLE_RATE * self.CHUNK_S)  # Audio chunk size
        self._stream = None  # Audio input stream
        self._stop_event = threading.Event()  # Cross-thread stop coordination

        # Session logging components
        self._session_logger = None
        self._session_log_sink_id = None

        # Apply noise profile optimizations for local processing
        from speech.noise_profiles import get_noise_profile, make_suppressor_config
        prof = get_noise_profile(self._noise_profile_name)
        for attr in ("VAD_MODE", "MIN_SPEECH_S", "MAX_SEGMENT_S", "PADDING_MS"):
            if attr in prof:
                setattr(self, attr, prof[attr])

        suppressor_cfg = make_suppressor_config(prof, self.SAMPLE_RATE)

        # Initialize local noise controller for pre-processing
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

        # Build phrase hints for improved fish species recognition
        self._phrase_hints = self._build_phrase_hints()

        logger.info(
            f"AssemblyAI recognizer initialized with noise_profile={self._noise_profile_name}"
        )

    def set_last_species(self, species: str) -> None:
        """Set the last detected fish species for context in number-only transcriptions.

        Used by the UI to provide context when users speak measurements without
        explicitly stating the species name.

        Args:
            species: Name of the fish species to use as context, or None to clear
        """
        try:
            self._last_fish_specie = str(species) if species else None
        except Exception:
            self._last_fish_specie = None

    def begin(self) -> None:
        """Reset internal state and start/restart the recognizer thread.

        Performs complete reinitialization of the recognizer, applying current
        noise profile settings and restarting all processing threads.
        """
        # Reset control flags for clean restart
        self._stop_flag = False

        # Reinitialize noise controller with current profile settings
        from speech.noise_profiles import get_noise_profile, make_suppressor_config
        prof = get_noise_profile(self._noise_profile_name)
        for attr in ("VAD_MODE", "MIN_SPEECH_S", "MAX_SEGMENT_S", "PADDING_MS"):
            if attr in prof:
                setattr(self, attr, prof[attr])
        suppressor_cfg = make_suppressor_config(prof, self.SAMPLE_RATE)

        # Recreate noise controller with updated settings
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

        # Initialize session logging
        from logger.session_logger import SessionLogger
        self._session_logger = SessionLogger.get()
        self._session_logger.log_start(self.get_config())

        # Start the recognition thread if not already running
        if not self.isRunning():
            try:
                self.start()
            except Exception as e:
                logger.error(f"Failed to (re)start recognizer: {e}")

    def get_config(self) -> dict:
        """Return comprehensive configuration for logging and debugging.

        Returns:
            dict: Complete configuration including audio settings, API details,
                 and noise processing parameters
        """
        return {
            # Audio configuration
            "SAMPLE_RATE": self.SAMPLE_RATE,
            "CHANNELS": self.CHANNELS,
            "CHUNK_S": self.CHUNK_S,

            # Noise processing settings
            "VAD_MODE": self.VAD_MODE,
            "MIN_SPEECH_S": self.MIN_SPEECH_S,
            "MAX_SEGMENT_S": self.MAX_SEGMENT_S,
            "PADDING_MS": self.PADDING_MS,
            "NOISE_PROFILE": self._noise_profile_name,

            # API configuration
            "API": "AssemblyAI",
            "LANGUAGE": self.language,
        }

    def is_stopped(self) -> bool:
        """Check if a stop request has been made.

        Returns:
            bool: True if stop has been requested, False otherwise
        """
        return self._stop_flag

    def pause(self) -> None:
        """Pause transcription processing.

        Called when the "WAIT" voice command is detected. The recognizer continues
        listening but stops processing transcriptions until resume() is called.
        """
        self._paused = True
        self.status_changed.emit("paused")

    def resume(self) -> None:
        """Resume transcription processing.

        Called when the "START" voice command is detected. Resumes normal
        processing of transcriptions from AssemblyAI.
        """
        self._paused = False
        self.status_changed.emit("listening")

    def _build_phrase_hints(self) -> List[str]:
        """Build phrase hints from configuration files for improved recognition accuracy.

        Loads fish species names, number words, units, and control commands from
        configuration files to create a vocabulary boost list for AssemblyAI.
        This significantly improves recognition accuracy for domain-specific terms.

        Returns:
            List[str]: List of phrase hints to boost recognition accuracy
        """
        hints: List[str] = []

        # Load fish species names from configuration
        try:
            with open(os.path.join("config", "species.json"), "r", encoding="utf-8") as f:
                species_cfg = json.load(f)
            for item in species_cfg.get("items", []):
                name = item.get("name")
                if name:
                    hints.append(str(name))
        except Exception as e:
            logger.warning(f"Could not load species hints: {e}")

        # Load number words and variants
        try:
            with open(os.path.join("config", "numbers.json"), "r", encoding="utf-8") as f:
                numbers_cfg = json.load(f)
            hints.extend(list(numbers_cfg.get("number_words", {}).keys()))
        except Exception as e:
            logger.warning(f"Could not load number hints: {e}")

        # Load measurement units and synonyms
        try:
            with open(os.path.join("config", "units.json"), "r", encoding="utf-8") as f:
                units_cfg = json.load(f)
            hints.extend(list(units_cfg.get("synonyms", {}).keys()))
        except Exception as e:
            logger.warning(f"Could not load unit hints: {e}")

        # Add voice control commands
        hints.extend(["cancel", "wait", "start"])

        return hints

    def _audio_callback(self, indata, frames, time_info, status):
        """Sounddevice callback for real-time audio capture.

        This callback is invoked by sounddevice for each audio chunk. It converts
        the audio to the appropriate format and queues it for streaming to AssemblyAI.

        Args:
            indata: Input audio data as float32 numpy array
            frames: Number of audio frames in this chunk
            time_info: Timing information from sounddevice (unused)
            status: Status flags from sounddevice for error detection
        """
        if status:
            logger.warning(f"Audio callback status: {status}")

        # Only queue audio if not stopped and not paused
        if not self._stop_event.is_set() and not self._paused:
            # Convert float32 to int16 bytes for AssemblyAI
            audio_bytes = (indata * 32767).astype(np.int16).tobytes()
            self._audio_queue.put(audio_bytes)

    def _websocket_thread(self):
        """WebSocket communication thread for real-time streaming to AssemblyAI.

        This method runs in a separate thread and handles:
        1. Establishing WebSocket connection with phrase hints
        2. Streaming audio data from the queue
        3. Processing incoming transcription messages
        4. Handling connection errors and cleanup
        """
        try:
            import websocket
            from urllib.parse import urlencode

            # Prepare connection parameters with phrase hints
            params = {
                "sample_rate": self.SAMPLE_RATE,
                # Limit to 100 phrase hints to stay within AssemblyAI limits
                "word_boost": json.dumps(self._phrase_hints[:100]) if self._phrase_hints else None,
            }

            # Remove None values from parameters
            params = {k: v for k, v in params.items() if v is not None}

            # Construct WebSocket URL with parameters
            url = f"wss://streaming.assemblyai.com/v3/ws?{urlencode(params)}"

            def on_open(ws):
                """Handle WebSocket connection opening."""
                logger.info("AssemblyAI WebSocket connection opened")
                self.status_changed.emit("listening")

                # Start audio streaming thread
                def send_audio():
                    """Continuously send audio data to AssemblyAI."""
                    while not self._stop_event.is_set():
                        try:
                            # Get audio data from queue with timeout
                            audio_data = self._audio_queue.get(timeout=0.1)
                            # Send binary audio data if connection is active
                            if audio_data and ws.sock and ws.sock.connected:
                                ws.send(audio_data, websocket.ABNF.OPCODE_BINARY)
                        except Exception:
                            continue

                # Start daemon thread for audio streaming
                audio_thread = threading.Thread(target=send_audio, daemon=True)
                audio_thread.start()

            def on_message(ws, message):
                """Handle incoming messages from AssemblyAI.

                Processes both partial and final transcripts, handling various
                message formats across different AssemblyAI API versions.
                """
                try:
                    data = json.loads(message)

                    # Extract message type and transcript text (handles API variations)
                    msg_type = data.get("type") or data.get("message_type")
                    transcript = data.get("transcript") or data.get("text") or ""

                    # Debug logging for message structure analysis
                    logger.debug(f"AssemblyAI message: {json.dumps(data)}")

                    # Determine if transcript is partial or final
                    is_partial = None
                    if "partial" in data:
                        is_partial = bool(data.get("partial"))
                    elif "final" in data:
                        is_partial = not bool(data.get("final"))

                    # Handle session management messages
                    if msg_type in ("Begin", "SessionBegins"):
                        session_id = data.get("id") or data.get("session_id")
                        logger.info(f"AssemblyAI session started: {session_id}")
                        return

                    if msg_type in ("Termination", "SessionTerminated"):
                        logger.info("AssemblyAI session terminated")
                        self._stop_event.set()
                        return

                    # Handle turn-based messages with end_of_turn indicator
                    if msg_type == "Turn":
                        if "end_of_turn" in data:
                            is_partial = not bool(data.get("end_of_turn"))

                    # Fallback partial/final determination based on message type
                    if is_partial is None:
                        if msg_type in ("PartialTranscript", "Partial"):
                            is_partial = True
                        elif msg_type in ("Transcript", "FinalTranscript"):
                            is_partial = False

                    # Process transcript if available
                    if transcript:
                        if is_partial:
                            logger.debug(f"Emitting partial: {transcript}")
                            # Note: Partial transcripts are logged but not currently emitted to UI
                            # self.partial_text.emit(transcript)
                        else:
                            # Extract confidence score (prefer end_of_turn_confidence)
                            confidence = float(
                                data.get("end_of_turn_confidence", data.get("confidence", 0.85))
                            )
                            logger.info(f"Processing final transcript: {transcript}")
                            self._process_final_transcript(transcript, confidence)

                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding AssemblyAI message: {e}")
                except Exception as e:
                    logger.error(f"Error handling AssemblyAI message: {e}")

            def on_error(ws, error):
                """Handle WebSocket errors."""
                logger.error(f"AssemblyAI WebSocket error: {error}")
                self.error.emit(f"AssemblyAI error: {error}")

            def on_close(ws, close_status_code, close_msg):
                """Handle WebSocket connection closure."""
                logger.info(f"AssemblyAI WebSocket closed: {close_status_code} - {close_msg}")
                self.status_changed.emit("stopped")

            # Create and configure WebSocket client
            self._ws = websocket.WebSocketApp(
                url,
                header={"Authorization": self._api_key},
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )

            # Start WebSocket event loop (blocking)
            self._ws.run_forever()

        except Exception as e:
            logger.error(f"AssemblyAI WebSocket thread error: {e}")
            self.error.emit(f"AssemblyAI connection error: {e}")

    def _process_final_transcript(self, text: str, confidence: float):
        """Process final transcript from AssemblyAI and emit appropriate signals.

        Applies fish-specific ASR corrections, parses for species and measurements,
        and emits structured results. Handles voice commands for pause/resume.

        Args:
            text: Raw transcript text from AssemblyAI
            confidence: Confidence score from AssemblyAI (0.0 to 1.0)
        """
        # Skip processing if no text or currently paused
        if not text or self._paused:
            return

        text = text.strip()
        logger.info(f"Raw transcription: {text}")

        # Handle voice commands for pause/resume functionality
        text_lower = text.lower()
        if "wait" in text_lower:
            self.pause()
            self.final_text.emit("Waiting until 'start' is said.", 0.85)
            return
        elif "start" in text_lower:
            self.resume()
            return

        # Skip processing if paused (double-check after command processing)
        if self._paused:
            logger.debug("Paused: ignoring transcription")
            return

        # Apply domain-specific processing and parsing
        try:
            from parser import FishParser, TextNormalizer, ParserResult
            fish_parser = FishParser()
            text_normalizer = TextNormalizer()

            # Apply fish-specific ASR corrections for improved accuracy
            corrected_text = text_normalizer.apply_fish_asr_corrections(text)
            if corrected_text != text.lower():
                logger.info(f"After ASR corrections: {corrected_text}")

            # Parse corrected text for fish species and measurements
            result: ParserResult = fish_parser.parse_text(corrected_text)

            # Update species context if detected
            if result.species is not None:
                self._last_fish_specie = result.species
                self.specie_detected.emit(result.species)

            # Format and emit results
            if result.length_cm is not None:
                # Format numeric measurement with species context
                raw_val = float(result.length_cm)
                num_str = (f"{raw_val:.1f}").rstrip("0").rstrip(".")
                formatted = f"{self._last_fish_specie} {num_str} cm"
                logger.info(f">> {formatted}")
                self.final_text.emit(formatted, confidence)
            else:
                # Fallback to corrected text if parsing incomplete
                logger.info(f">> {corrected_text} (partial parse)")
                self.final_text.emit(corrected_text, confidence)

        except Exception as e:
            logger.error(f"Parser error: {e}")
            # Final fallback to raw transcription
            logger.info(f">> {text}")
            self.final_text.emit(text, confidence)

    def run(self) -> None:
        """Main recognition thread execution method.

        Orchestrates the entire recognition process by:
        1. Initializing WebSocket communication thread
        2. Starting audio capture stream
        3. Coordinating threads until stop is requested
        4. Cleaning up resources on exit
        """
        try:
            self.status_changed.emit("initializing")
            logger.info("Starting AssemblyAI recognizer...")

            # Reset stop event for new session
            self._stop_event.clear()

            # Start WebSocket communication thread
            self._ws_thread = threading.Thread(target=self._websocket_thread, daemon=True)
            self._ws_thread.start()

            # Initialize and start audio capture stream
            self._stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=self.CHANNELS,
                dtype=np.float32,
                blocksize=self._chunk_frames,
                callback=self._audio_callback,
            )
            self._stream.start()

            logger.info("AssemblyAI recognizer started successfully")

            # Keep main thread alive while recognition is active
            while not self._stop_flag and not self._stop_event.is_set():
                time.sleep(0.1)

        except Exception as e:
            logger.error(f"AssemblyAI recognizer error: {e}")
            self.error.emit(f"Recognition error: {e}")

        finally:
            # Ensure cleanup happens regardless of how we exit
            self._cleanup()

    def _cleanup(self):
        """Clean up all resources and stop all threads.

        Ensures proper shutdown of:
        - Audio input stream
        - WebSocket connection
        - Background threads
        - Session logging
        """
        logger.info("Cleaning up AssemblyAI recognizer...")

        # Signal all threads to stop
        self._stop_event.set()

        # Stop and close audio stream
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.error(f"Error closing audio stream: {e}")
            self._stream = None

        # Close WebSocket connection gracefully
        if self._ws:
            try:
                # Send termination message to AssemblyAI
                terminate_msg = {"type": "Terminate"}
                self._ws.send(json.dumps(terminate_msg))
                time.sleep(0.5)  # Allow time for message to be sent
                self._ws.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
            self._ws = None

        # Wait for WebSocket thread to complete
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=2.0)

        # Clean up session logging
        if hasattr(self, '_session_logger') and self._session_logger:
            self._session_logger.log_end()
        if hasattr(self, '_session_log_sink_id') and self._session_log_sink_id:
            import loguru
            loguru.logger.remove(self._session_log_sink_id)

        # Emit final status
        self.status_changed.emit("stopped")
        logger.info("AssemblyAI recognizer cleanup complete")

    def stop(self) -> None:
        """Request the recognizer to stop and begin cleanup.

        This method initiates a graceful shutdown by setting stop flags
        and triggering the cleanup process.
        """
        logger.info("Stopping AssemblyAI recognizer...")
        super().stop()  # Call parent class stop method
        self._stop_event.set()  # Signal all threads to stop

    def set_noise_profile(self, name: Optional[str]) -> None:
        """Update the noise profile configuration.

        Note: Changes take effect on the next restart of the recognizer.

        Args:
            name: New noise profile name ("clean", "human", "engine", "mixed")
        """
        self._noise_profile_name = (name or "mixed").lower()
        logger.info(f"Noise profile set to: {self._noise_profile_name} (restart required)")
