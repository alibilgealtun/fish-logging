"""AssemblyAI Streaming Speech Recognition Integration."""
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
    Optimized for high-noise environments with built-in noise handling.
    """

    # PyQt signals
    partial_text = pyqtSignal(str)
    final_text = pyqtSignal(str, float)
    error = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    specie_detected = pyqtSignal(str)

    # ===== CONFIG =====
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    CHUNK_S: float = 0.05  # 50ms chunks for streaming

    # Noise controller settings
    VAD_MODE: int = 2
    MIN_SPEECH_S: float = 0.4
    MAX_SEGMENT_S: float = 3.0
    PADDING_MS: int = 600

    def __init__(
        self,
        language: str = "en",
        api_key: Optional[str] = None,
        noise_profile: Optional[str] = None
    ) -> None:
        """Initialize AssemblyAI recognizer.

        Args:
            language: Language code (e.g., "en", "es").
            api_key: AssemblyAI API key. If None, reads from ASSEMBLYAI_API_KEY env var.
            noise_profile: Noise profile name (clean|human|engine|mixed).
        """
        super().__init__(language=f"{language}-US" if language == "en" else language)

        self._noise_profile_name = (noise_profile or "mixed").lower()
        self._api_key = api_key or os.getenv("ASSEMBLYAI_API_KEY")
        self._last_fish_specie: Optional[str] = None  # Track last detected species

        if not self._api_key:
            raise ValueError(
                "AssemblyAI API key not provided. Set ASSEMBLYAI_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self._ws = None
        self._ws_thread: Optional[threading.Thread] = None
        self._audio_queue: Queue = Queue()
        self._chunk_frames = int(self.SAMPLE_RATE * self.CHUNK_S)
        self._stream = None
        self._stop_event = threading.Event()
        self._session_logger = None
        self._session_log_sink_id = None

        # Apply noise profile overrides
        from speech.noise_profiles import get_noise_profile, make_suppressor_config
        prof = get_noise_profile(self._noise_profile_name)
        for attr in ("VAD_MODE", "MIN_SPEECH_S", "MAX_SEGMENT_S", "PADDING_MS"):
            if attr in prof:
                setattr(self, attr, prof[attr])

        suppressor_cfg = make_suppressor_config(prof, self.SAMPLE_RATE)

        # Initialize noise controller
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

        # Build phrase hints for better accuracy
        self._phrase_hints = self._build_phrase_hints()

        logger.info(
            f"AssemblyAI recognizer initialized with noise_profile={self._noise_profile_name}"
        )

    def set_last_species(self, species: str) -> None:
        """Public setter for last species (used by UI selector)."""
        try:
            self._last_fish_specie = str(species) if species else None
        except Exception:
            self._last_fish_specie = None

    def begin(self) -> None:
        """Reset internal state and start the recognizer thread."""
        # Reset flags/state for a clean restart
        self._stop_flag = False

        # Reinitialize noise controller with current profile
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

        # Initialize session logger
        from logger.session_logger import SessionLogger
        self._session_logger = SessionLogger()
        self._session_logger.log_start(self.get_config())
        import loguru
        self._session_log_sink_id = loguru.logger.add(
            self._session_logger.log_path,
            format="[{time:YYYY-MM-DD HH:mm:ss}] {level}: {message}",
            level="INFO"
        )

        if not self.isRunning():
            try:
                self.start()
            except Exception as e:
                logger.error(f"Failed to (re)start recognizer: {e}")

    def get_config(self) -> dict:
        """Return all relevant config parameters for logging/export."""
        return {
            "SAMPLE_RATE": self.SAMPLE_RATE,
            "CHANNELS": self.CHANNELS,
            "CHUNK_S": self.CHUNK_S,
            "VAD_MODE": self.VAD_MODE,
            "MIN_SPEECH_S": self.MIN_SPEECH_S,
            "MAX_SEGMENT_S": self.MAX_SEGMENT_S,
            "PADDING_MS": self.PADDING_MS,
            "NOISE_PROFILE": self._noise_profile_name,
            "API": "AssemblyAI",
            "LANGUAGE": self.language,
        }

    def is_stopped(self) -> bool:
        """Return True if stop has been requested."""
        return self._stop_flag

    def pause(self) -> None:
        """Pause transcription (after hearing 'WAIT')."""
        self._paused = True
        self.status_changed.emit("paused")

    def resume(self) -> None:
        """Resume transcription (after hearing 'START')."""
        self._paused = False
        self.status_changed.emit("listening")

    def _build_phrase_hints(self) -> List[str]:
        """Build phrase hints from species, numbers, and units config."""
        hints: List[str] = []

        # Load species
        try:
            with open(os.path.join("config", "species.json"), "r", encoding="utf-8") as f:
                species_cfg = json.load(f)
            for item in species_cfg.get("items", []):
                name = item.get("name")
                if name:
                    hints.append(str(name))
        except Exception as e:
            logger.warning(f"Could not load species hints: {e}")

        # Load numbers
        try:
            with open(os.path.join("config", "numbers.json"), "r", encoding="utf-8") as f:
                numbers_cfg = json.load(f)
            hints.extend(list(numbers_cfg.get("number_words", {}).keys()))
        except Exception as e:
            logger.warning(f"Could not load number hints: {e}")

        # Load units
        try:
            with open(os.path.join("config", "units.json"), "r", encoding="utf-8") as f:
                units_cfg = json.load(f)
            hints.extend(list(units_cfg.get("synonyms", {}).keys()))
        except Exception as e:
            logger.warning(f"Could not load unit hints: {e}")

        # Add control words
        hints.extend(["cancel", "wait", "start"])

        return hints

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice to capture audio."""
        if status:
            logger.warning(f"Audio callback status: {status}")

        if not self._stop_event.is_set() and not self._paused:
            # Convert to bytes and queue
            audio_bytes = (indata * 32767).astype(np.int16).tobytes()
            self._audio_queue.put(audio_bytes)

    def _websocket_thread(self):
        """WebSocket communication thread."""
        try:
            import websocket
            from urllib.parse import urlencode

            # Connection parameters
            params = {
                "sample_rate": self.SAMPLE_RATE,
                # Limit to 100 to stay within AAI limits
                "word_boost": json.dumps(self._phrase_hints[:100]) if self._phrase_hints else None,
            }

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            url = f"wss://streaming.assemblyai.com/v3/ws?{urlencode(params)}"

            def on_open(ws):
                logger.info("AssemblyAI WebSocket connection opened")
                self.status_changed.emit("listening")

                # Start sending audio
                def send_audio():
                    while not self._stop_event.is_set():
                        try:
                            audio_data = self._audio_queue.get(timeout=0.1)
                            if audio_data and ws.sock and ws.sock.connected:
                                ws.send(audio_data, websocket.ABNF.OPCODE_BINARY)
                        except Exception:
                            continue

                audio_thread = threading.Thread(target=send_audio, daemon=True)
                audio_thread.start()

            def on_message(ws, message):
                try:
                    data = json.loads(message)

                    # Robustly read message type and text across AAI versions
                    msg_type = data.get("type") or data.get("message_type")
                    transcript = data.get("transcript") or data.get("text") or ""

                    # Debug: log the full message structure
                    logger.debug(f"AssemblyAI message: {json.dumps(data)}")

                    # Derive partial/final across variants
                    is_partial = None
                    if "partial" in data:
                        is_partial = bool(data.get("partial"))
                    elif "final" in data:
                        # Some versions provide "final": true/false
                        is_partial = not bool(data.get("final"))

                    if msg_type in ("Begin", "SessionBegins"):
                        session_id = data.get("id") or data.get("session_id")
                        logger.info(f"AssemblyAI session started: {session_id}")
                        return

                    if msg_type in ("Termination", "SessionTerminated"):
                        logger.info("AssemblyAI session terminated")
                        self._stop_event.set()
                        return

                    # Normalize partial/final based on known message types if not explicitly provided
                    if msg_type == "Turn":
                        # Use end_of_turn to determine final vs partial
                        if "end_of_turn" in data:
                            is_partial = not bool(data.get("end_of_turn"))
                    if is_partial is None:
                        if msg_type in ("PartialTranscript", "Partial"):
                            is_partial = True
                        elif msg_type in ("Transcript", "FinalTranscript"):
                            is_partial = False

                    if transcript:
                        if is_partial:
                            logger.debug(f"Emitting partial: {transcript}")
                            #self.partial_text.emit(transcript)
                        else:
                            # Prefer end_of_turn_confidence, else general confidence, else default to 0.85 like faster-whisper
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
                logger.error(f"AssemblyAI WebSocket error: {error}")
                self.error.emit(f"AssemblyAI error: {error}")

            def on_close(ws, close_status_code, close_msg):
                logger.info(f"AssemblyAI WebSocket closed: {close_status_code} - {close_msg}")
                self.status_changed.emit("stopped")

            # Create and run WebSocket
            self._ws = websocket.WebSocketApp(
                url,
                header={"Authorization": self._api_key},
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )

            self._ws.run_forever()

        except Exception as e:
            logger.error(f"AssemblyAI WebSocket thread error: {e}")
            self.error.emit(f"AssemblyAI connection error: {e}")

    def _process_final_transcript(self, text: str, confidence: float):
        """Process final transcript and emit signals."""
        if not text or self._paused:
            return

        text = text.strip()
        logger.info(f"Raw transcription: {text}")

        # --- Check for pause/resume commands ---
        text_lower = text.lower()
        if "wait" in text_lower:
            self.pause()
            self.final_text.emit("Waiting until 'start' is said.", 0.85)
            return
        elif "start" in text_lower:
            self.resume()
            return

        # If paused, skip processing
        if self._paused:
            logger.debug("Paused: ignoring transcription")
            return

        # Apply ASR corrections and parsing
        try:
            from parser import FishParser, TextNormalizer, ParserResult
            fish_parser = FishParser()
            text_normalizer = TextNormalizer()

            # Apply fish-specific ASR corrections
            corrected_text = text_normalizer.apply_fish_asr_corrections(text)
            if corrected_text != text.lower():
                logger.info(f"After ASR corrections: {corrected_text}")

            result: ParserResult = fish_parser.parse_text(corrected_text)

            if result.species is not None:
                self._last_fish_specie = result.species
                self.specie_detected.emit(result.species)

            if result.length_cm is not None:
                # Format numeric output
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
            # Fallback to raw text
            logger.info(f">> {text}")
            self.final_text.emit(text, confidence)

    def run(self) -> None:
        """Main recognition loop."""
        try:
            self.status_changed.emit("initializing")
            logger.info("Starting AssemblyAI recognizer...")

            self._stop_event.clear()

            # Start WebSocket thread
            self._ws_thread = threading.Thread(target=self._websocket_thread, daemon=True)
            self._ws_thread.start()

            # Start audio capture
            self._stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=self.CHANNELS,
                dtype=np.float32,
                blocksize=self._chunk_frames,
                callback=self._audio_callback,
            )
            self._stream.start()

            logger.info("AssemblyAI recognizer started successfully")

            # Keep thread alive
            while not self._stop_flag and not self._stop_event.is_set():
                time.sleep(0.1)

        except Exception as e:
            logger.error(f"AssemblyAI recognizer error: {e}")
            self.error.emit(f"Recognition error: {e}")

        finally:
            self._cleanup()

    def _cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up AssemblyAI recognizer...")

        self._stop_event.set()

        # Stop audio stream
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.error(f"Error closing audio stream: {e}")
            self._stream = None

        # Close WebSocket
        if self._ws:
            try:
                # Send termination message
                terminate_msg = {"type": "Terminate"}
                self._ws.send(json.dumps(terminate_msg))
                time.sleep(0.5)
                self._ws.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
            self._ws = None

        # Wait for WebSocket thread
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=2.0)

        # Clean up session logger
        if hasattr(self, '_session_logger') and self._session_logger:
            self._session_logger.log_end()
        if hasattr(self, '_session_log_sink_id') and self._session_log_sink_id:
            import loguru
            loguru.logger.remove(self._session_log_sink_id)

        self.status_changed.emit("stopped")
        logger.info("AssemblyAI recognizer cleanup complete")

    def stop(self) -> None:
        """Stop the recognizer."""
        logger.info("Stopping AssemblyAI recognizer...")
        super().stop()
        self._stop_event.set()

    def set_noise_profile(self, name: Optional[str]) -> None:
        """Update noise profile (requires restart to take effect)."""
        self._noise_profile_name = (name or "mixed").lower()
        logger.info(f"Noise profile set to: {self._noise_profile_name} (restart required)")
