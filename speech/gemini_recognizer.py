"""Google Gemini 2.5 Pro Speech Recognition Integration."""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Optional, List
from queue import Queue
import tempfile

import numpy as np
import sounddevice as sd
import soundfile as sf
from loguru import logger
from PyQt6.QtCore import pyqtSignal

from .base_recognizer import BaseSpeechRecognizer
from noise.controller import NoiseController
from services import get_audio_saver


class GeminiRecognizer(BaseSpeechRecognizer):
    """
    Real-time speech recognizer using Google Gemini 2.5 Pro multimodal API.
    Supports audio understanding with advanced context awareness.
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
    CHUNK_S: float = 0.5

    # Noise controller settings
    VAD_MODE: int = 2
    MIN_SPEECH_S: float = 0.4
    MAX_SEGMENT_S: float = 3.0
    PADDING_MS: int = 600

    def __init__(
        self,
        language: str = "en",
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-pro",
        noise_profile: Optional[str] = None
    ) -> None:
        """Initialize Gemini recognizer.

        Args:
            language: Language code (e.g., "en", "es").
            api_key: Google AI API key. If None, reads from GEMINI_API_KEY env var.
            model: Gemini model name (default: gemini-2.5-pro).
            noise_profile: Noise profile name (clean|human|engine|mixed).
        """
        super().__init__(language=f"{language}-US" if language == "en" else language)

        self._noise_profile_name = (noise_profile or "mixed").lower()
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        self._model_name = model

        if not self._api_key:
            raise ValueError(
                "Gemini API key not provided. Set GEMINI_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self._client = None
        self._model = None
        self._chunk_frames = int(self.SAMPLE_RATE * self.CHUNK_S)
        self._stream = None
        self._stop_event = threading.Event()
        self._audio_buffer: List[np.ndarray] = []
        self._buffer_lock = threading.Lock()
        self._processing_thread: Optional[threading.Thread] = None

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

        # Build context prompt
        self._context_prompt = self._build_context_prompt()

        logger.info(
            f"Gemini recognizer initialized with model={self._model_name}, "
            f"noise_profile={self._noise_profile_name}"
        )

    def _build_context_prompt(self) -> str:
        """Build context prompt with fish species vocabulary."""
        context = self.FISH_PROMPT + "\n\nKnown fish species:\n"

        try:
            with open(os.path.join("config", "species.json"), "r", encoding="utf-8") as f:
                species_cfg = json.load(f)

            species_list = [item.get("name") for item in species_cfg.get("items", []) if item.get("name")]
            if species_list:
                context += ", ".join(species_list[:50])  # Limit to first 50 species
        except Exception as e:
            logger.warning(f"Could not load species for context: {e}")

        context += "\n\nTranscribe the audio accurately, focusing on fish species names and measurements."
        return context

    def _init_client(self):
        """Initialize Gemini client."""
        if self._client is not None:
            return

        try:
            import google.generativeai as genai
            from google.generativeai.types import GenerationConfig  # type: ignore

            genai.configure(api_key=self._api_key)

            # Configure generation settings for speech transcription
            generation_config = GenerationConfig(
                temperature=0.1,  # Low temperature for accurate transcription
                top_p=0.95,
                top_k=40,
                max_output_tokens=256,
            )

            self._model = genai.GenerativeModel(
                model_name=self._model_name,
                generation_config=generation_config,
            )

            self._client = genai
            logger.info(f"Gemini client initialized with model: {self._model_name}")

        except Exception as e:
            msg = f"Failed to initialize Gemini client: {e}"
            logger.error(msg)
            self.error.emit(msg)
            raise

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice to capture audio."""
        if status:
            logger.warning(f"Audio callback status: {status}")

        if not self._stop_event.is_set() and not self._paused:
            with self._buffer_lock:
                self._audio_buffer.append(indata.copy())

    def _processing_loop(self):
        """Process audio segments using Gemini."""
        last_process_time = time.time()

        while not self._stop_event.is_set():
            try:
                # Wait for enough audio data
                time.sleep(0.1)

                # Check if we should process (every MAX_SEGMENT_S seconds)
                if time.time() - last_process_time < self.MAX_SEGMENT_S:
                    continue

                # Get audio buffer
                with self._buffer_lock:
                    if not self._audio_buffer:
                        continue

                    audio_data = np.concatenate(self._audio_buffer, axis=0)
                    self._audio_buffer.clear()

                if len(audio_data) < self.SAMPLE_RATE * self.MIN_SPEECH_S:
                    continue

                # Process with noise controller
                segments = self._noise_controller.process_audio(audio_data)

                for segment in segments:
                    if self._stop_event.is_set() or self._paused:
                        break

                    # Transcribe segment with Gemini
                    self._transcribe_segment(segment)

                last_process_time = time.time()

            except Exception as e:
                logger.error(f"Error in Gemini processing loop: {e}")
                time.sleep(0.5)

    def _transcribe_segment(self, audio_data: np.ndarray):
        """Transcribe audio segment using Gemini."""
        try:
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name
                sf.write(tmp_path, audio_data, self.SAMPLE_RATE)

            try:
                # Upload audio file
                audio_file = self._client.upload_file(tmp_path)

                # Create prompt with audio
                prompt = f"{self._context_prompt}\n\nAudio file to transcribe:"

                # Generate transcription
                response = self._model.generate_content([prompt, audio_file])

                if response.text:
                    transcript = response.text.strip()

                    # Emit partial during processing
                    self.partial_text.emit(transcript)

                    # Process as final transcript
                    confidence = 0.90  # Gemini doesn't provide confidence scores
                    self._process_final_transcript(transcript, confidence, audio_data)

                # Delete the uploaded file
                try:
                    self._client.delete_file(audio_file.name)
                except Exception:
                    pass

            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Error transcribing segment with Gemini: {e}")
            # Don't emit error to avoid spamming, just log

    def _process_final_transcript(self, text: str, confidence: float, audio_data: np.ndarray):
        """Process final transcript and emit signals."""
        if not text or self._paused:
            return

        text = text.strip()

        # Save audio if configured
        audio_saver = get_audio_saver()
        if audio_saver and audio_saver.should_save():
            try:
                audio_saver.save_segment(
                    audio_data=audio_data,
                    sample_rate=self.SAMPLE_RATE
                )
            except Exception as e:
                logger.error(f"Error saving audio: {e}")

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

        # Apply ASR corrections and parsing like faster-whisper
        try:
            from parser import FishParser, TextNormalizer, ParserResult
            fish_parser = FishParser()
            text_normalizer = TextNormalizer()

            corrected_text = text_normalizer.apply_fish_asr_corrections(text)
            if corrected_text != text.lower():
                logger.info(f"After ASR corrections: {corrected_text}")

            result: ParserResult = fish_parser.parse_text(corrected_text)

            if result.species is not None:
                self._last_fish_specie = result.species
                self.specie_detected.emit(result.species)

            if result.length_cm is not None:
                raw_val = float(result.length_cm)
                num_str = (f"{raw_val:.1f}").rstrip("0").rstrip(".")
                formatted = f"{self._last_fish_specie} {num_str} cm"
                logger.info(f">> {formatted}")
                self.final_text.emit(formatted, 0.85)
            else:
                logger.info(f">> {corrected_text} (partial parse)")
                self.final_text.emit(corrected_text, 0.85)
        except Exception as e:
            logger.error(f"Parser error: {e}")
            logger.info(f">> {text}")
            self.final_text.emit(text, 0.85)

    def run(self) -> None:
        """Main recognition loop."""
        try:
            self.status_changed.emit("initializing")
            logger.info("Starting Gemini recognizer...")

            # Initialize Gemini client
            self._init_client()

            self._stop_event.clear()
            self._audio_buffer.clear()

            # Start processing thread
            self._processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self._processing_thread.start()

            # Start audio capture
            self._stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=self.CHANNELS,
                dtype=np.float32,
                blocksize=self._chunk_frames,
                callback=self._audio_callback,
            )
            self._stream.start()

            self.status_changed.emit("listening")
            logger.info("Gemini recognizer started successfully")

            # Keep thread alive
            while not self._stop_flag and not self._stop_event.is_set():
                time.sleep(0.1)

        except Exception as e:
            logger.error(f"Gemini recognizer error: {e}")
            self.error.emit(f"Recognition error: {e}")

        finally:
            self._cleanup()

    def _cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up Gemini recognizer...")

        self._stop_event.set()

        # Stop audio stream
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.error(f"Error closing audio stream: {e}")
            self._stream = None

        # Wait for processing thread
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=2.0)

        self.status_changed.emit("stopped")
        logger.info("Gemini recognizer cleanup complete")

    def stop(self) -> None:
        """Stop the recognizer."""
        logger.info("Stopping Gemini recognizer...")
        super().stop()
        self._stop_event.set()

    def set_noise_profile(self, name: Optional[str]) -> None:
        """Update noise profile (requires restart to take effect)."""
        self._noise_profile_name = (name or "mixed").lower()
        logger.info(f"Noise profile set to: {self._noise_profile_name} (restart required)")
