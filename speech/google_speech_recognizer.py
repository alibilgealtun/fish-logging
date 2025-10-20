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
    # Reuse same signal names for consistency (redundant but explicit)
    partial_text = pyqtSignal(str)
    final_text = pyqtSignal(str, float)
    error = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    specie_detected = pyqtSignal(str)

    # ===== CONFIG (aligned with Whisper recognizer defaults) =====
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    CHUNK_S: float = 0.5

    # Noise controller settings
    VAD_MODE: int = 2
    MIN_SPEECH_S: float = 0.4
    MAX_SEGMENT_S: float = 3.0
    PADDING_MS: int = 600

    def __init__(self, language: str = "en-US", credentials_path: Optional[str] = None, numbers_only: bool = False, noise_profile: Optional[str] = None):
        """Google Cloud Speech-to-Text recognizer with optional noise profile.

        noise_profile: Optional name (clean|human|engine|mixed) controlling VAD/segment/suppression.
        """
        super().__init__(language=language)
        self.numbers_only = bool(numbers_only)
        self._noise_profile_name = (noise_profile or "mixed").lower()
        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        elif not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            candidate = os.path.join(os.getcwd(), "google.json")
            if os.path.exists(candidate) and self._is_valid_gcp_credentials_file(candidate):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = candidate
                logger.info("Using credentials at project root: google.json")
        self._client = None
        self._stream = None  # type: ignore
        self._chunk_frames: int = int(self.SAMPLE_RATE * self.CHUNK_S)
        self._last_status_msg: Optional[str] = None
        # Number prefix
        self._number_sound = self._load_number_prefix()
        # Apply noise profile overrides prior to creating controller
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
        self._phrase_hints = self._build_phrase_hints(numbers_only=self.numbers_only)

    @staticmethod
    def _is_valid_gcp_credentials_file(path: str) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cred_type = data.get("type")
            return cred_type in {
                "authorized_user",
                "service_account",
                "external_account",
                "external_account_authorized_user",
                "impersonated_service_account",
                "gdch_service_account",
            }
        except Exception:
            return False

    def _get_client(self):
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
        """Collect hints. If numbers_only, restrict to numbers/units and a few control words."""
        hints: List[str] = []
        try:
            import json as _json
            with open(os.path.join("config", "numbers.json"), "r", encoding="utf-8") as f:
                numbers_cfg = _json.load(f)
            number_words = list(numbers_cfg.get("number_words", {}).keys())
        except Exception:
            number_words = [
                "zero","one","two","three","four","five","six","seven","eight","nine",
                "ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen",
                "eighteen","nineteen","twenty","thirty","forty","fifty","sixty","seventy",
                "eighty","ninety","hundred","point","dot","comma"
            ]
        try:
            import json as _json
            with open(os.path.join("config", "units.json"), "r", encoding="utf-8") as f:
                units_cfg = _json.load(f)
            unit_words = list(units_cfg.get("synonyms", {}).keys())
        except Exception:
            unit_words = ["cm","centimeter","centimeters","mm","millimeter","millimeters"]

        if numbers_only:
            hints.extend(number_words)
            hints.extend(unit_words)
            hints.extend(["point", "dot", "comma"])  # decimals
            hints.extend(["wait", "start"])  # app control words
        else:
            # Include species as well for general mode
            try:
                import json as _json
                with open(os.path.join("config", "species.json"), "r", encoding="utf-8") as f:
                    species_cfg = _json.load(f)
                for item in species_cfg.get("items", []):
                    name = item.get("name")
                    if name:
                        hints.append(str(name))
            except Exception:
                pass
            hints.extend(number_words)
            hints.extend(unit_words)
            hints.extend(["wait", "start", "cancel"])  # app commands

        # Deduplicate and cap size
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
            if len(uniq) >= 500:
                break
        return uniq

    # -------- File/batch synchronous API (kept) --------
    def transcribe_file(self, file_path: str) -> Dict[str, Any]:
        """
        Transcribe a single audio file using Google Cloud Speech-to-Text.
        """
        from google.cloud import speech as speech

        with open(file_path, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)

        # Build phrase hints with strong boosting for numbers-only mode
        contexts = []
        if self._phrase_hints:
            try:
                contexts = [speech.SpeechContext(phrases=self._phrase_hints, boost=20.0)]
            except Exception:
                contexts = [speech.SpeechContext(phrases=self._phrase_hints)]

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.SAMPLE_RATE,
            language_code=self.language,
            enable_automatic_punctuation=not self.numbers_only,
            model="latest_short",  # optimized for short commands
            speech_contexts=contexts,
        )

        response = self._get_client().recognize(config=config, audio=audio)

        transcripts = [result.alternatives[0].transcript for result in response.results]
        full_text = " ".join(transcripts).strip()

        # If numbers_only, extract only digits/decimals
        if self.numbers_only:
            import re
            matches = re.findall(r"\d+(?:[.,]\d+)?", full_text)
            if matches:
                # Join numbers if multiple (e.g., "twenty five" might get split)
                full_text = " ".join(matches)
            else:
                # Optionally keep raw text if no number found
                full_text = full_text

        return {
            "text": full_text,
            "raw_response": response,
        }

    def transcribe_batch(self, files: List[str]) -> List[Dict[str, Any]]:
        """
        Transcribe multiple files and return a list of dicts.
        """
        results = []
        for file_path in files:
            results.append({
                "file": file_path,
                **self.transcribe_file(file_path)
            })
        return results

    # -------- Realtime support (mirrors Whisper flow) --------
    def set_last_species(self, species: str) -> None:
        try:
            self._last_fish_specie = str(species) if species else None
        except Exception:
            self._last_fish_specie = None

    def _load_number_prefix(self) -> np.ndarray:
        """Load a short number-beep/prompt to prepend before segments.
        Returns PCM16 mono numpy array.
        """
        candidates = [
            os.path.join(os.getcwd(), "assets/audio/number.wav"),
        ]
        for path in candidates:
            try:
                if os.path.exists(path):
                    data, sr = sf.read(path, dtype='int16')
                    if sr != self.SAMPLE_RATE:
                        # Resample quickly using numpy (nearest) – good enough for a tiny prefix
                        ratio = self.SAMPLE_RATE / sr
                        idx = (np.arange(int(len(data) * ratio)) / ratio).astype(int)
                        data = data[idx]
                    if data.ndim > 1:
                        data = data[:, 0]
                    return data.astype(np.int16)
            except Exception as e:
                logger.debug(f"Failed to load number prefix {path}: {e}")
        # Fallback to short silence
        return (np.zeros(int(self.SAMPLE_RATE * 0.05))).astype(np.int16)

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
                logger.error(f"Failed to emit status_changed message: {message}")

    def begin(self) -> None:
        self._stop_flag = False
        self._last_status_msg = None
        # Rebuild controller with (possibly re-applied) profile
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
        """Run the realtime loop: mic -> noise/vad -> Google STT -> parse -> UI."""
        # Open microphone stream
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
                    segment = next(segment_generator)
                    if segment is None or segment.size == 0:
                        continue

                    # Enforce min duration (already applied but keep safe)
                    if segment.size / self.SAMPLE_RATE < self.MIN_SPEECH_S:
                        continue

                    self._emit_status_once("processing")

                    # Prepend number prefix and send raw PCM to Google
                    #combined = np.concatenate((self._number_sound, segment)).astype(np.int16)
                    audio = speech.RecognitionAudio(content=segment.tobytes())

                    try:
                        response = self._get_client().recognize(config=base_config, audio=audio)
                    except Exception as e:
                        msg = f"Google STT error: {e}"
                        logger.error(msg)
                        self.error.emit(msg)
                        self._emit_status_once("listening")
                        continue

                    if not response.results:
                        self._emit_status_once("listening")
                        continue

                    # Use the first alternative of the first result as the most likely
                    first_alt = response.results[0].alternatives[0]
                    text_out = first_alt.transcript.strip()
                    confidence = float(getattr(first_alt, "confidence", 0.85) or 0.85)

                    if not text_out:
                        self._emit_status_once("listening")
                        continue

                    logger.info(f"Raw transcription: {text_out}")

                    # Handle control commands
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

                    if self._paused:
                        logger.debug("Paused: ignoring transcription")
                        self._emit_status_once("paused")
                        continue

                    # Normalize and parse for species + length
                    try:
                        from parser import FishParser, TextNormalizer
                        fish_parser = FishParser()
                        text_normalizer = TextNormalizer()
                        corrected_text = text_normalizer.apply_fish_asr_corrections(text_out)
                        result = fish_parser.parse_text(corrected_text)

                        if self.numbers_only:
                            if result.length_cm is None:
                                # Nothing numeric detected; skip
                                self._emit_status_once("listening")
                                continue
                            raw_val = float(result.length_cm)
                            num_str = (f"{raw_val:.1f}").rstrip("0").rstrip(".")
                            self.final_text.emit(num_str, confidence)
                        else:
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
                        # Fallback
                        if self.numbers_only:
                            # Extract any digits as a last resort
                            import re
                            m = re.search(r"\d+(?:[.,]\d+)?", text_out)
                            if m:
                                val = m.group(0).replace(",", ".")
                                self.final_text.emit(val, confidence)
                        else:
                            self.final_text.emit(text_out, confidence)

                    self._emit_status_once("listening")

                except StopIteration:
                    break
                except Exception as e:
                    logger.error(f"Main loop error: {e}")
                    self.error.emit(f"Processing error: {e}")
                    continue

        self._emit_status_once("stopped")
        logger.info("Google speech recognizer stopped")

    def get_config(self) -> Dict[str, Any]:
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
