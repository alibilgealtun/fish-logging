from __future__ import annotations

from typing import Optional, Any, List

from loguru import logger

from .base_recognizer import BaseSpeechRecognizer, TranscriptionSegment


class WhisperRecognizer(BaseSpeechRecognizer):
    """
    Realtime CPU-only speech recognizer using NoiseController + faster-whisper.
    Implements only backend-specific logic; pipeline handled by BaseSpeechRecognizer.
    """

    # ===== CONFIG  =====
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    CHUNK_S: float = 0.5

    # === Model specific configs ===
    MODEL_NAME: str = "base.en"
    DEVICE: str = "cpu"
    COMPUTE_TYPE: str = "int8"

    # Noise controller settings
    VAD_MODE: int = 2
    MIN_SPEECH_S: float = 0.2
    MAX_SEGMENT_S: float = 3.0
    PADDING_MS: int = 600

    # === Decoding parameters ===
    BEAM_SIZE: int = 3
    BEST_OF: int = 5
    TEMPERATURE: float = 0.0
    PATIENCE: float = 1.0
    LENGTH_PENALTY: float = 1.0
    REPETITION_PENALTY: float = 1.0
    WITHOUT_TIMESTAMPS: bool = True
    CONDITION_ON_PREVIOUS_TEXT: bool = True
    VAD_FILTER: bool = False
    VAD_PARAMETERS: Optional[dict] = None
    WORD_TIMESTAMPS: bool = False

    def _load_backend_model(self) -> None:
        try:
            logger.info("Loading faster-whisper model... (first run may download it)")
            from faster_whisper import WhisperModel  # type: ignore
            self._model = WhisperModel(
                self.MODEL_NAME,
                device=self.DEVICE,
                compute_type=self.COMPUTE_TYPE,
                download_root=None,
                local_files_only=False,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}")

    def _backend_transcribe(self, segment, wav_path: Optional[str]) -> List[TranscriptionSegment]:
        if self._model is None:
            raise RuntimeError("Whisper model not loaded")
        if not wav_path:
            raise RuntimeError("wav_path is required for faster-whisper backend")

        # Import type only for static linters; runtime already loaded
        segments, info = self._model.transcribe(
            wav_path,
            beam_size=self.BEAM_SIZE,
            best_of=self.BEST_OF,
            temperature=self.TEMPERATURE,
            patience=self.PATIENCE,
            length_penalty=self.LENGTH_PENALTY,
            repetition_penalty=self.REPETITION_PENALTY,
            language="en",
            condition_on_previous_text=self.CONDITION_ON_PREVIOUS_TEXT,
            initial_prompt=self.FISH_PROMPT,
            vad_filter=self.VAD_FILTER,
            vad_parameters=self.VAD_PARAMETERS,
            without_timestamps=self.WITHOUT_TIMESTAMPS,
            word_timestamps=self.WORD_TIMESTAMPS,
        )

        out: List[TranscriptionSegment] = []
        try:
            for s in segments:
                text = getattr(s, "text", "")
                if text:
                    out.append(TranscriptionSegment(text=text.strip(), confidence=0.85))
        except Exception:
            # Best-effort fallback
            out = [TranscriptionSegment(text=str(segments).strip(), confidence=0.85)]
        return out
