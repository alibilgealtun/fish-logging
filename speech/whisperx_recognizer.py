from __future__ import annotations

from typing import Optional, List

from loguru import logger

from .base_recognizer import BaseSpeechRecognizer, TranscriptionSegment


class WhisperXRecognizer(BaseSpeechRecognizer):
    """WhisperX-based recognizer using the shared realtime pipeline."""

    # ===== CONFIG  =====
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    CHUNK_S: float = 0.5

    VAD_MODE: int = 2
    MIN_SPEECH_S: float = 0.2
    MAX_SEGMENT_S: float = 3.0
    PADDING_MS: int = 600

    # === Model specific configs ===
    MODEL_NAME: str = "base.en"
    DEVICE: str = "cpu"
    COMPUTE_TYPE: str = "int8"

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
            logger.info("Loading whisperx model... (this may download files on first run)")
            import whisperx as _whisperx  # type: ignore
            try:
                self._model = _whisperx.load_model(self.MODEL_NAME, device=self.DEVICE, compute_type=self.COMPUTE_TYPE)
            except Exception:
                # fallback to simpler signature
                self._model = _whisperx.load_model(self.MODEL_NAME, device=self.DEVICE)
        except Exception as e:
            raise RuntimeError(f"Failed to load whisperx model: {e}")

    def _backend_transcribe(self, segment, wav_path: Optional[str]) -> List[TranscriptionSegment]:
        if self._model is None:
            raise RuntimeError("whisperx model not loaded")
        if not wav_path:
            raise RuntimeError("wav_path is required for whisperx backend")

        # Invoke whisperx, handling API variations
        try:
            transcription_result = self._model.transcribe(
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
        except Exception:
            transcription_result = self._model.transcribe(wav_path)

        out: List[TranscriptionSegment] = []
        # Normalize to segments list similar to faster-whisper
        try:
            if isinstance(transcription_result, dict):
                raw_segments = transcription_result.get("segments", None)
                if raw_segments is None and "text" in transcription_result:
                    text_out = (transcription_result.get("text") or "").strip()
                    if text_out:
                        out = [TranscriptionSegment(text=text_out, confidence=0.85)]
                else:
                    for seg in (raw_segments or []):
                        if isinstance(seg, dict):
                            text = (seg.get("text") or seg.get("sentence") or "").strip()
                            conf = float(seg.get("confidence", seg.get("score", 0.85)))
                            if text:
                                out.append(TranscriptionSegment(text=text, confidence=conf))
                        else:
                            text = getattr(seg, "text", str(seg)).strip()
                            conf = float(getattr(seg, "confidence", 0.85))
                            if text:
                                out.append(TranscriptionSegment(text=text, confidence=conf))
            elif isinstance(transcription_result, list):
                for seg in transcription_result:
                    if isinstance(seg, dict):
                        text = (seg.get("text") or "").strip()
                        conf = float(seg.get("confidence", 0.85))
                        if text:
                            out.append(TranscriptionSegment(text=text, confidence=conf))
                    else:
                        text = getattr(seg, "text", str(seg)).strip()
                        conf = float(getattr(seg, "confidence", 0.85))
                        if text:
                            out.append(TranscriptionSegment(text=text, confidence=conf))
            else:
                text = str(transcription_result).strip()
                if text:
                    out = [TranscriptionSegment(text=text, confidence=0.85)]
        except Exception:
            # Fallback best-effort
            out = [TranscriptionSegment(text=str(transcription_result).strip(), confidence=0.85)]

        return out
