from __future__ import annotations

from typing import Optional, List, Any

from loguru import logger

from .base_recognizer import BaseSpeechRecognizer, TranscriptionSegment


class InsanelyFastWhisperRecognizer(BaseSpeechRecognizer):
    """
    Realtime recognizer using insanely-fast-whisper, leveraging shared pipeline
    from BaseSpeechRecognizer. Only backend-specific code is implemented here.
    """

    # ===== CONFIG  =====
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    CHUNK_S: float = 0.5

    VAD_MODE: int = 2
    MIN_SPEECH_S: float = 0.4
    MAX_SEGMENT_S: float = 3.0
    PADDING_MS: int = 600

    # Model specific
    MODEL_NAME: str = "base.en"
    DEVICE: str = "cpu"
    COMPUTE_TYPE: str = "int8"

    # Decoding parameters
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
        """Load insanely-fast-whisper using common API variants."""
        try:
            import insanely_fast_whisper as _ifw  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "insanely-fast-whisper is not installed. Install with `pip install insanely-fast-whisper`."
            ) from e

        # Try Transcriber class
        try:
            if hasattr(_ifw, "Transcriber"):
                self._model = _ifw.Transcriber(model=self.MODEL_NAME, device=self.DEVICE)  # type: ignore[attr-defined]
                logger.info("Loaded insanely-fast-whisper via Transcriber")
                return
        except Exception as e:
            logger.debug(f"IFW Transcriber load failed: {e}")

        # Try load_model function
        try:
            if hasattr(_ifw, "load_model"):
                self._model = _ifw.load_model(self.MODEL_NAME, device=self.DEVICE)  # type: ignore[attr-defined]
                logger.info("Loaded insanely-fast-whisper via load_model()")
                return
        except Exception as e:
            logger.debug(f"IFW load_model failed: {e}")

        # As last resort, try faster_whisper (some forks may re-export or for tests)
        try:
            from faster_whisper import WhisperModel  # type: ignore
            self._model = WhisperModel(self.MODEL_NAME, device=self.DEVICE, compute_type=self.COMPUTE_TYPE)
            logger.info("Fell back to faster_whisper WhisperModel for IFW recognizer")
            return
        except Exception as e:
            logger.debug(f"Fallback to faster_whisper failed: {e}")

        raise RuntimeError("Unsupported insanely-fast-whisper API; could not load model")

    def _backend_transcribe(self, segment, wav_path: Optional[str]) -> List[TranscriptionSegment]:
        if self._model is None:
            raise RuntimeError("insanely-fast-whisper model not loaded")

        # Try rich signature first, then minimal
        try:
            result = self._model.transcribe(  # type: ignore[attr-defined]
                wav_path,
                beam_size=self.BEAM_SIZE,
                best_of=self.BEST_OF,
                temperature=self.TEMPERATURE,
                initial_prompt=self.FISH_PROMPT,
                without_timestamps=self.WITHOUT_TIMESTAMPS,
            )
        except Exception:
            result = self._model.transcribe(wav_path)  # type: ignore[attr-defined]

        out: List[TranscriptionSegment] = []
        # Normalize typical IFW outputs: dict with 'text' and optional 'segments'
        try:
            if isinstance(result, dict):
                segs = result.get("segments")
                if isinstance(segs, list):
                    for s in segs:
                        if isinstance(s, dict):
                            text = (s.get("text") or "").strip()
                            conf = float(s.get("avg_logprob", s.get("confidence", 0.85)))
                            if text:
                                out.append(TranscriptionSegment(text=text, confidence=conf))
                        else:
                            text = getattr(s, "text", str(s)).strip()
                            conf = float(getattr(s, "confidence", 0.85))
                            if text:
                                out.append(TranscriptionSegment(text=text, confidence=conf))
                else:
                    text = (result.get("text") or "").strip()
                    if text:
                        out = [TranscriptionSegment(text=text, confidence=0.85)]
            elif isinstance(result, list):
                for s in result:
                    text = (s.get("text") if isinstance(s, dict) else getattr(s, "text", str(s))).strip()
                    if text:
                        out.append(TranscriptionSegment(text=text, confidence=0.85))
            else:
                text = str(result).strip()
                if text:
                    out = [TranscriptionSegment(text=text, confidence=0.85)]
        except Exception:
            out = [TranscriptionSegment(text=str(result).strip(), confidence=0.85)]

        return out
