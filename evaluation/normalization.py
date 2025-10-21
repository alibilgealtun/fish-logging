from __future__ import annotations
"""Normalization utilities for ASR numeric evaluation.

This module adapts existing parsing/normalization logic (NumberParser, TextNormalizer)
into reusable helpers for the evaluation pipeline.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import re

from parser.number_parser import NumberParser
from parser.text_normalizer import TextNormalizer
from parser.config import ConfigManager
from parser.text_utils import tokenize_text


@dataclass
class NormalizationResult:
    """Result of text normalization for evaluation.

    Attributes:
        raw_text: Original recognized text
        corrected_text: Text after ASR corrections
        normalized_text: Final normalized text
        predicted_number: Extracted numeric value if any
        debug: Additional debug information
    """
    raw_text: str
    corrected_text: str
    normalized_text: str
    predicted_number: Optional[float]
    debug: Dict[str, Any] = field(default_factory=dict)


class ASRNormalizer:
    """Wraps fish logging project's normalization utilities for evaluation usage."""
    def __init__(self, config: ConfigManager | None = None):
        self.config = config or ConfigManager()
        self.number_parser = NumberParser(self.config)
        self.text_normalizer = TextNormalizer(self.config)

    @staticmethod
    def _strip_punctuation(text: str) -> str:
        return re.sub(r"[^0-9a-zA-Z. ]+", " ", text).strip()

    def normalize(self, text: str) -> NormalizationResult:
        raw = text or ""
        corrected = self.text_normalizer.apply_fish_asr_corrections(raw)
        stripped = self._strip_punctuation(corrected.lower())

        # Attempt spoken number parsing (including units) first
        value, _unit = self.number_parser.extract_number_with_units(stripped)
        debug: Dict[str, Any] = {"unit": _unit}

        # Fallback: direct numeric regex if parser failed
        if value is None:
            m = re.search(r"\d+(?:[.,]\d+)?", stripped)
            if m:
                try:
                    value = float(m.group(0).replace(",", "."))
                    debug["fallback_numeric"] = True
                except Exception:
                    value = None

        normalized_text = stripped
        return NormalizationResult(
            raw_text=raw,
            corrected_text=corrected,
            normalized_text=normalized_text,
            predicted_number=value,
            debug=debug,
        )

    # ----- Error classification helpers -----
    def classify_error(self, prediction: Optional[float], reference: Optional[float], raw_text: str, ref_text: str = "") -> str:
        """Classify the type of prediction error.

        Args:
            prediction: Predicted numeric value
            reference: Reference numeric value
            raw_text: Raw recognized text
            ref_text: Reference text

        Returns:
            Error type: 'none', 'deletion', 'insertion', 'exact', 'ordering', 'formatting', or 'substitution'
        """
        if prediction is None and reference is None:
            return "none"
        if prediction is None and reference is not None:
            return "deletion"
        if prediction is not None and reference is None:
            return "insertion"
        if prediction == reference:
            return "exact"
        # Ordering vs substitution vs formatting
        if prediction is not None and reference is not None:
            p_int = int(round(prediction * 1000))  # scale to preserve decimals ordering notion
            r_int = int(round(reference * 1000))
            if sorted(str(p_int)) == sorted(str(r_int)):
                return "ordering"
            # formatting: textual vs numeric forms present
            tokens = tokenize_text(raw_text)
            if any(t in self.config.number_words for t in tokens) and re.search(r"\d", raw_text) is None:
                return "formatting"
        return "substitution"
