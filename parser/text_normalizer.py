"""Text normalization and ASR correction functionality."""

import re

from .config import ConfigManager


class TextNormalizer:
    """Handles text normalization and ASR corrections."""

    def __init__(self, config: ConfigManager = ConfigManager("config/")) -> None:
        """Initialize text normalizer with configuration."""
        self.config = config
        self._compile_correction_patterns()

    def _compile_correction_patterns(self) -> None:
        """Pre-compile regex patterns for efficiency."""
        self._ordinal_pattern = re.compile(r"(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)
        self._see_em_pattern = re.compile(r"\b(see[ -]?em|seeem|cem|c m|c- m)\b", re.IGNORECASE)
        self._centi_space_pattern = re.compile(r"\bcenti\s*meters\b", re.IGNORECASE)
        self._centi_metre_pattern = re.compile(r"\bcenti\s*metres\b", re.IGNORECASE)
        self._numeric_unit_pattern = re.compile(
            r"\b(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|cm|centi\w*|mm|milli\w*)\b"
        )

    def _apply_unit_and_number_corrections(self, text: str) -> str:
        """Apply targeted ASR corrections for units and numbers."""
        corrected_text = text

        # Common pattern: pre-centimeters -> three centimeters
        corrections = [
            (
            r"\b(?:pre|pre-|tree|tri|tre|free|thre)\s*[-]?\s*(?:centi(?:metre|meter|metres|meters)|centimeters?|centimetres?)\b",
            "three centimeters"),
            (r"\b(?:tree|tri|tre|free|thre)\s+(?:centi(?:metre|meter|metres|meters)|centimeters?|centimetres?)\b",
             "three centimeters"),
        ]

        for pattern, replacement in corrections:
            corrected_text = re.sub(pattern, replacement, corrected_text, flags=re.IGNORECASE)

        # Apply specific pattern corrections
        corrected_text = self._see_em_pattern.sub("cm", corrected_text)
        corrected_text = self._centi_space_pattern.sub("centimeters", corrected_text)
        corrected_text = self._centi_metre_pattern.sub("centimetres", corrected_text)
        corrected_text = self._ordinal_pattern.sub(r"\1", corrected_text)

        return corrected_text

    def apply_fish_asr_corrections(self, text: str) -> str:
        """
        Apply targeted ASR corrections for fish-related utterances.

        This normalizes common mis-hearings for species names while keeping the
        rest of the phrase intact. Corrections are applied conservatively, mostly
        when there is evidence of a measurement in the same utterance.

        Args:
            text: Raw ASR text

        Returns:
            Normalized text suitable for parsing
        """
        if not text:
            return ""

        corrected_text = text.strip().lower()

        # Check if utterance contains numeric/unit context
        has_numeric_context = bool(self._numeric_unit_pattern.search(corrected_text))

        # Apply species corrections only with numeric context
        if has_numeric_context:
            species_corrections = self.config.species_corrections
            for pattern, replacement in species_corrections.items():
                corrected_text = re.sub(pattern, replacement, corrected_text, flags=re.IGNORECASE)

        # Apply unit/number corrections
        corrected_text = self._apply_unit_and_number_corrections(corrected_text)

        return corrected_text