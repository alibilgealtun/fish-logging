"""
Fish measurement parser with ASR correction capabilities.

This module provides functionality to parse fish species and measurements
from potentially noisy ASR (Automatic Speech Recognition) text input.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, List

from .config import ConfigManager
from .species_matcher import SpeciesMatcher
from .number_parser import NumberParser
from .text_normalizer import TextNormalizer


@dataclass
class ParserResult:
    """Result of parsing fish measurement text."""
    cancel: bool
    species: Optional[str]
    length_cm: Optional[float]


class FishParser:
    """Main parser for fish species and measurements from ASR text."""

    def __init__(self, config: ConfigManager = None):
        """Initialize parser with configuration."""
        if config is None:
            config = ConfigManager()
        self.config = config
        self.species_matcher = SpeciesMatcher(self.config)
        self.number_parser = NumberParser(self.config)
        self.text_normalizer = TextNormalizer(self.config)

        # Compile regex patterns once
        self._cancel_pattern = re.compile(r"\bcancel\b", re.IGNORECASE)

    def parse_text(self, text: str) -> ParserResult:
        """
        Parse text to extract fish species and length measurements.

        Args:
            text: Raw input text (potentially from ASR)

        Returns:
            ParserResult containing parsed information
        """
        if not text:
            return ParserResult(cancel=False, species=None, length_cm=None)

        text_norm = text.strip()

        # Check for cancel command
        if self._is_cancel_command(text_norm):
            return ParserResult(cancel=True, species=None, length_cm=None)

        # Extract species and length
        species = self._extract_species(text_norm)
        length_cm = self._extract_length(text_norm)

        return ParserResult(cancel=False, species=species, length_cm=length_cm)

    def _is_cancel_command(self, text: str) -> bool:
        """Check if text contains a cancel command.

        Args:
            text: Normalized text to check

        Returns:
            True if cancel command detected
        """
        return bool(self._cancel_pattern.search(text))

    def _extract_species(self, text: str) -> Optional[str]:
        """Extract and normalize species name from text.

        Args:
            text: Text to extract species from

        Returns:
            Normalized species name or None
        """
        species = self.species_matcher.fuzzy_match_species(text)
        if species:
            return species.title()
        return None

    def _extract_length(self, text: str) -> Optional[float]:
        """Extract length measurement from text and convert to centimeters.

        Args:
            text: Text to extract length from

        Returns:
            Length in centimeters or None
        """
        length_val, unit = self.number_parser.extract_number_with_units(text)

        if length_val is not None:
            # If no unit specified, assume cm (default)
            unit = unit or "cm"
            # Only return if unit is cm (already converted by parser)
            if unit == "cm":
                return float(length_val)

        return None
