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

        # Normalize whitespace
        text_norm = text.strip()

        # Check for cancel command
        if self._cancel_pattern.search(text_norm):
            return ParserResult(cancel=True, species=None, length_cm=None)

        # Extract measurements
        length_val, unit = self.number_parser.extract_number_with_units(text_norm)

        # If a number was found but no unit was specified, assume the default unit 'cm'.
        if length_val is not None and unit is None:
            unit = "cm"

        # Finalize the length in cm if the unit is correct
        if length_val is not None and unit == "cm":
            length_cm = float(length_val)

        # Extract species
        species = self.species_matcher.fuzzy_match_species(text_norm)
        if species:
            species = species.title()

        # Return complete result if both species and measurement found
        if length_val is not None and unit == "cm" and species is not None:
            return ParserResult(cancel=False, species=species, length_cm=float(length_val))

        return ParserResult(cancel=False, species=species, length_cm=length_val)
