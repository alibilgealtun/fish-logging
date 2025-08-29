"""
Fish Parser - A robust parser for fish species and measurements from ASR text.

This package provides functionality to parse fish species names and length
measurements from potentially noisy Automatic Speech Recognition (ASR) text input.

Main Components:
- FishParser: Main parser class that coordinates all parsing operations
- SpeciesMatcher: Fuzzy matching for fish species names
- NumberParser: Parsing of numeric measurements and units
- TextNormalizer: ASR corrections and text normalization
- ConfigManager: Configuration management from JSON files

Usage:
    from parser import FishParser

    parser = FishParser()
    result = parser.parse_text("I caught a sea bass twenty five centimeters")
    print(f"Species: {result.species}, Length: {result.length_cm}cm")
"""

from .parser import FishParser, ParserResult
from .config import ConfigManager
from .species_matcher import SpeciesMatcher
from .number_parser import NumberParser
from .text_normalizer import TextNormalizer
from .text_utils import tokenize_text

__all__ = [
    "FishParser",
    "ParserResult",
    "ConfigManager",
    "SpeciesMatcher",
    "NumberParser",
    "TextNormalizer",
    "tokenize_text",
]