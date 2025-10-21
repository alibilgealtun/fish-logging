"""
Fish Parser Package for Fish Logging Application.

This package provides comprehensive text parsing capabilities for extracting fish
species and measurement data from speech recognition text. It implements robust
parsing algorithms with fuzzy matching, text normalization, and ASR error correction
specifically designed for marine and fishing terminology.

Main Components:
    FishParser: Main parser class that coordinates all parsing operations
    SpeciesMatcher: Fuzzy matching for fish species names with taxonomic support
    NumberParser: Advanced parsing of numeric measurements and units
    TextNormalizer: ASR corrections and text normalization for marine terminology
    TextUtils: Utility functions for text processing and analysis
    ConfigManager: Configuration management from JSON files

Architecture:
    - Pipeline pattern for sequential text processing
    - Strategy pattern for different parsing approaches
    - Factory pattern for parser component creation
    - Configuration-driven behavior for flexibility

Features:
    - Fuzzy species name matching with confidence scoring
    - Multiple unit parsing (cm, inches, feet, etc.)
    - ASR error correction for common speech recognition mistakes
    - Text normalization for consistent processing
    - Configurable species database with aliases
    - Robust number extraction from natural language

Use Cases:
    - Speech-to-text fish logging applications
    - Voice-controlled fishing data entry
    - Automated fish measurement recording
    - Marine biology data collection
    - Commercial fishing log automation

Design Philosophy:
    - Robust parsing with graceful error handling
    - Fuzzy matching for real-world speech input
    - Configurable for different fishing regions
    - Performance optimized for real-time use
    - Extensible for new species and units
"""

from __future__ import annotations

# Core parser components
from .parser import FishParser, ParseResult
from .species_matcher import SpeciesMatcher
from .number_parser import NumberParser
from .text_normalizer import TextNormalizer
from .text_utils import TextUtils
from .config import ConfigManager

__all__ = [
    # Main parser interface
    "FishParser",
    "ParseResult",

    # Core parsing components
    "SpeciesMatcher",
    "NumberParser",
    "TextNormalizer",
    "TextUtils",

    # Configuration management
    "ConfigManager",
]

__version__ = "1.0.0"
__author__ = "Fish Logging Team"

# Common parsing patterns and examples
EXAMPLE_INPUTS = [
    "I caught a sea bass twenty five centimeters",
    "Got a red snapper about thirty inches long",
    "Twelve point five inch bluegill just now",
    "Large mouth bass forty two cm",
    "Twenty three centimeter yellow perch",
]

SUPPORTED_UNITS = [
    "centimeters", "cm", "centimetres",
    "inches", "inch", "in",
    "feet", "foot", "ft",
    "millimeters", "mm", "millimetres"
]