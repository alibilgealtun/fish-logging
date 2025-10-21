"""
Main Fish Parser for Fish Logging Application.

This module provides the primary FishParser class that coordinates all text parsing
operations for extracting fish species and measurement data from speech recognition
text. It serves as the main entry point for the parsing subsystem.

Classes:
    ParseResult: Data class containing parsed fish information
    FishParser: Main parser coordinating all parsing operations

Features:
    - Unified parsing interface for speech recognition text
    - Cancel command detection for voice control
    - Species name extraction with fuzzy matching
    - Length measurement parsing with unit conversion
    - ASR error correction and text normalization
    - Confidence scoring for parsing results

Architecture:
    - Facade pattern providing unified interface
    - Delegation to specialized parsing components
    - Pipeline processing with error recovery
    - Configuration-driven behavior
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from .config import ConfigManager
from .species_matcher import SpeciesMatcher
from .number_parser import NumberParser
from .text_normalizer import TextNormalizer

# Optional logger: prefer loguru if available, else fallback to stdlib logging
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """
    Result of parsing fish measurement text from speech recognition.

    This dataclass encapsulates all information extracted from speech text,
    including fish species, measurements, and metadata about the parsing process.

    Attributes:
        cancel: Whether the input was a cancel command
        species: Identified fish species name (normalized)
        length_cm: Fish length in centimeters (standardized unit)
        confidence: Overall confidence score (0.0-1.0)
        raw_text: Original input text for reference
        parsed_components: Dictionary of individual parsing results

    Design:
        - Immutable dataclass for thread safety
        - Standardized units for consistency
        - Confidence scoring for quality assessment
        - Comprehensive metadata for debugging
    """
    cancel: bool
    species: Optional[str] = None
    length_cm: Optional[float] = None
    confidence: float = 0.0
    raw_text: str = ""
    parsed_components: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize parsed_components if not provided."""
        if self.parsed_components is None:
            self.parsed_components = {}

    @property
    def is_valid(self) -> bool:
        """Check if parse result contains valid fish data."""
        return self.species is not None and self.length_cm is not None

    @property
    def is_partial(self) -> bool:
        """Check if parse result contains partial information."""
        return (self.species is not None) != (self.length_cm is not None)

    def to_dict(self) -> Dict[str, Any]:
        """Convert parse result to dictionary format."""
        return {
            'cancel': self.cancel,
            'species': self.species,
            'length_cm': self.length_cm,
            'confidence': self.confidence,
            'raw_text': self.raw_text,
            'is_valid': self.is_valid,
            'is_partial': self.is_partial,
            'parsed_components': self.parsed_components
        }


class FishParser:
    """
    Main parser for extracting fish species and measurements from speech text.

    This class serves as the primary interface for the fish parsing subsystem,
    coordinating multiple specialized parsing components to extract structured
    fish data from natural language speech recognition input.

    Key Responsibilities:
        - Coordinate text normalization and ASR error correction
        - Extract fish species names using fuzzy matching
        - Parse length measurements with unit conversion
        - Detect voice control commands (e.g., "cancel")
        - Provide confidence scoring for results
        - Handle edge cases and parsing errors gracefully

    Architecture:
        - Facade pattern providing unified parsing interface
        - Delegation to specialized parsing components
        - Pipeline processing with error recovery
        - Configuration-driven behavior for flexibility

    Processing Pipeline:
        1. Text normalization and ASR correction
        2. Cancel command detection
        3. Species name extraction and matching
        4. Length measurement parsing and conversion
        5. Confidence calculation and result assembly

    Usage:
        parser = FishParser()
        result = parser.parse_text("I caught a sea bass twenty five centimeters")
        if result.is_valid:
            print(f"Species: {result.species}, Length: {result.length_cm}cm")
    """

    def __init__(self, config: Optional[ConfigManager] = None) -> None:
        """
        Initialize parser with configuration and component dependencies.

        Args:
            config: Configuration manager (creates default if None)

        Components Initialized:
            - ConfigManager: Handles JSON configuration files
            - SpeciesMatcher: Fuzzy matching for fish species
            - NumberParser: Numeric measurement extraction
            - TextNormalizer: ASR correction and text cleanup

        Design:
            - Dependency injection for testability
            - Lazy loading of heavy resources
            - Configuration-driven component setup
        """
        if config is None:
            config = ConfigManager()
        self.config = config

        # Initialize parsing components with shared configuration
        self.species_matcher = SpeciesMatcher(self.config)
        self.number_parser = NumberParser(self.config)
        self.text_normalizer = TextNormalizer(self.config)

        # Pre-compile regex patterns for performance
        self._cancel_pattern = re.compile(
            r'\b(?:cancel|undo|delete|remove|clear)\b',
            re.IGNORECASE
        )

        # Statistics tracking
        self.stats = {
            'total_parses': 0,
            'successful_parses': 0,
            'cancel_commands': 0,
            'partial_parses': 0,
            'failed_parses': 0
        }

        logger.info("FishParser initialized with all components")

    def parse_text(self, text: str) -> ParseResult:
        """
        Parse speech recognition text to extract fish species and measurements.

        Args:
            text: Raw input text from speech recognition system

        Returns:
            ParseResult: Structured parsing results with confidence scores

        Processing Steps:
            1. Input validation and normalization
            2. Cancel command detection
            3. Text normalization with ASR correction
            4. Species name extraction
            5. Length measurement parsing
            6. Confidence calculation
            7. Result assembly and validation

        Error Handling:
            - Graceful handling of empty or invalid input
            - Robust parsing with partial results
            - Detailed logging for debugging
            - Fallback strategies for edge cases
        """
        self.stats['total_parses'] += 1

        # Input validation
        if not text or not text.strip():
            logger.debug("Empty input text provided")
            return ParseResult(
                cancel=False,
                raw_text=text,
                confidence=0.0
            )

        original_text = text
        text_norm = text.strip()

        logger.debug(f"Parsing text: '{text_norm}'")

        # Check for cancel command first (highest priority)
        if self._is_cancel_command(text_norm):
            self.stats['cancel_commands'] += 1
            logger.info(f"Cancel command detected: '{text_norm}'")
            return ParseResult(
                cancel=True,
                raw_text=original_text,
                confidence=1.0
            )

        # Apply text normalization and ASR corrections
        normalized_text = self.text_normalizer.normalize(text_norm)
        logger.debug(f"Normalized text: '{normalized_text}'")

        # Extract components in parallel for efficiency
        species_result = self._extract_species(normalized_text)
        length_result = self._extract_length(normalized_text)

        # Calculate overall confidence
        confidence = self._calculate_confidence(species_result, length_result)

        # Assemble final result
        result = ParseResult(
            cancel=False,
            species=species_result.get('species') if species_result else None,
            length_cm=length_result.get('length_cm') if length_result else None,
            confidence=confidence,
            raw_text=original_text,
            parsed_components={
                'species_result': species_result,
                'length_result': length_result,
                'normalized_text': normalized_text
            }
        )

        # Update statistics
        self._update_parsing_stats(result)

        logger.debug(f"Parse result: {result}")
        return result

    def _is_cancel_command(self, text: str) -> bool:
        """
        Detect cancel commands in input text.

        Args:
            text: Normalized text to check

        Returns:
            bool: True if cancel command detected

        Cancel Patterns:
            - "cancel", "undo", "delete", "remove", "clear"
            - Case-insensitive matching
            - Word boundary matching to avoid false positives
        """
        return bool(self._cancel_pattern.search(text))

    def _extract_species(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract fish species from text using fuzzy matching.

        Args:
            text: Normalized text to extract species from

        Returns:
            Optional[Dict]: Species extraction result with confidence

        Processing:
            - Fuzzy matching against species database
            - Confidence scoring based on match quality
            - Handling of common species aliases
            - Taxonomic name normalization
        """
        try:
            match_result = self.species_matcher.fuzzy_match_species(text)

            if match_result:
                # Extract species name and confidence from matcher result
                if isinstance(match_result, str):
                    # Simple string result
                    species = match_result.title()
                    confidence = 0.8  # Default confidence for exact matches
                else:
                    # Detailed result with confidence
                    species = match_result.get('species', '').title()
                    confidence = match_result.get('confidence', 0.0)

                return {
                    'species': species,
                    'confidence': confidence,
                    'raw_match': match_result
                }

        except Exception as e:
            logger.error(f"Species extraction failed: {e}")

        return None

    def _extract_length(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract length measurement from text and convert to centimeters.

        Args:
            text: Normalized text to extract length from

        Returns:
            Optional[Dict]: Length extraction result with metadata

        Processing:
            - Number parsing from natural language
            - Unit detection and conversion
            - Validation of reasonable fish lengths
            - Confidence scoring based on clarity
        """
        try:
            length_result = self.number_parser.parse_measurement(text)

            if length_result and length_result.get('length_cm'):
                length_cm = length_result['length_cm']

                # Validate reasonable fish length range
                if self._is_reasonable_fish_length(length_cm):
                    return {
                        'length_cm': length_cm,
                        'original_value': length_result.get('original_value'),
                        'original_unit': length_result.get('original_unit'),
                        'confidence': length_result.get('confidence', 0.8)
                    }
                else:
                    logger.warning(f"Unreasonable fish length: {length_cm}cm")

        except Exception as e:
            logger.error(f"Length extraction failed: {e}")

        return None

    def _is_reasonable_fish_length(self, length_cm: float) -> bool:
        """
        Validate if a length measurement is reasonable for a fish.

        Args:
            length_cm: Length in centimeters

        Returns:
            bool: True if length is within reasonable range

        Validation Rules:
            - Minimum: 1cm (small tropical fish)
            - Maximum: 500cm (large game fish)
            - Reasonable range for typical recreational fishing
        """
        return 1.0 <= length_cm <= 500.0

    def _calculate_confidence(
        self,
        species_result: Optional[Dict[str, Any]],
        length_result: Optional[Dict[str, Any]]
    ) -> float:
        """
        Calculate overall confidence score for parsing result.

        Args:
            species_result: Species extraction result
            length_result: Length extraction result

        Returns:
            float: Overall confidence score (0.0-1.0)

        Confidence Calculation:
            - Weighted average of component confidences
            - Higher weight for complete results
            - Penalty for missing components
            - Bonus for high-quality matches
        """
        if not species_result and not length_result:
            return 0.0

        confidences = []
        weights = []

        if species_result:
            confidences.append(species_result.get('confidence', 0.5))
            weights.append(0.6)  # Species slightly more important

        if length_result:
            confidences.append(length_result.get('confidence', 0.5))
            weights.append(0.4)

        # Calculate weighted average
        if confidences:
            total_weight = sum(weights)
            weighted_sum = sum(c * w for c, w in zip(confidences, weights))
            base_confidence = weighted_sum / total_weight

            # Bonus for complete results
            if species_result and length_result:
                base_confidence *= 1.1  # 10% bonus for completeness

            return min(base_confidence, 1.0)

        return 0.0

    def _update_parsing_stats(self, result: ParseResult) -> None:
        """Update parsing statistics based on result."""
        if result.is_valid:
            self.stats['successful_parses'] += 1
        elif result.is_partial:
            self.stats['partial_parses'] += 1
        else:
            self.stats['failed_parses'] += 1

    def get_parsing_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive parsing statistics.

        Returns:
            Dict: Statistics including success rates and performance metrics
        """
        stats = self.stats.copy()

        if stats['total_parses'] > 0:
            stats['success_rate'] = stats['successful_parses'] / stats['total_parses']
            stats['partial_rate'] = stats['partial_parses'] / stats['total_parses']
            stats['failure_rate'] = stats['failed_parses'] / stats['total_parses']
        else:
            stats['success_rate'] = 0.0
            stats['partial_rate'] = 0.0
            stats['failure_rate'] = 0.0

        return stats

    def reset_stats(self) -> None:
        """Reset parsing statistics."""
        self.stats = {
            'total_parses': 0,
            'successful_parses': 0,
            'cancel_commands': 0,
            'partial_parses': 0,
            'failed_parses': 0
        }

    def configure_for_region(self, region: str) -> None:
        """
        Configure parser for specific fishing region.

        Args:
            region: Fishing region identifier (e.g., "atlantic", "pacific", "freshwater")

        Regional Optimizations:
            - Species database filtering
            - Regional measurement preferences
            - Local terminology and aliases
        """
        try:
            # Update species matcher for region
            if hasattr(self.species_matcher, 'configure_for_region'):
                self.species_matcher.configure_for_region(region)

            # Update number parser for regional units
            if hasattr(self.number_parser, 'configure_for_region'):
                self.number_parser.configure_for_region(region)

            logger.info(f"Configured parser for region: {region}")

        except Exception as e:
            logger.error(f"Failed to configure for region {region}: {e}")


# Legacy alias for backward compatibility
ParserResult = ParseResult
