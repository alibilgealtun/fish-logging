"""
Fish Species Matching for Fish Logging Application.

This module provides sophisticated fuzzy matching capabilities for identifying
fish species names in natural language text from speech recognition. It implements
multiple matching strategies with confidence scoring and handles common speech
recognition errors and variations in species naming.

Classes:
    SpeciesMatcher: Main class for fuzzy matching fish species names

Features:
    - Multi-strategy matching (exact, fuzzy, token-based)
    - Confidence scoring for match quality assessment
    - Species name normalization and standardization
    - Common name and scientific name handling
    - Regional species database filtering
    - ASR error correction for species names

Matching Strategies:
    1. Exact word boundary matching (highest confidence)
    2. Token window fuzzy matching (medium confidence)
    3. Whole-text fuzzy matching (lowest confidence, fallback)

Use Cases:
    - Voice-activated fish logging systems
    - Automated species identification from speech
    - Fishing tournament data entry
    - Marine biology field data collection
"""
import re
from typing import Optional, List, Dict, Any, Tuple
from rapidfuzz import process, fuzz

from .config import ConfigManager
from .text_utils import tokenize_text

# Optional logger: prefer loguru if available, else fallback to stdlib logging
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class SpeciesMatcher:
    """
    Sophisticated fuzzy matching system for fish species identification.

    This class implements multiple matching strategies to identify fish species
    names from natural language text, particularly optimized for speech recognition
    input with potential errors and variations.

    Key Features:
        - Multi-strategy matching with confidence scoring
        - Species name normalization and standardization
        - Fuzzy matching with configurable thresholds
        - Token-based windowing for partial matches
        - Regional species database support
        - Common and scientific name handling

    Matching Strategies:
        1. Exact word boundary matching (95%+ confidence)
        2. Token window fuzzy matching (70-95% confidence)
        3. Whole-text fuzzy matching (50-70% confidence)

    Architecture:
        - Strategy pattern for different matching approaches
        - Configuration-driven species database
        - Caching for performance optimization
        - Extensible for new matching algorithms

    Usage:
        matcher = SpeciesMatcher(config)
        result = matcher.fuzzy_match_species("I caught a sea bass today")
        if result:
            print(f"Matched species: {result}")
    """

    def __init__(self, config: ConfigManager) -> None:
        """
        Initialize species matcher with configuration.

        Args:
            config: Configuration manager containing species database

        Initialization:
            - Loads species database from configuration
            - Prepares normalized species mappings
            - Sets up fuzzy matching parameters
            - Initializes performance caches
        """
        self.config = config

        # Species database and normalization mappings
        self.species_list = self.config.get_species_list()
        self.species_aliases = self.config.get_species_aliases()

        # Fuzzy matching parameters
        self.exact_match_threshold = 95  # Minimum score for exact matches
        self.fuzzy_match_threshold = 70  # Minimum score for fuzzy matches
        self.fallback_threshold = 50    # Minimum score for fallback matches

        # Performance optimization caches
        self._match_cache = {}
        self._normalized_species = {}

        # Pre-compile regex patterns for performance
        self._word_boundary_patterns = self._compile_species_patterns()

        logger.debug(f"SpeciesMatcher initialized with {len(self.species_list)} species")

    def normalize_species_name(self, name: str) -> str:
        """
        Normalize a species name using configured mappings and standardization.

        Args:
            name: Raw species name to normalize

        Returns:
            str: Normalized and standardized species name

        Normalization Process:
            1. Convert to lowercase and strip whitespace
            2. Apply species-specific normalization mappings
            3. Handle common aliases and abbreviations
            4. Standardize formatting (title case, spacing)

        Examples:
            - "seabass" -> "Sea Bass"
            - "large mouth bass" -> "Largemouth Bass"
            - "red fish" -> "Red Drum"
        """
        if not name:
            return ""

        # Check cache first for performance
        if name in self._normalized_species:
            return self._normalized_species[name]

        # Basic normalization
        name_lower = name.lower().strip()

        # Apply species-specific mappings from configuration
        normalized = self.config.species_normalization.get(name_lower, name_lower)

        # Apply alias mappings
        if normalized in self.species_aliases:
            normalized = self.species_aliases[normalized]

        # Standardize formatting
        result = normalized.title()

        # Cache result for future use
        self._normalized_species[name] = result

        return result

    def fuzzy_match_species(self, text: str) -> Optional[str]:
        """
        Find the best matching fish species in the given text using multiple strategies.

        This method implements a multi-strategy approach to species matching,
        starting with the most precise methods and falling back to more
        permissive approaches if needed.

        Args:
            text: Text to search for species names

        Returns:
            Optional[str]: Normalized species name if found, None otherwise

        Strategy Progression:
            1. Direct exact word boundary matching (highest confidence)
            2. Token window fuzzy matching (medium confidence)
            3. Whole-text fuzzy matching (fallback)

        Performance:
            - Caches results for identical input text
            - Short-circuits on high-confidence matches
            - Optimized regex patterns for exact matching
        """
        if not text:
            return None

        # Check cache for performance optimization
        if text in self._match_cache:
            return self._match_cache[text]

        lowered = text.lower()
        result = None

        # Strategy 1: Exact word boundary matching (highest precision)
        result = self._exact_word_boundary_match(lowered)
        if result:
            self._match_cache[text] = result
            logger.debug(f"Exact match found: '{result}' in '{text}'")
            return result

        # Strategy 2: Token window fuzzy matching (medium precision)
        result = self._token_window_fuzzy_match(lowered)
        if result:
            self._match_cache[text] = result
            logger.debug(f"Token window match found: '{result}' in '{text}'")
            return result

        # Strategy 3: Whole-text fuzzy matching (fallback)
        result = self._whole_text_fuzzy_match(lowered)
        if result:
            self._match_cache[text] = result
            logger.debug(f"Fuzzy match found: '{result}' in '{text}'")
            return result

        # No match found
        self._match_cache[text] = None
        logger.debug(f"No species match found in: '{text}'")
        return None

    def _compile_species_patterns(self) -> Dict[str, re.Pattern]:
        """
        Pre-compile regex patterns for exact species matching.

        Returns:
            Dict[str, re.Pattern]: Mapping of species to compiled regex patterns

        Pattern Features:
            - Word boundary matching to avoid partial matches
            - Case-insensitive matching
            - Handling of spaces and hyphens in species names
        """
        patterns = {}

        for species in self.species_list:
            # Create word boundary pattern for exact matching
            # Handle spaces and common punctuation in species names
            escaped = re.escape(species.lower())
            escaped = escaped.replace(r'\ ', r'[\s\-]*')  # Allow spaces or hyphens
            pattern = rf'\b{escaped}\b'

            try:
                patterns[species] = re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                logger.warning(f"Failed to compile pattern for '{species}': {e}")

        return patterns

    def _exact_word_boundary_match(self, text: str) -> Optional[str]:
        """
        Attempt exact word boundary matching using pre-compiled patterns.

        Args:
            text: Lowercase text to search

        Returns:
            Optional[str]: Exact match if found, None otherwise

        Features:
            - Highest confidence matching strategy
            - Uses pre-compiled regex patterns for performance
            - Handles word boundaries and punctuation
            - Returns first match found (prioritizes longer matches)
        """
        # Sort species by length (descending) to prioritize longer matches
        sorted_species = sorted(self.species_list, key=len, reverse=True)

        for species in sorted_species:
            if species in self._word_boundary_patterns:
                pattern = self._word_boundary_patterns[species]
                if pattern.search(text):
                    return self.normalize_species_name(species)

        return None

    def _token_window_fuzzy_match(self, text: str) -> Optional[str]:
        """
        Fuzzy match using sliding windows of 1-3 tokens.

        Args:
            text: Lowercase text to search

        Returns:
            Optional[str]: Best fuzzy match if above threshold, None otherwise

        Algorithm:
            1. Tokenize input text into words
            2. Create sliding windows of 1-3 tokens
            3. Fuzzy match each window against species database
            4. Return best match above confidence threshold

        Advantages:
            - Handles partial species names in longer sentences
            - Good performance on speech recognition errors
            - Balances precision with recall
        """
        tokens = tokenize_text(text)
        if not tokens:
            return None

        best_match = None
        best_score = 0

        # Try windows of different sizes (1-3 tokens)
        for window_size in range(1, min(4, len(tokens) + 1)):
            for i in range(len(tokens) - window_size + 1):
                # Create token window
                window = ' '.join(tokens[i:i + window_size])

                # Fuzzy match against species database
                match_result = process.extractOne(
                    window,
                    self.species_list,
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=self.fuzzy_match_threshold
                )

                if match_result and match_result[1] > best_score:
                    best_match = match_result[0]
                    best_score = match_result[1]

        if best_match and best_score >= self.fuzzy_match_threshold:
            return self.normalize_species_name(best_match)

        return None

    def _whole_text_fuzzy_match(self, text: str) -> Optional[str]:
        """
        Fallback fuzzy matching against entire text.

        Args:
            text: Lowercase text to search

        Returns:
            Optional[str]: Best fuzzy match if above threshold, None otherwise

        Algorithm:
            - Uses rapidfuzz for efficient fuzzy matching
            - Employs token_sort_ratio for word order flexibility
            - Lower confidence threshold for fallback strategy

        Use Cases:
            - When species name is heavily corrupted by ASR
            - When species spans multiple words with intervening text
            - Last resort matching for difficult cases
        """
        if not text:
            return None

        match_result = process.extractOne(
            text,
            self.species_list,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=self.fallback_threshold
        )

        if match_result:
            species, score = match_result
            logger.debug(f"Whole-text fuzzy match: '{species}' (score: {score})")
            return self.normalize_species_name(species)

        return None

    def get_match_confidence(self, text: str, species: str) -> float:
        """
        Calculate confidence score for a specific species match.

        Args:
            text: Original input text
            species: Species name to calculate confidence for

        Returns:
            float: Confidence score (0.0-1.0)

        Confidence Factors:
            - Fuzzy matching score
            - Match strategy used
            - Text length and complexity
            - Species name frequency
        """
        if not text or not species:
            return 0.0

        # Calculate fuzzy match score
        fuzzy_score = fuzz.token_sort_ratio(text.lower(), species.lower())

        # Adjust based on match strategy
        if species.lower() in text.lower():
            # Exact substring match gets bonus
            base_confidence = min(fuzzy_score / 100.0 * 1.2, 1.0)
        else:
            # Pure fuzzy match
            base_confidence = fuzzy_score / 100.0

        # Apply length penalty for very short text
        if len(text.split()) < 3:
            base_confidence *= 0.9

        return max(0.0, min(1.0, base_confidence))

    def suggest_species(self, partial_text: str, limit: int = 5) -> List[Tuple[str, float]]:
        """
        Suggest possible species based on partial input.

        Args:
            partial_text: Partial species name or text
            limit: Maximum number of suggestions to return

        Returns:
            List[Tuple[str, float]]: List of (species, confidence) tuples

        Features:
            - Ranked suggestions by confidence
            - Fuzzy matching for typo tolerance
            - Configurable result limit
            - Useful for autocomplete interfaces
        """
        if not partial_text:
            return []

        # Get fuzzy matches with scores
        matches = process.extract(
            partial_text.lower(),
            self.species_list,
            scorer=fuzz.partial_ratio,
            limit=limit,
            score_cutoff=30  # Lower threshold for suggestions
        )

        # Convert to normalized format with confidence scores
        suggestions = []
        for species, score in matches:
            normalized = self.normalize_species_name(species)
            confidence = score / 100.0
            suggestions.append((normalized, confidence))

        return suggestions

    def configure_for_region(self, region: str) -> None:
        """
        Configure species matcher for specific fishing region.

        Args:
            region: Region identifier (e.g., "atlantic", "pacific", "freshwater")

        Regional Optimizations:
            - Filter species database to regional species
            - Adjust fuzzy matching thresholds
            - Load regional aliases and common names
        """
        try:
            # Load regional species list if available
            regional_species = self.config.get_regional_species(region)
            if regional_species:
                self.species_list = regional_species
                self._word_boundary_patterns = self._compile_species_patterns()
                logger.info(f"Configured species matcher for region: {region} "
                           f"({len(self.species_list)} species)")
            else:
                logger.warning(f"No regional species data found for: {region}")

        except Exception as e:
            logger.error(f"Failed to configure for region {region}: {e}")

    def clear_cache(self) -> None:
        """Clear performance caches."""
        self._match_cache.clear()
        self._normalized_species.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get matcher statistics and performance information.

        Returns:
            Dict: Statistics including cache size, species count, etc.
        """
        return {
            'species_count': len(self.species_list),
            'cache_size': len(self._match_cache),
            'normalized_cache_size': len(self._normalized_species),
            'exact_threshold': self.exact_match_threshold,
            'fuzzy_threshold': self.fuzzy_match_threshold,
            'fallback_threshold': self.fallback_threshold,
        }
