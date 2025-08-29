"""Species matching functionality using fuzzy matching algorithms."""

import re
from typing import Optional, List
from rapidfuzz import process, fuzz

from .config import ConfigManager
from .text_utils import tokenize_text


class SpeciesMatcher:
    """Handles fuzzy matching of fish species names."""

    def __init__(self, config: ConfigManager):
        """Initialize species matcher with configuration."""
        self.config = config

    def normalize_species_name(self, name: str) -> str:
        """Normalize a species name using configured mappings."""
        name_lower = name.lower().strip()
        return self.config.species_normalization.get(name_lower, name_lower)

    def fuzzy_match_species(self, text: str) -> Optional[str]:
        """
        Find the best matching fish species in the given text.

        Uses multiple strategies:
        1. Direct exact word boundary matching
        2. Token window fuzzy matching (1-3 token windows)
        3. Whole-text fuzzy matching as fallback

        Args:
            text: Text to search for species names

        Returns:
            Normalized species name if found, None otherwise
        """
        lowered = text.lower()
        species_list = self.config.known_species

        # Strategy 1: Direct exact word boundary matching
        for candidate in species_list:
            normalized_candidate = self.normalize_species_name(candidate)
            pattern = r"\b" + re.escape(normalized_candidate) + r"\b"
            if re.search(pattern, lowered):
                return self.config.species_normalization.get(
                    normalized_candidate, normalized_candidate
                )

        # Strategy 2: Token window fuzzy matching
        tokens = tokenize_text(lowered)
        n = len(tokens)

        # Check windows of different sizes (3, 2, 1 tokens)
        for window_size in (3, 2, 1):
            for i in range(0, n - window_size + 1):
                window = " ".join(tokens[i:i + window_size])
                best_match = process.extractOne(
                    window, species_list, scorer=fuzz.token_set_ratio
                )
                if best_match and best_match[1] >= 80:
                    normalized = self.normalize_species_name(best_match[0])
                    return self.config.species_normalization.get(normalized, normalized)

        # Strategy 3: Whole-text fuzzy matching fallback
        best_whole_match = process.extractOne(
            lowered, species_list, scorer=fuzz.token_set_ratio
        )
        if best_whole_match and best_whole_match[1] >= 85:
            normalized = self.normalize_species_name(best_whole_match[0])
            return self.config.species_normalization.get(normalized, normalized)

        return None