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

        Uses multiple strategies in order of specificity:
        1. Direct exact word boundary matching
        2. Token window fuzzy matching (1-3 token windows)
        3. Whole-text fuzzy matching as fallback

        Args:
            text: Text to search for species names

        Returns:
            Normalized species name if found, None otherwise
        """
        lowered = text.lower()

        # Try strategies in order of specificity
        strategies = [
            self._exact_word_boundary_match,
            self._token_window_fuzzy_match,
            self._whole_text_fuzzy_match,
        ]

        for strategy in strategies:
            result = strategy(lowered)
            if result:
                return result

        return None

    def _exact_word_boundary_match(self, text: str) -> Optional[str]:
        """Strategy 1: Direct exact word boundary matching."""
        species_list = self.config.species

        for candidate in species_list:
            normalized_candidate = self.normalize_species_name(candidate)
            pattern = r"\b" + re.escape(normalized_candidate) + r"\b"
            if re.search(pattern, text):
                return self.config.species_normalization.get(
                    normalized_candidate, normalized_candidate
                )

        return None

    def _token_window_fuzzy_match(self, text: str) -> Optional[str]:
        """Strategy 2: Token window fuzzy matching (checks 1-3 token windows)."""
        species_list = self.config.species
        tokens = tokenize_text(text)
        n = len(tokens)

        # Check windows of different sizes (3, 2, 1 tokens)
        for window_size in (3, 2, 1):
            for i in range(0, n - window_size + 1):
                window = " ".join(tokens[i : i + window_size])
                best_match = process.extractOne(
                    window, species_list, scorer=fuzz.token_set_ratio
                )
                if best_match and best_match[1] >= 80:
                    normalized = self.normalize_species_name(best_match[0])
                    return self.config.species_normalization.get(normalized, normalized)

        return None

    def _whole_text_fuzzy_match(self, text: str) -> Optional[str]:
        """Strategy 3: Whole-text fuzzy matching fallback (higher threshold)."""
        species_list = self.config.species

        best_whole_match = process.extractOne(
            text, species_list, scorer=fuzz.token_set_ratio
        )
        if best_whole_match and best_whole_match[1] >= 85:
            normalized = self.normalize_species_name(best_whole_match[0])
            return self.config.species_normalization.get(normalized, normalized)

        return None

