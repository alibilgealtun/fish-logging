"""Number and unit parsing functionality."""

import re
from typing import Optional, Tuple, List
from rapidfuzz import process, fuzz

# Assuming config and text_utils are in the same directory or installed package
from .config import ConfigManager
from .text_utils import tokenize_text


class NumberParser:
    """Handles parsing of numbers and units from text."""

    def __init__(self, config: ConfigManager):
        """Initialize number parser with configuration."""
        self.config = config
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for number extraction."""
        self._numeric_with_unit_pattern = re.compile(
            r"(\d+(?:[.,]\d+)?)\s*(cm|centimeters?|centimetres?|centimetre|"
            r"centi[- ]?meters?|mm|millimeters?|millimetres?|milimeters?)\b",
            re.IGNORECASE,
        )
        self._numeric_pattern = re.compile(r"(\d+(?:[.,]\d+)?)")
        self._ordinal_pattern = re.compile(r"(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)

    def word2num(self, tokens: List[str]) -> Optional[float]:
        """
        Convert number words to a numeric value.
        This version correctly handles 'and' as a conjunction, not a decimal point.
        """
        if not tokens:
            return None

        normalized_tokens = [t.lower().strip() for t in tokens]

        total = 0
        current_chunk = 0
        decimal_str = ""
        in_decimal_part = False
        is_digit_sequence = True

        for i, token in enumerate(normalized_tokens):
            # Map misheard words to their correct counterparts
            token = self.config.misheard_number_tokens.get(token, token)

            if token in self.config.decimal_tokens:
                # FIX: 'and' is a conjunction after 'hundred'/'thousand', not a decimal.
                if token == 'and' and i > 0 and normalized_tokens[i-1] in ['hundred', 'thousand']:
                    continue
                if in_decimal_part: return None # Multiple decimals
                in_decimal_part = True
                is_digit_sequence = True
                continue

            if token in self.config.ignored_tokens:
                continue

            if token not in self.config.number_words:
                continue

            value = self.config.number_words[token]

            if in_decimal_part:
                if value >= 10:
                    for digit in str(value): decimal_str += digit
                else:
                    decimal_str += str(value)
            else:
                if value >= 10: is_digit_sequence = False

                if value == 100:
                    current_chunk = (current_chunk or 1) * 100
                    is_digit_sequence = False
                elif value == 1000:
                    total += (current_chunk or 1) * 1000
                    current_chunk = 0
                    is_digit_sequence = False
                else:
                    if is_digit_sequence and current_chunk > 0 and value < 10:
                        current_chunk = current_chunk * 10 + value
                    else:
                        current_chunk += value

        total += current_chunk
        if decimal_str:
            total += float("0." + decimal_str)

        return float(total) if tokens else None

    def _find_longest_spoken_number(self, tokens: List[str]) -> Optional[float]:
        """Find and parse the longest contiguous sequence of number words."""
        if not tokens:
            return None

        best_value = None
        best_length = 0
        n = len(tokens)

        for i in range(n):
            for j in range(i + 1, n + 1):
                subsequence = tokens[i:j]
                if any(t.lower() in self.config.number_words or t.lower() in self.config.decimal_tokens for t in subsequence):
                    parsed_value = self.word2num(subsequence)
                    if parsed_value is not None and len(subsequence) >= best_length:
                        best_value = parsed_value
                        best_length = len(subsequence)
        return best_value

    def _fuzzy_find_unit_index(self, tokens: List[str]) -> Optional[int]:
        """Find token index that represents a unit using fuzzy matching."""
        if not tokens or len(tokens) > 2:
            return None
        text = " ".join(t.lower().strip() for t in tokens)
        if text in self.config.unit_synonyms: return 0
        unit_candidates = list(self.config.unit_synonyms.keys())
        best_match = process.extractOne(text, unit_candidates, scorer=fuzz.ratio)
        if best_match and best_match[1] >= 80: return 0
        if re.search(r"(^cm$|^mm$|cent|millim)", text): return 0
        return None

    def _normalize_unit(self, unit_token: str) -> str:
        """Normalize a unit token to standard form."""
        unit_lower = unit_token.lower().strip()
        if unit_lower in self.config.unit_synonyms:
            return self.config.unit_synonyms[unit_lower]
        unit_candidates = list(self.config.unit_synonyms.keys())
        best_match = process.extractOne(unit_lower, unit_candidates, scorer=fuzz.ratio)
        return self.config.unit_synonyms.get(best_match[0], "cm") if best_match and best_match[1] >= 80 else "cm"

    def extract_number_with_units(self, text: str) -> Tuple[Optional[float], Optional[str]]:
        """Extract numeric measurement and convert to standard units."""
        if not text:
            return None, None

        text_lower = text.strip().lower()

        def finalize(number: float, unit_str: str) -> Tuple[float, str]:
            normalized_unit = self._normalize_unit(unit_str)
            value_in_cm = number / 10.0 if normalized_unit == "mm" else number
            return round(value_in_cm, 4), "cm"

        # Strategy 1: Numeric with unit regex (e.g., "35.5cm")
        match = self._numeric_with_unit_pattern.search(text_lower)
        if match:
            return finalize(float(match.group(1).replace(",", ".")), match.group(2))

        # FIX: Re-add Ordinal number strategy (e.g., "35th")
        ordinal_match = self._ordinal_pattern.search(text_lower)
        if ordinal_match:
            number = float(ordinal_match.group(1))
            unit_match = re.search(r"\b(cm|centimeters?|mm|millimeters?)\b", text_lower, re.IGNORECASE)
            unit = unit_match.group(1) if unit_match else "cm"
            return finalize(number, unit)

        # Spoken Number Strategy
        tokens = tokenize_text(text_lower)
        unit_indices = [i for i, token in enumerate(tokens) if self._fuzzy_find_unit_index([token]) is not None]

        if unit_indices:
            last_unit_index = unit_indices[-1]
            unit_token = tokens[last_unit_index]
            # FIX: Increase search window to 10 to catch longer numbers
            start_index = max(0, last_unit_index - 10)
            search_tokens = tokens[start_index:last_unit_index]
            parsed_number = self._find_longest_spoken_number(search_tokens)
            if parsed_number is not None:
                return finalize(parsed_number, unit_token)

        parsed_number = self._find_longest_spoken_number(tokens)
        if parsed_number is not None:
            unit_match = re.search(r"\b(cm|centimeters?|mm|millimeters?)\b", text_lower, re.IGNORECASE)
            unit = unit_match.group(1) if unit_match else "cm"
            return finalize(parsed_number, unit)

        return None, None