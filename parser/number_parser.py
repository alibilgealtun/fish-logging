"""Number and unit parsing functionality."""

import re
from typing import Optional, Tuple, List
from rapidfuzz import process, fuzz

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

    def words_to_number(self, tokens: List[str]) -> Optional[float]:
        """
        Convert number words to numeric value.

        Handles: "twenty seven point five", "one hundred and twenty", etc.

        Args:
            tokens: List of word tokens to parse

        Returns:
            Numeric value or None if parsing fails
        """
        if not tokens:
            return None

        # Normalize tokens using ASR corrections
        normalized_tokens = []
        for token in tokens:
            token_lower = token.lower()
            if token_lower in self.config.ignored_tokens:
                continue
            if token_lower in self.config.misheard_number_tokens:
                normalized_tokens.append(self.config.misheard_number_tokens[token_lower])
            else:
                normalized_tokens.append(token_lower)

        total = 0
        current = 0
        decimal_digits = []
        in_decimal = False

        for token in normalized_tokens:
            if token in self.config.decimal_tokens:
                if in_decimal:
                    return None  # Multiple decimal points
                in_decimal = True
                continue

            if token in self.config.number_words:
                value = self.config.number_words[token]
                if in_decimal:
                    # Handle decimal digits
                    if value < 10:
                        decimal_digits.append(value)
                    else:
                        # Split multi-digit numbers into individual digits
                        for digit in str(value):
                            decimal_digits.append(int(digit))
                else:
                    if value == 100:
                        if current == 0:
                            current = 1
                        current *= 100
                    else:
                        current += value
            else:
                # Unknown token - parsing failed
                return None

        total += current

        if in_decimal:
            if not decimal_digits:
                return None
            decimal_str = "0." + "".join(str(d) for d in decimal_digits)
            return total + float(decimal_str)

        return float(total) if total != 0 or current != 0 else None

    def _find_longest_spoken_number(self, tokens: List[str]) -> Optional[float]:
        """Find and parse the longest contiguous sequence of number words."""
        best_value = None
        i = 0
        n = len(tokens)

        while i < n:
            token_lower = tokens[i].lower()
            if (token_lower in self.config.number_words or
                    token_lower in self.config.decimal_tokens or
                    token_lower in self.config.misheard_number_tokens):

                # Find end of number sequence
                j = i
                while j < n:
                    token_j = tokens[j].lower()
                    if (token_j in self.config.number_words or
                            token_j in self.config.decimal_tokens or
                            token_j in self.config.misheard_number_tokens or
                            token_j in self.config.ignored_tokens):
                        j += 1
                    else:
                        break

                # Try to parse this sequence
                parsed_value = self.words_to_number(tokens[i:j])
                if parsed_value is not None:
                    best_value = parsed_value
                i = j
            else:
                i += 1

        return best_value

    def _fuzzy_find_unit_index(self, tokens: List[str]) -> Optional[int]:
        """Find token index that represents a unit using fuzzy matching."""
        unit_candidates = list(self.config.unit_synonyms.keys())

        # Check single tokens
        for i, token in enumerate(tokens):
            # Exact match
            if token in self.config.unit_synonyms:
                return i

            # Fuzzy match
            best_match = process.extractOne(token, unit_candidates, scorer=fuzz.ratio)
            if best_match and best_match[1] >= 80:
                return i

            # Pattern-based detection
            if re.search(r"(^cm$|^mm$|cent|millim)", token):
                return i

        # Check two-token combinations
        for i in range(len(tokens) - 1):
            pair = f"{tokens[i]} {tokens[i + 1]}"
            best_match = process.extractOne(pair, unit_candidates, scorer=fuzz.ratio)
            if best_match and best_match[1] >= 80:
                return i

        return None

    def extract_number_with_units(self, text: str) -> Tuple[Optional[float], Optional[str]]:
        """
        Extract numeric measurement and convert to standard units.

        Args:
            text: Text to parse for measurements

        Returns:
            Tuple of (value_in_cm, unit_name) or (None, None)
        """
        if not text:
            return None, None

        text_lower = text.strip().lower()

        # Strategy 1: Numeric with unit regex
        match = self._numeric_with_unit_pattern.search(text_lower)
        if match:
            number = float(match.group(1).replace(",", "."))
            unit = match.group(2).lower()
            normalized_unit = self.config.unit_synonyms.get(
                unit, self.config.unit_synonyms.get(unit.replace(" ", ""), "cm")
            )
            if normalized_unit == "mm":
                return number / 10.0, "cm"
            return number, "cm"

        # Strategy 2: Ordinal numbers with units
        ordinal_match = self._ordinal_pattern.search(text_lower)
        if ordinal_match:
            number = float(ordinal_match.group(1))
            unit_match = re.search(
                r"\b(cm|centimeters?|centimetres?|mm|millimeters?|millimetres?)\b",
                text_lower, re.IGNORECASE
            )
            if unit_match:
                unit = unit_match.group(1).lower()
                normalized_unit = self.config.unit_synonyms.get(unit, "cm")
                if normalized_unit == "mm":
                    return number / 10.0, "cm"
                return number, "cm"
            return number, "cm"

        # Strategy 3: Numeric with separate unit
        numeric_match = self._numeric_pattern.search(text_lower)
        if numeric_match:
            try:
                number = float(numeric_match.group(1).replace(",", "."))
            except ValueError:
                number = None

            if number is not None:
                unit_match = re.search(
                    r"\b(cm|centimeters?|centimetres?|mm|millimeters?|millimetres?|milimeters?)\b",
                    text_lower, re.IGNORECASE
                )
                if unit_match:
                    unit = unit_match.group(1).lower()
                    normalized_unit = self.config.unit_synonyms.get(unit, "cm")
                    if normalized_unit == "mm":
                        return number / 10.0, "cm"
                    return number, "cm"
                # No explicit unit - default to cm
                return number, "cm"

        # Strategy 4: Spoken numbers with units
        tokens = tokenize_text(text_lower)

        # Expand tokens that contain unit substrings
        expanded_tokens = []
        for token in tokens:
            if re.search(r"(centi|cent|millim|mm)", token):
                match = re.search(r"(cent|centi|millim|mm)", token)
                if match:
                    idx = match.start()
                    prefix = token[:idx]
                    suffix = token[idx:]
                    if prefix:
                        expanded_tokens.append(prefix)
                    expanded_tokens.append(suffix)
                else:
                    expanded_tokens.append(token)
            else:
                expanded_tokens.append(token)

        tokens = [t for t in expanded_tokens if t]

        # Find unit and parse associated number
        unit_index = self._fuzzy_find_unit_index(tokens)
        if unit_index is not None:
            # Try number tokens before unit
            number_tokens = tokens[:unit_index]
            if not number_tokens:
                # Try number tokens after unit
                number_tokens = tokens[unit_index + 1:]

            parsed_number = self.words_to_number(number_tokens) if number_tokens else None

            # If no number found near unit, try finding longest number anywhere
            if parsed_number is None:
                parsed_number = self._find_longest_spoken_number(tokens)

            if parsed_number is not None:
                unit_token = tokens[unit_index].lower()
                normalized_unit = self.config.unit_synonyms.get(unit_token)

                if normalized_unit is None:
                    # Fuzzy resolve unit
                    best_match = process.extractOne(
                        unit_token, list(self.config.unit_synonyms.keys()), scorer=fuzz.ratio
                    )
                    if best_match and best_match[1] >= 80:
                        normalized_unit = self.config.unit_synonyms.get(best_match[0], "cm")
                    else:
                        normalized_unit = "cm"

                if normalized_unit == "mm":
                    return float(parsed_number) / 10.0, "cm"
                return float(parsed_number), "cm"

        # Strategy 5: Spoken number without explicit unit (assume cm)
        spoken_number = self._find_longest_spoken_number(tokens)
        if spoken_number is not None:
            return float(spoken_number), "cm"

        return None, None