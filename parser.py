from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import re
from rapidfuzz import process, fuzz

# Known species list (TODO can be extended or loaded from a file later)
KNOWN_SPECIES = [
    "sea bass", "seabass", "cod", "mackerel", "trout", "salmon", "tuna",
    "snapper", "grouper", "bream", "carp", "pike", "perch", "haddock",
    "halibut", "flounder", "mullet", "anchovy", "sardine", "herring",
]

# Map normalized names for common variants
SPECIES_NORMALIZATION = {
    "seabass": "sea bass",
    "sea-bass": "sea bass",
}

# Acceptable unit variations -> normalized
UNIT_SYNONYMS = {
    # centimeters
    "cm": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "centimetre": "cm",
    "centimetres": "cm",
    "centi": "cm",
    "centi-meters": "cm",
    # millimeters (incl. common misspelling)
    "mm": "mm",
    "millimeter": "mm",
    "millimetre": "mm",
    "millimeters": "mm",
    "millimetres": "mm",
    "milimeter": "mm",
    "milimeters": "mm",
    # common ASR forms for "cm"
    "see": "cm",  # part of "see em" handling - used with neighbor logic
    "seeem": "cm",
    "see-em": "cm",
    "see em": "cm",
    "cem": "cm",
}

# Basic number words
NUMBER_WORDS: Dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
}

# tokens that indicate decimal point
DECIMAL_TOKENS = {"point", "dot", "comma"}

# tokens allowed but ignored (like "and" in "one hundred and five")
IGNORED_TOKENS = {"and"}

# common ASR mishearings for numbers (applied only in numeric contexts, not globally)
MISHEARD_NUMBER_TOKENS = {
    "pre": "three",
    "tree": "three",
    "tri": "three",
    "tre": "three",
    "free": "three",
    "thre": "three",
    "too": "two",
    "to": "two",
    "for": "four",
    "fore": "four",
    "won": "one",
    "ate": "eight",
    "zeroo": "zero",
}



# Regular expressions compiled once
_NUMERIC_WITH_UNIT_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(cm|centimeters?|centimetres?|centimetre|centi[- ]?meters?|mm|millimeters?|millimetres?|milimeters?)\b",
    re.IGNORECASE,
)
_NUMERIC_RE = re.compile(r"(\d+(?:[.,]\d+)?)")
_ORDINAL_RE = re.compile(r"(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)

# result dataclass
@dataclass
class ParserResult:
    cancel: bool
    species: Optional[str]
    length_cm: Optional[float]


def normalize_species_name(name: str) -> str:
    name_l = name.lower().strip()
    name_l = SPECIES_NORMALIZATION.get(name_l, name_l)
    return name_l


def _tokenize_text_for_parser(text: str) -> List[str]:
    """Tokenize while preserving hyphenated tokens; split hyphens later."""
    # Keep letters and hyphens
    tokens = re.findall(r"[a-zA-Z]+(?:-[a-zA-Z]+)?", text)
    split_tokens: List[str] = []
    for t in tokens:
        if "-" in t:
            parts = t.split("-")
            for p in parts:
                if p:
                    split_tokens.append(p.lower())
        else:
            split_tokens.append(t.lower())
    return split_tokens


def fuzzy_match_species(text: str, species_list: List[str] = KNOWN_SPECIES) -> Optional[str]:
    """
    Improved fuzzy species match:
    - direct containment (word boundary)
    - scan 1-3 token windows and fuzzy match each window to the species list
    - fallback: best whole-text fuzzy match (as before)
    """
    lowered = text.lower()

    # 1) direct exact contains (word-boundary) for faster exact hits
    for candidate in species_list:
        norm_c = normalize_species_name(candidate)
        if re.search(r"\b" + re.escape(norm_c) + r"\b", lowered):
            return SPECIES_NORMALIZATION.get(norm_c, norm_c)

    # 2) token-window fuzzy matching (unigrams, bigrams, trigrams)
    tokens = _tokenize_text_for_parser(lowered)
    n = len(tokens)
    for window_size in (3, 2, 1):
        for i in range(0, n - window_size + 1):
            window = " ".join(tokens[i : i + window_size])
            best = process.extractOne(window, species_list, scorer=fuzz.token_set_ratio)
            if best and best[1] >= 80:
                normalized = normalize_species_name(best[0])
                return SPECIES_NORMALIZATION.get(normalized, normalized)

    # 3) whole-text fuzzy match fallback
    best_whole = process.extractOne(lowered, species_list, scorer=fuzz.token_set_ratio)
    if best_whole and best_whole[1] >= 85:
        normalized = normalize_species_name(best_whole[0])
        return SPECIES_NORMALIZATION.get(normalized, normalized)

    return None


def _apply_asr_unit_and_number_corrections(text: str) -> str:
    """
    Apply targeted corrections that commonly appear from ASR:
    - "pre-centimeters", "tree centimeters", "free-centimeters" -> "three centimeters"
    - "see em", "see-em", "seeem" -> "cm"
    - join/separate some centi variants
    This function is conservative: it targets unit/number contexts only.
    """
    t = text

    # common pattern: pre-centimeters -> three centimeters
    t = re.sub(
        r"\b(?:pre|pre-|tree|tri|tre|free|thre)\s*[-]?\s*(?:centi(?:metre|meter|metres|meters)|centimeters?|centimetres?)\b",
        "three centimeters",
        t,
        flags=re.IGNORECASE,
    )

    # "tree centimeters" etc (space)
    t = re.sub(
        r"\b(?:tree|tri|tre|free|thre)\s+(?:centi(?:metre|meter|metres|meters)|centimeters?|centimetres?)\b",
        "three centimeters",
        t,
        flags=re.IGNORECASE,
    )

    # "see em", "see-em", "seeem" => "cm" when used as unit-like
    t = re.sub(r"\b(see[ -]?em|seeem|cem|c m|c- m)\b", "cm", t, flags=re.IGNORECASE)

    # common spelled variants / spaces before "centimeters"
    t = re.sub(r"\bcenti\s*meters\b", "centimeters", t, flags=re.IGNORECASE)
    t = re.sub(r"\bcenti\s*metres\b", "centimetres", t, flags=re.IGNORECASE)

    # ordinal forms like '3rd' -> '3'
    t = re.sub(r"(\d+)(?:st|nd|rd|th)\b", r"\1", t, flags=re.IGNORECASE)

    return t


def apply_fish_asr_corrections(text: str) -> str:
    """Apply targeted ASR corrections for fish-related utterances.

    This normalizes common mis-hearings for species names while keeping the
    rest of the phrase intact. Corrections are applied conservatively, mostly
    when there is evidence of a measurement (digits/number words/units) in the
    same utterance to avoid over-correction in unrelated sentences.

    Examples
    --------
    - "Throughout 5 cm" -> "trout 5 cm"
    - "sea bus 27 centimeters" -> "sea bass 27 centimeters"

    Parameters
    ----------
    text: str
        Raw ASR text.

    Returns
    -------
    str
        Lowercased, lightly normalized text suitable for parsing.
    """
    t = (text or "").strip().lower()

    if not t:
        return t

    # Helper: does the utterance contain a number or a unit token?
    has_numeric_or_unit = bool(
        re.search(r"\b(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|cm|centi\w*|mm|milli\w*)\b", t)
    )

    # Only apply aggressive species corrections if there is numeric/unit context
    if has_numeric_or_unit:
        # Common confusion: "throughout" -> "trout"
        t = re.sub(r"\bthru\s*out\b", "trout", t)
        t = re.sub(r"\btrue\s*out\b", "trout", t)
        t = re.sub(r"\bthroughout\b", "trout", t)

        # Sea bass variations
        t = re.sub(r"\bsea\s+bus\b", "sea bass", t)
        t = re.sub(r"\bseabass\b", "sea bass", t)

    # Light unit/number cleanups that help downstream parsing
    t = _apply_asr_unit_and_number_corrections(t)

    return t

def words_to_number(tokens: List[str]) -> Optional[float]:
    """
    Parse number-word tokens into numeric value.
    Handles: "twenty seven point five", "one hundred and twenty", "three", etc.
    Returns None if cannot parse.
    """
    if not tokens:
        return None

    # Pre-normalize common ASR misheard number tokens
    tokens_normalized: List[str] = []
    for tok in tokens:
        t = tok.lower()
        if t in IGNORED_TOKENS:
            continue
        if t in MISHEARD_NUMBER_TOKENS:
            tokens_normalized.append(MISHEARD_NUMBER_TOKENS[t])
        else:
            tokens_normalized.append(t)

    total = 0
    current = 0
    decimal_part_digits: List[int] = []
    in_decimal = False

    for token in tokens_normalized:
        t = token.lower()
        if t in DECIMAL_TOKENS:
            if in_decimal:
                # repeated decimal token -> invalid
                return None
            in_decimal = True
            continue
        if t in NUMBER_WORDS:
            value = NUMBER_WORDS[t]
            if in_decimal:
                # append decimal digits (value may be >=10 => split)
                if value < 10:
                    decimal_part_digits.append(value)
                else:
                    for d in str(value):
                        decimal_part_digits.append(int(d))
            else:
                if value == 100:
                    if current == 0:
                        current = 1
                    current *= 100
                else:
                    current += value
        else:
            # unknown token in numeric region -> cannot parse
            return None

    total += current
    if in_decimal:
        if not decimal_part_digits:
            return None
        decimal_value = float("0." + "".join(str(d) for d in decimal_part_digits))
        return total + decimal_value
    return float(total) if total != 0 or current != 0 else None


def _longest_spoken_number(tokens: List[str]) -> Optional[float]:
    """
    Find the longest contiguous run of number words and parse it.
    """
    best_val: Optional[float] = None
    i = 0
    n = len(tokens)
    while i < n:
        if tokens[i].lower() in NUMBER_WORDS or tokens[i].lower() in DECIMAL_TOKENS or tokens[i].lower() in MISHEARD_NUMBER_TOKENS:
            j = i
            while j < n and (tokens[j].lower() in NUMBER_WORDS or tokens[j].lower() in DECIMAL_TOKENS or tokens[j].lower() in MISHEARD_NUMBER_TOKENS or tokens[j].lower() in IGNORED_TOKENS):
                j += 1
            val = words_to_number(tokens[i:j])
            if val is not None:
                # prefer a later/bigger parse (keeps longest)
                best_val = val
            i = j
        else:
            i += 1
    return best_val


def _fuzzy_find_unit_index(tokens: List[str]) -> Optional[int]:
    """
    Return index of a token that looks like a unit (fuzzy match against UNIT_SYNONYMS keys).
    """
    candidates = list(UNIT_SYNONYMS.keys())
    for i, tok in enumerate(tokens):
        # exact quick checks
        if tok in UNIT_SYNONYMS:
            return i
        # fuzzy match single token
        best = process.extractOne(tok, candidates, scorer=fuzz.ratio)
        if best and best[1] >= 80:
            return i
        # if token endswith 'cm' or 'mm' or contains 'cent' treat as unit
        if re.search(r"(^cm$|^mm$|cent|millim)", tok):
            return i
    # also detect two-token unit like "see em"
    for i in range(len(tokens) - 1):
        pair = f"{tokens[i]} {tokens[i+1]}"
        best = process.extractOne(pair, candidates, scorer=fuzz.ratio)
        if best and best[1] >= 80:
            return i
    return None


def extract_number_with_units(text: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Extract numeric length (converted to cm) and unit from free text.
    Returns (length_value_in_cm, 'cm' | None)
    """
    if not text:
        return None, None

    t = text.strip().lower()

    # stage 0: quick numeric + unit regex (handles digits)
    m = _NUMERIC_WITH_UNIT_RE.search(t)
    if m:
        num = float(m.group(1).replace(",", "."))
        unit = m.group(2).lower()
        norm = UNIT_SYNONYMS.get(unit, UNIT_SYNONYMS.get(unit.replace(" ", ""), "cm"))
        if norm == "mm":
            return num / 10.0, "cm"
        return num, "cm"

    # stage 0.5: ordinal digits like "3rd cm"
    m = _ORDINAL_RE.search(t)
    if m:
        num = float(m.group(1))
        unit_match = re.search(r"\b(cm|centimeters?|centimetres?|mm|millimeters?|millimetres?)\b", t, re.IGNORECASE)
        if unit_match:
            unit = unit_match.group(1).lower()
            norm = UNIT_SYNONYMS.get(unit, "cm")
            if norm == "mm":
                return num / 10.0, "cm"
            return num, "cm"
        else:
            return num, "cm"

    # stage 1: numeric alone + scattered unit word
    m = _NUMERIC_RE.search(t)
    if m:
        try:
            num = float(m.group(1).replace(",", "."))
        except ValueError:
            num = None
        if num is not None:
            unit_match = re.search(r"\b(cm|centimeters?|centimetres?|mm|millimeters?|millimetres?|milimeters?)\b", t, re.IGNORECASE)
            if unit_match:
                unit = unit_match.group(1).lower()
                norm = UNIT_SYNONYMS.get(unit, "cm")
                if norm == "mm":
                    return num / 10.0, "cm"
                return num, "cm"
            # no explicit unit -> default to cm
            return num, "cm"

    # stage 2: attempt ASR targeted corrections and re-run numeric regexes on corrected text
    corrected = _apply_asr_unit_and_number_corrections(t)
    if corrected != t:
        # try again with corrected
        m2 = _NUMERIC_WITH_UNIT_RE.search(corrected)
        if m2:
            num = float(m2.group(1).replace(",", "."))
            unit = m2.group(2).lower()
            norm = UNIT_SYNONYMS.get(unit, "cm")
            if norm == "mm":
                return num / 10.0, "cm"
            return num, "cm"
        m2 = _NUMERIC_RE.search(corrected)
        if m2:
            try:
                num = float(m2.group(1).replace(",", "."))
            except ValueError:
                num = None
            if num is not None:
                unit_match = re.search(r"\b(cm|centimeters?|centimetres?|mm|millimeters?|millimetres?|milimeters?)\b", corrected, re.IGNORECASE)
                if unit_match:
                    unit = unit_match.group(1).lower()
                    norm = UNIT_SYNONYMS.get(unit, "cm")
                    if norm == "mm":
                        return num / 10.0, "cm"
                    return num, "cm"
                return num, "cm"

    # stage 3: spoken number + unit detection using token parsing + fuzzy unit matching
    tokens = _tokenize_text_for_parser(t)
    # split tokens further if they contain unit-substrings (like "precentimeters")
    expanded_tokens: List[str] = []
    for tok in tokens:
        # split out tokens that contain 'cent' or 'mm' etc.
        if re.search(r"(centi|cent|millim|mm)", tok):
            # try to find the first occurrence of 'cent' or 'm' as boundary
            mcent = re.search(r"(cent|centi|millim|mm)", tok)
            if mcent:
                idx = mcent.start()
                prefix = tok[:idx]
                suffix = tok[idx:]
                if prefix:
                    expanded_tokens.append(prefix)
                expanded_tokens.append(suffix)
            else:
                expanded_tokens.append(tok)
        else:
            expanded_tokens.append(tok)
    tokens = [tok for tok in expanded_tokens if tok]

    # attempt to find a unit index via fuzzy matching
    unit_idx = _fuzzy_find_unit_index(tokens)
    if unit_idx is not None:
        # number is likely before unit
        # gather tokens before unit_idx as the number
        number_tokens = tokens[:unit_idx]
        if not number_tokens:
            # maybe number after unit: e.g., "cm twenty three"
            number_tokens = tokens[unit_idx + 1 :]
        num_val = words_to_number(number_tokens) if number_tokens else None

        # If still None, try longest spoken number anywhere
        if num_val is None:
            num_val = _longest_spoken_number(tokens)

        if num_val is not None:
            unit_tok = tokens[unit_idx].lower()
            unit_norm = UNIT_SYNONYMS.get(unit_tok, None)
            if unit_norm is None:
                # fuzzy-resolve unit token
                best = process.extractOne(unit_tok, list(UNIT_SYNONYMS.keys()), scorer=fuzz.ratio)
                if best and best[1] >= 80:
                    unit_norm = UNIT_SYNONYMS.get(best[0], "cm")
                else:
                    unit_norm = "cm"
            if unit_norm == "mm":
                return float(num_val) / 10.0, "cm"
            return float(num_val), "cm"

    # stage 4: spoken number without unit -> assume cm
    num_val = _longest_spoken_number(tokens)
    if num_val is not None:
        return float(num_val), "cm"

    return None, None


def parse_text(text: str) -> ParserResult:
    if not text:
        return ParserResult(cancel=False, species=None, length_cm=None)

    # normalize whitespace and lowercase for command parsing
    text_norm = text.strip()

    # Cancel command (case-insensitive)
    if re.search(r"\bcancel\b", text_norm, re.IGNORECASE):
        return ParserResult(cancel=True, species=None, length_cm=None)

    # Extract number and unit (returns cm)
    length_val, unit = extract_number_with_units(text_norm)

    # Extract species via improved fuzzy match
    species = fuzzy_match_species(text_norm)
    if species:
        species = species.title()

    if length_val is not None and unit == "cm" and species is not None:
        return ParserResult(cancel=False, species=species, length_cm=float(length_val))

    return ParserResult(cancel=False, species=species, length_cm=length_val)


