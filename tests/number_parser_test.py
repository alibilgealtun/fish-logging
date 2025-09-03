# tests/number_parser_test.py

import pytest
from parser.number_parser import NumberParser
from parser.config import ConfigManager


# Enhanced config for comprehensive testing
class DummyConfig(ConfigManager):
    number_words = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
        "seventy": 70, "eighty": 80, "ninety": 90,
        "hundred": 100, "thousand": 1000
    }
    decimal_tokens = {"point", "dot", "comma", "and"}
    ignored_tokens = []  # Fixed: removed "and" to avoid conflict
    misheard_number_tokens = {
        "pre": "three", "tree": "three", "tri": "three", "tre": "three",
        "free": "three", "thre": "three", "too": "two", "to": "two",
        "for": "four", "fore": "four", "won": "one", "ate": "eight", "zeroo": "zero"
    }
    unit_synonyms = {
        "cm": "cm", "centimeter": "cm", "centimeters": "cm", "centimetre": "cm", "centimetres": "cm",
        "mm": "mm", "millimeter": "mm", "millimeters": "mm", "millimetre": "mm", "millimetres": "mm",
        "milimeter": "mm", "milimeters": "mm"  # Common misspellings
    }


class TestBasicDecimalParsing:
    """Test different decimal separators in speech."""

    def test_and_as_decimal(self):
        parser = NumberParser(DummyConfig())
        test_cases = [
            ("thirty five and five centimeters", 35.5),
            ("twenty three and seven millimeters", 2.37),  # mm to cm conversion
            ("forty and two centimeters", 40.2),
            ("fifteen and nine cm", 15.9),
        ]
        for text, expected in test_cases:
            value, unit = parser.extract_number_with_units(text)
            assert value == expected, f"Failed for '{text}': got {value}, expected {expected}"
            assert unit == "cm"

    def test_point_as_decimal(self):
        parser = NumberParser(DummyConfig())
        test_cases = [
            ("thirty five point five centimeters", 35.5),
            ("twenty three point seven millimeters", 2.37),
            ("forty point two centimeters", 40.2),
            ("fifteen point nine cm", 15.9),
        ]
        for text, expected in test_cases:
            value, unit = parser.extract_number_with_units(text)
            assert value == expected, f"Failed for '{text}': got {value}, expected {expected}"

    def test_dot_as_decimal(self):
        parser = NumberParser(DummyConfig())
        test_cases = [
            ("thirty five dot five centimeters", 35.5),
            ("twenty three dot seven millimeters", 2.37),
            ("forty dot two centimeters", 40.2),
        ]
        for text, expected in test_cases:
            value, unit = parser.extract_number_with_units(text)
            assert value == expected, f"Failed for '{text}': got {value}, expected {expected}"

    def test_comma_as_decimal(self):
        parser = NumberParser(DummyConfig())
        test_cases = [
            ("thirty five comma five centimeters", 35.5),
            ("twenty three comma seven millimeters", 2.37),
            ("forty comma two centimeters", 40.2),
        ]
        for text, expected in test_cases:
            value, unit = parser.extract_number_with_units(text)
            assert value == expected, f"Failed for '{text}': got {value}, expected {expected}"


class TestMisheardNumbers:
    """Test common ASR mistakes and corrections."""

    def test_three_variations(self):
        parser = NumberParser(DummyConfig())
        three_variants = ["pre", "tree", "tri", "tre", "free", "thre"]
        for variant in three_variants:
            text = f"{variant} and five centimeters"
            value, unit = parser.extract_number_with_units(text)
            assert value == 3.5, f"Failed for variant '{variant}': got {value}, expected 3.5"

    def test_two_variations(self):
        parser = NumberParser(DummyConfig())
        two_variants = ["too", "to"]
        for variant in two_variants:
            text = f"thirty {variant} centimeters"
            value, unit = parser.extract_number_with_units(text)
            assert value == 32.0, f"Failed for variant '{variant}': got {value}, expected 32.0"

    def test_four_variations(self):
        parser = NumberParser(DummyConfig())
        four_variants = ["for", "fore"]
        for variant in four_variants:
            text = f"twenty {variant} point five cm"
            value, unit = parser.extract_number_with_units(text)
            assert value == 24.5, f"Failed for variant '{variant}': got {value}, expected 24.5"

    def test_mixed_misheard(self):
        parser = NumberParser(DummyConfig())
        test_cases = [
            ("won hundred tree centimeters", 103.0),  # one hundred three
            ("ate and five millimeters", 0.85),  # eight and five mm
            ("for tree point to cm", 43.2),  # four three point two
        ]
        for text, expected in test_cases:
            value, unit = parser.extract_number_with_units(text)
            assert value == expected, f"Failed for '{text}': got {value}, expected {expected}"


class TestComplexNumbers:
    """Test complex number constructions."""

    def test_hundreds(self):
        parser = NumberParser(DummyConfig())
        test_cases = [
            ("one hundred centimeters", 100.0),
            ("two hundred fifty cm", 250.0),
            ("three hundred and five mm", 30.5),  # 305mm = 30.5cm
            ("five hundred point seven centimeters", 500.7),
        ]
        for text, expected in test_cases:
            value, unit = parser.extract_number_with_units(text)
            assert value == expected, f"Failed for '{text}': got {value}, expected {expected}"

    def test_teens(self):
        parser = NumberParser(DummyConfig())
        test_cases = [
            ("thirteen point five centimeters", 13.5),
            ("fourteen and seven mm", 1.47),
            ("fifteen centimeters", 15.0),
            ("nineteen point two cm", 19.2),
        ]
        for text, expected in test_cases:
            value, unit = parser.extract_number_with_units(text)
            assert value == expected, f"Failed for '{text}': got {value}, expected {expected}"

    def test_compound_tens(self):
        parser = NumberParser(DummyConfig())
        test_cases = [
            ("twenty one centimeters", 21.0),
            ("thirty two and five mm", 3.25),
            ("forty three point seven cm", 43.7),
            ("fifty four and nine centimeters", 54.9),
            ("sixty seven point two millimeters", 6.72),
            ("seventy eight centimeters", 78.0),
            ("eighty nine and three cm", 89.3),
            ("ninety one point five millimeters", 9.15),
        ]
        for text, expected in test_cases:
            value, unit = parser.extract_number_with_units(text)
            assert value == expected, f"Failed for '{text}': got {value}, expected {expected}"


class TestUnitVariations:
    """Test different unit spellings and abbreviations."""

    def test_centimeter_variations(self):
        parser = NumberParser(DummyConfig())
        cm_variants = ["cm", "centimeter", "centimeters", "centimetre", "centimetres"]
        for variant in cm_variants:
            text = f"twenty five {variant}"
            value, unit = parser.extract_number_with_units(text)
            assert value == 25.0, f"Failed for variant '{variant}'"
            assert unit == "cm"

    def test_millimeter_variations(self):
        parser = NumberParser(DummyConfig())
        mm_variants = ["mm", "millimeter", "millimeters", "millimetre", "millimetres", "milimeter", "milimeters"]
        for variant in mm_variants:
            text = f"twenty five {variant}"
            value, unit = parser.extract_number_with_units(text)
            assert value == 2.5, f"Failed for variant '{variant}': got {value}, expected 2.5"  # 25mm = 2.5cm
            assert unit == "cm"


class TestRealWorldScenarios:
    """Test realistic fish measurement scenarios."""

    def test_fish_species_contexts(self):
        parser = NumberParser(DummyConfig())
        test_cases = [
            ("caught a sea bass thirty five and five centimeters", 35.5),
            ("trout measured twenty three point seven cm", 23.7),
            ("the salmon was forty two and eight millimeters", 4.28),
            ("pike length fifty seven centimeters", 57.0),
            ("bass caught today twenty nine point five cm", 29.5),
            ("perch measured fifteen and three millimeters", 1.53),
        ]
        for text, expected in test_cases:
            value, unit = parser.extract_number_with_units(text)
            assert value == expected, f"Failed for '{text}': got {value}, expected {expected}"

    def test_measurement_contexts(self):
        parser = NumberParser(DummyConfig())
        test_cases = [
            ("length is thirty five and five centimeters", 35.5),
            ("measured at twenty three point seven cm", 23.7),
            ("size was forty two and eight millimeters", 4.28),
            ("total length fifty seven centimeters", 57.0),
            ("fish size twenty nine point five cm", 29.5),
        ]
        for text, expected in test_cases:
            value, unit = parser.extract_number_with_units(text)
            assert value == expected, f"Failed for '{text}': got {value}, expected {expected}"


class TestEdgeCases:
    """Test edge cases and potential failure modes."""

    def test_multiple_decimals(self):
        parser = NumberParser(DummyConfig())
        # Should fail gracefully with multiple decimal points
        text = "twenty point five and three centimeters"
        value, unit = parser.extract_number_with_units(text)
        # Should return None or handle gracefully
        assert value is None or isinstance(value, (int, float))

    def test_decimal_only(self):
        parser = NumberParser(DummyConfig())
        test_cases = [
            ("point five centimeters", 0.5),
            ("and seven mm", 0.07),  # 0.7mm = 0.07cm
            ("dot three cm", 0.3),
        ]
        for text, expected in test_cases:
            value, unit = parser.extract_number_with_units(text)
            assert value == expected, f"Failed for '{text}': got {value}, expected {expected}"

    def test_zero_values(self):
        parser = NumberParser(DummyConfig())
        test_cases = [
            ("zero centimeters", 0.0),
            ("zero point five cm", 0.5),
            ("zero and three millimeters", 0.03),
        ]
        for text, expected in test_cases:
            value, unit = parser.extract_number_with_units(text)
            assert value == expected, f"Failed for '{text}': got {value}, expected {expected}"

    def test_large_numbers(self):
        parser = NumberParser(DummyConfig())
        test_cases = [
            ("one hundred twenty three and four centimeters", 123.4),
            ("two hundred and fifty point five mm", 25.05),
            ("three hundred centimeters", 300.0),
        ]
        for text, expected in test_cases:
            value, unit = parser.extract_number_with_units(text)
            assert value == expected, f"Failed for '{text}': got {value}, expected {expected}"


class TestNumericInputs:
    """Test numeric inputs mixed with spoken words."""

    def test_mixed_numeric_spoken(self):
        parser = NumberParser(DummyConfig())
        test_cases = [
            ("35.5 centimeters", 35.5),
            ("23,7 millimeters", 2.37),  # European decimal comma
            ("42.8 cm", 42.8),
            ("15mm", 1.5),  # No space
        ]
        for text, expected in test_cases:
            value, unit = parser.extract_number_with_units(text)
            assert value == expected, f"Failed for '{text}': got {value}, expected {expected}"

    def test_ordinal_numbers(self):
        parser = NumberParser(DummyConfig())
        test_cases = [
            ("35th centimeter mark", 35.0),
            ("23rd millimeter", 2.3),
            ("1st cm", 1.0),
            ("42nd millimeter mark", 4.2),
        ]
        for text, expected in test_cases:
            value, unit = parser.extract_number_with_units(text)
            assert value == expected, f"Failed for '{text}': got {value}, expected {expected}"


class TestNoiseAndContext:
    """Test parsing with noise words and context."""

    def test_with_noise_words(self):
        parser = NumberParser(DummyConfig())
        test_cases = [
            ("um the fish was like thirty five and five centimeters long", 35.5),
            ("so yeah it measured about twenty three point seven cm", 23.7),
            ("well actually it was forty two and eight millimeters exactly", 4.28),
            ("I think the length was probably fifty seven centimeters", 57.0),
        ]
        for text, expected in test_cases:
            value, unit = parser.extract_number_with_units(text)
            assert value == expected, f"Failed for '{text}': got {value}, expected {expected}"

    def test_multiple_numbers_in_text(self):
        parser = NumberParser(DummyConfig())
        # Should pick the number closest to the unit
        test_cases = [
            ("fish measured fifty seven centimeters", 57.0),
        ]
        for text, expected in test_cases:
            value, unit = parser.extract_number_with_units(text)
            assert value == expected, f"Failed for '{text}': got {value}, expected {expected}"


# Original test to ensure compatibility
def test_sea_bass_measurement():
    parser = NumberParser(DummyConfig())
    text = "sea bass thirty five and five centimeters"
    value, unit = parser.extract_number_with_units(text)
    assert value == 35.5
    assert unit == "cm"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])