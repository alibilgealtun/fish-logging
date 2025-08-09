from parser import parse_text, apply_fish_asr_corrections


def test_parse_direct_numeric():
    r = parse_text("Sea bass 27.5 centimeters")
    assert not r.cancel
    assert r.species == "Sea Bass"
    assert r.length_cm == 27.5


def test_parse_numeric_first():
    r = parse_text("27.5 cm sea bass")
    assert r.species == "Sea Bass"
    assert r.length_cm == 27.5


def test_parse_spoken_number():
    r = parse_text("Sea bass length twenty seven point five")
    assert r.species == "Sea Bass"
    assert r.length_cm == 27.5


def test_cancel_command():
    r = parse_text("cancel")
    assert r.cancel


def test_asr_correction_throughout_to_trout_digits():
    """ASR 'throughout 5 cm' should normalize to 'trout 5 cm'."""
    s = apply_fish_asr_corrections("Throughout 5 cm")
    assert "trout" in s
    assert "5" in s and "cm" in s


def test_asr_correction_sea_bus_to_sea_bass():
    """ASR 'sea bus 27 centimeters' should normalize to 'sea bass 27 centimeters'."""
    s = apply_fish_asr_corrections("sea bus 27 centimeters")
    assert "sea bass" in s
    assert "27" in s and "centimeters" in s
