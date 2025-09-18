"""Configuration management for the fish parser."""

import json
from pathlib import Path
from typing import Dict, List, Set


class ConfigManager:
    """Manages configuration data loaded from JSON files."""

    def __init__(self, config_path: str = "config/"):
        """Initialize configuration manager."""
        self.config_path = Path(config_path)
        self._load_all_configs()

    def _load_all_configs(self) -> None:
        """Load all configuration files."""
        self.species_data = self._load_json("../config/species.json")
        self.units_data = self._load_json("../config/units.json")
        self.numbers_data = self._load_json("../config/numbers.json")
        self.asr_corrections = self._load_json("../config/asr_corrections.json")

    def _load_json(self, filename: str) -> Dict:
        """Load a JSON configuration file."""
        file_path = self.config_path / filename
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {e}")

    @property
    def species(self) -> List[str]:
        """Get list of known fish species.

        Supports both legacy schema with a flat "species" list and
        the new structured schema with an ordered "items" list of
        objects containing name and code.
        """
        data = self.species_data
        if isinstance(data.get("items"), list) and data["items"]:
            return [str(entry.get("name", "")).strip() for entry in data["items"] if str(entry.get("name", "")).strip()]
        # Legacy fallback
        return data.get("species", [])

    @property
    def species_normalization(self) -> Dict[str, str]:
        """Get species name normalization mappings."""
        return self.species_data["normalization"]

    @property
    def unit_synonyms(self) -> Dict[str, str]:
        """Get unit synonym mappings."""
        return self.units_data["synonyms"]

    @property
    def number_words(self) -> Dict[str, int]:
        """Get number word mappings."""
        return self.numbers_data["number_words"]

    @property
    def decimal_tokens(self) -> Set[str]:
        """Get decimal point tokens."""
        return set(self.numbers_data["decimal_tokens"])

    @property
    def ignored_tokens(self) -> Set[str]:
        """Get tokens to ignore during parsing."""
        return set(self.numbers_data["ignored_tokens"])

    @property
    def misheard_number_tokens(self) -> Dict[str, str]:
        """Get common ASR mishearings for numbers."""
        return self.numbers_data["misheard_tokens"]

    @property
    def species_corrections(self) -> Dict[str, str]:
        """Get ASR corrections for species names."""
        return self.asr_corrections["species"]

    @property
    def unit_corrections(self) -> Dict[str, str]:
        """Get ASR corrections for units."""
        return self.asr_corrections["units"]