"""Configuration management for the fish parser."""

import json
from pathlib import Path
from typing import Dict, List, Set


class ConfigManager:
    """Manages configuration data from centralized config object."""

    def __init__(self, app_config=None):
        """Initialize configuration manager with centralized config."""
        if app_config is None:
            # Fallback for backward compatibility - load config directly
            from config.config import ConfigLoader
            loader = ConfigLoader()
            app_config, _ = loader.load([])
        
        self.app_config = app_config
        self.species_data = app_config.species_data
        self.units_data = app_config.units_data  
        self.numbers_data = app_config.numbers_data
        self.asr_corrections = app_config.asr_corrections

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
        return self.species_data.get("normalization", {})

    @property
    def species_corrections(self) -> Dict[str, str]:
        """Get species name corrections from ASR corrections."""
        return self.asr_corrections.get("species", {})

    @property
    def unit_synonyms(self) -> Dict[str, str]:
        """Get unit synonyms mapping from units data."""
        return self.units_data.get("synonyms", {})

    @property
    def number_words(self) -> Dict[str, int]:
        """Get number words mapping from numbers data."""
        return self.numbers_data.get("number_words", {})

    @property
    def decimal_tokens(self) -> List[str]:
        """Get decimal tokens from numbers data."""
        return self.numbers_data.get("decimal_tokens", [])

    @property
    def ignored_tokens(self) -> List[str]:
        """Get ignored tokens from numbers data."""
        return self.numbers_data.get("ignored_tokens", [])

    @property
    def misheard_number_tokens(self) -> Dict[str, str]:
        """Get misheard number tokens mapping from numbers data."""
        return self.numbers_data.get("misheard_tokens", {})
