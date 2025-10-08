"""Enhanced configuration system with comprehensive loading and validation."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import argparse


@dataclass(frozen=True)
class SpeechConfig:
    """Speech recognition configuration."""
    engine: str
    numbers_only: bool
    model_path: Optional[str] = None
    language: str = "en"
    noise_profile: str = "mixed"  # clean | human | engine | mixed

    def __post_init__(self):
        if self.engine not in ("whisper", "whisperx", "vosk", "google"):
            raise ValueError(f"Invalid engine: {self.engine}")
        if self.noise_profile not in {"clean", "human", "engine", "mixed"}:
            raise ValueError(f"Invalid noise_profile: {self.noise_profile}")


@dataclass(frozen=True)
class DatabaseConfig:
    """Database and logging configuration."""
    excel_output_path: str = "logs/hauls/logs.xlsx"
    session_log_dir: str = "logs/sessions"
    backup_enabled: bool = True


@dataclass(frozen=True)
class UIConfig:
    """User interface configuration."""
    theme: str = "default"
    window_size: Tuple[int, int] = (1200, 800)
    auto_save_interval: int = 30  # seconds


@dataclass(frozen=True)
class AppConfig:
    """Complete application configuration."""
    speech: SpeechConfig
    database: DatabaseConfig
    ui: UIConfig

    # Loaded from JSON files
    asr_corrections: Dict[str, str] = field(default_factory=dict)
    species_data: Dict[str, Any] = field(default_factory=dict)
    numbers_data: Dict[str, Any] = field(default_factory=dict)
    units_data: Dict[str, Any] = field(default_factory=dict)
    google_sheets_config: Dict[str, Any] = field(default_factory=dict)

    debug: bool = False
    log_level: str = "INFO"


class ConfigLoader:
    """Centralized configuration loader with validation and hierarchy."""

    def __init__(self, config_dir: Path = Path("config")):
        self.config_dir = config_dir

    def load(self, argv: List[str]) -> Tuple[AppConfig, List[str]]:
        """Load configuration with proper hierarchy: defaults → files → env → CLI."""
        # 1. Start with defaults
        config_dict = self._get_defaults()

        # 2. Load from JSON files (deep merge)
        json_data = self._load_json_configs()
        self._deep_update(config_dict, json_data)

        # 3. Override with environment variables (deep merge)
        env_overrides = self._load_env_overrides()
        self._deep_update(config_dict, env_overrides)

        # 4. Parse CLI arguments (highest priority, deep merge)
        cli_overrides, unknown_args = self._parse_cli_args(argv)
        self._deep_update(config_dict, cli_overrides)

        # 5. Build and validate final config
        config = self._build_config(config_dict)
        return config, unknown_args

    def _get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            "speech": {
                "engine": "whisper",
                "numbers_only": False,
                "language": "en",
                "noise_profile": "mixed",
            },
            "database": {
                "excel_output_path": "logs/hauls/logs.xlsx",
                "session_log_dir": "logs/sessions",
                "backup_enabled": True,
            },
            "ui": {
                "theme": "default",
                "window_size": (1200, 800),
                "auto_save_interval": 30,
            },
            "debug": False,
            "log_level": "INFO",
        }

    def _load_json_configs(self) -> Dict[str, Any]:
        """Load all JSON configuration files."""
        json_configs = {}

        json_files = {
            "asr_corrections": "asr_corrections.json",
            "species_data": "species.json",
            "numbers_data": "numbers.json",
            "units_data": "units.json",
            "google_sheets_config": "google_sheets.json",
        }

        for key, filename in json_files.items():
            file_path = self.config_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json_configs[key] = json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Warning: Failed to load {filename}: {e}")
                    json_configs[key] = {}
            else:
                json_configs[key] = {}

        return json_configs

    def _load_env_overrides(self) -> Dict[str, Any]:
        """Load configuration overrides from environment variables."""
        overrides: Dict[str, Any] = {}

        # Speech configuration
        speech_engine = os.getenv("SPEECH_ENGINE")
        if speech_engine:
            speech_overrides = overrides.setdefault("speech", {})
            if isinstance(speech_overrides, dict):
                speech_overrides["engine"] = speech_engine

        if self._env_bool("SPEECH_NUMBERS_ONLY"):
            speech_overrides = overrides.setdefault("speech", {})
            if isinstance(speech_overrides, dict):
                speech_overrides["numbers_only"] = True

        speech_language = os.getenv("SPEECH_LANGUAGE")
        if speech_language:
            speech_overrides = overrides.setdefault("speech", {})
            if isinstance(speech_overrides, dict):
                speech_overrides["language"] = speech_language

        noise_profile = os.getenv("SPEECH_NOISE_PROFILE")
        if noise_profile:
            speech_overrides = overrides.setdefault("speech", {})
            if isinstance(speech_overrides, dict):
                speech_overrides["noise_profile"] = noise_profile

        # Debug and logging
        if self._env_bool("DEBUG"):
            overrides["debug"] = True

        log_level = os.getenv("LOG_LEVEL")
        if log_level:
            overrides["log_level"] = log_level.upper()

        return overrides

    def _parse_cli_args(self, argv: List[str]) -> Tuple[Dict[str, Any], List[str]]:
        """Parse CLI arguments."""
        parser = argparse.ArgumentParser(description="Fish logging application")

        parser.add_argument(
            "--engine", "--model",
            choices=["whisper", "whisperx", "vosk", "google"],
            help="Speech recognition engine"
        )
        parser.add_argument(
            "--numbers-only",
            action="store_true",
            help="Enable numbers-only mode"
        )
        parser.add_argument(
            "--noise-profile",
            choices=["clean", "human", "engine", "mixed"],
            help="Noise environment profile (defaults to mixed)"
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug mode"
        )
        parser.add_argument(
            "--log-level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="Set logging level"
        )

        known, unknown = parser.parse_known_args(argv)

        overrides: Dict[str, Any] = {}
        if known.engine:
            speech_overrides = overrides.setdefault("speech", {})
            if isinstance(speech_overrides, dict):
                speech_overrides["engine"] = known.engine
        if known.numbers_only:
            speech_overrides = overrides.setdefault("speech", {})
            if isinstance(speech_overrides, dict):
                speech_overrides["numbers_only"] = True
        if known.noise_profile:
            speech_overrides = overrides.setdefault("speech", {})
            if isinstance(speech_overrides, dict):
                speech_overrides["noise_profile"] = known.noise_profile
        if known.debug:
            overrides["debug"] = True
        if known.log_level:
            overrides["log_level"] = known.log_level

        return overrides, unknown

    def _build_config(self, config_dict: Dict[str, Any]) -> AppConfig:
        """Build and validate the final configuration object."""
        # Extract nested configs
        speech_config = SpeechConfig(**config_dict.get("speech", {}))
        database_config = DatabaseConfig(**config_dict.get("database", {}))
        ui_config = UIConfig(**config_dict.get("ui", {}))

        # Build main config
        return AppConfig(
            speech=speech_config,
            database=database_config,
            ui=ui_config,
            asr_corrections=config_dict.get("asr_corrections", {}),
            species_data=config_dict.get("species_data", {}),
            numbers_data=config_dict.get("numbers_data", {}),
            units_data=config_dict.get("units_data", {}),
            google_sheets_config=config_dict.get("google_sheets_config", {}),
            debug=config_dict.get("debug", False),
            log_level=config_dict.get("log_level", "INFO"),
        )

    @staticmethod
    def _env_bool(name: str, default: bool = False) -> bool:
        """Parse boolean from environment variable."""
        val = os.getenv(name)
        if val is None:
            return default
        return val.strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _deep_update(target: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Recursively update mapping 'target' with 'updates' without clobbering nested dicts.

        - Only keys present in updates are applied.
        - For dict values, merge recursively.
        - For non-dict values, assign directly.
        """
        for key, new_val in updates.items():
            if isinstance(new_val, dict) and isinstance(target.get(key), dict):
                ConfigLoader._deep_update(target[key], new_val)  # type: ignore[index]
            else:
                target[key] = new_val


def parse_app_args(argv: List[str]) -> Tuple[AppConfig, List[str]]:
    """Parse application configuration with enhanced loading."""
    loader = ConfigLoader()
    return loader.load(argv)


__all__ = ["AppConfig", "SpeechConfig", "DatabaseConfig", "UIConfig", "ConfigLoader", "parse_app_args"]
