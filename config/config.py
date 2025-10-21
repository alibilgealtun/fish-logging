"""Enhanced configuration system with comprehensive loading and validation.

Implements a hierarchical configuration system with the following precedence:
1. Default values (lowest priority)
2. JSON configuration files
3. Environment variables
4. Command-line arguments (highest priority)

Configuration is deep-merged across all sources, allowing partial overrides
at any level of the configuration hierarchy.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import argparse

from core.exceptions import ConfigurationError


@dataclass(frozen=True)
class SpeechConfig:
    """Speech recognition configuration.

    Attributes:
        engine: Speech recognition engine name
        numbers_only: Enable numbers-only parsing mode
        model_path: Optional custom model path
        language: Recognition language code
        noise_profile: Acoustic environment profile for optimized parameters
    """
    engine: str
    numbers_only: bool
    model_path: Optional[str] = None
    language: str = "en"
    noise_profile: str = "mixed"  # clean | human | engine | mixed

    def __post_init__(self):
        if self.engine not in ("whisper", "whisperx", "vosk", "google", "assemblyai", "gemini", "chirp", "wav2vec2"):
            raise ConfigurationError(f"Invalid engine: {self.engine}")
        if self.noise_profile not in {"clean", "human", "engine", "mixed"}:
            raise ConfigurationError(f"Invalid noise_profile: {self.noise_profile}")


@dataclass(frozen=True)
class DatabaseConfig:
    """Database and logging configuration.

    Attributes:
        excel_output_path: Path to Excel output file for fish data
        session_log_dir: Directory for session log files
        backup_enabled: Enable automatic backup functionality
    """
    excel_output_path: str = "logs/hauls/logs.xlsx"
    session_log_dir: str = "logs/sessions"
    backup_enabled: bool = True


@dataclass(frozen=True)
class AudioConfig:
    """Audio storage configuration.

    Attributes:
        segments_dir: Directory for saving audio segments
        save_segments: Enable saving of audio segments for debugging/analysis
    """
    segments_dir: str = "audio/segments"
    save_segments: bool = False


@dataclass(frozen=True)
class UIConfig:
    """User interface configuration.

    Attributes:
        theme: UI theme name
        window_size: Default window dimensions (width, height)
        auto_save_interval: Auto-save interval in seconds
    """
    theme: str = "default"
    window_size: Tuple[int, int] = (1200, 800)
    auto_save_interval: int = 30


@dataclass(frozen=True)
class AppConfig:
    """Complete application configuration.

    Aggregates all configuration sections and loaded JSON data.

    Attributes:
        speech: Speech recognition settings
        database: Database and logging settings
        audio: Audio storage settings
        ui: User interface settings
        asr_corrections: ASR text corrections mapping
        species_data: Species configuration data
        numbers_data: Number parsing configuration
        units_data: Unit conversion configuration
        google_sheets_config: Google Sheets integration config
        debug: Debug mode flag
        log_level: Logging verbosity level
    """
    speech: SpeechConfig
    database: DatabaseConfig
    audio: AudioConfig
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
    """Centralized configuration loader with validation and hierarchy.

    Implements the configuration loading strategy with proper precedence
    and deep merging of nested configuration dictionaries.
    """

    def __init__(self, config_dir: Path = Path("config")):
        self.config_dir = config_dir

    def load(self, argv: List[str]) -> Tuple[AppConfig, List[str]]:
        """Load configuration with proper hierarchy: defaults → files → env → CLI.

        Args:
            argv: Command-line arguments to parse

        Returns:
            Tuple of (AppConfig instance, unknown CLI arguments)
        """
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
            "audio": {
                "segments_dir": "audio/segments",
                "save_segments": False,
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
        """Load all JSON configuration files.

        Returns:
            Dictionary containing loaded JSON data, with empty dicts for missing files
        """
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
        """Load configuration overrides from environment variables.

        Supported environment variables:
        - SPEECH_ENGINE: Speech recognition engine
        - SPEECH_NUMBERS_ONLY: Enable numbers-only mode
        - SPEECH_LANGUAGE: Recognition language
        - SPEECH_NOISE_PROFILE: Noise environment profile
        - DEBUG: Enable debug mode
        - LOG_LEVEL: Set logging level

        Returns:
            Dictionary with environment-based overrides
        """
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
        """Parse CLI arguments.

        Args:
            argv: Command-line arguments

        Returns:
            Tuple of (overrides dictionary, unknown arguments)
        """
        parser = argparse.ArgumentParser(description="Fish logging application")

        parser.add_argument(
            "--engine", "--model",
            choices=["whisper", "whisperx", "vosk", "google", "assemblyai", "gemini", "chirp", "wav2vec2"],
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
        parser.add_argument(
            "--save-audio",
            action="store_true",
            help="Save audio segments to disk for debugging/analysis"
        )

        known, unknown = parser.parse_known_args(argv)

        # Build overrides from parsed arguments
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
        if known.save_audio:
            audio_overrides = overrides.setdefault("audio", {})
            if isinstance(audio_overrides, dict):
                audio_overrides["save_segments"] = True

        return overrides, unknown

    def _build_config(self, config_dict: Dict[str, Any]) -> AppConfig:
        """Build and validate the final configuration object.

        Args:
            config_dict: Merged configuration dictionary

        Returns:
            Validated AppConfig instance

        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Extract nested configs
        speech_config = SpeechConfig(**config_dict.get("speech", {}))
        database_config = DatabaseConfig(**config_dict.get("database", {}))
        audio_config = AudioConfig(**config_dict.get("audio", {}))
        ui_config = UIConfig(**config_dict.get("ui", {}))

        # Build main config
        return AppConfig(
            speech=speech_config,
            database=database_config,
            audio=audio_config,
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
        """Parse boolean from environment variable.

        Args:
            name: Environment variable name
            default: Default value if not set

        Returns:
            Boolean value (True for "1", "true", "yes", "y", "on")
        """
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

        Args:
            target: Dictionary to update (modified in place)
            updates: Dictionary with new values
        """
        for key, new_val in updates.items():
            if isinstance(new_val, dict) and isinstance(target.get(key), dict):
                ConfigLoader._deep_update(target[key], new_val)  # type: ignore[index]
            else:
                target[key] = new_val


def parse_app_args(argv: List[str]) -> Tuple[AppConfig, List[str]]:
    """Parse application configuration with enhanced loading.

    Convenience function for creating a ConfigLoader and loading configuration.

    Args:
        argv: Command-line arguments

    Returns:
        Tuple of (AppConfig instance, unknown arguments)
    """
    loader = ConfigLoader()
    return loader.load(argv)


__all__ = ["AppConfig", "SpeechConfig", "DatabaseConfig", "AudioConfig", "UIConfig", "ConfigLoader", "parse_app_args"]
