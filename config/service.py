"""Configuration service facade for simplified configuration access.

Implements the Facade pattern to provide a clean, simple interface
to the complex configuration system. Reduces boilerplate and nesting
in client code.
"""
from __future__ import annotations

from typing import Any, Optional
from pathlib import Path

from config.config import AppConfig, ConfigLoader


class ConfigurationService:
    """Facade for application configuration management.

    Provides simplified access to configuration values without
    deep nesting and verbose attribute access. All properties
    delegate to the underlying AppConfig instance.

    Example:
        config_service = ConfigurationService(config)
        engine = config_service.engine  # Instead of config.speech.engine

    Attributes:
        _config: Underlying AppConfig instance
    """

    def __init__(self, config: AppConfig):
        self._config = config

    # Speech configuration shortcuts
    @property
    def engine(self) -> str:
        """Get speech recognition engine name."""
        return self._config.speech.engine

    @property
    def numbers_only(self) -> bool:
        """Get whether numbers-only mode is enabled."""
        return self._config.speech.numbers_only

    @property
    def noise_profile(self) -> str:
        """Get noise profile setting."""
        return self._config.speech.noise_profile

    @property
    def language(self) -> str:
        """Get language setting."""
        return self._config.speech.language

    # Database/logging configuration
    @property
    def excel_path(self) -> str:
        """Get Excel output path."""
        return self._config.database.excel_output_path

    @property
    def session_log_dir(self) -> str:
        """Get session log directory."""
        return self._config.database.session_log_dir

    @property
    def backup_enabled(self) -> bool:
        """Get whether backup is enabled."""
        return self._config.database.backup_enabled

    # Audio configuration
    @property
    def audio_segments_dir(self) -> str:
        """Get audio segments directory."""
        return self._config.audio.segments_dir

    @property
    def save_audio_segments(self) -> bool:
        """Get whether to save audio segments."""
        return self._config.audio.save_segments

    # UI configuration
    @property
    def theme(self) -> str:
        """Get UI theme."""
        return self._config.ui.theme

    @property
    def window_size(self) -> tuple[int, int]:
        """Get default window size."""
        return self._config.ui.window_size

    # General configuration
    @property
    def debug(self) -> bool:
        """Get debug mode status."""
        return self._config.debug

    @property
    def log_level(self) -> str:
        """Get log level."""
        return self._config.log_level

    # Data access
    def get_species_data(self) -> dict[str, Any]:
        """Get species configuration data."""
        return self._config.species_data

    def get_asr_corrections(self) -> dict[str, str]:
        """Get ASR corrections mapping."""
        return self._config.asr_corrections

    def get_numbers_data(self) -> dict[str, Any]:
        """Get numbers configuration data."""
        return self._config.numbers_data

    def get_units_data(self) -> dict[str, Any]:
        """Get units configuration data."""
        return self._config.units_data

    def get_google_sheets_config(self) -> dict[str, Any]:
        """Get Google Sheets configuration."""
        return self._config.google_sheets_config

    # Direct config access (for advanced use)
    @property
    def raw_config(self) -> AppConfig:
        """Get raw configuration object.

        Returns:
            Underlying AppConfig instance for direct access
        """
        return self._config

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for serialization.

        Returns:
            Dictionary representation of current configuration
        """
        return {
            "speech": {
                "engine": self.engine,
                "numbers_only": self.numbers_only,
                "noise_profile": self.noise_profile,
                "language": self.language,
            },
            "database": {
                "excel_path": self.excel_path,
                "session_log_dir": self.session_log_dir,
                "backup_enabled": self.backup_enabled,
            },
            "audio": {
                "segments_dir": self.audio_segments_dir,
                "save_segments": self.save_audio_segments,
            },
            "ui": {
                "theme": self.theme,
                "window_size": self.window_size,
            },
            "debug": self.debug,
            "log_level": self.log_level,
        }


class ConfigurationServiceFactory:
    """Factory for creating ConfigurationService instances.

    Provides static factory methods for common creation patterns,
    encapsulating the construction logic.
    """

    @staticmethod
    def create_from_args(args: list[str]) -> tuple[ConfigurationService, list[str]]:
        """Create configuration service from command-line arguments.

        Args:
            args: Command-line arguments

        Returns:
            Tuple of (ConfigurationService, unknown_args)
        """
        loader = ConfigLoader()
        config, unknown_args = loader.load(args)
        return ConfigurationService(config), unknown_args

    @staticmethod
    def create_from_config(config: AppConfig) -> ConfigurationService:
        """Create configuration service from existing config.

        Args:
            config: AppConfig instance

        Returns:
            ConfigurationService instance
        """
        return ConfigurationService(config)

    @staticmethod
    def create_default() -> ConfigurationService:
        """Create configuration service with defaults.

        Returns:
            ConfigurationService with default configuration
        """
        loader = ConfigLoader()
        config, _ = loader.load([])
        return ConfigurationService(config)
