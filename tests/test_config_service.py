"""Unit tests for configuration service."""
import pytest
from unittest.mock import Mock, patch

from config.service import ConfigurationService, ConfigurationServiceFactory
from config.config import AppConfig, SpeechConfig, DatabaseConfig, AudioConfig, UIConfig


class TestConfigurationService:
    """Tests for ConfigurationService facade."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock AppConfig for testing."""
        return AppConfig(
            speech=SpeechConfig(
                engine="whisper",
                numbers_only=False,
                model_path=None,
                language="en",
                noise_profile="mixed"
            ),
            database=DatabaseConfig(
                excel_output_path="logs/hauls/test.xlsx",
                session_log_dir="logs/sessions",
                backup_enabled=True
            ),
            audio=AudioConfig(
                segments_dir="audio/segments",
                save_segments=True
            ),
            ui=UIConfig(
                theme="default",
                window_size=(1200, 800),
                auto_save_interval=30
            ),
            debug=False,
            log_level="INFO"
        )

    def test_engine_property(self, mock_config):
        # Arrange
        service = ConfigurationService(mock_config)

        # Assert
        assert service.engine == "whisper"

    def test_numbers_only_property(self, mock_config):
        # Arrange
        service = ConfigurationService(mock_config)

        # Assert
        assert service.numbers_only is False

    def test_noise_profile_property(self, mock_config):
        # Arrange
        service = ConfigurationService(mock_config)

        # Assert
        assert service.noise_profile == "mixed"

    def test_language_property(self, mock_config):
        # Arrange
        service = ConfigurationService(mock_config)

        # Assert
        assert service.language == "en"

    def test_excel_path_property(self, mock_config):
        # Arrange
        service = ConfigurationService(mock_config)

        # Assert
        assert service.excel_path == "logs/hauls/test.xlsx"

    def test_session_log_dir_property(self, mock_config):
        # Arrange
        service = ConfigurationService(mock_config)

        # Assert
        assert service.session_log_dir == "logs/sessions"

    def test_audio_segments_dir_property(self, mock_config):
        # Arrange
        service = ConfigurationService(mock_config)

        # Assert
        assert service.audio_segments_dir == "audio/segments"

    def test_save_audio_segments_property(self, mock_config):
        # Arrange
        service = ConfigurationService(mock_config)

        # Assert
        assert service.save_audio_segments is True

    def test_theme_property(self, mock_config):
        # Arrange
        service = ConfigurationService(mock_config)

        # Assert
        assert service.theme == "default"

    def test_window_size_property(self, mock_config):
        # Arrange
        service = ConfigurationService(mock_config)

        # Assert
        assert service.window_size == (1200, 800)

    def test_debug_property(self, mock_config):
        # Arrange
        service = ConfigurationService(mock_config)

        # Assert
        assert service.debug is False

    def test_raw_config_property(self, mock_config):
        # Arrange
        service = ConfigurationService(mock_config)

        # Assert
        assert service.raw_config is mock_config

    def test_to_dict(self, mock_config):
        # Arrange
        service = ConfigurationService(mock_config)

        # Act
        result = service.to_dict()

        # Assert
        assert result["speech"]["engine"] == "whisper"
        assert result["speech"]["numbers_only"] is False
        assert result["speech"]["noise_profile"] == "mixed"
        assert result["database"]["excel_path"] == "logs/hauls/test.xlsx"
        assert result["audio"]["segments_dir"] == "audio/segments"
        assert result["ui"]["theme"] == "default"
        assert result["debug"] is False


class TestConfigurationServiceFactory:
    """Tests for ConfigurationServiceFactory."""

    @patch('config.service.ConfigLoader')
    def test_create_from_args(self, mock_loader_class):
        # Arrange
        mock_loader = Mock()
        mock_config = Mock(spec=AppConfig)
        mock_loader.load.return_value = (mock_config, ["--unknown"])
        mock_loader_class.return_value = mock_loader

        # Act
        service, unknown = ConfigurationServiceFactory.create_from_args(["--engine", "whisper"])

        # Assert
        assert isinstance(service, ConfigurationService)
        assert unknown == ["--unknown"]
        mock_loader.load.assert_called_once()

    def test_create_from_config(self):
        # Arrange
        mock_config = Mock(spec=AppConfig)

        # Act
        service = ConfigurationServiceFactory.create_from_config(mock_config)

        # Assert
        assert isinstance(service, ConfigurationService)
        assert service.raw_config is mock_config

    @patch('config.service.ConfigLoader')
    def test_create_default(self, mock_loader_class):
        # Arrange
        mock_loader = Mock()
        mock_config = Mock(spec=AppConfig)
        mock_loader.load.return_value = (mock_config, [])
        mock_loader_class.return_value = mock_loader

        # Act
        service = ConfigurationServiceFactory.create_default()

        # Assert
        assert isinstance(service, ConfigurationService)
        mock_loader.load.assert_called_once_with([])

