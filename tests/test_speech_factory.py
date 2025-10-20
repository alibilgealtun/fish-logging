"""Unit tests for speech factory and registry."""
import pytest
from unittest.mock import Mock, patch

from speech.factory import RecognizerRegistry, create_recognizer
from core.exceptions import RecognizerError


class TestRecognizerRegistry:
    """Tests for RecognizerRegistry."""

    def setup_method(self):
        """Clear registry before each test."""
        # Save original state
        self.original_factories = RecognizerRegistry._factories.copy()
        self.original_metadata = RecognizerRegistry._metadata.copy()
        # Clear for testing
        RecognizerRegistry._factories.clear()
        RecognizerRegistry._metadata.clear()

    def teardown_method(self):
        """Restore registry after each test."""
        RecognizerRegistry._factories = self.original_factories
        RecognizerRegistry._metadata = self.original_metadata

    def test_register_engine(self):
        # Arrange
        factory = Mock()

        # Act
        RecognizerRegistry.register(
            "test_engine",
            factory,
            "Test Engine",
            ["test-package"]
        )

        # Assert
        assert RecognizerRegistry.is_registered("test_engine")
        metadata = RecognizerRegistry.get_metadata("test_engine")
        assert metadata["description"] == "Test Engine"
        assert metadata["requires"] == ["test-package"]

    def test_register_case_insensitive(self):
        # Arrange
        factory = Mock()

        # Act
        RecognizerRegistry.register("TestEngine", factory)

        # Assert
        assert RecognizerRegistry.is_registered("testengine")
        assert RecognizerRegistry.is_registered("TESTENGINE")

    def test_create_registered_engine(self):
        # Arrange
        mock_recognizer = Mock()
        factory = Mock(return_value=mock_recognizer)
        RecognizerRegistry.register("test", factory)

        # Act
        result = RecognizerRegistry.create("test", numbers_only=True, noise_profile="clean")

        # Assert
        assert result is mock_recognizer
        factory.assert_called_once_with(numbers_only=True, noise_profile="clean")

    def test_create_unknown_engine_raises_error(self):
        # Assert
        with pytest.raises(RecognizerError) as exc_info:
            RecognizerRegistry.create("unknown_engine")

        assert "Unknown speech recognition engine" in str(exc_info.value)

    def test_create_factory_error_raises_recognizer_error(self):
        # Arrange
        factory = Mock(side_effect=ImportError("Package not found"))
        RecognizerRegistry.register("failing", factory)

        # Assert
        with pytest.raises(RecognizerError) as exc_info:
            RecognizerRegistry.create("failing")

        assert "Failed to create" in str(exc_info.value)

    def test_get_available_engines(self):
        # Arrange
        RecognizerRegistry.register("engine1", Mock())
        RecognizerRegistry.register("engine2", Mock())
        RecognizerRegistry.register("engine3", Mock())

        # Act
        engines = RecognizerRegistry.get_available_engines()

        # Assert
        assert len(engines) == 3
        assert "engine1" in engines
        assert "engine2" in engines
        assert "engine3" in engines

    def test_get_metadata_for_unregistered_returns_empty(self):
        # Act
        metadata = RecognizerRegistry.get_metadata("nonexistent")

        # Assert
        assert metadata == {}


class TestCreateRecognizer:
    """Tests for create_recognizer function (backward compatibility)."""

    @patch('speech.factory.RecognizerRegistry.create')
    def test_create_recognizer_delegates_to_registry(self, mock_create):
        # Arrange
        mock_recognizer = Mock()
        mock_create.return_value = mock_recognizer

        # Act
        result = create_recognizer("whisper", numbers_only=True, noise_profile="clean")

        # Assert
        assert result is mock_recognizer
        mock_create.assert_called_once_with("whisper", numbers_only=True, noise_profile="clean")

