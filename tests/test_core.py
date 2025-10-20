"""Unit tests for core infrastructure components."""
import pytest
from unittest.mock import Mock

from core.container import Container
from core.result import Success, Failure
from core.error_handler import handle_exceptions, as_result, ErrorHandler


class TestContainer:
    """Tests for dependency injection container."""

    def test_register_and_get_singleton(self):
        # Arrange
        container = Container()
        instance = {"name": "test"}

        # Act
        container.register_singleton("service", instance)
        result = container.get("service")

        # Assert
        assert result is instance

    def test_register_and_get_service(self):
        # Arrange
        container = Container()
        instance = Mock()

        # Act
        container.register("mock_service", instance)
        result = container.get("mock_service")

        # Assert
        assert result is instance

    def test_register_factory(self):
        # Arrange
        container = Container()
        factory = Mock(return_value="created_instance")

        # Act
        container.register_factory("factory_service", factory)
        result1 = container.get("factory_service")
        result2 = container.get("factory_service")

        # Assert
        factory.assert_called_once()  # Should only be called once (cached)
        assert result1 == "created_instance"
        assert result1 is result2  # Same instance (cached)

    def test_has_service(self):
        # Arrange
        container = Container()
        container.register("existing", Mock())

        # Assert
        assert container.has("existing")
        assert not container.has("non_existing")

    def test_get_nonexistent_raises_error(self):
        # Arrange
        container = Container()

        # Assert
        with pytest.raises(KeyError):
            container.get("nonexistent")

    def test_clear(self):
        # Arrange
        container = Container()
        container.register("service1", Mock())
        container.register_singleton("service2", Mock())

        # Act
        container.clear()

        # Assert
        assert not container.has("service1")
        assert not container.has("service2")


class TestResult:
    """Tests for Result type."""

    def test_success_creation(self):
        # Act
        result = Success(42)

        # Assert
        assert result.is_success()
        assert not result.is_failure()
        assert result.unwrap() == 42

    def test_success_unwrap_or(self):
        # Act
        result = Success(42)

        # Assert
        assert result.unwrap_or(0) == 42

    def test_success_map(self):
        # Act
        result = Success(5)
        mapped = result.map(lambda x: x * 2)

        # Assert
        assert mapped.is_success()
        assert mapped.unwrap() == 10

    def test_failure_creation(self):
        # Act
        error = ValueError("test error")
        result = Failure(error)

        # Assert
        assert result.is_failure()
        assert not result.is_success()
        assert result.error is error

    def test_failure_unwrap_raises(self):
        # Arrange
        error = ValueError("test error")
        result = Failure(error)

        # Assert
        with pytest.raises(ValueError):
            result.unwrap()

    def test_failure_unwrap_or(self):
        # Act
        result = Failure(ValueError("error"))

        # Assert
        assert result.unwrap_or(99) == 99

    def test_failure_map(self):
        # Act
        result = Failure(ValueError("error"))
        mapped = result.map(lambda x: x * 2)

        # Assert
        assert mapped.is_failure()
        assert mapped is result


class TestErrorHandlingDecorators:
    """Tests for error handling decorators."""

    def test_handle_exceptions_success(self):
        # Arrange
        @handle_exceptions(default_return=None)
        def test_func():
            return "success"

        # Act
        result = test_func()

        # Assert
        assert result == "success"

    def test_handle_exceptions_with_error(self):
        # Arrange
        @handle_exceptions(default_return="default")
        def test_func():
            raise ValueError("test error")

        # Act
        result = test_func()

        # Assert
        assert result == "default"

    def test_handle_exceptions_reraise(self):
        # Arrange
        @handle_exceptions(reraise=True)
        def test_func():
            raise ValueError("test error")

        # Assert
        with pytest.raises(ValueError):
            test_func()

    def test_as_result_success(self):
        # Arrange
        @as_result
        def test_func(x):
            return x * 2

        # Act
        result = test_func(5)

        # Assert
        assert result.is_success()
        assert result.unwrap() == 10

    def test_as_result_failure(self):
        # Arrange
        @as_result
        def test_func():
            raise ValueError("error")

        # Act
        result = test_func()

        # Assert
        assert result.is_failure()
        assert isinstance(result.error, ValueError)


class TestErrorHandler:
    """Tests for ErrorHandler class."""

    def test_handle_logs_error(self):
        # Arrange
        mock_logger = Mock()
        handler = ErrorHandler(mock_logger)
        error = ValueError("test error")

        # Act
        handler.handle(error, context="test_context")

        # Assert
        mock_logger.error.assert_called_once()

    def test_handle_reraise(self):
        # Arrange
        handler = ErrorHandler()
        error = ValueError("test error")

        # Assert
        with pytest.raises(ValueError):
            handler.handle(error, reraise=True)

    def test_safe_execute_success(self):
        # Arrange
        handler = ErrorHandler()
        func = Mock(return_value=42)

        # Act
        result = handler.safe_execute(func, context="test")

        # Assert
        assert result == 42
        func.assert_called_once()

    def test_safe_execute_with_error_returns_default(self):
        # Arrange
        handler = ErrorHandler()
        func = Mock(side_effect=ValueError("error"))

        # Act
        result = handler.safe_execute(func, default=99)

        # Assert
        assert result == 99

