"""Utility decorators and error handlers for consistent error handling."""
from __future__ import annotations

import functools
import time
from typing import Any, Callable, Optional, TypeVar

from loguru import logger

from core.result import Success, Failure, Result

T = TypeVar('T')


def handle_exceptions(
    logger_instance=logger,
    default_return: Optional[Any] = None,
    reraise: bool = False,
    message: Optional[str] = None
):
    """Decorator to handle exceptions consistently.

    Args:
        logger_instance: Logger to use for error logging
        default_return: Value to return on exception
        reraise: Whether to re-raise the exception after logging
        message: Custom error message prefix
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = message or f"Error in {func.__name__}"
                logger_instance.error(f"{error_msg}: {e}", exc_info=True)
                if reraise:
                    raise
                return default_return
        return wrapper
    return decorator


def as_result(func: Callable[..., T]) -> Callable[..., Result[T, Exception]]:
    """Decorator to convert function output to Result type.

    Success values are wrapped in Success, exceptions in Failure.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Result[T, Exception]:
        try:
            result = func(*args, **kwargs)
            return Success(result)
        except Exception as e:
            return Failure(e)
    return wrapper


def log_execution_time(logger_instance=logger, level: str = "DEBUG"):
    """Decorator to log function execution time.

    Args:
        logger_instance: Logger to use
        level: Log level (DEBUG, INFO, etc.)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.time() - start_time
                log_func = getattr(logger_instance, level.lower(), logger_instance.debug)
                log_func(f"{func.__name__} executed in {elapsed:.3f}s")
        return wrapper
    return decorator


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)):
    """Decorator to retry function execution on failure.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay in seconds between retries
        exceptions: Tuple of exception types to catch and retry

    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt}/{max_retries}): {e}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_retries} attempts: {e}"
                        )
            raise last_exception
        return wrapper
    return decorator


class ErrorHandler:
    """Centralized error handling for consistent error management."""

    def __init__(self, logger_instance=logger):
        self.logger = logger_instance

    def handle(self, error: Exception, context: str = "", reraise: bool = False) -> None:
        """Handle an error with logging.

        Args:
            error: The exception to handle
            context: Additional context information
            reraise: Whether to re-raise after handling
        """
        message = f"Error in {context}: {error}" if context else str(error)
        self.logger.error(message, exc_info=True)
        if reraise:
            raise error

    def safe_execute(
        self,
        func: Callable[..., T],
        *args,
        default: Optional[T] = None,
        context: str = "",
        **kwargs
    ) -> Optional[T]:
        """Execute a function safely with error handling.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            default: Default value to return on error
            context: Context information for logging
            **kwargs: Keyword arguments for the function

        Returns:
            Function result or default value on error
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.handle(e, context=context or func.__name__)
            return default
