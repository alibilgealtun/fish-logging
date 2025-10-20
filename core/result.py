"""Result type for error handling without exceptions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar, Generic, Callable, Union

T = TypeVar('T')
E = TypeVar('E')


@dataclass(frozen=True)
class Success(Generic[T]):
    """Represents a successful result."""
    value: T

    def is_success(self) -> bool:
        return True

    def is_failure(self) -> bool:
        return False

    def map(self, fn: Callable[[T], any]) -> Result[any, E]:
        """Transform the success value."""
        try:
            return Success(fn(self.value))
        except Exception as e:
            return Failure(e)

    def unwrap(self) -> T:
        """Get the value or raise if failure."""
        return self.value

    def unwrap_or(self, default: T) -> T:
        """Get the value or return default."""
        return self.value


@dataclass(frozen=True)
class Failure(Generic[E]):
    """Represents a failed result."""
    error: E

    def is_success(self) -> bool:
        return False

    def is_failure(self) -> bool:
        return True

    def map(self, fn: Callable) -> Result[any, E]:
        """Pass through the failure."""
        return self

    def unwrap(self):
        """Get the value or raise if failure."""
        if isinstance(self.error, Exception):
            raise self.error
        raise Exception(str(self.error))

    def unwrap_or(self, default):
        """Get the value or return default."""
        return default


# Type alias for Result
Result = Union[Success[T], Failure[E]]

