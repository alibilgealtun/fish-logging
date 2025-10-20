"""Dependency Injection Container for managing application dependencies."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, TypeVar

T = TypeVar('T')


class Container:
    """Simple DI container for dependency management."""

    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}

    def register_singleton(self, name: str, instance: Any) -> None:
        """Register a singleton instance."""
        self._singletons[name] = instance

    def register_factory(self, name: str, factory: Callable) -> None:
        """Register a factory function that creates instances."""
        self._factories[name] = factory

    def register(self, name: str, service: Any) -> None:
        """Register a service instance."""
        self._services[name] = service

    def get(self, name: str) -> Any:
        """Get a service by name."""
        # Check singletons first
        if name in self._singletons:
            return self._singletons[name]

        # Check registered services
        if name in self._services:
            return self._services[name]

        # Use factory if available
        if name in self._factories:
            instance = self._factories[name]()
            # Cache as singleton
            self._singletons[name] = instance
            return instance

        raise KeyError(f"Service '{name}' not found in container")

    def has(self, name: str) -> bool:
        """Check if a service is registered."""
        return name in self._singletons or name in self._services or name in self._factories

    def clear(self) -> None:
        """Clear all registrations (useful for testing)."""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()

