"""Main entry point for the fish logging application."""
from __future__ import annotations

from app.startup import run_application
from speech.factory import create_recognizer


def main() -> None:
    """Application entry point."""
    # Import PyQt WebEngine widgets to ensure they're available
    from PyQt6 import QtWebEngineWidgets  # noqa: F401

    run_application()


__all__ = ["main", "create_recognizer"]

if __name__ == "__main__":
    main()
