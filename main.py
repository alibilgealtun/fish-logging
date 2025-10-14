"""Main entry point for the fish logging application."""
from __future__ import annotations

# Load environment variables from .env early so config and recognizers can see them
try:
    from dotenv import load_dotenv

    load_dotenv()  # Searches for a .env file in the current and parent directories
except Exception:
    # If python-dotenv isn't installed, proceed; env vars can still be provided by the shell
    pass

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
