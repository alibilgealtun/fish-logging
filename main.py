"""Main entry point for the fish logging application."""
from __future__ import annotations

from app.startup import run_application


def main() -> None:
    """Application entry point."""
    # Import PyQt WebEngine widgets to ensure they're available
    from PyQt6 import QtWebEngineWidgets  # noqa: F401

    run_application()


if __name__ == "__main__":
    main()
