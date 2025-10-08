"""Application startup and configuration."""
from __future__ import annotations

import sys

from loguru import logger

from app.application import Application
from config.config import parse_app_args
from speech.factory import create_recognizer


def run_application() -> None:
    """Main application entry point with clean separation of concerns."""
    # Parse configuration
    config, unknown_args = parse_app_args(sys.argv[1:])
    
    # Create recognizer with fallback, including noise profile selection
    recognizer = _create_recognizer_with_fallback(
        config.speech.engine,
        config.speech.numbers_only,
        config.speech.noise_profile,
    )

    # Initialize application
    app_instance = Application(recognizer)
    
    # Log session configuration (include noise profile)
    app_instance.log_session_info({
        "engine": config.speech.engine,
        "numbers_only": config.speech.numbers_only,
        "noise_profile": config.speech.noise_profile,
    })
    
    # Create Qt application
    qt_args = [sys.argv[0], *unknown_args]
    qt_app = app_instance.create_qt_app(qt_args)
    
    # Create and show main window
    main_window = app_instance.create_main_window()
    main_window.show()
    
    # Run application
    sys.exit(qt_app.exec())


def _create_recognizer_with_fallback(engine: str, numbers_only: bool, noise_profile: str):
    """Create recognizer with fallback to whisper if the requested engine fails."""
    try:
        return create_recognizer(engine, numbers_only=numbers_only, noise_profile=noise_profile)
    except Exception:
        logger.exception("Failed to create recognizer with engine {}. Falling back to whisper.", engine)
        return create_recognizer("whisper", numbers_only=False, noise_profile=noise_profile)
