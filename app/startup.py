"""Application startup and configuration.

Main entry point that orchestrates application initialization including
configuration parsing, service setup, and GUI creation.
"""
from __future__ import annotations

import sys

from loguru import logger

from app.application import Application
from config.service import ConfigurationServiceFactory
from speech.factory import create_recognizer
from services import initialize_audio_saver


def run_application() -> None:
    """Main application entry point with clean separation of concerns.
    
    Orchestrates the startup sequence:
    1. Parse configuration from all sources (defaults, files, env, CLI)
    2. Initialize audio saver service if enabled
    3. Create speech recognizer with fallback handling
    4. Initialize application infrastructure
    5. Create and launch Qt GUI
    
    Raises:
        SystemExit: On application termination
    """
    # Parse configuration using the new service facade
    config_service, unknown_args = ConfigurationServiceFactory.create_from_args(sys.argv[1:])

    # Initialize audio saver service
    initialize_audio_saver(
        segments_dir=config_service.audio_segments_dir,
        enabled=config_service.save_audio_segments
    )

    # Create recognizer with fallback, including noise profile selection
    recognizer = _create_recognizer_with_fallback(
        config_service.engine,
        config_service.numbers_only,
        config_service.noise_profile,
    )

    # Initialize application
    app_instance = Application(recognizer)
    
    # Log session configuration (include noise profile)
    app_instance.log_session_info(config_service.to_dict())

    # Create Qt application
    qt_args = [sys.argv[0], *unknown_args]
    qt_app = app_instance.create_qt_app(qt_args)
    
    # Create and show main window
    main_window = app_instance.create_main_window()
    main_window.show()
    
    # Run application
    sys.exit(qt_app.exec())


def _create_recognizer_with_fallback(engine: str, numbers_only: bool, noise_profile: str):
    """Create recognizer with fallback to whisper if the requested engine fails.
    
    Args:
        engine: Speech recognition engine name
        numbers_only: Whether to enable numbers-only mode
        noise_profile: Noise environment profile (clean, human, engine, mixed)
        
    Returns:
        Initialized speech recognizer instance
        
    Note:
        Falls back to whisper engine with numbers_only=False on any failure.
    """
    try:
        return create_recognizer(engine, numbers_only=numbers_only, noise_profile=noise_profile)
    except Exception:
        logger.exception("Failed to create recognizer with engine {}. Falling back to whisper.", engine)
        return create_recognizer("whisper", numbers_only=False, noise_profile=noise_profile)
