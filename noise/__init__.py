"""
Noise Processing Package for Fish Logging Application.

This package provides comprehensive noise reduction and audio processing capabilities
for the fish logging application's speech recognition system. It implements adaptive
noise suppression algorithms to improve speech recognition accuracy in marine
environments.

Modules:
    controller: Main noise controller with profile management
    simple_controller: Simplified noise controller for basic operations
    suppressor: Core noise suppression algorithms and configuration

Architecture:
    - Strategy pattern for different noise suppression approaches
    - Profile-based configuration for various marine environments
    - Adaptive algorithms that learn from audio characteristics
    - Real-time processing for live speech recognition

Use Cases:
    - Engine noise reduction on fishing vessels
    - Wind and weather noise suppression
    - Multiple speaker environment handling
    - Background conversation filtering
    - Equipment noise isolation

Design Philosophy:
    - Real-time performance for live applications
    - Adaptive learning from environment characteristics
    - Profile-based configuration for different scenarios
    - Minimal latency impact on speech recognition
    - Robust handling of varying noise conditions
"""

from __future__ import annotations

# Core noise suppression components
from .suppressor import AdaptiveNoiseSuppressor, SuppressorConfig
from .controller import NoiseController
from .simple_controller import SimpleNoiseController

__all__ = [
    # Configuration and core algorithms
    "SuppressorConfig",
    "AdaptiveNoiseSuppressor",

    # Controllers for different complexity levels
    "NoiseController",
    "SimpleNoiseController",
]

__version__ = "1.0.0"
__author__ = "Fish Logging Team"

# Deprecated compatibility - use direct imports instead
# Legacy support for existing imports like:
# from noise.noise_controller import NoiseController
