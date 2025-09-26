"""Compatibility shim for noise components.

Deprecated: use
- noise.suppressor for SuppressorConfig and AdaptiveNoiseSuppressor
- noise.controller for NoiseController

This module re-exports the public classes to preserve existing imports like:
    from noise.noise_controller import NoiseController
"""

from __future__ import annotations
from .suppressor import AdaptiveNoiseSuppressor, SuppressorConfig
from .controller import NoiseController

__all__ = [
    "SuppressorConfig",
    "AdaptiveNoiseSuppressor",
    "NoiseController",
]
