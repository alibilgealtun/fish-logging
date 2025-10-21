"""
Custom Widgets Package for Fish Logging Application.

This package contains custom PyQt6 widgets designed specifically for the fish logging
application. These widgets provide a modern, cohesive user interface with glassmorphism
design elements and specialized functionality.

Widgets:
    AnimatedButton: Button with smooth animations and visual feedback
    GlassPanel: Semi-transparent panel with blur effects
    ModernLabel: Styled label with typography hierarchy
    ModernTable: Enhanced table with modern styling
    ModernTextEdit: Styled text input with modern appearance
    PulsingStatusIndicator: Animated status indicator with color coding
    BoatNameInput: Specialized input for boat name entry
    SpeciesSelector: Searchable dropdown for fish species selection
    StationIdInput: Specialized input for station ID entry
    SettingsWidget: Complete settings panel with all configurations
    NoiseProfileInput: Input for noise profile selection
    ReportWidget: Data visualization and reporting interface

Design Philosophy:
    - Consistent glassmorphism aesthetic
    - High contrast for accessibility
    - Smooth animations and transitions
    - Touch-friendly sizing and interactions
    - Professional typography hierarchy
    - Semantic color coding
"""

from __future__ import annotations

# Core modern widgets
from .AnimatedButton import AnimatedButton
from .GlassPanel import GlassPanel
from .ModernLabel import ModernLabel
from .ModernTable import ModernTable
from .ModernTextEdit import ModernTextEdit
from .PulsingStatusIndicator import PulsingStatusIndicator

# Specialized input widgets
from .BoatNameInput import BoatNameInput
from .SpeciesSelector import SpeciesSelector
from .StationIdInput import StationIdInput
from .NoiseProfileInput import NoiseProfileInput

# Complex widgets
from .SettingsWidget import SettingsWidget
from .ReportWidget import ReportWidget

__all__ = [
    # Core widgets
    'AnimatedButton',
    'GlassPanel',
    'ModernLabel',
    'ModernTable',
    'ModernTextEdit',
    'PulsingStatusIndicator',

    # Input widgets
    'BoatNameInput',
    'SpeciesSelector',
    'StationIdInput',
    'NoiseProfileInput',

    # Complex widgets
    'SettingsWidget',
    'ReportWidget',
]

__version__ = "1.0.0"
__author__ = "Fish Logging Team"
