"""
Glass Panel Widget for Fish Logging Application.

This module contains the GlassPanel class which provides a semi-transparent
panel with glassmorphism effects. It serves as a base container for other
widgets, creating visual depth and modern aesthetics.

Classes:
    GlassPanel: Semi-transparent panel with blur effects and modern styling

Features:
    - Glassmorphism visual effects
    - Semi-transparent backgrounds
    - Smooth border radius
    - Backdrop blur simulation
    - Modern shadow effects
"""
from __future__ import annotations

from PyQt6.QtWidgets import QWidget, QFrame
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette


class GlassPanel(QFrame):
    """
    Modern glass panel widget with glassmorphism effects.

    This widget provides a semi-transparent container with modern glassmorphism
    styling, creating visual depth and hierarchy in the application interface.
    It serves as a base for other content panels.

    Key Features:
        - Semi-transparent background with blur effect simulation
        - Modern border radius for smooth corners
        - Subtle border highlighting
        - Consistent with application's glassmorphism theme
        - Optimal contrast for content readability

    Design Philosophy:
        - Visual depth through transparency layers
        - Modern aesthetics with subtle effects
        - Content-first design with background enhancement
        - Accessibility through proper contrast ratios

    Usage:
        Can be used as a container for any content that needs modern
        glassmorphism styling. Commonly used for forms, panels, and
        content sections throughout the application.
    """

    def __init__(self, parent=None, opacity: float = 0.35) -> None:
        """
        Initialize the glass panel with glassmorphism styling.

        Args:
            parent: Parent widget (optional)
            opacity: Background opacity level (0.0-1.0)

        Design:
            - Configurable opacity for different use cases
            - Automatic modern styling application
            - Frame-based for proper layout integration
        """
        super().__init__(parent)
        self.opacity = max(0.0, min(1.0, opacity))  # Clamp to valid range
        self._setup_glass_styling()

    def _setup_glass_styling(self) -> None:
        """
        Configure the glassmorphism styling for the panel.

        Styling Features:
            - Semi-transparent background
            - Subtle border with transparency
            - Smooth border radius
            - Modern color palette
            - Backdrop blur effect simulation
        """
        # Calculate opacity values for consistent theming
        bg_opacity = self.opacity
        border_opacity = min(0.5, self.opacity + 0.15)

        style = f"""
            QFrame {{
                background-color: rgba(0, 0, 0, {bg_opacity});
                border: 2px solid rgba(255, 255, 255, {border_opacity});
                border-radius: 16px;
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
            }}
            
            QFrame:hover {{
                border: 2px solid rgba(255, 255, 255, {min(0.6, border_opacity + 0.1)});
            }}
        """

        self.setStyleSheet(style)

        # Set frame properties for proper rendering
        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setLineWidth(0)

    def set_opacity(self, opacity: float) -> None:
        """
        Update the panel opacity and refresh styling.

        Args:
            opacity: New opacity level (0.0-1.0)

        Features:
            - Dynamic opacity adjustment
            - Automatic styling refresh
            - Clamped to valid range
        """
        self.opacity = max(0.0, min(1.0, opacity))
        self._setup_glass_styling()

    def set_blur_radius(self, radius: int) -> None:
        """
        Set the backdrop blur radius (visual effect only).

        Args:
            radius: Blur radius in pixels

        Note:
            This is primarily for CSS-based styling. Actual blur
            effects may vary based on platform support.
        """
        current_style = self.styleSheet()
        # Update blur values in existing stylesheet
        updated_style = current_style.replace(
            "backdrop-filter: blur(10px)",
            f"backdrop-filter: blur({radius}px)"
        ).replace(
            "-webkit-backdrop-filter: blur(10px)",
            f"-webkit-backdrop-filter: blur({radius}px)"
        )
        self.setStyleSheet(updated_style)

    def add_subtle_shadow(self) -> None:
        """
        Add subtle shadow effect for enhanced depth perception.

        Features:
            - Subtle drop shadow
            - Enhanced visual hierarchy
            - Modern elevation effect
        """
        current_style = self.styleSheet()
        shadow_style = """
            QFrame {
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            }
        """
        self.setStyleSheet(current_style + shadow_style)

    def set_theme_variant(self, variant: str = "dark") -> None:
        """
        Set the color theme variant for the panel.

        Args:
            variant: Theme variant ("dark", "light", "blue")

        Variants:
            - dark: Standard dark glassmorphism (default)
            - light: Light glassmorphism for bright themes
            - blue: Blue-tinted glassmorphism for accents
        """
        if variant == "light":
            bg_color = f"rgba(255, 255, 255, {self.opacity})"
            border_color = f"rgba(0, 0, 0, {min(0.3, self.opacity + 0.1)})"
        elif variant == "blue":
            bg_color = f"rgba(59, 130, 246, {self.opacity * 0.8})"
            border_color = f"rgba(147, 197, 253, {min(0.6, self.opacity + 0.2)})"
        else:  # dark (default)
            bg_color = f"rgba(0, 0, 0, {self.opacity})"
            border_color = f"rgba(255, 255, 255, {min(0.5, self.opacity + 0.15)})"

        style = f"""
            QFrame {{
                background-color: {bg_color};
                border: 2px solid {border_color};
                border-radius: 16px;
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
            }}
        """

        self.setStyleSheet(style)

    def get_recommended_text_color(self) -> str:
        """
        Get recommended text color for optimal contrast.

        Returns:
            str: Hex color code for optimal text readability

        Logic:
            - Returns high contrast color based on background opacity
            - Ensures accessibility compliance
            - Adapts to panel transparency level
        """
        if self.opacity > 0.5:
            return "#ffffff"  # White text for dark backgrounds
        else:
            return "#1f2937"  # Dark text for light/transparent backgrounds

    def apply_content_padding(self) -> None:
        """
        Apply standard content padding for child widgets.

        Features:
            - Consistent spacing around content
            - Touch-friendly margins
            - Proper visual breathing room
        """
        self.setContentsMargins(24, 20, 24, 20)

    def enable_interactive_effects(self) -> None:
        """
        Enable interactive hover and focus effects.

        Features:
            - Enhanced border on hover
            - Smooth transition effects
            - Interactive visual feedback
        """
        current_style = self.styleSheet()
        interactive_style = """
            QFrame:hover {
                border: 2px solid rgba(100, 200, 255, 0.6);
                transition: border-color 0.2s ease;
            }
            
            QFrame:focus-within {
                border: 2px solid rgba(59, 130, 246, 0.8);
            }
        """
        self.setStyleSheet(current_style + interactive_style)
