"""
Animated Button Widget for Fish Logging Application.

This module contains the AnimatedButton class which provides a modern button
with smooth animations, visual feedback, and semantic color variants for
different action types.

Classes:
    AnimatedButton: Modern button with animations and color variants

Features:
    - Smooth hover and press animations
    - Semantic color variants (primary, success, warning, info, danger)
    - Modern glassmorphism styling
    - Touch-friendly sizing
    - Accessibility support
"""
from __future__ import annotations

from PyQt6.QtCore import QPropertyAnimation, QEasingCurve, QRect, pyqtSignal
from PyQt6.QtWidgets import QPushButton
from PyQt6.QtGui import QFont


class AnimatedButton(QPushButton):
    """
    Modern button widget with smooth animations and semantic color variants.

    This button provides enhanced visual feedback through smooth animations,
    semantic color coding for different action types, and modern styling
    that fits the application's glassmorphism design theme.

    Key Features:
        - Smooth hover and press animations
        - Semantic color variants for different action types
        - Modern glassmorphism styling with gradients
        - Touch-friendly sizing and interactions
        - Backward compatibility with primary flag
        - Accessibility-friendly contrast ratios

    Supported Variants:
        - primary: Main action button (blue gradient)
        - success: Positive actions (green gradient)
        - warning: Caution actions (orange gradient)
        - info: Informational actions (light blue gradient)
        - danger: Destructive actions (red gradient)

    Design Philosophy:
        - Visual hierarchy through color coding
        - Smooth transitions for premium feel
        - High contrast for accessibility
        - Consistent with application theme
    """

    # Custom signals for enhanced interaction
    hoverEntered = pyqtSignal()
    hoverLeft = pyqtSignal()

    def __init__(self, text: str, primary: bool = False, variant: str | None = None) -> None:
        """
        Initialize the animated button.

        Args:
            text: Button text to display
            primary: Legacy flag for primary styling (deprecated)
            variant: Semantic variant name (preferred over primary flag)

        Design:
            - Backward compatible with primary flag
            - Prefers semantic variant naming
            - Automatic animation setup
            - Modern styling application
        """
        super().__init__(text)

        # Determine variant (prefer explicit variant over legacy primary flag)
        self.variant = variant if variant else ("primary" if primary else "info")

        # Setup hover animation for smooth visual feedback
        self.hover_animation = QPropertyAnimation(self, b"geometry")
        self.hover_animation.setDuration(200)
        self.hover_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        # Store original geometry for animation reference
        self.original_geometry = QRect()

        # Apply initial styling
        self._apply_style()
        self._setup_font()

    def set_variant(self, variant: str) -> None:
        """
        Change the button's semantic variant.

        Args:
            variant: New variant name to apply

        Supported variants: primary, success, warning, info, danger
        """
        if variant in self._get_available_variants():
            self.variant = variant
            self._apply_style()
        else:
            raise ValueError(f"Unsupported variant: {variant}")

    def _get_available_variants(self) -> list[str]:
        """Get list of available button variants."""
        return ["primary", "success", "warning", "info", "danger"]

    def _palette_for_variant(self) -> dict[str, tuple[str, str] | str]:
        """
        Get color palette for the current variant.

        Returns:
            dict: Color palette with normal, hover, pressed states and text color

        Color Design:
            - Uses gradient stops for depth and modern appearance
            - High contrast ratios for accessibility
            - Semantic color associations
        """
        palettes = {
            "primary": {
                "normal": ("#667eea", "#764ba2"),  # Blue to purple gradient
                "hover": ("#7c9df0", "#8b5fbf"),   # Lighter version
                "pressed": ("#5a6fd8", "#6a4190"), # Darker version
                "text": "#ffffff",
            },
            "success": {
                "normal": ("#34d399", "#059669"),  # Emerald gradient
                "hover": ("#6ee7b7", "#10b981"),
                "pressed": ("#2fb37f", "#047857"),
                "text": "#ffffff",
            },
            "warning": {
                "normal": ("#f59e0b", "#d97706"),  # Amber gradient
                "hover": ("#fbbf24", "#f59e0b"),
                "pressed": ("#d97706", "#b45309"),
                "text": "#ffffff",
            },
            "info": {
                "normal": ("#60a5fa", "#3b82f6"),  # Blue gradient
                "hover": ("#93c5fd", "#60a5fa"),
                "pressed": ("#3b82f6", "#2563eb"),
                "text": "#ffffff",
            },
            "danger": {
                "normal": ("#ef4444", "#dc2626"),  # Red gradient
                "hover": ("#f87171", "#ef4444"),
                "pressed": ("#dc2626", "#b91c1c"),
                "text": "#ffffff",
            },
        }

        return palettes.get(self.variant, palettes["primary"])

    def _apply_style(self) -> None:
        """
        Apply modern styling based on current variant.

        Styling Features:
            - Gradient backgrounds for depth
            - Smooth border radius for modern look
            - Proper padding for touch interfaces
            - State-based color transitions
            - Shadow effects for elevation
        """
        palette = self._palette_for_variant()

        style = f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {palette['normal'][0]},
                    stop:1 {palette['normal'][1]});
                border: none;
                border-radius: 8px;
                color: {palette['text']};
                font-weight: 600;
                padding: 12px 24px;
                min-height: 20px;
                font-size: 14px;
            }}
            
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {palette['hover'][0]},
                    stop:1 {palette['hover'][1]});
            }}
            
            QPushButton:pressed {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {palette['pressed'][0]},
                    stop:1 {palette['pressed'][1]});
            }}
            
            QPushButton:disabled {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(156, 163, 175, 0.5),
                    stop:1 rgba(107, 114, 128, 0.5));
                color: rgba(255, 255, 255, 0.5);
            }}
        """

        self.setStyleSheet(style)

    def _setup_font(self) -> None:
        """
        Configure button font for optimal readability.

        Font Features:
            - Medium weight for good visibility
            - Appropriate size for touch interfaces
            - System font for consistency
        """
        font = QFont()
        font.setWeight(QFont.Weight.Medium)
        font.setPointSize(10)
        self.setFont(font)

    def enterEvent(self, event) -> None:
        """
        Handle mouse enter events for hover animation.

        Args:
            event: Qt mouse event

        Animation:
            - Slight scale increase for hover feedback
            - Smooth transition for premium feel
        """
        super().enterEvent(event)
        self.hoverEntered.emit()

        # Animate slight scale increase on hover
        if self.original_geometry.isValid():
            target_rect = self.original_geometry.adjusted(-2, -2, 2, 2)
            self.hover_animation.setStartValue(self.geometry())
            self.hover_animation.setEndValue(target_rect)
            self.hover_animation.start()

    def leaveEvent(self, event) -> None:
        """
        Handle mouse leave events for hover animation.

        Args:
            event: Qt mouse event

        Animation:
            - Return to original size
            - Smooth transition back to normal state
        """
        super().leaveEvent(event)
        self.hoverLeft.emit()

        # Animate back to original size
        if self.original_geometry.isValid():
            self.hover_animation.setStartValue(self.geometry())
            self.hover_animation.setEndValue(self.original_geometry)
            self.hover_animation.start()

    def resizeEvent(self, event) -> None:
        """
        Handle resize events to update animation reference.

        Args:
            event: Qt resize event

        Updates the original geometry reference for consistent animations.
        """
        super().resizeEvent(event)
        self.original_geometry = self.geometry()

    def set_loading(self, loading: bool = True) -> None:
        """
        Set button loading state with visual feedback.

        Args:
            loading: Whether button should show loading state

        Features:
            - Disables interaction during loading
            - Visual loading indicator
            - Preserves original text for restoration
        """
        if loading:
            self.setEnabled(False)
            self._original_text = self.text()
            self.setText("⏳ Loading...")
        else:
            self.setEnabled(True)
            if hasattr(self, '_original_text'):
                self.setText(self._original_text)

    def set_icon_text(self, icon: str, text: str) -> None:
        """
        Set button text with emoji icon.

        Args:
            icon: Emoji or symbol to display
            text: Text to display after icon

        Example:
            button.set_icon_text("🎣", "Start Fishing")
        """
        self.setText(f"{icon} {text}")

    def get_variant(self) -> str:
        """
        Get the current button variant.

        Returns:
            str: Current semantic variant name
        """
        return self.variant

