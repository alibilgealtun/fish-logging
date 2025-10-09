from __future__ import annotations

from PyQt6.QtCore import QPropertyAnimation, QEasingCurve, QRect
from PyQt6.QtWidgets import QPushButton


class AnimatedButton(QPushButton):
    """Custom button with hover animations, modern styling, and color variants.

    Backward compatible with signature (text: str, primary: bool = False).
    Prefer passing variant: one of {"primary", "success", "warning", "info", "danger"}.
    """

    def __init__(self, text: str, primary: bool = False, variant: str | None = None):
        super().__init__(text)
        # Map old primary flag to a variant if variant not provided
        self.variant = variant if variant else ("primary" if primary else "danger")
        self.hover_animation = QPropertyAnimation(self, b"geometry")
        self.hover_animation.setDuration(200)
        self.hover_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._apply_style()
        self.original_geometry = self.geometry()

    def set_variant(self, variant: str) -> None:
        self.variant = variant
        self._apply_style()

    def _palette_for_variant(self) -> dict:
        # gradient stops for normal, hover, pressed, and disabled colors
        palettes = {
            "primary": {
                "normal": ("#667eea", "#764ba2"),
                "hover": ("#7c9df0", "#8b5fbf"),
                "pressed": ("#5a6fd8", "#6a4190"),
                "text": "#ffffff",
            },
            "success": {
                "normal": ("#34d399", "#059669"),  # emerald
                "hover": ("#6ee7b7", "#10b981"),
                "pressed": ("#2fb37f", "#047857"),
                "text": "#ffffff",
            },
            "warning": {
                "normal": ("#f59e0b", "#d97706"),  # amber
                "hover": ("#fbbf24", "#f59e0b"),
                "pressed": ("#d97706", "#b45309"),
                "text": "#ffffff",
            },
            "info": {
                "normal": ("#60a5fa", "#3b82f6"),  # blue
                "hover": ("#93c5fd", "#60a5fa"),
                "pressed": ("#3b82f6", "#2563eb"),
                "text": "#ffffff",
            },
            "danger": {
                "normal": ("#ff6b6b", "#ee5a52"),
                "hover": ("#ff7979", "#fd6c5d"),
                "pressed": ("#e55656", "#d63447"),
                "text": "#ffffff",
            },
        }
        return palettes.get(self.variant, palettes["primary"])  # default to primary

    def _apply_style(self):
        p = self._palette_for_variant()
        normal_a, normal_b = p["normal"]
        hover_a, hover_b = p["hover"]
        pressed_a, pressed_b = p["pressed"]
        text_color = p["text"]
        self.setStyleSheet(f"""
            AnimatedButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {normal_a}, stop:1 {normal_b});
                color: {text_color};
                border: none;
                border-radius: 12px;
                padding: 12px 24px;
                font-weight: 600;
                font-size: 14px;
                min-width: 120px;
            }}
            AnimatedButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {hover_a}, stop:1 {hover_b});
            }}
            AnimatedButton:pressed {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {pressed_a}, stop:1 {pressed_b});
            }}
            AnimatedButton:disabled {{
                background: #e5e7eb;
                color: #9ca3af;
            }}
        """)

    def enterEvent(self, event) -> None:  # type: ignore[override]
        """Handle mouse enter event to start hover animation"""
        self.original_geometry = self.geometry()
        target_geometry = QRect(
            self.original_geometry.x(),
            self.original_geometry.y() - 2,
            self.original_geometry.width(),
            self.original_geometry.height()
        )
        self.hover_animation.setStartValue(self.original_geometry)
        self.hover_animation.setEndValue(target_geometry)
        self.hover_animation.start()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        """Handle mouse leave event to revert animation"""
        self.hover_animation.setStartValue(self.geometry())
        self.hover_animation.setEndValue(self.original_geometry)
        self.hover_animation.start()
        super().leaveEvent(event)