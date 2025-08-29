from __future__ import annotations

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
)


class PulsingStatusIndicator(QFrame):
    """Animated status indicator with pulsing effect"""

    def __init__(self):
        super().__init__()
        self.setFixedSize(20, 20)
        self._setup_style()

        # Pulsing animation
        self.pulse_timer = QTimer()
        self.pulse_timer.timeout.connect(self._pulse)
        self.pulse_timer.start(1000)
        self.pulse_state = 0

    def _setup_style(self):
        self.setStyleSheet("""
            PulsingStatusIndicator {
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.8,
                    fx:0.3, fy:0.3, stop:0 #4CAF50, stop:1 #2E7D32);
                border-radius: 10px;
                border: 3px solid rgba(255, 255, 255, 0.3);
            }
        """)

        # Add glow effect
        glow = QGraphicsDropShadowEffect()
        glow.setBlurRadius(15)
        glow.setColor(QColor(76, 175, 80, 100))
        glow.setOffset(0, 0)
        self.setGraphicsEffect(glow)

    def _pulse(self):
        self.pulse_state = (self.pulse_state + 1) % 3
        alpha = [100, 150, 200][self.pulse_state]

        if hasattr(self, 'current_color'):
            color = self.current_color
        else:
            color = "#4CAF50"

        glow = QGraphicsDropShadowEffect()
        glow.setBlurRadius(20)
        glow.setColor(QColor(color).lighter(120))
        glow.setOffset(0, 0)
        self.setGraphicsEffect(glow)

    def set_status_color(self, color: str, bg_color: str):
        self.current_color = color
        self.setStyleSheet(f"""
            PulsingStatusIndicator {{
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.8,
                    fx:0.3, fy:0.3, stop:0 {color}, stop:1 {bg_color});
                border-radius: 10px;
                border: 3px solid rgba(255, 255, 255, 0.4);
            }}
        """)
