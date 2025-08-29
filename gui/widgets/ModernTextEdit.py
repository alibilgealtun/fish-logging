from __future__ import annotations

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QTextEdit,
    QGraphicsDropShadowEffect,
)


class ModernTextEdit(QTextEdit):
    """Styled text edit with modern appearance"""

    def __init__(self):
        super().__init__()
        self._setup_style()

    def _setup_style(self):
        self.setStyleSheet("""
            ModernTextEdit {
                background: white;
                border: 2px solid rgba(108, 117, 125, 0.3);
                border-radius: 16px;
                padding: 16px;
                font-family: 'Segoe UI', system-ui, sans-serif;
                font-size: 14px;
                line-height: 1.5;
                color: #2c3e50;
                selection-background-color: #667eea;
                selection-color: white;
            }
            ModernTextEdit:focus {
                border: 2px solid #667eea;
                background: white;
            }
        """)

        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)
