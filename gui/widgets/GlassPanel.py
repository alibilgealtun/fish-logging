from __future__ import annotations

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPainter, QPen, QLinearGradient, QColor
from PyQt6.QtWidgets import (
    QFrame,
)


class GlassPanel(QFrame):
    """Glassmorphism panel effect"""

    def __init__(self):
        super().__init__()
        self._setup_style()

    def _setup_style(self):
        # Let us paint our own rounded background and avoid square artifacts
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)
        self.setFrameShape(QFrame.Shape.NoFrame)

    def paintEvent(self, event):  # type: ignore[override]
        radius = 20.0
        rect = self.rect()
        if rect.isEmpty():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Gradient translucent fill
        gradient = QLinearGradient(rect.topLeft().toPointF(), rect.bottomRight().toPointF())       
        gradient.setColorAt(0.0, QColor(255, 255, 255, int(255 * 0.25)))
        gradient.setColorAt(1.0, QColor(255, 255, 255, int(255 * 0.10)))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        painter.drawRoundedRect(QRectF(rect), radius, radius)

        # Subtle border
        border_color = QColor(255, 255, 255, int(255 * 0.30))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(border_color, 1))
        inset = QRectF(rect).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.drawRoundedRect(QRectF(inset), radius, radius)
        painter.end()
