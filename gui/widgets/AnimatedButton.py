from __future__ import annotations

from PyQt6.QtCore import QPropertyAnimation, QEasingCurve
from PyQt6.QtWidgets import QPushButton


class AnimatedButton(QPushButton):
    """Custom button with hover animations and modern styling"""

    def __init__(self, text: str, primary: bool = False):
        super().__init__(text)
        self.primary = primary
        self.hover_animation = QPropertyAnimation(self, b"geometry")
        self.hover_animation.setDuration(200)
        self.hover_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._setup_style()

    def _setup_style(self):
        if self.primary:
            self.setStyleSheet("""
                AnimatedButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #667eea, stop:1 #764ba2);
                    color: white;
                    border: none;
                    border-radius: 12px;
                    padding: 12px 24px;
                    font-weight: 600;
                    font-size: 14px;
                    min-width: 100px;
                }
                AnimatedButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #7c9df0, stop:1 #8b5fbf);
                    transform: translateY(-2px);
                }
                AnimatedButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #5a6fd8, stop:1 #6a4190);
                }
                AnimatedButton:disabled {
                    background: #cccccc;
                    color: #666666;
                }
            """)
        else:
            self.setStyleSheet("""
                AnimatedButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #ff6b6b, stop:1 #ee5a52);
                    color: white;
                    border: none;
                    border-radius: 12px;
                    padding: 12px 24px;
                    font-weight: 600;
                    font-size: 14px;
                    min-width: 100px;
                }
                AnimatedButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #ff7979, stop:1 #fd6c5d);
                }
                AnimatedButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #e55656, stop:1 #d63447);
                }
            """)
