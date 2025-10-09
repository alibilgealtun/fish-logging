from __future__ import annotations

from PyQt6.QtWidgets import (
    QLabel,
)

class ModernLabel(QLabel):
    """Styled label with modern typography"""

    def __init__(self, text: str, style: str = "normal"):
        super().__init__(text)
        self._setup_style(style)

    def _setup_style(self, style: str):
        if style == "header":
            self.setStyleSheet("""
                ModernLabel {
                    color: #ffffff;
                    font-family: 'Segoe UI', system-ui, sans-serif;
                    font-size: 20px;
                    font-weight: 700;
                    margin: 8px 0px;
                }
            """)
        elif style == "subheader":
            self.setStyleSheet("""
                ModernLabel {
                    color: #ffffff;
                    font-family: 'Segoe UI', system-ui, sans-serif;
                    font-size: 16px;
                    font-weight: 600;
                    margin: 6px 0px;
                }
            """)
        else:
            self.setStyleSheet("""
                ModernLabel {
                    color: #495057;
                    font-family: 'Segoe UI', system-ui, sans-serif;
                    font-size: 14px;
                    font-weight: 500;
                }
            """)
