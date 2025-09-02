from __future__ import annotations

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QTableWidget,
    QHeaderView,
    QGraphicsDropShadowEffect,
)


class ModernTable(QTableWidget):
    """Beautifully styled table with modern appearance"""

    def __init__(self, rows: int, columns: int):
        super().__init__(rows, columns)
        self._setup_style()

    def _setup_style(self):
        self.setStyleSheet("""
            ModernTable {
                background: white;
                gridline-color: rgba(108, 117, 125, 0.2);
                border: none;
                border-radius: 16px;
                font-family: 'Segoe UI', system-ui, sans-serif;
                font-size: 13px;
                color: #2c3e50;
            }
            ModernTable::item {
                padding: 12px 16px;
                border: none;
                border-bottom: 1px solid rgba(108, 117, 125, 0.1);
                color: #2c3e50;
                background: white;
            }
            ModernTable::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(102, 126, 234, 0.2),
                    stop:1 rgba(118, 75, 162, 0.2));
                color: #2c3e50;
            }
            ModernTable::item:alternate {
                background: #f8f9fa;
                color: #2c3e50;
            }
            QHeaderView::section {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8f9fa, stop:1 #e9ecef);
                color: #495057;
                padding: 16px;
                border: none;
                border-bottom: 2px solid #dee2e6;
                font-weight: 600;
                font-size: 13px;
                text-align: left;
            }
            QHeaderView::section:first {
                border-top-left-radius: 16px;
            }
            QHeaderView::section:last {
                border-top-right-radius: 16px;
            }
        """)

        # Configure header
        header = self.horizontalHeader()
        header.setDefaultSectionSize(150)
        header.setStretchLastSection(False)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

        # Make all columns except the last one stretch to fill the viewport
        try:
            column_count = self.columnCount()
            if column_count > 0:
                for section in range(max(0, column_count - 1)):
                    header.setSectionResizeMode(section, QHeaderView.ResizeMode.Stretch)
                # Last column stays fixed (icon / actions)
                header.setSectionResizeMode(column_count - 1, QHeaderView.ResizeMode.Fixed)
                # Give a compact width for the last column (trash icon)
                header.resizeSection(column_count - 1, 44)
        except Exception as e:
            pass # TODO log this

        # Add shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 0, 0, 20))
        shadow.setOffset(0, 8)
        self.setGraphicsEffect(shadow)
