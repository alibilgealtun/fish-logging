"""Table management for the fish logging GUI."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import (
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QStyle,
)
from loguru import logger

from core.error_handler import handle_exceptions


class TableManager:
    """Manages the fish logging table operations.

    Responsibilities:
    - Adding rows to the table
    - Removing rows from the table
    - Managing table state
    """

    def __init__(self, table: QTableWidget, style_provider):
        """Initialize the table manager.

        Args:
            table: The QTableWidget to manage
            style_provider: Object that provides style().standardIcon()
        """
        self.table = table
        self.style_provider = style_provider

    @handle_exceptions(default_return=None, message="Failed to add table row")
    def add_entry(
        self,
        species: str,
        length_cm: float,
        confidence: float,
        boat: str = "",
        station_id: str = "",
        delete_callback: Optional[callable] = None
    ) -> None:
        """Add a fish entry to the table.

        Args:
            species: Fish species name
            length_cm: Length in centimeters
            confidence: Recognition confidence (for metadata, not displayed)
            boat: Boat name (logged but not displayed in table)
            station_id: Station ID (logged but not displayed in table)
            delete_callback: Callback for delete button clicks
        """
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        # Insert at top
        self.table.insertRow(0)

        # Create centered items
        it_date = QTableWidgetItem(date_str)
        it_time = QTableWidgetItem(time_str)
        it_species = QTableWidgetItem(species)
        it_length = QTableWidgetItem(f"{length_cm:.1f}")

        for item in (it_date, it_time, it_species, it_length):
            try:
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            except Exception:
                pass

        self.table.setItem(0, 0, it_date)
        self.table.setItem(0, 1, it_time)
        self.table.setItem(0, 2, it_species)
        self.table.setItem(0, 3, it_length)
        self.table.setRowHeight(0, 46)

        # Add delete button
        if delete_callback:
            btn_delete = self._create_delete_button(delete_callback)
            self.table.setCellWidget(0, 4, btn_delete)

    def _create_delete_button(self, callback: callable) -> QPushButton:
        """Create a delete button for table rows.

        Args:
            callback: Function to call when button is clicked

        Returns:
            Configured delete button
        """
        btn_delete = QPushButton()
        btn_delete.setIcon(
            self.style_provider.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon)
        )
        btn_delete.setIconSize(QSize(18, 18))
        btn_delete.setFixedSize(28, 28)
        btn_delete.setStyleSheet("""
            QPushButton { border: none; background: transparent; }
            QPushButton:hover { color: red; }
        """)
        btn_delete.setFlat(True)
        btn_delete.clicked.connect(callback)
        return btn_delete

    @handle_exceptions(default_return=False, message="Failed to remove last row")
    def remove_last_row(self) -> bool:
        """Remove the last (most recent) row from the table.

        Returns:
            True if a row was removed, False if table was empty
        """
        if self.table.rowCount() > 0:
            self.table.removeRow(0)
            return True
        return False

    @handle_exceptions(default_return=False, message="Failed to remove row")
    def remove_row(self, row_index: int) -> bool:
        """Remove a specific row from the table.

        Args:
            row_index: Index of the row to remove

        Returns:
            True if row was removed, False otherwise
        """
        if 0 <= row_index < self.table.rowCount():
            self.table.removeRow(row_index)
            logger.info(f"Deleted row {row_index} from table")
            return True
        return False

    def get_row_count(self) -> int:
        """Get the current number of rows in the table.

        Returns:
            Number of rows
        """
        return self.table.rowCount()

    def clear(self) -> None:
        """Clear all rows from the table."""
        self.table.setRowCount(0)

