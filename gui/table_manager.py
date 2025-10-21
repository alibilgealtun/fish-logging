"""
Table Manager for Fish Logging Application.

This module contains the TableManager class which manages the fish entries
table display and data operations. It implements the Manager pattern to
encapsulate table-specific logic and provide a clean interface for table
operations.

Classes:
    TableManager: Manages fish entries table display and operations

Architecture:
    - Manager pattern for table-specific operations
    - Separation of table logic from main window
    - Clean API for table data management
    - Robust error handling and validation
"""
from __future__ import annotations

from typing import Optional, List, Dict, Any
from datetime import datetime
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QPushButton, QMessageBox
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon

# Optional logger: prefer loguru if available, else fallback to stdlib logging
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class TableManager:
    """
    Manages the fish entries table display and operations.

    This class encapsulates all table-related functionality, providing
    a clean interface for adding, removing, and managing fish entry
    data in the table widget while maintaining data integrity.

    Key Responsibilities:
        - Manage table data display and formatting
        - Handle row insertion and deletion operations
        - Provide table state management and validation
        - Coordinate with UI for user interactions
        - Ensure data consistency and integrity

    Design Patterns:
        - Manager pattern: Encapsulates table operations
        - Observer pattern: Responds to data changes
        - Command pattern: Encapsulates table operations

    Attributes:
        table: The QTableWidget being managed
        parent_window: Parent window for dialogs and context
        _row_data: Cache of row data for operations
    """

    def __init__(self, table: QTableWidget, parent_window=None) -> None:
        """
        Initialize the table manager with table widget and parent.

        Args:
            table: QTableWidget instance to manage
            parent_window: Parent window for dialogs and context

        Design:
            - Direct widget management for performance
            - Parent window for dialog context
            - Initialization validation
        """
        if not table:
            raise ValueError("Table widget is required")

        self.table = table
        self.parent_window = parent_window
        self._row_data: List[Dict[str, Any]] = []

        # Configure table if not already done
        self._ensure_table_configuration()

        logger.debug(f"TableManager initialized for table with {table.rowCount()} rows")

    def add_fish_entry(
        self,
        species: str,
        length_cm: float,
        confidence: float = 0.0,
        boat_name: str = "",
        station_id: str = "",
        raw_text: str = "",
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Add a new fish entry to the table.

        Adds a new row at the top of the table with properly formatted
        fish entry data and interactive controls.

        Args:
            species: Fish species name
            length_cm: Fish length in centimeters
            confidence: Recognition confidence (0.0-1.0)
            boat_name: Boat name for context
            station_id: Station ID for context
            raw_text: Original speech text
            timestamp: Entry timestamp (uses current time if None)

        Returns:
            bool: True if entry added successfully, False otherwise

        Features:
            - Chronological ordering (newest first)
            - Formatted data display
            - Interactive delete buttons
            - Data validation and sanitization
            - Comprehensive error handling
        """
        try:
            # Validate input data
            if not species or not species.strip():
                logger.warning("Cannot add entry: species is required")
                return False

            if length_cm <= 0:
                logger.warning(f"Cannot add entry: invalid length {length_cm}")
                return False

            # Use current time if not provided
            if timestamp is None:
                timestamp = datetime.now()

            # Format timestamp for display
            date_str = timestamp.strftime("%Y-%m-%d")
            time_str = timestamp.strftime("%H:%M:%S")

            # Insert new row at top (chronological order)
            self.table.insertRow(0)

            # Create and configure table items
            self._set_table_items(0, date_str, time_str, species, length_cm)

            # Add interactive delete button
            self._add_delete_button(0)

            # Store row data for operations
            row_data = {
                'species': species.strip(),
                'length_cm': length_cm,
                'confidence': confidence,
                'boat_name': boat_name,
                'station_id': station_id,
                'raw_text': raw_text,
                'timestamp': timestamp
            }
            self._row_data.insert(0, row_data)

            logger.info(f"Added fish entry: {species} ({length_cm:.1f}cm)")
            return True

        except Exception as e:
            logger.error(f"Failed to add fish entry: {e}")
            return False

    def remove_entry_at_row(self, row_index: int, confirm: bool = True) -> bool:
        """
        Remove fish entry at the specified row index.

        Args:
            row_index: Index of the row to remove
            confirm: Whether to show confirmation dialog

        Returns:
            bool: True if removed successfully, False otherwise

        Features:
            - Bounds checking for safety
            - Optional user confirmation
            - Data consistency maintenance
            - Error handling and recovery
        """
        try:
            # Validate row index
            if not (0 <= row_index < self.table.rowCount()):
                logger.warning(f"Invalid row index: {row_index}")
                return False

            # Show confirmation dialog if requested
            if confirm and not self._confirm_deletion(row_index):
                logger.debug("Row deletion cancelled by user")
                return False

            # Remove from table and data cache
            self.table.removeRow(row_index)

            # Update data cache if available
            if row_index < len(self._row_data):
                removed_data = self._row_data.pop(row_index)
                logger.info(f"Removed entry: {removed_data.get('species', 'Unknown')} "
                           f"({removed_data.get('length_cm', 0):.1f}cm)")
            else:
                logger.info(f"Removed table row {row_index}")

            return True

        except Exception as e:
            logger.error(f"Failed to remove entry at row {row_index}: {e}")
            return False

    def remove_last_entry(self) -> bool:
        """
        Remove the most recently added entry (top row).

        Convenience method for quickly undoing the last entry,
        useful for correction workflows.

        Returns:
            bool: True if removed successfully, False otherwise
        """
        if self.table.rowCount() == 0:
            logger.debug("No entries to remove")
            return False

        return self.remove_entry_at_row(0, confirm=False)

    def clear_all_entries(self, confirm: bool = True) -> bool:
        """
        Clear all entries from the table.

        Args:
            confirm: Whether to show confirmation dialog

        Returns:
            bool: True if cleared successfully, False otherwise

        Features:
            - Bulk deletion with confirmation
            - Complete data cleanup
            - Error handling and logging
        """
        try:
            if self.table.rowCount() == 0:
                logger.debug("Table already empty")
                return True

            # Show confirmation for bulk deletion
            if confirm:
                reply = QMessageBox.question(
                    self.parent_window,
                    "Clear All Entries",
                    f"Are you sure you want to delete all {self.table.rowCount()} entries?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )

                if reply != QMessageBox.StandardButton.Yes:
                    logger.debug("Clear all cancelled by user")
                    return False

            # Clear table and data
            entry_count = self.table.rowCount()
            self.table.setRowCount(0)
            self._row_data.clear()

            logger.info(f"Cleared {entry_count} entries from table")
            return True

        except Exception as e:
            logger.error(f"Failed to clear entries: {e}")
            return False

    def get_entry_count(self) -> int:
        """
        Get the number of entries in the table.

        Returns:
            int: Number of entries
        """
        return self.table.rowCount()

    def get_entry_data(self, row_index: int) -> Optional[Dict[str, Any]]:
        """
        Get the data for a specific entry.

        Args:
            row_index: Index of the entry to retrieve

        Returns:
            Optional[Dict[str, Any]]: Entry data or None if invalid index
        """
        if 0 <= row_index < len(self._row_data):
            return self._row_data[row_index].copy()
        return None

    def get_all_entries_data(self) -> List[Dict[str, Any]]:
        """
        Get data for all entries in the table.

        Returns:
            List[Dict[str, Any]]: List of all entry data

        Features:
            - Deep copy for data safety
            - Chronological ordering (newest first)
        """
        return [entry.copy() for entry in self._row_data]

    def find_entries_by_species(self, species: str) -> List[int]:
        """
        Find all entries matching the specified species.

        Args:
            species: Species name to search for

        Returns:
            List[int]: List of row indices matching the species
        """
        if not species:
            return []

        matching_rows = []
        species_lower = species.lower().strip()

        for i, entry in enumerate(self._row_data):
            entry_species = entry.get('species', '').lower().strip()
            if entry_species == species_lower:
                matching_rows.append(i)

        return matching_rows

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the entries in the table.

        Returns:
            Dict[str, Any]: Statistics including counts, averages, etc.
        """
        if not self._row_data:
            return {
                'total_entries': 0,
                'species_count': 0,
                'average_length': 0.0,
                'length_range': (0.0, 0.0)
            }

        try:
            lengths = [entry['length_cm'] for entry in self._row_data if 'length_cm' in entry]
            species_set = set(entry['species'] for entry in self._row_data if 'species' in entry)

            stats = {
                'total_entries': len(self._row_data),
                'species_count': len(species_set),
                'average_length': sum(lengths) / len(lengths) if lengths else 0.0,
                'length_range': (min(lengths), max(lengths)) if lengths else (0.0, 0.0),
                'species_list': sorted(list(species_set))
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to calculate statistics: {e}")
            return {'error': str(e)}

    def _ensure_table_configuration(self) -> None:
        """
        Ensure the table is properly configured for fish entries.

        Sets up the table with appropriate headers, sizing, and behavior
        if not already configured.
        """
        try:
            # Check if table needs configuration
            if self.table.columnCount() == 0:
                self.table.setColumnCount(5)
                self.table.setHorizontalHeaderLabels([
                    "📅 Date", "⏰ Time", "🐟 Species", "📏 Length (cm)", "🗑"
                ])

            # Configure table behavior
            self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
            self.table.setAlternatingRowColors(True)
            self.table.verticalHeader().setVisible(False)

            # Configure column sizing
            header = self.table.horizontalHeader()
            if header:
                header.resizeSection(0, 120)  # Date
                header.resizeSection(1, 100)  # Time
                header.resizeSection(4, 44)   # Delete button

        except Exception as e:
            logger.error(f"Failed to configure table: {e}")

    def _set_table_items(self, row: int, date_str: str, time_str: str, species: str, length_cm: float) -> None:
        """
        Set table items for the specified row.

        Args:
            row: Row index to set items for
            date_str: Formatted date string
            time_str: Formatted time string
            species: Species name
            length_cm: Fish length in centimeters
        """
        try:
            # Create table items with proper formatting
            items = [
                QTableWidgetItem(date_str),
                QTableWidgetItem(time_str),
                QTableWidgetItem(species),
                QTableWidgetItem(f"{length_cm:.1f}")
            ]

            # Apply center alignment for professional appearance
            for item in items:
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            # Set items in table
            for col, item in enumerate(items):
                self.table.setItem(row, col, item)

            # Set optimal row height
            self.table.setRowHeight(row, 46)

        except Exception as e:
            logger.error(f"Failed to set table items for row {row}: {e}")

    def _add_delete_button(self, row: int) -> None:
        """
        Add delete button to the specified row.

        Args:
            row: Row index to add delete button to
        """
        try:
            # Create styled delete button
            btn_delete = QPushButton()

            # Use standard trash icon if available
            if self.parent_window:
                icon = self.parent_window.style().standardIcon(
                    self.parent_window.style().StandardPixmap.SP_TrashIcon
                )
                btn_delete.setIcon(icon)

            btn_delete.setFixedSize(28, 28)
            btn_delete.setFlat(True)
            btn_delete.setStyleSheet("""
                QPushButton { 
                    border: none; 
                    background: transparent; 
                }
                QPushButton:hover { 
                    color: red; 
                    background: rgba(255, 0, 0, 0.1);
                }
            """)

            # Connect to deletion handler
            btn_delete.clicked.connect(lambda: self._handle_delete_button_click(btn_delete))

            # Add to table
            self.table.setCellWidget(row, 4, btn_delete)

        except Exception as e:
            logger.error(f"Failed to add delete button to row {row}: {e}")

    def _handle_delete_button_click(self, button: QPushButton) -> None:
        """
        Handle delete button click events.

        Args:
            button: The delete button that was clicked
        """
        try:
            # Find the row containing the clicked button
            for row in range(self.table.rowCount()):
                if self.table.cellWidget(row, 4) is button:
                    self.remove_entry_at_row(row, confirm=True)
                    break

        except Exception as e:
            logger.error(f"Failed to handle delete button click: {e}")

    def _confirm_deletion(self, row_index: int) -> bool:
        """
        Show confirmation dialog for entry deletion.

        Args:
            row_index: Index of the row to delete

        Returns:
            bool: True if user confirmed deletion, False otherwise
        """
        try:
            # Get entry information for confirmation
            entry_info = "this entry"
            if row_index < len(self._row_data):
                entry_data = self._row_data[row_index]
                species = entry_data.get('species', 'Unknown')
                length = entry_data.get('length_cm', 0)
                entry_info = f"{species} ({length:.1f}cm)"

            reply = QMessageBox.question(
                self.parent_window,
                "Delete Entry",
                f"Are you sure you want to delete {entry_info}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            return reply == QMessageBox.StandardButton.Yes

        except Exception as e:
            logger.error(f"Failed to show confirmation dialog: {e}")
            return False

