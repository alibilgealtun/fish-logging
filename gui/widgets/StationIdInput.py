from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QLineEdit, QWidget, QSizePolicy
)


class StationIdInput(QWidget):
    """Station ID input that remembers value and saves on Enter/blur."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.settings = QSettings("Voice2FishLog", "StationConfig")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        label = QLabel("📍 Station:")
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("Enter station ID…")
        self.edit.setMinimumHeight(36)
        self.edit.setFixedWidth(200)
        self.edit.setStyleSheet("""
            QLineEdit {
                background: white;
                color: black;
                border-radius: 6px;
                padding: 6px 10px;
            }
        """)

        # Load last saved
        self.edit.setText(self.settings.value("station_id", ""))

        # Save on Enter + stop cursor blinking by removing focus
        self.edit.returnPressed.connect(self._on_enter)
        # Also save when leaving the field
        self.edit.editingFinished.connect(self.save_station_id)

        layout.addWidget(label)
        layout.addWidget(self.edit)

        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def get_station_id(self) -> str:
        return self.edit.text().strip()

    def save_station_id(self) -> None:
        self.settings.setValue("station_id", self.get_station_id())

    def _on_enter(self) -> None:
        self.save_station_id()
        # move focus away so the caret stops blinking
        self.edit.clearFocus()
        w = self.window()
        if isinstance(w, QWidget):
            w.setFocus()

