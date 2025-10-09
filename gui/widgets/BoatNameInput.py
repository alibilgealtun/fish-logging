from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QLineEdit, QWidget, QSizePolicy
)


class BoatNameInput(QWidget):
    """Boat name input that remembers value and behaves nicely on Enter."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.settings = QSettings("Voice2FishLog", "BoatConfig")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        label = QLabel("⛵ Boat:")
        label.setStyleSheet("QLabel { color: white; }")
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("Enter boat name...")
        self.edit.setMinimumHeight(36)
        self.edit.setFixedWidth(240)
        self.edit.setStyleSheet("""
            QLineEdit {
                background: white;
                color: black;
                border-radius: 6px;
                padding: 6px 10px;
            }
        """)

        # load last saved
        self.edit.setText(self.settings.value("boat_name", ""))

        # save on Enter + stop cursor blinking by removing focus
        self.edit.returnPressed.connect(self._on_enter)
        # also save when leaving the field
        self.edit.editingFinished.connect(self.save_boat_name)

        layout.addWidget(label)
        layout.addWidget(self.edit)

        # FIX: use QSizePolicy.Policy.Fixed
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def get_boat_name(self) -> str:
        return self.edit.text().strip()

    def save_boat_name(self) -> None:
        self.settings.setValue("boat_name", self.get_boat_name())

    def _on_enter(self) -> None:
        self.save_boat_name()
        # move focus away so the caret stops blinking
        self.edit.clearFocus()
        w = self.window()
        if isinstance(w, QWidget):
            w.setFocus()