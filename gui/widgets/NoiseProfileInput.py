from __future__ import annotations

from typing import Optional, List

from PyQt6.QtCore import QSettings, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QComboBox, QWidget, QSizePolicy
)

from speech.noise_profiles import get_manager


class NoiseProfileInput(QWidget):
    """Noise profile selector with persistent value and white label text."""

    profileChanged = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.settings = QSettings("Voice2FishLog", "NoiseConfig")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.label = QLabel("🔊 Noise Profile:")
        self.label.setStyleSheet("QLabel { color: white; }")

        self.combo = QComboBox()
        self.combo.setMinimumHeight(36)
        self.combo.setFixedWidth(180)
        self.combo.setStyleSheet(
            "QComboBox { background: white; color: black; border-radius: 6px; padding: 6px 10px; }"
        )

        # Populate from available profiles
        manager = get_manager()
        profiles: List[str] = manager.list_profiles()
        # Keep mapping original lower-case names but show nice capitalization
        self._profiles: List[str] = profiles
        for name in profiles:
            self.combo.addItem(name.capitalize(), userData=name)

        # Load last saved
        saved = str(self.settings.value("profile", "mixed"))
        # Set current index by userData
        idx = max(0, next((i for i in range(self.combo.count()) if self.combo.itemData(i) == saved), 0))
        self.combo.setCurrentIndex(idx)

        # Events
        self.combo.currentIndexChanged.connect(self._on_changed)

        layout.addWidget(self.label)
        layout.addWidget(self.combo)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def _on_changed(self, _idx: int) -> None:
        self.save_noise_profile()
        self.profileChanged.emit(self.get_noise_profile())

    def get_noise_profile(self) -> str:
        data = self.combo.currentData()
        return str(data) if data else "mixed"

    def save_noise_profile(self) -> None:
        self.settings.setValue("profile", self.get_noise_profile())

