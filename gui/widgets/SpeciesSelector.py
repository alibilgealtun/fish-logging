from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtWidgets import QComboBox, QCompleter, QWidget

import json


@dataclass
class SpeciesItem:
    number: int
    name: str
    code: Optional[str] = None

    @property
    def display(self) -> str:
        num = f"{self.number}"
        return f"{num} — {self.name} ({self.code})" if self.code else f"{num} — {self.name}"

    def search_blob(self, synonyms: List[str]) -> str:
        parts = [self.name, self.name.lower(), str(self.number), f"{self.number:02d}"]
        if self.code:
            parts.extend([self.code, (self.code or "").lower()])
        parts.extend(synonyms)
        seen = set()
        uniq = []
        for p in parts:
            if p and p not in seen:
                uniq.append(p)
                seen.add(p)
        return " | ".join(uniq)


class SpeciesSelector(QComboBox):
    speciesChanged = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        # Width reduced by ~20%
        self.setMinimumWidth(240)
        self.setMinimumHeight(36)
        # White/black styling and night-blue drop-down
        self.setStyleSheet(
            """
            QComboBox {
                background: #ffffff;
                color: #000000;
                border: 1px solid #cfcfcf;
                border-radius: 6px;
                padding: 6px 36px 6px 8px;
                font-size: 14px;
            }
            QComboBox:focus, QComboBox:hover, QComboBox:on, QComboBox:pressed {
                background: #ffffff;
                color: #000000;
                border: 1px solid #7aa2ff;
            }
            QComboBox QLineEdit {
                background: #ffffff;
                color: #000000;
                selection-background-color: #e6f0ff;
                selection-color: #000000;
                border: none;
            }
            QComboBox QAbstractItemView {
                background: #ffffff;
                color: #000000;
                selection-background-color: #e6f0ff;
                selection-color: #000000;
                outline: none;
                border: 1px solid #cfcfcf;
            }
            QComboBox QAbstractItemView::item { color: #000000; }
            QComboBox QAbstractItemView::item:selected { background: #e6f0ff; color: #000000; }
            QComboBox::drop-down {
                background: #0b1d3a;
                width: 30px;
                border: none;
                border-left: 1px solid #cfcfcf;
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
            }
            """
        )
        self.setPlaceholderText("Search species or number…")
        try:
            self.lineEdit().setPlaceholderText("Search species name, code, or number…")  # type: ignore[union-attr]
        except Exception:
            pass

        self._items: List[SpeciesItem] = []
        self._name_to_row: Dict[str, int] = {}

        self._model = QStandardItemModel(self)
        self.setModel(self._model)
        self.setModelColumn(0)

        data = self._load_species_data()
        self._populate(data)
        self._setup_completer()
        self.currentIndexChanged.connect(self._emit_change)

    def currentSpecies(self) -> Optional[str]:
        idx = self.currentIndex()
        if idx < 0 or idx >= len(self._items):
            return None
        return self._items[idx].name

    def setCurrentByName(self, name: str) -> None:
        if not name:
            return
        key = name.strip().lower()
        if key in self._name_to_row:
            row = self._name_to_row[key]
            if 0 <= row < self.count():
                self.setCurrentIndex(row)

    def _emit_change(self, _idx: int) -> None:
        name = self.currentSpecies() or ""
        self.speciesChanged.emit(name)

    def _setup_completer(self) -> None:
        completer = QCompleter(self._model, self)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        completer.setCompletionRole(Qt.ItemDataRole.UserRole)
        completer.setCompletionColumn(0)
        self.setCompleter(completer)
        try:
            self.lineEdit().returnPressed.connect(self._accept_completion)  # type: ignore[union-attr]
        except Exception:
            pass

    def _accept_completion(self) -> None:
        c = self.completer()
        if not c:
            return
        idx = c.currentIndex()
        if idx.isValid():
            self.setCurrentIndex(idx.row())

    def _load_species_data(self) -> Dict:
        cfg_path = Path("config/species.json")
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"items": [], "normalization": {}, "species": []}

    def _generate_code(self, name: str) -> str:
        s = ''.join(ch for ch in name.upper() if ch.isalpha())
        return (s + "XXX")[:3]

    def _populate(self, data: Dict) -> None:
        # Support new schema with "items": [ {name, code}... ] and legacy "species": ["name"]
        norm_map: Dict[str, str] = {str(k): str(v) for k, v in data.get("normalization", {}).items()}

        ordered_items: List[SpeciesItem] = []
        if isinstance(data.get("items"), list) and data["items"]:
            for i, entry in enumerate(data["items"], start=1):
                name = str(entry.get("name", "")).strip()
                if not name:
                    continue
                code = str(entry.get("code", "")).strip() or None
                ordered_items.append(SpeciesItem(i, name, code))
        else:
            # Legacy fallback
            seen = set()
            names: List[str] = [str(x) for x in data.get("species", [])]
            flat: List[str] = []
            for n in names:
                key = n.strip().lower()
                if key and key not in seen:
                    flat.append(n)
                    seen.add(key)
            for i, name in enumerate(flat, start=1):
                ordered_items.append(SpeciesItem(i, name, None))

        self._items.clear()
        self._model.clear()
        self._name_to_row.clear()
        self._model.setColumnCount(2)

        for item in ordered_items:
            display_name = item.name  # keep canonical case from JSON
            code = item.code or self._generate_code(item.name)
            # Build synonyms from normalization map
            synonyms: List[str] = []
            name_lc = item.name.lower()
            for k, v in norm_map.items():
                if v.lower() == name_lc:
                    synonyms.extend([k, k.lower()])
            synonyms.append(item.name.replace("-", " ").lower())

            search_blob = SpeciesItem(item.number, item.name, code).search_blob(synonyms)
            row_disp = QStandardItem(f"{item.number} — {display_name} ({code})")
            row_disp.setData(item.name, Qt.ItemDataRole.UserRole + 1)
            row_disp.setData(search_blob, Qt.ItemDataRole.UserRole)
            row_search = QStandardItem(search_blob)

            self._model.appendRow([row_disp, row_search])
            self._items.append(SpeciesItem(item.number, item.name, code))
            self._name_to_row[name_lc] = item.number - 1

        if self._items:
            self.setCurrentIndex(0)
