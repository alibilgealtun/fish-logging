from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal, QModelIndex
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtWidgets import QComboBox, QCompleter, QWidget, QListView

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


class SpeciesCompleter(QCompleter):
    """Custom completer that uses search blob (UserRole) for matching but returns
    the pretty display text for insertion so the line edit never shows the blob."""
    def pathFromIndex(self, index):  # type: ignore[override]
        try:
            if not index.isValid():
                return ""
            model = index.model()
            # Always pull display from first column (0) even if role is UserRole
            disp_index = model.index(index.row(), 0)
            txt = model.data(disp_index, Qt.ItemDataRole.DisplayRole)
            return txt if isinstance(txt, str) else super().pathFromIndex(index)
        except Exception:
            return super().pathFromIndex(index)


class SpeciesSelector(QComboBox):
    speciesChanged = pyqtSignal(str)
    # Unified list popup style applied to BOTH dropdown and search popup
    _POPUP_STYLE = (
        "QListView {"
        " background:#ffffff; color:#111827; border:1px solid #cbd5e1;"
        " outline:none; font-size:14px; font-family:'Segoe UI','Helvetica Neue',Arial,sans-serif;"
        " }"
        "QListView::item { padding:6px 10px; }"
        "QListView::item:selected { background:#e6f0ff; color:#111827; }"
        "QListView::item:hover { background:#f1f5f9; }"
    )

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.setMinimumWidth(240)
        self.setMinimumHeight(36)

        # Minimal, parse-safe combobox style (no comments, no complex SVG)
        self.setStyleSheet(
            "QComboBox { background:#ffffff; color:#111827; border:1px solid #cbd5e1; border-radius:6px;"
            " padding:6px 40px 6px 12px; font-size:14px; font-family:'Segoe UI','Helvetica Neue',Arial,sans-serif; }"
            "QComboBox:focus, QComboBox:hover, QComboBox:on { border:1px solid #2563eb; }"
            "QComboBox QLineEdit { background:#ffffff; color:#111827; border:none; selection-background-color:#e6f0ff; selection-color:#111827; font-size:14px; }"
            "QComboBox::drop-down { subcontrol-origin:padding; subcontrol-position:top right; width:34px; border-left:1px solid #cbd5e1;"
            " background:#f1f5f9; border-top-right-radius:6px; border-bottom-right-radius:6px; margin:0; padding:0; }"
            "QComboBox::drop-down:hover { background:#e2e8f0; }"
            "QComboBox::down-arrow { width:0; height:0; border-left:7px solid transparent; border-right:7px solid transparent;"
            " border-top:12px solid #1f2937; margin-right:10px; }"
        )

        # Initialize internal storage BEFORE population
        self._items: List[SpeciesItem] = []
        self._name_to_row: Dict[str, int] = {}
        self._model = QStandardItemModel(self)
        self.setModel(self._model)
        self.setModelColumn(0)

        # Dedicated QListView for dropdown; style unified
        list_view = QListView(self)
        list_view.setUniformItemSizes(True)
        list_view.setStyleSheet(self._POPUP_STYLE)
        self.setView(list_view)

        # Placeholder
        try:
            self.setPlaceholderText("Search species or number…")
            self.lineEdit().setPlaceholderText("Search species name, code, or number…")  # type: ignore[union-attr]
        except Exception:
            pass

        # Load + populate
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
        # Always enforce pretty display text
        row = self.currentIndex()
        if 0 <= row < self._model.rowCount():
            display_text = self._model.item(row, 0).text()
            if self.isEditable() and self.lineEdit():
                self.blockSignals(True)
                self.lineEdit().setText(display_text)
                self.blockSignals(False)
        self.speciesChanged.emit(self.currentSpecies() or "")

    def _setup_completer(self) -> None:
        completer = SpeciesCompleter(self._model, self)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        # We keep search blob in UserRole of first column item; set completionRole to that, but pathFromIndex gives pretty text
        completer.setCompletionRole(Qt.ItemDataRole.UserRole)
        completer.setCompletionColumn(0)
        popup = QListView(self)
        popup.setUniformItemSizes(True)
        popup.setStyleSheet(self._POPUP_STYLE)
        completer.setPopup(popup)
        self.setCompleter(completer)
        try:
            self.lineEdit().returnPressed.connect(self._accept_completion)  # type: ignore[union-attr]
        except Exception:
            pass
        # connect highlight index to preview pretty text (first column) not search blob
        try:
            popup = completer.popup()
            completer.highlighted[QModelIndex].connect(self._on_completer_highlight_index)  # type: ignore
        except Exception:
            pass

    def _accept_completion(self) -> None:
        c = self.completer()
        if not c:
            return
        idx = c.currentIndex()
        if idx.isValid():
            self._select_from_completer_index(idx)
            return
        # If nothing selected in popup but popup visible, choose first
        popup = c.popup()
        if popup and popup.model() and popup.model().rowCount() > 0:
            first = popup.model().index(0, 0)
            self._select_from_completer_index(first)

    def _load_species_data(self) -> Dict:
        try:
            # Get data from centralized config
            from config.config import ConfigLoader
            loader = ConfigLoader()
            config, _ = loader.load([])
            return config.species_data
        except Exception:
            return {"items": [], "normalization": {}, "species": []}

    def _generate_code(self, name: str) -> str:
        s = ''.join(ch for ch in name.upper() if ch.isalpha())
        return (s + "XXX")[:3]

    def _populate(self, data: Dict) -> None:
        # Safe if called multiple times
        if not hasattr(self, '_items'):
            self._items = []
        if not hasattr(self, '_name_to_row'):
            self._name_to_row = {}
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

    def _select_species_by_name(self, name: str) -> None:
        key = name.strip().lower()
        row = self._name_to_row.get(key)
        if row is not None and 0 <= row < self.count():
            self.setCurrentIndex(row)

    def _select_from_completer_index(self, idx) -> None:
        try:
            if not idx or not idx.isValid():
                return
            # Try direct species name from UserRole+1
            name_data = idx.data(Qt.ItemDataRole.UserRole + 1)
            if isinstance(name_data, str) and name_data.strip():
                self._select_species_by_name(name_data)
                return
            # Fallback: parse display text up to the dash
            disp = idx.data(Qt.ItemDataRole.DisplayRole)
            if isinstance(disp, str):
                # attempt to extract number — Name (CODE)
                parts = disp.split('—', 1)
                if len(parts) == 2:
                    # after dash, before '('
                    name_part = parts[1].strip()
                    if '(' in name_part:
                        name_part = name_part.split('(')[0].strip()
                    self._select_species_by_name(name_part)
        except Exception:
            pass

    def _on_completer_highlight_index(self, idx: QModelIndex) -> None:
        try:
            if not idx.isValid():
                return
            row = idx.row()
            if 0 <= row < self._model.rowCount():
                disp = self._model.item(row, 0).text()
                if self.isEditable() and self.lineEdit():
                    self.blockSignals(True)
                    self.lineEdit().setText(disp)
                    self.blockSignals(False)
        except Exception:
            pass

    def keyPressEvent(self, event):  # type: ignore[override]
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            c = self.completer()
            if c and c.popup() and c.popup().isVisible():
                idx = c.popup().currentIndex()
                if not idx.isValid() and c.popup().model().rowCount() > 0:
                    idx = c.popup().model().index(0, 0)
                if idx.isValid():
                    self._select_from_completer_index(idx)
                    event.accept()
                    return
        super().keyPressEvent(event)
