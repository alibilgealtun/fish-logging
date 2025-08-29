from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget, QPushButton, QLabel, QGraphicsBlurEffect, QHeaderView,
)
from loguru import logger
from parser import FishParser
from .widgets import *


class MainWindow(QMainWindow):
    def __init__(self, speech_recognizer, excel_logger) -> None:
        super().__init__()
        self.speech_recognizer = speech_recognizer
        self.excel_logger = excel_logger
        self.setWindowTitle("🎣 Voice2FishLog ")
        self.resize(1100, 700)
        self.fish_parser = FishParser()

        self._setup_background()
        self._setup_modern_theme()
        self._setup_ui()
        self._connect_signals()

        # Start listening immediately (clean restart-friendly)
        if not self.speech_recognizer.isRunning():
            try:
                if hasattr(self.speech_recognizer, "begin"):
                    self.speech_recognizer.begin()
                else:
                    self.speech_recognizer.start()
            except Exception as e:
                logger.error(f"Failed to start recognizer: {e}")

    def _setup_modern_theme(self) -> None:
        """Apply modern dark/light theme to the entire application"""
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #667eea, stop:0.5 #764ba2, stop:1 #667eea);
            }
            QSplitter::handle {
                background: rgba(255, 255, 255, 0.3);
                border-radius: 3px;
                margin: 2px;
            }
            QSplitter::handle:horizontal {
                width: 6px;
            }
            QSplitter::handle:vertical {
                height: 6px;
            }
            QStatusBar {
                background: rgba(255, 255, 255, 0.1);
                color: white;
                border: none;
                border-top: 1px solid rgba(255, 255, 255, 0.2);
                font-weight: 500;
                padding: 8px;
            }
        """)

    def _setup_background(self) -> None:
        """Set blurred image background"""
        self.bg_label = QLabel(self)
        self.bg_pixmap = QPixmap("assets/bg.jpg")

        # set first time
        self.bg_label.setPixmap(
            self.bg_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation
            )
        )
        self.bg_label.setGeometry(self.rect())
        self.bg_label.setScaledContents(True)

        # blur effect
        blur = QGraphicsBlurEffect()
        blur.setBlurRadius(75)
        self.bg_label.setGraphicsEffect(blur)

        # send behind everything
        self.bg_label.lower()

    def resizeEvent(self, event):
        """Keep background scaled when window resizes"""
        super().resizeEvent(event) # Qt calls it automatically
        if hasattr(self, "bg_label") and hasattr(self, "bg_pixmap"):
            self.bg_label.setGeometry(self.rect())
            self.bg_label.setPixmap(
                self.bg_pixmap.scaled(
                    self.size(),
                    Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    Qt.TransformationMode.SmoothTransformation
                )
            )

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)


        # Live transcription section with glassmorphism
        transcription_panel = GlassPanel()
        transcription_layout = QVBoxLayout(transcription_panel)
        transcription_layout.setSpacing(12)
        transcription_layout.setContentsMargins(24, 20, 24, 20)

        transcription_header = ModernLabel("🎤 Live Transcription", "header")
        transcription_layout.addWidget(transcription_header)

        self.live_text = ModernTextEdit()
        self.live_text.setReadOnly(True)
        self.live_text.setPlaceholderText("🎧 Listening for your voice... speak naturally about your catch!")
        self.live_text.setMinimumHeight(50)
        self.live_text.setMaximumHeight(60)
        try:
            self.live_text.setLineWrapMode(self.live_text.LineWrapMode.NoWrap)
        except Exception:
            pass
        self.live_text.setStyleSheet(self.live_text.styleSheet() + "\nModernTextEdit { font-size: 16px;  }\n")
        transcription_layout.addWidget(self.live_text)

        # Table section with glassmorphism
        table_panel = GlassPanel()
        table_layout = QVBoxLayout(table_panel)
        table_layout.setSpacing(16)
        table_layout.setContentsMargins(24, 20, 24, 20)



        header_row = QHBoxLayout()
        table_header = ModernLabel("📊 Logged Entries", "header")

        self.boat_input = BoatNameInput()

        header_row.addWidget(table_header)
        header_row.addStretch(1)
        header_row.addWidget(self.boat_input)
        table_layout.addLayout(header_row)


        self.table = ModernTable(0, 7)
        self.table.setHorizontalHeaderLabels(["📅 Date", "⏰ Time","⛵ Boat", "🐟 Species", "📏 Length (cm)", "🎯 Confidence", "🗑"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSortingEnabled(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        # Set a tight width for the trash column and let others expand
        header = self.table.horizontalHeader()
        try:
            header.setSectionResizeMode(6, QHeaderView.ResizeMode.Fixed)
        except Exception:
            pass
        header.resizeSection(6, 44)
        table_layout.addWidget(self.table)

        # Control panel with glassmorphism
        control_panel = GlassPanel()
        control_layout = QHBoxLayout(control_panel)
        control_layout.setContentsMargins(24, 16, 24, 16)
        control_layout.setSpacing(20)

        self.btn_start = AnimatedButton("▶️ Start Listening", primary=True)
        self.btn_start.setEnabled(False)
        self.btn_stop = AnimatedButton("⏹️ Stop Listening")

        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)

        # Status indicator
        status_container = QWidget()
        status_layout = QHBoxLayout(status_container)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(12)

        self.status_panel = PulsingStatusIndicator()
        self.status_label = ModernLabel("🟢 Ready to Listen", "subheader")
        self.status_label.setStyleSheet("""
            ModernLabel {
                color: white;
                font-weight: 600;
                font-size: 15px;
            }
        """)

        status_layout.addWidget(self.status_panel)
        status_layout.addWidget(self.status_label)

        control_layout.addWidget(status_container)
        control_layout.addStretch(1)

        table_layout.addWidget(control_panel)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(transcription_panel)
        splitter.addWidget(table_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([150, 550])

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.addWidget(splitter)
        central.setLayout(main_layout)

        self.statusBar().showMessage("🎧 Ready to capture your fishing stories...")

        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_start.clicked.connect(self._on_start)

    def _connect_signals(self) -> None:
        # Avoid duplicate connections by disconnecting if already connected
        try:
            self.speech_recognizer.partial_text.disconnect(self._on_partial_text)
        except Exception:
            pass
        try:
            self.speech_recognizer.final_text.disconnect(self._on_final_text)
        except Exception:
            pass
        try:
            self.speech_recognizer.error.disconnect(self._on_error)
        except Exception:
            pass
        self.speech_recognizer.partial_text.connect(self._on_partial_text)
        self.speech_recognizer.final_text.connect(self._on_final_text)
        self.speech_recognizer.error.connect(self._on_error)
        # Live status updates (Listening, Capturing, Finishing, Stopped)
        if hasattr(self.speech_recognizer, "status_changed"):
            try:
                self.speech_recognizer.status_changed.disconnect(self._on_status)
            except Exception:
                pass
            self.speech_recognizer.status_changed.connect(self._on_status)

    def _on_partial_text(self, text: str) -> None:
        # Show last partial text in live panel and log to console
        self.live_text.setPlainText(text)
        self.live_text.verticalScrollBar().setValue(self.live_text.verticalScrollBar().maximum())
        logger.info(f"[GUI partial] {text}")

    def _on_final_text(self, text: str, confidence: float) -> None:
        self.live_text.setPlainText(text)
        self.live_text.verticalScrollBar().setValue(self.live_text.verticalScrollBar().maximum())
        logger.info(f"[GUI final] conf={confidence:.2f} text={text}")

        result = self.fish_parser.parse_text(text)
        if result.cancel:
            ok = self.excel_logger.cancel_last()
            if ok:
                self._remove_last_table_row()
                self.statusBar().showMessage("✅ Last entry cancelled successfully", 3000)
                logger.info("[parsed] CANCEL -> removed last entry")
            else:
                self.statusBar().showMessage("ℹ️ Nothing to cancel", 3000)
                logger.info("[parsed] CANCEL -> nothing to remove")
            return

        logger.info(f"[parsed] species={result.species} length_cm={result.length_cm}")

        if result.species and result.length_cm is not None:
            # Log and update table (newest first)
            try:
                boat_name = self.boat_input.get_boat_name()
                self.boat_input.save_boat_name()
                self.excel_logger.log_entry(result.species, result.length_cm, confidence, boat_name)
                logger.info(f"[excel] Logged {result.species} {result.length_cm:.1f} cm conf={confidence:.2f}")
                self._prepend_table_row(result.species, result.length_cm, confidence, boat_name)
                self.statusBar().showMessage("🎣 Great catch logged successfully!", 2000)
            except Exception as e:
                self._alert(f"Failed to log to Excel: {e}")

    def _prepend_table_row(self, species: str, length_cm: float, confidence: float, boat: str) -> None:
        # Excel has date/time; here we show current
        from datetime import datetime
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        self.table.insertRow(0)
        self.table.setItem(0, 0, QTableWidgetItem(date_str))
        self.table.setItem(0, 1, QTableWidgetItem(time_str))
        self.table.setItem(0, 2, QTableWidgetItem(boat or "—"))
        self.table.setItem(0, 3, QTableWidgetItem(species))
        self.table.setItem(0, 4, QTableWidgetItem(f"{length_cm:.1f}"))
        self.table.setItem(0, 5, QTableWidgetItem(f"{confidence:.2f}"))
        self.table.setRowHeight(0, 46)


        btn_delete = QPushButton("🗑")
        btn_delete.setFixedSize(15,30)
        btn_delete.setStyleSheet("""
            QPushButton {
                font-size: 25px; 
                border: none;
                background: transparent;
            }
            QPushButton:hover {
                color: red;  /* Make it turn red when hovered */
                cursor: pointer;
            }
        """)

        btn_delete.setFlat(True)
        btn_delete.clicked.connect(lambda _, row=0: self._remove_table_row(row))
        self.table.setCellWidget(0, 6, btn_delete)

    def _remove_last_table_row(self) -> None:
        if self.table.rowCount() > 0:
            self.table.removeRow(0)

    def _remove_table_row(self, row_index: int) -> None:
        if 0 <= row_index < self.table.rowCount():
            reply = QMessageBox.question(
                self, "Delete Entry",
                "Are you sure you want to delete this entry?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.table.removeRow(row_index)

    def _on_error(self, message: str) -> None:
        self._alert(message)
        self.statusBar().showMessage("❌ Error: " + message)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_panel.set_status_color("#f44336", "#d32f2f")

    def _on_status(self, message: str) -> None:
        # Update color and text to communicate state at a glance
        status_config = {
            "listening": {
                "color": "#2196F3",
                "bg_color": "#1976D2",
                "label": "🎧 Listening for speech...",
                "status": "Ready to capture your words"
            },
            "capturing": {
                "color": "#4CAF50",
                "bg_color": "#388E3C",
                "label": "🎤 Capturing speech...",
                "status": "Recording your fishing story"
            },
            "finishing": {
                "color": "#FF9800",
                "bg_color": "#F57C00",
                "label": "⚡ Processing...",
                "status": "Analyzing your catch data"
            },
            "stopped": {
                "color": "#9E9E9E",
                "bg_color": "#616161",
                "label": "⏸️ Listening stopped",
                "status": "Voice recognition paused"
            },
        }

        config = status_config.get(message, status_config["listening"])
        self.status_panel.set_status_color(config["color"], config["bg_color"])
        self.status_label.setText(config["label"])
        self.statusBar().showMessage(config["status"])

    def _on_stop(self) -> None:
        try:
            self.speech_recognizer.stop()
        finally:
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.statusBar().showMessage("⏸️ Voice recognition stopped")
            self.status_panel.set_status_color("#9E9E9E", "#616161")

    def _on_start(self) -> None:
        if not self.speech_recognizer.isRunning():
            try:
                if hasattr(self.speech_recognizer, "begin"):
                    self.speech_recognizer.begin()
                else:
                    self.speech_recognizer.start()
                self.statusBar().showMessage("🎧 Listening for your fishing stories...")
            except Exception as e:
                self._alert(f"Failed to start listening: {e}")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status_panel.set_status_color("#2196F3", "#1976D2")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        try:
            self.speech_recognizer.stop()
        except Exception:
            pass
        return super().closeEvent(event)

    def _alert(self, message: str) -> None:
        logger.error(message)
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Voice2FishLog - Error")
        msg_box.setText(message)
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setStyleSheet("""
            QMessageBox {
                background: white;
                color: #333;
                font-family: 'Segoe UI', system-ui, sans-serif;
            }
            QMessageBox QPushButton {
                background: #667eea;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
                min-width: 80px;
            }
            QMessageBox QPushButton:hover {
                background: #7c9df0;
            }
        """)
        msg_box.exec()