from __future__ import annotations

from typing import Optional

from loguru import logger
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, pyqtProperty, QTimer
from PyQt6.QtGui import QFont, QPalette, QColor, QLinearGradient, QPainter, QBrush
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QFrame,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QHeaderView,
    QGraphicsDropShadowEffect,
)

from parser import parse_text


class AnimatedButton(QPushButton):
    """Custom button with hover animations and modern styling"""
    
    def __init__(self, text: str, primary: bool = False):
        super().__init__(text)
        self.primary = primary
        self.hover_animation = QPropertyAnimation(self, b"geometry")
        self.hover_animation.setDuration(200)
        self.hover_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._setup_style()
        
    def _setup_style(self):
        if self.primary:
            self.setStyleSheet("""
                AnimatedButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #667eea, stop:1 #764ba2);
                    color: white;
                    border: none;
                    border-radius: 12px;
                    padding: 12px 24px;
                    font-weight: 600;
                    font-size: 14px;
                    min-width: 100px;
                }
                AnimatedButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #7c9df0, stop:1 #8b5fbf);
                    transform: translateY(-2px);
                }
                AnimatedButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #5a6fd8, stop:1 #6a4190);
                }
                AnimatedButton:disabled {
                    background: #cccccc;
                    color: #666666;
                }
            """)
        else:
            self.setStyleSheet("""
                AnimatedButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #ff6b6b, stop:1 #ee5a52);
                    color: white;
                    border: none;
                    border-radius: 12px;
                    padding: 12px 24px;
                    font-weight: 600;
                    font-size: 14px;
                    min-width: 100px;
                }
                AnimatedButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #ff7979, stop:1 #fd6c5d);
                }
                AnimatedButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #e55656, stop:1 #d63447);
                }
            """)

class PulsingStatusIndicator(QFrame):
    """Animated status indicator with pulsing effect"""
    
    def __init__(self):
        super().__init__()
        self.setFixedSize(20, 20)
        self._setup_style()
        
        # Pulsing animation
        self.pulse_timer = QTimer()
        self.pulse_timer.timeout.connect(self._pulse)
        self.pulse_timer.start(1000)
        self.pulse_state = 0
        
    def _setup_style(self):
        self.setStyleSheet("""
            PulsingStatusIndicator {
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.8,
                    fx:0.3, fy:0.3, stop:0 #4CAF50, stop:1 #2E7D32);
                border-radius: 10px;
                border: 3px solid rgba(255, 255, 255, 0.3);
            }
        """)
        
        # Add glow effect
        glow = QGraphicsDropShadowEffect()
        glow.setBlurRadius(15)
        glow.setColor(QColor(76, 175, 80, 100))
        glow.setOffset(0, 0)
        self.setGraphicsEffect(glow)
        
    def _pulse(self):
        self.pulse_state = (self.pulse_state + 1) % 3
        alpha = [100, 150, 200][self.pulse_state]
        
        if hasattr(self, 'current_color'):
            color = self.current_color
        else:
            color = "#4CAF50"
            
        glow = QGraphicsDropShadowEffect()
        glow.setBlurRadius(20)
        glow.setColor(QColor(color).lighter(120))
        glow.setOffset(0, 0)
        self.setGraphicsEffect(glow)
        
    def set_status_color(self, color: str, bg_color: str):
        self.current_color = color
        self.setStyleSheet(f"""
            PulsingStatusIndicator {{
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.8,
                    fx:0.3, fy:0.3, stop:0 {color}, stop:1 {bg_color});
                border-radius: 10px;
                border: 3px solid rgba(255, 255, 255, 0.4);
            }}
        """)

class ModernTextEdit(QTextEdit):
    """Styled text edit with modern appearance"""
    
    def __init__(self):
        super().__init__()
        self._setup_style()
        
    def _setup_style(self):
        self.setStyleSheet("""
            ModernTextEdit {
                background: white;
                border: 2px solid rgba(108, 117, 125, 0.3);
                border-radius: 16px;
                padding: 16px;
                font-family: 'Segoe UI', system-ui, sans-serif;
                font-size: 14px;
                line-height: 1.5;
                color: #2c3e50;
                selection-background-color: #667eea;
                selection-color: white;
            }
            ModernTextEdit:focus {
                border: 2px solid #667eea;
                background: white;
            }
        """)
        
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)

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
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        
        # Add shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 0, 0, 20))
        shadow.setOffset(0, 8)
        self.setGraphicsEffect(shadow)

class GlassPanel(QFrame):
    """Glassmorphism panel effect"""
    
    def __init__(self):
        super().__init__()
        self._setup_style()
        
    def _setup_style(self):
        self.setStyleSheet("""
            GlassPanel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255, 255, 255, 0.25),
                    stop:1 rgba(255, 255, 255, 0.1));
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 20px;
                backdrop-filter: blur(10px);
            }
        """)

class ModernLabel(QLabel):
    """Styled label with modern typography"""
    
    def __init__(self, text: str, style: str = "normal"):
        super().__init__(text)
        self._setup_style(style)
        
    def _setup_style(self, style: str):
        if style == "header":
            self.setStyleSheet("""
                ModernLabel {
                    color: #2c3e50;
                    font-family: 'Segoe UI', system-ui, sans-serif;
                    font-size: 20px;
                    font-weight: 700;
                    margin: 8px 0px;
                }
            """)
        elif style == "subheader":
            self.setStyleSheet("""
                ModernLabel {
                    color: #34495e;
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


class MainWindow(QMainWindow):
    def __init__(self, speech_recognizer, excel_logger) -> None:
        super().__init__()
        self.speech_recognizer = speech_recognizer
        self.excel_logger = excel_logger
        self.setWindowTitle("🎣 Voice2FishLog - Modern Edition")
        self.resize(1100, 700)
        
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
        self.live_text.setMinimumHeight(160)
        transcription_layout.addWidget(self.live_text)

        # Table section with glassmorphism
        table_panel = GlassPanel()
        table_layout = QVBoxLayout(table_panel)
        table_layout.setSpacing(16)
        table_layout.setContentsMargins(24, 20, 24, 20)
        
        table_header = ModernLabel("📊 Logged Entries", "header")
        table_layout.addWidget(table_header)

        self.table = ModernTable(0, 5)
        self.table.setHorizontalHeaderLabels(["📅 Date", "⏰ Time", "🐟 Species", "📏 Length (cm)", "🎯 Confidence"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSortingEnabled(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
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
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([250, 450])

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

        result = parse_text(text)
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
                self.excel_logger.log_entry(result.species, result.length_cm, confidence)
                logger.info(f"[excel] Logged {result.species} {result.length_cm:.1f} cm conf={confidence:.2f}")
                self._prepend_table_row(result.species, result.length_cm, confidence)
                self.statusBar().showMessage("🎣 Great catch logged successfully!", 2000)
            except Exception as e:
                self._alert(f"Failed to log to Excel: {e}")

    def _prepend_table_row(self, species: str, length_cm: float, confidence: float) -> None:
        # Excel has date/time; here we show current
        from datetime import datetime
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        self.table.insertRow(0)
        self.table.setItem(0, 0, QTableWidgetItem(date_str))
        self.table.setItem(0, 1, QTableWidgetItem(time_str))
        self.table.setItem(0, 2, QTableWidgetItem(species))
        self.table.setItem(0, 3, QTableWidgetItem(f"{length_cm:.1f}"))
        self.table.setItem(0, 4, QTableWidgetItem(f"{confidence:.2f}"))

    def _remove_last_table_row(self) -> None:
        if self.table.rowCount() > 0:
            self.table.removeRow(0)

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