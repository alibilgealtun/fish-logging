from __future__ import annotations

from PyQt6.QtCore import Qt, QSize
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
    QTabWidget, QStyle,
)
# Optional logger: prefer loguru if available, else fallback to stdlib logging
try:  # pragma: no cover
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("main_window")

from parser import FishParser
from .widgets import *
from .widgets.ReportWidget import ReportWidget
from .widgets.SettingsWidget import SettingsWidget
from .table_manager import TableManager
from .speech_event_handler import SpeechEventHandler
from .status_presenter import StatusPresenter
from .settings_provider import SettingsProvider
from .recognizer_controller import RecognizerController
from app.use_cases import (
    ProcessFinalTextUseCase,
    LogFishEntryUseCase,
    CancelLastEntryUseCase,
)


class MainWindow(QMainWindow):
    def __init__(self, speech_recognizer, excel_logger, fish_parser: FishParser = None) -> None:
        super().__init__()
        self.speech_recognizer = speech_recognizer
        self.excel_logger = excel_logger
        self.setWindowTitle("🎣 Voice2FishLog ")
        self.resize(1100, 700)

        # Initialize domain components (with dependency injection)
        self.fish_parser = fish_parser if fish_parser is not None else FishParser()

        # Initialize use cases
        self.process_text_use_case = ProcessFinalTextUseCase(self.fish_parser)
        self.log_entry_use_case = LogFishEntryUseCase(self.excel_logger)
        self.cancel_entry_use_case = CancelLastEntryUseCase(self.excel_logger)

        # Initialize speech event handler
        self.speech_handler = SpeechEventHandler(
            self.process_text_use_case,
            self.log_entry_use_case,
            self.cancel_entry_use_case,
        )

        # Setup callbacks for speech handler (will be set after UI is created)
        self._setup_speech_handler_callbacks()

        self._setup_background()
        self._setup_modern_theme()
        self._setup_ui()

        # Initialize helper classes after UI is created
        self.table_manager = TableManager(self.table, self)
        self.status_presenter = StatusPresenter(self.status_panel, self.status_label, self.statusBar())
        self.settings_provider = SettingsProvider(self.settings_widget)
        self.recognizer_controller = RecognizerController(self.speech_recognizer)

        self._connect_signals()

        # Start listening immediately using the controller
        self.recognizer_controller.ensure_started()

    def _setup_speech_handler_callbacks(self) -> None:
        """Setup callbacks for the speech event handler."""
        # These will update the UI based on events
        self.speech_handler.on_partial_update = self._update_live_text
        self.speech_handler.on_entry_logged = self._on_entry_logged_success
        self.speech_handler.on_entry_cancelled = self._on_entry_cancelled_success
        self.speech_handler.on_cancel_failed = self._on_cancel_failed
        self.speech_handler.on_error = self._show_error_message
        self.speech_handler.on_species_detected = self._on_species_detected_internal

    def _setup_modern_theme(self) -> None:
        """Apply modern dark/light theme to the entire application"""
        self.setStyleSheet("""
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
            QTabWidget::pane {
                border: 2px solid rgba(255, 255, 255, 0.15);
                border-radius: 12px;
                background: rgba(0, 0, 0, 0.35);
                top: -2px;
            }
            QTabBar::tab {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(0, 0, 0, 0.4),
                    stop:1 rgba(0, 0, 0, 0.5));
                border: 2px solid rgba(255, 255, 255, 0.2);
                border-bottom: none;
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
                padding: 18px 32px;
                margin-right: 6px;
                color: rgba(255, 255, 255, 0.9);
                font-size: 16px;
                font-weight: 600;
                min-width: 160px;
            }
            QTabBar::tab:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 255, 255, 0.2),
                    stop:1 rgba(255, 255, 255, 0.15));
                border: 2px solid rgba(255, 255, 255, 0.35);
                color: white;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(100, 180, 255, 0.5),
                    stop:1 rgba(80, 150, 255, 0.4));
                border: 2px solid rgba(100, 200, 255, 0.6);
                color: white;
                font-weight: 700;
                padding: 20px 36px;
            }
            QTabBar::tab:!selected {
                margin-top: 4px;
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

        # Build main logging tab content
        logging_tab = self._create_main_tab()
        self.logging_tab = logging_tab

        # Reports tab
        self.report_widget = ReportWidget()

        # Settings tab
        self.settings_widget = SettingsWidget()

        # Tabs container
        self.tabs = QTabWidget()
        self.tabs.addTab(logging_tab, "🎣 Fish Logging")
        self.tabs.addTab(self.report_widget, "📊 Reports")
        self.tabs.addTab(self.settings_widget, "⚙️ Settings")
        # Start/stop recognizer when switching tabs
        self.tabs.currentChanged.connect(self._on_tab_changed)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.addWidget(self.tabs)
        central.setLayout(main_layout)

        self.statusBar().showMessage("🎧 Ready to capture your fishing stories...")

        # Initialize recognizer's last species from current selector value
        try:
            init_species = getattr(self, 'species_selector', None)
            if init_species is not None:
                cur = self.species_selector.currentSpecies()
                if cur and hasattr(self.speech_recognizer, 'set_last_species'):
                    self.speech_recognizer.set_last_species(cur)
        except Exception:
            pass

        # Button signals belong to logging tab controls
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_start.clicked.connect(self._on_start)

        # React to noise profile changes from Settings
        try:
            self.settings_widget.noiseProfileChanged.connect(self._on_noise_profile_changed)
        except Exception:
            pass

    def _on_noise_profile_changed(self, profile_key: str) -> None:
        """Apply noise profile by stopping recognizer, updating, and restarting if on logging tab."""
        try:
            is_logging_tab = (self.tabs.currentWidget() is self.logging_tab)
        except Exception:
            is_logging_tab = True
        try:
            # Stop to rebuild noise controller safely
            self._on_stop()
            if hasattr(self.speech_recognizer, 'set_noise_profile'):
                self.speech_recognizer.set_noise_profile(profile_key)
            self.statusBar().showMessage(f"🔊 Noise profile set to {profile_key}")
        finally:
            # Restart only if user is on logging tab
            if is_logging_tab:
                self._on_start()

    def _on_tab_changed(self, index: int) -> None:
        """Pause recognizer on non-logging tabs; render default chart in Reports. When returning, sync current species."""
        try:
            widget = self.tabs.widget(index)
            if widget is self.logging_tab:
                # Sync current species to recognizer to avoid "None" on numbers-only utterances
                try:
                    cur = self.species_selector.currentSpecies()
                    if cur and hasattr(self.speech_recognizer, 'set_last_species'):
                        self.speech_recognizer.set_last_species(cur)
                except Exception:
                    pass
                # Returning to Fish Logging -> start listening
                self._on_start()
            else:
                # Entering non-logging tabs -> stop listening
                self._on_stop()
                if widget is self.report_widget:
                    # Render a default chart once
                    if hasattr(self.report_widget, "show_default_chart"):
                        self.report_widget.show_default_chart()
        except Exception as e:
            logger.error(f"Tab change handler failed: {e}")

    def _create_main_tab(self) -> QWidget:
        """Create and return the main logging tab UI as a QWidget."""
        main_tab = QWidget()

        # Create UI panels using extracted methods
        transcription_panel = self._create_transcription_panel()
        table_panel = self._create_table_panel()

        # Main splitter for the logging tab
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(transcription_panel)
        splitter.addWidget(table_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([150, 550])

        main_tab_layout = QVBoxLayout()
        main_tab_layout.setContentsMargins(20, 20, 20, 20)
        main_tab_layout.addWidget(splitter)
        main_tab.setLayout(main_tab_layout)

        return main_tab

    def _create_transcription_panel(self) -> QWidget:
        """Create the live transcription panel."""
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

        return transcription_panel

    def _create_table_panel(self) -> QWidget:
        """Create the logged entries table panel."""
        table_panel = GlassPanel()
        table_layout = QVBoxLayout(table_panel)
        table_layout.setSpacing(16)
        table_layout.setContentsMargins(24, 20, 24, 20)

        # Add header with species selector
        header_row = self._create_table_header()
        table_layout.addLayout(header_row)

        # Create and configure table
        self._create_and_configure_table()
        table_layout.addWidget(self.table)

        # Add control panel
        control_panel = self._create_control_panel()
        table_layout.addWidget(control_panel)

        return table_panel

    def _create_table_header(self) -> QHBoxLayout:
        """Create the table header with species selector."""
        header_row = QHBoxLayout()
        table_header = ModernLabel("📊 Logged Entries", "header")

        self.current_specie_label = ModernLabel("🐟 Current:", style="subheader")

        # Replace label with SpeciesSelector (searchable dropdown with numbering + codes)
        self.species_selector = SpeciesSelector()
        self.species_selector.setMinimumWidth(240)

        header_row.addWidget(table_header)
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(6)
        header_row.addWidget(self.current_specie_label)
        header_row.addWidget(self.species_selector)
        header_row.addStretch()

        return header_row

    def _create_and_configure_table(self) -> None:
        """Create and configure the main logging table."""
        # Update table to show only visible columns (Date, Time, Species, Length, Trash)
        self.table = ModernTable(0, 5)
        self.table.setHorizontalHeaderLabels(["📅 Date", "⏰ Time", "🐟 Species", "📏 Length (cm)", "🗑"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSortingEnabled(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)

        header = self.table.horizontalHeader()
        try:
            # Make Date and Time columns a bit narrower
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
            header.resizeSection(0, 200)
            header.resizeSection(1, 200)
            # Fixed width for trash column
            header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        except Exception as e:
            logger.error(str(e))
        header.resizeSection(4, 44)

    def _create_control_panel(self) -> QWidget:
        """Create the control panel with start/stop buttons and status indicator."""
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
        status_container = self._create_status_indicator()
        control_layout.addWidget(status_container)
        control_layout.addStretch(1)

        return control_panel

    def _create_status_indicator(self) -> QWidget:
        """Create the status indicator widget."""
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

        return status_container

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

        if hasattr(self.speech_recognizer, "specie_detected"):
            try:
                self.speech_recognizer.specie_detected.disconnect(self._on_specie_detected)
            except Exception:
                pass
            self.speech_recognizer.specie_detected.connect(self._on_specie_detected)

        # When user changes selection manually, update status bar and recognizer
        try:
            self.species_selector.speciesChanged.connect(self._on_species_selected)
        except Exception:
            pass

    def _on_specie_detected(self, specie: str) -> None:
        # Sync detected species with selector (best-effort)
        try:
            self.species_selector.setCurrentByName(specie)
        except Exception:
            pass

    def _on_species_selected(self, name: str) -> None:
        # Update recognizer last species and status
        try:
            if hasattr(self.speech_recognizer, 'set_last_species'):
                self.speech_recognizer.set_last_species(name)
        except Exception:
            pass
        self.statusBar().showMessage(f"Current species: {name}")

    def _on_partial_text(self, text: str) -> None:
        # Delegate to speech handler
        self.speech_handler.handle_partial_text(text)

    def _on_final_text(self, text: str, confidence: float) -> None:
        """Handle final text from speech recognition - delegated to handler."""
        # Use SettingsProvider to get boat and station data (no Law of Demeter violation)
        boat_name, station_id = self.settings_provider.get_and_save_all()

        # Delegate to speech handler with context
        self.speech_handler.handle_final_text(text, confidence, boat_name, station_id)

    def _prepend_table_row(self, species: str, length_cm: float, confidence: float, boat: str, station: str) -> None:
        # Show only Date, Time, Species, Length in table (Boat/Station/Confidence hidden, but logged)
        from datetime import datetime
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        self.table.insertRow(0)
        # Create centered items
        it_date = QTableWidgetItem(date_str)
        it_time = QTableWidgetItem(time_str)
        it_species = QTableWidgetItem(species)
        it_length = QTableWidgetItem(f"{length_cm:.1f}")
        for it in (it_date, it_time, it_species, it_length):
            try:
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            except Exception:
                pass
        self.table.setItem(0, 0, it_date)
        self.table.setItem(0, 1, it_time)
        self.table.setItem(0, 2, it_species)
        self.table.setItem(0, 3, it_length)
        self.table.setRowHeight(0, 46)

        # Trash button using standard icon; bind robust row deletion handler
        btn_delete = QPushButton()
        btn_delete.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon))
        btn_delete.setIconSize(QSize(18, 18))
        btn_delete.setFixedSize(28, 28)
        btn_delete.setStyleSheet("""
            QPushButton { border: none; background: transparent; }
            QPushButton:hover { color: red; }
        """)
        btn_delete.setFlat(True)
        btn_delete.clicked.connect(self._on_delete_clicked)
        self.table.setCellWidget(0, 4, btn_delete)

    def _on_delete_clicked(self) -> None:
        """Delete the row that contains the clicked trash button."""
        try:
            sender = self.sender()
            if sender is None:
                return
            last_col = 4
            for r in range(self.table.rowCount()):
                if self.table.cellWidget(r, last_col) is sender:
                    self._remove_table_row(r)
                    break
        except Exception as e:
            logger.error(f"Delete click failed: {e}")

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
        """Handle error from speech recognizer."""
        self._alert(message)
        self.status_presenter.show_error(message)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _on_status(self, message: str) -> None:
        """Handle status change from speech recognizer - delegate to StatusPresenter."""
        self.status_presenter.show_status(message)

    def _on_stop(self) -> None:
        """Stop speech recognition."""
        self.recognizer_controller.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_presenter.show_status("stopped")

    def _on_start(self) -> None:
        """Start speech recognition."""
        success = self.recognizer_controller.start()
        if success:
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.status_presenter.show_status("listening")
        else:
            self._alert("Failed to start listening")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """Handle window close event."""
        self.recognizer_controller.stop()
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

    def _update_live_text(self, text: str) -> None:
        """Update the live transcription text area."""
        self.live_text.setPlainText(text)
        self.live_text.verticalScrollBar().setValue(self.live_text.verticalScrollBar().maximum())

    def _on_entry_logged_success(self, entry) -> None:
        """Handle successful logging of a fish entry."""
        # Update species selector if species was detected
        try:
            self.species_selector.setCurrentByName(entry.species)
        except Exception:
            pass

        # Add to table using table manager
        self.table_manager.add_entry(
            entry.species,
            entry.length_cm,
            entry.confidence,
            entry.boat,
            entry.station_id,
            delete_callback=self._on_delete_clicked
        )
        self.statusBar().showMessage("🎣 Great catch logged successfully!", 2000)

    def _on_entry_cancelled_success(self) -> None:
        """Handle successful cancellation of the last entry."""
        self.table_manager.remove_last_row()
        self.statusBar().showMessage("✅ Last entry cancelled successfully", 3000)

    def _on_cancel_failed(self) -> None:
        """Handle failure to cancel the last entry."""
        self.statusBar().showMessage("ℹ️ Nothing to cancel", 3000)

    def _show_error_message(self, message: str) -> None:
        """Show an error message to the user."""
        self._alert(message)
        self.statusBar().showMessage("❌ Error: " + message)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_panel.set_status_color("#f44336", "#d32f2f")

    def _on_species_detected_internal(self, species: str) -> None:
        """Handle species detection from speech recognizer."""
        try:
            self.species_selector.setCurrentByName(species)
        except Exception:
            pass
