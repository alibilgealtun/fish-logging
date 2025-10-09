from __future__ import annotations

from pathlib import Path
from typing import Optional
import json

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFileDialog,
    QMessageBox,
    QInputDialog,
    QComboBox,
)
from PyQt6.QtCore import QObject, pyqtSignal, QThread, Qt, QTimer
from PyQt6.QtGui import QCursor

# Optional logger: prefer loguru if available, else fallback to stdlib logging
try:  # pragma: no cover
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("settings_widget")

from .GlassPanel import GlassPanel
from .ModernLabel import ModernLabel
from .AnimatedButton import AnimatedButton
from services.google_sheets_backup import GoogleSheetsBackup, GoogleSheetsConfig


class _BackupWorker(QObject):
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, svc: GoogleSheetsBackup, excel_path: Path, cfg: Optional[GoogleSheetsConfig]):
        super().__init__()
        self._svc = svc
        self._excel_path = excel_path
        self._cfg = cfg

    def run(self):
        try:
            result = self._svc.backup_excel_to_sheet(self._excel_path, self._cfg)
            self.finished.emit(result)
        except Exception as e:
            self.failed.emit(str(e))


class SettingsWidget(QWidget):
    noiseProfileChanged = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._svc = GoogleSheetsBackup()
        self._cfg: Optional[GoogleSheetsConfig] = self._svc.load_config()
        self._thread: Optional[QThread] = None
        self._worker: Optional[_BackupWorker] = None
        self._user_settings_path = Path("config/user_settings.json")
        # Suppress auto-emission of noise profile change during initial widget setup
        self._suppress_profile_signal: bool = True
        self._build_ui()
        self._refresh_config_labels()
        self._load_saved_noise_profile()
        # Re-enable emission after event loop starts to ensure recognizer is fully initialized
        QTimer.singleShot(0, self._enable_profile_signal)
        # Ensure background QThread is stopped if app quits due to OS signals
        try:
            from PyQt6.QtWidgets import QApplication
            app = QApplication.instance()
            if app is not None:
                app.aboutToQuit.connect(self._on_app_quit)
        except Exception:
            pass

    def _enable_profile_signal(self) -> None:
        # Allow future user-driven noise profile changes to emit
        self._suppress_profile_signal = False

    def closeEvent(self, event):  # type: ignore[override]
        """Ensure the background backup thread is stopped before widget is destroyed."""
        try:
            if self._thread is not None and self._thread.isRunning():
                try:
                    # Ask the thread's event loop to exit (worker may finish soon)
                    self._thread.quit()
                except Exception:
                    pass
                try:
                    # Wait a bit for graceful shutdown
                    if not self._thread.wait(3000):
                        # Last resort: force termination to prevent Qt abort
                        logger.warning("Settings backup thread still running on close; forcing terminate()")
                        try:
                            self._thread.terminate()
                        except Exception:
                            pass
                        try:
                            self._thread.wait(1000)
                        except Exception:
                            pass
                except Exception:
                    pass
        finally:
            try:
                if self._worker is not None:
                    self._worker.deleteLater()
            except Exception:
                pass
            try:
                if self._thread is not None:
                    self._thread.deleteLater()
            except Exception:
                pass
            self._worker = None
            self._thread = None
        return super().closeEvent(event)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        header = ModernLabel("⚙️ Settings", style="header")
        layout.addWidget(header)

        # NEW: Configs panel (Station ID & Boat Name)
        configs_panel = GlassPanel()
        configs_layout = QVBoxLayout(configs_panel)
        configs_layout.setContentsMargins(20, 16, 20, 16)
        configs_layout.setSpacing(12)

        configs_title = ModernLabel("🔧 Configs", style="subheader")
        configs_layout.addWidget(configs_title)

        # Import the widgets
        from gui.widgets import StationIdInput, BoatNameInput

        # Station ID and Boat Name in the same row
        configs_row = QHBoxLayout()
        configs_row.setSpacing(12)
        self.station_input = StationIdInput()
        self.boat_input = BoatNameInput()
        configs_row.addWidget(self.station_input)
        configs_row.addWidget(self.boat_input)
        configs_row.addStretch(1)
        configs_layout.addLayout(configs_row)

        # Noise profile selector row
        profile_row = QHBoxLayout()
        profile_label = ModernLabel("Noise Profile:", style="subheader")
        self.noise_profile_combo = QComboBox()
        self.noise_profile_combo.addItem("Mixed (human + engine)", "mixed")
        self.noise_profile_combo.addItem("Human Voices", "human")
        self.noise_profile_combo.addItem("Engine Noise", "engine")
        self.noise_profile_combo.addItem("Clean / Quiet", "clean")
        self.noise_profile_combo.setCurrentIndex(0)
        self.noise_profile_combo.setToolTip("Select acoustic environment to adapt VAD & suppression")
        self.noise_profile_combo.currentIndexChanged.connect(self._on_noise_profile_changed)
        profile_row.addWidget(profile_label)
        profile_row.addWidget(self.noise_profile_combo)
        profile_row.addStretch(1)
        configs_layout.addLayout(profile_row)

        layout.addWidget(configs_panel)

        # Config panel (Google Sheets)
        cfg_panel = GlassPanel()
        cfg_layout = QVBoxLayout(cfg_panel)
        cfg_layout.setContentsMargins(20, 16, 20, 16)
        cfg_layout.setSpacing(8)

        cfg_title = ModernLabel("Backup - Google Sheets Configuration", style="subheader")
        cfg_layout.addWidget(cfg_title)

        self.lbl_credentials = QLabel()
        self.lbl_spreadsheet = QLabel()
        self.lbl_worksheet = QLabel()

        for lbl in (self.lbl_credentials, self.lbl_spreadsheet, self.lbl_worksheet):
            lbl.setStyleSheet(
                "QLabel { color:#1f2937; background: rgba(255,255,255,0.8); border:1px solid #e5e7eb;"
                " padding:8px; border-radius:8px; }"
            )
        cfg_layout.addWidget(self.lbl_credentials)
        cfg_layout.addWidget(self.lbl_spreadsheet)
        cfg_layout.addWidget(self.lbl_worksheet)

        # Actions
        actions_row = QHBoxLayout()
        self.btn_backup = AnimatedButton("🔄 Backup Now", variant="primary")
        self.btn_change_api = AnimatedButton("🔧 Change API…", variant="info")
        self.btn_test = AnimatedButton("🧪 Test Connection", variant="success")
        self.btn_help = AnimatedButton("📘 How to do it", variant="warning")
        self.btn_backup.clicked.connect(self._on_backup)
        self.btn_change_api.clicked.connect(self._on_change_api)
        self.btn_test.clicked.connect(self._on_test_connection)
        self.btn_help.clicked.connect(self._on_show_howto)
        actions_row.addWidget(self.btn_backup)
        actions_row.addWidget(self.btn_change_api)
        actions_row.addWidget(self.btn_test)
        actions_row.addWidget(self.btn_help)
        actions_row.addStretch(1)
        cfg_layout.addLayout(actions_row)

        # Status label
        self.lbl_status = QLabel("Ready.")
        self.lbl_status.setStyleSheet(
            "QLabel { color:#065f46; background: white; border:1px solid rgba(5,150,105,0.3);"
            " padding:8px; border-radius:8px; }"
        )
        cfg_layout.addWidget(self.lbl_status)

        layout.addWidget(cfg_panel)
        layout.addStretch(1)

    def _refresh_config_labels(self) -> None:
        if self._cfg is None:
            self.lbl_credentials.setText("Credentials: Not configured")
            self.lbl_spreadsheet.setText("Spreadsheet ID: Not configured")
            self.lbl_worksheet.setText("Worksheet: Not configured")
        else:
            self.lbl_credentials.setText(f"Credentials: {Path(self._cfg.credentials_path).resolve()}")
            self.lbl_spreadsheet.setText(f"Spreadsheet ID: {self._cfg.spreadsheet_id}")
            self.lbl_worksheet.setText(f"Worksheet: {self._cfg.worksheet_name}")

    def _set_busy(self, busy: bool) -> None:
        for btn in (self.btn_backup, self.btn_change_api, self.btn_test, self.btn_help):
            btn.setDisabled(busy)
        if busy:
            self.lbl_status.setText("Working… Please wait.")
            self.setCursor(QCursor(Qt.CursorShape.BusyCursor))
        else:
            self.unsetCursor()

    def _on_backup(self) -> None:
        try:
            excel_path = Path("logs/hauls/logs.xlsx")
            if not excel_path.exists():
                QMessageBox.warning(self, "Missing Logs", f"Excel file not found: {excel_path}")
                return
            # Async run via QThread
            self._set_busy(True)
            self._thread = QThread(self)
            self._worker = _BackupWorker(self._svc, excel_path, self._cfg)
            self._worker.moveToThread(self._thread)
            self._thread.started.connect(self._worker.run)
            self._worker.finished.connect(self._on_backup_finished)
            self._worker.failed.connect(self._on_backup_failed)
            # Ensure cleanup
            self._worker.finished.connect(self._thread.quit)
            self._worker.failed.connect(self._thread.quit)
            self._thread.finished.connect(self._cleanup_thread)
            self._thread.start()
        except Exception as e:
            logger.exception("Backup failed to start: {}", e)
            self._set_busy(False)
            QMessageBox.critical(self, "Backup Failed", str(e))

    def _cleanup_thread(self) -> None:
        # avoid lingering references
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None
        self._set_busy(False)

    def _on_backup_finished(self, result: dict) -> None:
        self.lbl_status.setText(
            f"Backup complete: {result.get('rows', 0)} rows to {result.get('spreadsheet_id','?')}/{result.get('worksheet_name','?')}"
        )
        QMessageBox.information(
            self,
            "Backup Complete",
            (
                f"Successfully backed up {result.get('rows', 0)} rows to\n"
                f"Sheet: {result.get('spreadsheet_id', '?')} / {result.get('worksheet_name', '?')}"
            ),
        )

    def _on_backup_failed(self, message: str) -> None:
        self.lbl_status.setText(f"Backup failed: {message}")
        QMessageBox.critical(self, "Backup Failed", message)

    def _on_change_api(self) -> None:
        # 1) pick credentials json
        cred_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Service Account JSON",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not cred_path:
            return

        # 2) Spreadsheet ID
        spreadsheet_id, ok = QInputDialog.getText(
            self,
            "Google Spreadsheet ID",
            "Enter the Spreadsheet ID (from the URL):",
            text=(self._cfg.spreadsheet_id if self._cfg else ""),
        )
        if not ok or not spreadsheet_id.strip():
            return

        # 3) Worksheet name
        worksheet_name, ok = QInputDialog.getText(
            self,
            "Worksheet Name",
            "Enter the target worksheet name:",
            text=(self._cfg.worksheet_name if self._cfg else "Logs"),
        )
        if not ok or not worksheet_name.strip():
            return

        try:
            new_cfg = GoogleSheetsConfig(
                credentials_path=Path(cred_path),
                spreadsheet_id=spreadsheet_id.strip(),
                worksheet_name=worksheet_name.strip(),
            )
            self._svc.save_config(new_cfg)
            self._cfg = new_cfg
            self._refresh_config_labels()
            self.lbl_status.setText("Settings saved. You can now test the connection.")
            QMessageBox.information(self, "Settings Saved", "Google Sheets API settings updated.")
        except Exception as e:
            logger.exception("Failed to save API config: {}", e)
            QMessageBox.critical(self, "Save Failed", str(e))

    def _on_test_connection(self) -> None:
        try:
            result = self._svc.test_connection(self._cfg)
            if result.get("ok"):
                worksheet = (
                    "exists" if result.get("worksheet_exists") else "will be created on first backup"
                )
                self.lbl_status.setText("Connection OK. " + ("Worksheet exists." if result.get("worksheet_exists") else "Worksheet will be created."))
                QMessageBox.information(
                    self,
                    "Connection OK",
                    (
                        f"Access confirmed for spreadsheet: {result.get('spreadsheet_id')}\n"
                        f"Worksheet: {worksheet}"
                    ),
                )
            else:
                detail = result.get("detail", "Unknown error")
                self.lbl_status.setText(f"Connection failed: {detail}")
                QMessageBox.warning(
                    self,
                    "Connection Failed",
                    f"Could not access Google Sheets.\n{detail}",
                )
        except Exception as e:
            logger.exception("Test connection failed: {}", e)
            QMessageBox.critical(self, "Test Failed", str(e))

    def _on_show_howto(self) -> None:
        steps = (
            "1) Create a Google Cloud project (console.cloud.google.com).\n"
            "2) Enable APIs: Google Sheets API and Google Drive API.\n"
            "3) Create a Service Account (IAM & Admin -> Service Accounts).\n"
            "   - Grant basic role 'Editor' (or minimum needed).\n"
            "   - Create a JSON key and download it to your computer.\n"
            "4) Create a Google Spreadsheet in Drive.\n"
            "   - Open the sheet and copy the Spreadsheet ID from the URL.\n"
            "     (The long string after /spreadsheets/d/ and before /edit).\n"
            "5) Share the spreadsheet with the service account email.\n"
            "   - The email is in your JSON file under 'client_email'.\n"
            "   - Give it at least 'Editor' access.\n"
            "6) In this app, open Settings -> 'Change API…'.\n"
            "   - Select the JSON credentials file you downloaded.\n"
            "   - Paste the Spreadsheet ID.\n"
            "   - Enter (or create) the Worksheet name (e.g., 'Logs').\n"
            "7) Click 'Test Connection' to verify access and the sheet.\n"
            "8) Click 'Backup Now' to push logs/hauls/logs.xlsx to Google Sheets.\n\n"
            "Notes:\n"
            "- Your config is saved to config/google_sheets.json.\n"
            "- We auto-create the worksheet if it doesn't exist.\n"
            "- Dates/times are formatted for Google Sheets; numbers stay numeric.\n"
            "- If backup fails, ensure the sheet is shared with the service account.\n"
        )
        QMessageBox.information(self, "How to set up Google Sheets Backup", steps)

    def _load_saved_noise_profile(self) -> None:
        """Load and apply last saved noise profile from user_settings.json."""
        try:
            if self._user_settings_path.exists():
                with open(self._user_settings_path, 'r') as f:
                    settings = json.load(f)
                saved_profile = settings.get("noise_profile", "mixed")
                # Find and select the matching item WITHOUT emitting (suppressed by flag)
                for i in range(self.noise_profile_combo.count()):
                    if self.noise_profile_combo.itemData(i) == saved_profile:
                        if i != self.noise_profile_combo.currentIndex():
                            self.noise_profile_combo.setCurrentIndex(i)
                        break
        except Exception as e:
            logger.debug(f"Could not load saved noise profile: {e}")

    def _save_noise_profile(self, profile_key: str) -> None:
        """Persist the selected noise profile to user_settings.json."""
        try:
            settings = {}
            if self._user_settings_path.exists():
                with open(self._user_settings_path, 'r') as f:
                    settings = json.load(f)
            settings["noise_profile"] = profile_key
            self._user_settings_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._user_settings_path, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save noise profile: {e}")

    def _on_noise_profile_changed(self, idx: int) -> None:
        try:
            if self._suppress_profile_signal:
                return  # Ignore programmatic changes during initialization
            key = self.noise_profile_combo.currentData()
            if isinstance(key, str):
                self._save_noise_profile(key)
                self.noiseProfileChanged.emit(key)
                logger.info(f"Noise profile changed to {key}")
        except Exception as e:
            logger.error(f"Failed to emit noise profile change: {e}")

    def _on_app_quit(self) -> None:
        """App-wide quit hook: stop and clean up the backup thread if running."""
        try:
            th = getattr(self, "_thread", None)
            if th is not None and th.isRunning():
                try:
                    th.quit()
                except Exception:
                    pass
                try:
                    if not th.wait(2000):
                        try:
                            th.terminate()
                        except Exception:
                            pass
                        try:
                            th.wait(800)
                        except Exception:
                            pass
                except Exception:
                    pass
        finally:
            try:
                if getattr(self, "_worker", None) is not None:
                    self._worker.deleteLater()
            except Exception:
                pass
            try:
                if getattr(self, "_thread", None) is not None:
                    self._thread.deleteLater()
            except Exception:
                pass
            self._worker = None
            self._thread = None

