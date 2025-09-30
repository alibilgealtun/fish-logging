"""
ReportWidget: Qt UI to generate length distribution reports (graphs + raw data)
using logs/hauls/logs.xlsx. Fits existing GUI structure and SOLID-aligned
reports/length_distribution_report.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer, QObject, pyqtSlot
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QTextEdit,
    QComboBox,
    QSplitter,
    QGraphicsDropShadowEffect,
    QScrollArea,
    QSizePolicy,
    QCheckBox,
)

# Logger: try loguru, fallback to stdlib logging
try:  # pragma: no cover
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("report_widget")

import sys




class ReportGenerationWorker(QThread):
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    finished_success = pyqtSignal(dict)
    finished_error = pyqtSignal(str)

    def __init__(self, data_path: Path, output_dir: Path, formats: List[str]):
        super().__init__()
        self.data_path = data_path
        self.output_dir = output_dir
        self.formats = formats

    def run(self) -> None:  # type: ignore[override]
        try:
            self.status_updated.emit("Initializing report generator…")
            self.progress_updated.emit(10)

            # Lazy import to avoid import-time failures breaking GUI startup
            from reports.length_distribution_report import LengthDistributionReportGenerator

            logger.info("[Report] Creating generator with data_path={}", self.data_path)
            generator = LengthDistributionReportGenerator(self.data_path)

            self.status_updated.emit("Loading and analyzing data…")
            self.progress_updated.emit(35)

            report_data = generator.generate_report(self.output_dir, self.formats)

            self.status_updated.emit("Finalizing…")
            self.progress_updated.emit(90)

            logger.info("[Report] Generation succeeded. Exported files: {}", list(report_data.get("exported_files", {}).values()))
            self.finished_success.emit(report_data)
            self.progress_updated.emit(100)
        except Exception as e:
            logger.exception("[Report] Generation failed: {}", e)
            self.finished_error.emit(str(e))

class ChartRenderWorker(QThread):
    finished_success = pyqtSignal(bytes)
    finished_error = pyqtSignal(str)

    def __init__(self, df, chart_key: str, species: str | None, width: int, height: int):
        super().__init__()
        self.df = df
        self.chart_key = chart_key
        self.species = species
        self.width = width
        self.height = height

    def run(self) -> None:  # type: ignore[override]
        try:
            from reports.length_distribution_report import LengthDistributionReportGenerator
            generator = LengthDistributionReportGenerator()  # data_path not used for chart creation
            if self.chart_key == "species_length_distribution":
                fig = generator.create_species_length_chart(self.df, self.species or "")
            else:
                fig = generator.create_plotly_chart(self.df, self.chart_key)
            png_bytes = fig.to_image(format="png", width=int(self.width), height=int(self.height), scale=1)
            self.finished_success.emit(png_bytes)
        except Exception as e:
            self.finished_error.emit(str(e))

class ReportWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker: ReportGenerationWorker | None = None
        self._generator = None
        self._df = None
        self._placeholder: QLabel | None = None
        self._chart_frame: QWidget | None = None
        self._content_widget: QWidget | None = None  # holds current chart widget (image or web)
        self._webview = None  # type: ignore
        self._web_channel = None  # type: ignore
        # Track real paths independent of display labels
        self._data_path: Path | None = None
        self._out_dir: Path | None = None
        # Debounced resize re-render timer (used for image mode)
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(180)  # ms
        self._resize_timer.timeout.connect(lambda: self._render_plotly_chart(self.chart_combo.currentData() or "species_pie"))
        # Render cache: (chart_key, species, width, height) -> PNG bytes
        self._render_worker: ChartRenderWorker | None = None
        self._render_task_id: int = 0
        self._render_cache: dict[tuple[str, str, int, int], bytes] = {}
        self._cache_quantum: int = 64  # px step to coalesce sizes
        # Track last rendered state
        self._last_key: str | None = None
        self._last_species: str | None = None
        self._last_width_q: int | None = None
        self._last_height: int | None = None
        # Debounce species changes
        self._species_timer = QTimer(self)
        self._species_timer.setSingleShot(True)
        self._species_timer.setInterval(250)
        self._species_timer.timeout.connect(self._render_species_if_needed)
        self._build_ui()

    class _ChartBridge(QObject):  # bridge for JS -> Python events
        js_event = pyqtSignal(dict)

        @pyqtSlot(str)
        def onJsEvent(self, message: str) -> None:
            import json
            try:
                data = json.loads(message)
                self.js_event.emit(data)
            except Exception:
                pass

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 0, 8, 8)
        main_layout.setSpacing(2)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # LEFT PANEL - Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 2, 8, 8)

        # Config group
        config_group = QGroupBox("Report Configuration")
        cfg = QGridLayout(config_group)

        # Data source selector
        data_src_label = QLabel("Data Source:")
        data_src_label.setStyleSheet("color:#fff;font-size:12px;font-weight:600;")
        cfg.addWidget(data_src_label, 0, 0)
        self.data_label = QLabel()
        self.data_label.setToolTip("Excel file that contains the haul logs")
        self.data_label.setStyleSheet(
            "background:#f7f7f7;border:1px solid #ddd;padding:6px;border-radius:6px;color:#000;font-size:11px;"
        )
        self.data_label.setWordWrap(True)
        self._update_data_label()
        cfg.addWidget(self.data_label, 0, 1)
        btn_browse_data = QPushButton("Browse…")
        btn_browse_data.setStyleSheet("QPushButton{background:#60a5fa;color:white;border:none;padding:6px 10px;border-radius:6px;}QPushButton:hover{background:#3b82f6;}")
        btn_browse_data.clicked.connect(self._browse_data)
        cfg.addWidget(btn_browse_data, 0, 2)

        # Output directory selector
        out_dir_label = QLabel("Output Directory:")
        out_dir_label.setStyleSheet("color:#fff;font-size:12px;font-weight:600;")
        cfg.addWidget(out_dir_label, 1, 0)
        self.out_label = QLabel()
        self.out_label.setToolTip("Where reports will be saved")
        self.out_label.setStyleSheet(
            "background:#f7f7f7;border:1px solid #ddd;padding:6px;border-radius:6px;color:#000;font-size:11px;"
        )
        self.out_label.setWordWrap(True)
        self._update_output_label()
        cfg.addWidget(self.out_label, 1, 1)
        btn_browse_out = QPushButton("Browse…")
        btn_browse_out.setStyleSheet("QPushButton{background:#34d399;color:#0b1f16;border:none;padding:6px 10px;border-radius:6px;}QPushButton:hover{background:#10b981;color:#052e1c;}")
        btn_browse_out.clicked.connect(self._browse_out)
        cfg.addWidget(btn_browse_out, 1, 2)

        left_layout.addWidget(config_group)

        # Chart selection group
        chart_group = QGroupBox("Chart Selection")
        chart_layout = QVBoxLayout(chart_group)

        chart_layout.addWidget(QLabel("Chart Type:"))
        self.chart_combo = QComboBox()
        # Load chart types lazily from generator module
        try:
            from reports.length_distribution_report import CHART_TYPES
            self._chart_keys = list(CHART_TYPES.keys())
            for key in self._chart_keys:
                self.chart_combo.addItem(CHART_TYPES[key], userData=key)
        except Exception as e:
            logger.error("[Report] Failed to load chart types: {}", e)
            self._chart_keys = [
                "species_pie",
                "species_avg_length_bar",
                "species_count_bar",
                "species_length_distribution",
            ]
            for key in self._chart_keys:
                self.chart_combo.addItem(key, userData=key)

        chart_layout.addWidget(self.chart_combo)

        # Open interactive in browser
        self.btn_open_interactive = QPushButton("Open interactive in browser")
        self.btn_open_interactive.setStyleSheet("QPushButton{background:#60a5fa;color:white;border:none;padding:6px 10px;border-radius:6px;}QPushButton:hover{background:#3b82f6;}")
        self.btn_open_interactive.clicked.connect(self._open_interactive)
        chart_layout.addWidget(self.btn_open_interactive)

        # Species selector (used when 'Length Distribution (Select Species)' is chosen)
        self.species_label = QLabel("Species:")
        chart_layout.addWidget(self.species_label)
        self.species_combo = QComboBox()
        self.species_combo.setEnabled(False)
        self.species_combo.currentIndexChanged.connect(self._on_species_combo_changed)
        chart_layout.addWidget(self.species_combo)
        # Hide species selector by default; will be toggled based on chart type
        self.species_label.setVisible(False)
        self.species_combo.setVisible(False)

        left_layout.addWidget(chart_group)

        # Progress group
        progress_group = QGroupBox("Generation Progress")
        pg = QVBoxLayout(progress_group)
        self.status = QLabel("Ready")
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        self.bar.setRange(0, 100)
        pg.addWidget(self.status)
        pg.addWidget(self.bar)
        left_layout.addWidget(progress_group)

        # Generate report button
        self.btn_generate = QPushButton("Generate Report")
        self.btn_generate.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
                margin: 10px 0;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.btn_generate.clicked.connect(self._on_generate)
        left_layout.addWidget(self.btn_generate)

        # Results
        results_group = QGroupBox("Results")
        rg = QVBoxLayout(results_group)
        self.results = QTextEdit()
        self.results.setReadOnly(True)
        self.results.setVisible(False)
        self.results.setMaximumHeight(150)
        rg.addWidget(self.results)
        left_layout.addWidget(results_group)

        left_layout.addStretch(1)

        # RIGHT PANEL - Visualization (only main chart canvas, no details section)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 0, 8, 8)

        self._chart_frame = QWidget()
        self._chart_frame.setStyleSheet(
            "background:#fff;border:1px solid #dcdcdc;border-radius:10px;"
        )
        shadow = QGraphicsDropShadowEffect(self._chart_frame)
        shadow.setBlurRadius(12)
        shadow.setOffset(0, 1)
        shadow.setColor(Qt.GlobalColor.lightGray)
        self._chart_frame.setGraphicsEffect(shadow)

        frame_layout = QVBoxLayout(self._chart_frame)
        frame_layout.setContentsMargins(8, 4, 8, 8)
        frame_layout.setSpacing(6)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        frame_layout.addWidget(self._scroll)

        self._holder = QWidget()
        self._holder_layout = QVBoxLayout(self._holder)
        self._holder_layout.setContentsMargins(0, 0, 0, 0)
        self._holder_layout.setSpacing(8)
        self._scroll.setWidget(self._holder)

        self._canvas_widget = QWidget()
        self._canvas_widget.setMinimumHeight(400)
        self._canvas_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas_container = QVBoxLayout(self._canvas_widget)
        self.canvas_container.setContentsMargins(0, 0, 0, 0)
        self.canvas_container.setSpacing(0)
        self._holder_layout.addWidget(self._canvas_widget)

        self._placeholder = QLabel("Select a chart type to display visualization")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(
            "QLabel { color: #666; font-size: 12px; border: 2px dashed #ccc; padding: 12px;"
            "border-radius: 8px; background-color: #f9f9f9; }"
        )
        self.canvas_container.addWidget(self._placeholder)

        right_layout.addWidget(self._chart_frame)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 700])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)

        main_layout.addWidget(splitter)

        self.chart_combo.currentIndexChanged.connect(self._on_chart_index_changed)
        # Ensure correct initial visibility for species selector
        self._update_species_selector_visibility()

        logger.info("Python executable: {}", sys.executable)
        QTimer.singleShot(0, self.show_default_chart)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        try:
            if self._resize_timer.isActive():
                self._resize_timer.stop()
            self._resize_timer.start()
        except Exception:
            pass
        return super().resizeEvent(event)

    def _ensure_generator_and_data(self) -> bool:
        """Ensure generator and DataFrame are available for visualization."""
        try:
            if self._data_path is None or not self._data_path.exists():
                raise FileNotFoundError("Data file not selected or not found. Please choose a valid Excel file.")
            if self._generator is None:
                from reports.length_distribution_report import LengthDistributionReportGenerator
                self._generator = LengthDistributionReportGenerator(self._data_path)
            # Always reload to reflect current file path
            self._df = self._generator.load_data()
            # Invalidate chart cache on new data
            try:
                self._render_cache.clear()
            except Exception:
                pass
            # Populate species list
            self._populate_species_list()
            return True
        except Exception as e:
            logger.exception("[Report] Failed to load data: {}", e)
            QMessageBox.critical(self, "Load Failed", f"Failed to load data:\n{e}")
            return False

    def _populate_species_list(self) -> None:
        try:
            if self._df is None:
                return
            species = sorted([str(s) for s in list(self._df["Species"].dropna().unique())])
            # Remember previous selection
            prev = self.species_combo.currentText() if hasattr(self, 'species_combo') else ''
            self.species_combo.blockSignals(True)
            self.species_combo.clear()
            for s in species:
                self.species_combo.addItem(s)
            self.species_combo.setEnabled(len(species) > 0)
            # Restore selection if possible
            if prev and prev in species:
                self.species_combo.setCurrentText(prev)
            self.species_combo.blockSignals(False)
        except Exception:
            try:
                self.species_combo.setEnabled(False)
            except Exception:
                pass

    def _clear_canvas(self) -> None:
        """Remove any existing chart widget (image)."""
        # Remove any other content widget (e.g., image)
        if getattr(self, "_content_widget", None) is not None:
            try:
                w2: QWidget = self._content_widget  # type: ignore
                self.canvas_container.removeWidget(w2)
                w2.setParent(None)
            except Exception:
                pass
            self._content_widget = None

    def _get_display_path(self, path: Path) -> str:
        """Get a display-friendly path - relative to project root if possible, else absolute."""
        try:
            project_root = Path(__file__).resolve().parent.parent.parent
            return str(path.resolve().relative_to(project_root))
        except Exception:
            return str(path.resolve())

    def _update_data_label(self) -> None:
        """Update the data label with default or current data source path."""
        default_path = Path("logs/hauls/logs.xlsx").resolve()
        if default_path.exists():
            self._data_path = default_path
            self.data_label.setText(self._get_display_path(default_path))
        else:
            self._data_path = None
            self.data_label.setText("No data file selected")

    def _update_output_label(self) -> None:
        """Update the output label with default or current output directory."""
        default_path = Path("reports/output").resolve()
        self._out_dir = default_path
        self.out_label.setText(self._get_display_path(default_path))

    # UI handlers
    def _browse_data(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Data File", "", "Excel Files (*.xlsx *.xls)")
        if path:
            p = Path(path).resolve()
            self._data_path = p
            self.data_label.setText(self._get_display_path(p))
            self._generator = None  # Reset generator so next render reloads from new file
            # Clear species until data is reloaded
            try:
                self.species_combo.clear(); self.species_combo.setEnabled(False)
            except Exception:
                pass

    def _browse_out(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory", "")
        if path:
            p = Path(path).resolve()
            self._out_dir = p
            self.out_label.setText(self._get_display_path(p))

    def _selected_formats(self) -> List[str]:
        # Always export PDF now
        return ["pdf"]

    def _on_generate(self) -> None:
        # Always export PDF
        fmts = ["pdf"]
        if self._data_path is None or not self._data_path.exists():
            QMessageBox.critical(self, "Data not found", "Data file not found. Please select a valid file.")
            return
        out_dir = self._out_dir or Path("reports/output").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        # Reset UI
        self.btn_generate.setEnabled(False)
        self.results.clear()
        self.results.setVisible(False)
        self.bar.setVisible(True)
        self.bar.setValue(0)
        self.status.setText("Starting…")
        logger.info("[Report] Starting generation. data={}, out={}, fmts={}", self._data_path, out_dir, fmts)
        # Start worker
        self.worker = ReportGenerationWorker(self._data_path, out_dir, fmts)
        self.worker.progress_updated.connect(self.bar.setValue)
        self.worker.status_updated.connect(self.status.setText)
        self.worker.finished_success.connect(self._on_success)
        self.worker.finished_error.connect(self._on_error)
        self.worker.start()

    def _on_success(self, report: Dict) -> None:
        self.btn_generate.setEnabled(True)
        self.bar.setVisible(False)
        self.status.setText("Done")
        logger.info("[Report] Done. Exported files: {}", report.get("exported_files"))
        QMessageBox.information(self, "Report Ready", "Report has been generated successfully.")
        # Show a compact summary of outputs
        files = report.get("exported_files", {})
        if files:
            text = "\n".join(f"• {k.upper()}: {v}" for k, v in files.items())
            self.results.setText(text)
            self.results.setVisible(True)

    def _on_error(self, message: str) -> None:
        self.btn_generate.setEnabled(True)
        self.bar.setVisible(False)
        self.status.setText("Error")
        logger.error("[Report] Error: {}", message)
        QMessageBox.critical(self, "Report Failed", f"Failed to generate report:\n{message}")

    def _open_interactive(self) -> None:
        """Open the current chart as an interactive Plotly HTML in the default browser."""
        try:
            if not self._ensure_generator_and_data():
                return
            key = self.chart_combo.currentData() or "species_pie"
            # Handle species-specific chart type
            if key == "species_length_distribution":
                sp = (self.species_combo.currentText() or "").strip()
                if not sp:
                    QMessageBox.information(self, "Select Species", "Please select a species to open its length distribution interactively.")
                    return
                fig = self._generator.create_species_length_chart(self._df, sp)  # type: ignore[arg-type]
            else:
                fig = self._generator.create_plotly_chart(self._df, key)  # type: ignore[arg-type]
            out_dir = Path("reports/output").resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            html_path = out_dir / "interactive_chart.html"
            fig.write_html(str(html_path), include_plotlyjs='cdn', full_html=True)
            import webbrowser
            webbrowser.open(html_path.as_uri())
        except Exception as e:
            logger.exception("[Report] Failed to open interactive chart: {}", e)
            QMessageBox.warning(self, "Open Failed", f"Could not open interactive chart:\n{e}")

    def _render_species_if_needed(self) -> None:
        try:
            key = self.chart_combo.currentData() or "species_pie"
            if key == "species_length_distribution":
                sp = (self.species_combo.currentText() or "").strip()
                if sp:
                    self._render_plotly_chart(key)
        except Exception:
            pass

    def _render_plotly_chart(self, key: str) -> None:
        # Prepare data first; if load fails just return
        if not self._ensure_generator_and_data():
            return
        try:
            use_species = (key == "species_length_distribution")
            if use_species:
                sp = (self.species_combo.currentText() or "").strip()
                if not sp:
                    # Show placeholder prompt for species
                    if self._placeholder is None:
                        self._placeholder = QLabel("Select a species to render length distribution")
                        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
                        self._placeholder.setStyleSheet(
                            "QLabel { color: #444; font-size: 13px; border: 2px dashed #ccc; padding: 14px;"
                            "border-radius: 8px; background-color: #fafafa; }"
                        )
                        self.canvas_container.addWidget(self._placeholder)
                    self.status.setText("Select a species to render length distribution")
                    return
                species = sp
            else:
                species = None

            # Compute target size up front in UI thread
            container_w = self._canvas_widget.width() or self._chart_frame.width() or 1000
            avail_w = max(700, int(container_w * 0.92))
            width_q = max(700, int(round(avail_w / self._cache_quantum) * self._cache_quantum))
            key_str = str(key)
            if use_species:
                target_h = 420
            elif "bar" in key_str:
                try:
                    categories = int(self._df["Species"].nunique())
                except Exception:
                    categories = 10
                target_h = min(max(22 * categories + 120, 320), 1000)
            elif "pie" in key_str:
                target_h = max(360, int(width_q * 0.6))
            else:
                target_h = 480

            # If nothing changed at the quantized resolution, scale current pixmap and return
            if (
                self._last_key == key
                and (self._last_species or "") == (species or "")
                and self._last_width_q == width_q
                and self._last_height == target_h
                and isinstance(self._content_widget, QLabel)
                and hasattr(self._content_widget, 'pixmap')
                and self._content_widget.pixmap() is not None
            ):
                try:
                    pm = self._content_widget.pixmap()
                    if pm and pm.width() != avail_w:
                        scaled = pm.scaledToWidth(int(avail_w), Qt.TransformationMode.SmoothTransformation)
                        self._content_widget.setPixmap(scaled)
                        self.status.setText("Resized")
                        return
                except Exception:
                    pass
                # Nothing to do
                return

            cache_key = (key, (species or ""), width_q, target_h)
            cached = self._render_cache.get(cache_key)
            if cached:
                # Show from cache immediately
                self._clear_canvas()
                from PyQt6.QtGui import QPixmap
                image_label = QLabel()
                pixmap = QPixmap()
                pixmap.loadFromData(cached)
                if pixmap.width() > avail_w:
                    pixmap = pixmap.scaledToWidth(int(avail_w), Qt.TransformationMode.SmoothTransformation)
                image_label.setPixmap(pixmap)
                image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                image_label.setMinimumHeight(320)
                image_label.setStyleSheet("QLabel{background:white;border:1px solid #e5e7eb;border-radius:6px;}")
                image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                self.canvas_container.addWidget(image_label)
                self._content_widget = image_label
                self.status.setText("Chart rendered (cached)")
                # Update last state
                self._last_key, self._last_species, self._last_width_q, self._last_height = key, (species or ""), width_q, target_h
                return

            # UI: show loading state and start async render
            if self._placeholder is not None:
                try:
                    self.canvas_container.removeWidget(self._placeholder)
                    self._placeholder.setParent(None)
                except Exception:
                    pass
                self._placeholder = None
            loading = QLabel("Rendering…")
            loading.setAlignment(Qt.AlignmentFlag.AlignCenter)
            loading.setStyleSheet("QLabel{color:#374151;background:#f3f4f6;border:1px dashed #d1d5db;border-radius:6px;padding:10px;}")
            self._clear_canvas()
            self.canvas_container.addWidget(loading)
            self._content_widget = loading

            try:
                self.bar.setVisible(True)
                self.bar.setRange(0, 0)
                self.status.setText("Rendering chart…")
            except Exception:
                pass

            self._render_task_id += 1
            current_tid = self._render_task_id
            self._render_worker = ChartRenderWorker(self._df, key, species, width_q, target_h)
            def _on_success(png_bytes: bytes, tid=current_tid, ckey=cache_key, w=width_q):
                if tid != self._render_task_id:
                    return
                try:
                    self.bar.setVisible(False)
                except Exception:
                    pass
                try:
                    self.canvas_container.removeWidget(loading)
                    loading.setParent(None)
                except Exception:
                    pass
                from PyQt6.QtGui import QPixmap
                image_label = QLabel()
                pixmap = QPixmap()
                pixmap.loadFromData(png_bytes)
                if pixmap.width() > w:
                    pixmap = pixmap.scaledToWidth(int(w), Qt.TransformationMode.SmoothTransformation)
                image_label.setPixmap(pixmap)
                image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                image_label.setMinimumHeight(320)
                image_label.setStyleSheet("QLabel{background:white;border:1px solid #e5e7eb;border-radius:6px;}")
                image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                self.canvas_container.addWidget(image_label)
                self._content_widget = image_label
                # Save to cache and update last state
                try:
                    self._render_cache[ckey] = png_bytes
                except Exception:
                    pass
                self._last_key = key
                self._last_species = (species or "")
                self._last_width_q = w
                self._last_height = target_h
                self.status.setText("Chart rendered")
            def _on_error(msg: str, tid=current_tid):
                if tid != self._render_task_id:
                    return
                try:
                    self.bar.setVisible(False)
                except Exception:
                    pass
                try:
                    self.canvas_container.removeWidget(loading)
                    loading.setParent(None)
                except Exception:
                    pass
                err = QLabel(f"Failed to render chart: {msg}")
                err.setAlignment(Qt.AlignmentFlag.AlignCenter)
                err.setStyleSheet("QLabel{color:#b91c1c;background:#fef2f2;border:1px solid #fecaca;border-radius:6px;padding:10px;}")
                self.canvas_container.addWidget(err)
                self._content_widget = err
                self.status.setText("Render failed")
            self._render_worker.finished_success.connect(_on_success)
            self._render_worker.finished_error.connect(_on_error)
            self._render_worker.start()
        except Exception as e:
            logger.exception("[Report] Failed to schedule render {}: {}", key, e)
            self._clear_canvas()
            err = QLabel("Failed to render chart: {}".format(str(e)))
            err.setAlignment(Qt.AlignmentFlag.AlignCenter)
            err.setStyleSheet("QLabel{color:#b91c1c;background:#fef2f2;border:1px solid #fecaca;border-radius:6px;padding:10px;}")
            self.canvas_container.addWidget(err)
            self._content_widget = err
            self.status.setText("Render failed")

    def _on_chart_index_changed(self, _index: int) -> None:
        # Toggle species selector visibility based on selected chart type
        self._update_species_selector_visibility()
        key = self.chart_combo.currentData() or "species_pie"
        self._render_plotly_chart(key)

    def _on_species_combo_changed(self, _index: int) -> None:
        try:
            key = self.chart_combo.currentData() or "species_pie"
            if key == "species_length_distribution":
                # debounce render to avoid thrash when scrolling
                if self._species_timer.isActive():
                    self._species_timer.stop()
                self._species_timer.start()
        except Exception:
            pass

    def _update_species_selector_visibility(self) -> None:
        try:
            key = self.chart_combo.currentData() or "species_pie"
            is_species = (key == "species_length_distribution")
            self.species_label.setVisible(is_species)
            self.species_combo.setVisible(is_species)
            # Enable only if we have items
            self.species_combo.setEnabled(is_species and self.species_combo.count() > 0)
            # Stop any pending debounce if hiding
            if not is_species and self._species_timer.isActive():
                self._species_timer.stop()
        except Exception:
            pass

    def show_default_chart(self) -> None:
        try:
            if self.chart_combo.count() == 0:
                return
            # Render only if nothing is currently displayed
            if self._content_widget is None and self._placeholder is not None:
                key = self.chart_combo.currentData() or "species_pie"
                self._render_plotly_chart(key)
        except Exception as e:
            logger.error("[Report] Failed to render default chart: {}", e)
