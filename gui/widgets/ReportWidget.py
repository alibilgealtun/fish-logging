"""
ReportWidget: Qt UI to generate length distribution reports (graphs + raw data)
using logs/hauls/logs.xlsx. Fits existing GUI structure and SOLID-aligned
reports/length_distribution_report.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QCheckBox,
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QTextEdit,
    QComboBox,
    QSplitter,
    QGraphicsDropShadowEffect,
    QScrollArea,
    QSizePolicy,
)
from PyQt6.QtGui import QFont
from loguru import logger
import sys

# Avoid importing WebEngine at module import time; set placeholder
QWebEngineView = None  # type: ignore

# Matplotlib canvas kept for PDF/Excel export use in generator; not used for UI anymore
# from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


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


class ReportWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker: ReportGenerationWorker | None = None
        self._generator = None
        self._df = None
        self._placeholder: QLabel | None = None
        self._stats_label: QLabel | None = None
        self._chart_frame: QWidget | None = None
        self._webview = None  # type: ignore
        # Track real paths independent of display labels
        self._data_path: Path | None = None
        self._out_dir: Path | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        # Maximize vertical space: ultra-tight top margin and spacing
        main_layout.setContentsMargins(8, 0, 8, 8)
        main_layout.setSpacing(2)

        # Create horizontal splitter for left-right layout
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # LEFT PANEL - Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        # Trim margins on left panel (smaller top)
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
        btn_browse_out.clicked.connect(self._browse_out)
        cfg.addWidget(btn_browse_out, 1, 2)

        # Formats
        fmt_row = QHBoxLayout()
        fmt_label = QLabel("Export Formats:")
        fmt_label.setStyleSheet("color:#fff;font-size:12px;font-weight:600;")
        fmt_row.addWidget(fmt_label)
        self.chk_pdf = QCheckBox("PDF")
        self.chk_pdf.setChecked(True)
        self.chk_excel = QCheckBox("Excel")
        self.chk_excel.setChecked(True)
        fmt_row.addWidget(self.chk_pdf)
        fmt_row.addWidget(self.chk_excel)
        fmt_row.addStretch(1)
        cfg.addLayout(fmt_row, 2, 0, 1, 3)

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
            ]
            for key in self._chart_keys:
                self.chart_combo.addItem(key, userData=key)

        chart_layout.addWidget(self.chart_combo)

        # Connect combo change to auto-show chart
        # self.chart_combo.currentTextChanged.connect(self._on_chart_type_changed)

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

        # RIGHT PANEL - Visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        # Remove top margin on right panel
        right_layout.setContentsMargins(8, 0, 8, 8)

        # Remove the chart header label to reclaim vertical space
        # chart_label = QLabel("Chart Visualization")
        # right_layout.addWidget(chart_label)

        # Canvas container frame with border and shadow
        self._chart_frame = QWidget()
        self._chart_frame.setStyleSheet(
            "background:#fff;border:1px solid #dcdcdc;border-radius:10px;"
        )
        shadow = QGraphicsDropShadowEffect(self._chart_frame)
        shadow.setBlurRadius(12)  # slightly reduced shadow blur
        shadow.setOffset(0, 1)
        shadow.setColor(Qt.GlobalColor.lightGray)
        self._chart_frame.setGraphicsEffect(shadow)

        frame_layout = QVBoxLayout(self._chart_frame)
        # Trim inner frame margins further (top=4)
        frame_layout.setContentsMargins(8, 4, 8, 8)
        frame_layout.setSpacing(6)

        # Scroll area containing stats + chart for vertical scrollability
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        frame_layout.addWidget(self._scroll)

        # Holder inside scroll area
        self._holder = QWidget()
        self._holder_layout = QVBoxLayout(self._holder)
        self._holder_layout.setContentsMargins(0, 0, 0, 0)
        self._holder_layout.setSpacing(8)
        self._scroll.setWidget(self._holder)

        # Stats label (hidden initially)
        self._stats_label = QLabel()
        self._stats_label.setVisible(False)
        self._stats_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._stats_label.setWordWrap(True)
        self._stats_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self._stats_label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                font-size: 13px;
                font-weight: 500;
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #f8f9fa, stop: 1 #e9ecef);
                border: 1px solid #dee2e6;
                border-radius: 10px;
                padding: 12px;
                margin: 4px;
            }
        """)
        # Enable rich text for HTML formatting
        self._stats_label.setTextFormat(Qt.TextFormat.RichText)
        self._holder_layout.addWidget(self._stats_label)

        # Canvas area layout under the stats label
        self._canvas_widget = QWidget()
        self._canvas_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas_container = QVBoxLayout(self._canvas_widget)
        self.canvas_container.setContentsMargins(0, 0, 0, 0)
        self.canvas_container.setSpacing(0)
        self._holder_layout.addWidget(self._canvas_widget)

        # Default placeholder inside canvas
        self._placeholder = QLabel("Select a chart type to display visualization")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(
            "QLabel { color: #666; font-size: 12px; border: 2px dashed #ccc; padding: 12px;"
            "border-radius: 8px; background-color: #f9f9f9; }"
        )
        self.canvas_container.addWidget(self._placeholder)

        right_layout.addWidget(self._chart_frame)

        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)

        # Set splitter proportions (left: 30%, right: 70%)
        splitter.setSizes([300, 700])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)

        main_layout.addWidget(splitter)

        # Reliable chart change signal
        self.chart_combo.currentIndexChanged.connect(self._on_chart_index_changed)

        logger.info("Python executable: {}", sys.executable)

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
            return True
        except Exception as e:
            logger.exception("[Report] Failed to load data: {}", e)
            QMessageBox.critical(self, "Load Failed", f"Failed to load data:\n{e}")
            return False

    def _clear_canvas(self) -> None:
        # Remove any existing web view
        if getattr(self, "_webview", None) is not None:
            try:
                w: QWidget = self._webview  # type: ignore
                self.canvas_container.removeWidget(w)
                w.setParent(None)
            except Exception:
                pass
            self._webview = None

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

    def _browse_out(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory", "")
        if path:
            p = Path(path).resolve()
            self._out_dir = p
            self.out_label.setText(self._get_display_path(p))

    def _selected_formats(self) -> List[str]:
        fmts: List[str] = []
        if self.chk_pdf.isChecked():
            fmts.append("pdf")
        if self.chk_excel.isChecked():
            fmts.append("excel")
        return fmts

    def _on_generate(self) -> None:
        fmts = self._selected_formats()
        if not fmts:
            QMessageBox.warning(self, "Select format", "Please select at least one export format (PDF/Excel)")
            return
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
        meta = report.get("metadata", {})
        files = report.get("exported_files", {})

        summary_lines = [
            "Report generated successfully.",
            f"Total Hauls: {meta.get('total_hauls', '—')}",
            f"Total Fish: {meta.get('total_fish', '—')}",
            f"Species Count: {meta.get('species_count', '—')}",
            f"Date Range: {meta.get('date_range', ('—','—'))[0]} to {meta.get('date_range', ('—','—'))[1]}",
            "",
            "Generated Files:",
        ]
        for k, v in files.items():
            summary_lines.append(f"• {k.upper()}: {v}")

        self.results.setText("\n".join(summary_lines))
        self.results.setVisible(True)
        self.status.setText("Done")
        logger.info("[Report] Done. Summary: {}", summary_lines)

        QMessageBox.information(self, "Report Ready", "Report has been generated successfully.")

    def _on_error(self, message: str) -> None:
        self.btn_generate.setEnabled(True)
        self.bar.setVisible(False)
        self.status.setText("Error")
        logger.error("[Report] Error: {}", message)
        QMessageBox.critical(self, "Report Failed", f"Failed to generate report:\n{message}")

    def _update_species_stats(self) -> None:
        """Compute and render species stats sorted by count with colorful HTML formatting."""
        try:
            if self._df is None or self._df.empty:
                return
            grouped = (
                self._df.groupby("Species")["Length (cm)"]
                .agg(count="count", avg="mean")
                .reset_index()
            )
            grouped = grouped.sort_values("count", ascending=False)

            # Color palette for different species
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22']

            lines = ['<div style="font-weight: 600; color: #2c3e50; margin-bottom: 8px;">📊 Species Summary</div>']
            for i, (_, row) in enumerate(grouped.iterrows()):
                color = colors[i % len(colors)]
                lines.append(
                    f'<div style="margin: 4px 0; padding: 6px; background: rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1); border-left: 3px solid {color}; border-radius: 4px;">'
                    f'<span style="color: {color}; font-weight: 700;">🐟 {row["Species"]}</span> '
                    f'<span style="color: #666;">–</span> '
                    f'<span style="color: #2c3e50; font-weight: 600;">count:</span> <span style="color: #e74c3c; font-weight: 700;">{int(row["count"])}</span> '
                    f'<span style="color: #666;">–</span> '
                    f'<span style="color: #2c3e50; font-weight: 600;">avg length:</span> <span style="color: #27ae60; font-weight: 700;">{row["avg"]:.1f} cm</span>'
                    f'</div>'
                )

            text = "".join(lines) if len(lines) > 1 else '<div style="color: #7f8c8d; font-style: italic;">No species data available</div>'
            if self._stats_label:
                self._stats_label.setText(text)
                self._stats_label.setVisible(True)
        except Exception as e:
            logger.error("[Report] Failed to compute species stats: {}", e)

    def _render_plotly_chart(self, key: str) -> None:
        # Import WebEngine only when needed, to keep startup fast
        global QWebEngineView
        if QWebEngineView is None:
            try:
                from PyQt6.QtWebEngineWidgets import QWebEngineView as _QWEV  # type: ignore
                QWebEngineView = _QWEV  # type: ignore
            except Exception as e:
                logger.error("[Report] Failed to import QWebEngineView: {}", e)
                QWebEngineView = None  # type: ignore
        if QWebEngineView is None:
            QMessageBox.warning(self, "Web Engine Missing", "PyQt6-WebEngine is required to render charts.\nInstall with: pip install PyQt6-WebEngine")
            return
        if not self._ensure_generator_and_data():
            return
        try:
            # Render to HTML and show without actions
            fig = self._generator.create_plotly_chart(self._df, key)  # type: ignore[arg-type]
            html = fig.to_html(full_html=True, include_plotlyjs='cdn', config={"displayModeBar": False, "responsive": True})
            # Clear previous widget/placeholder
            if self._placeholder is not None:
                self.canvas_container.removeWidget(self._placeholder)
                self._placeholder.setParent(None)
                self._placeholder = None
            self._clear_canvas()
            # Create web view lazily
            self._webview = QWebEngineView()
            # Make web view expand to container width/height
            self._webview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.canvas_container.addWidget(self._webview)
            self._webview.setHtml(html)
            self._update_species_stats()
            self.status.setText("Chart rendered")
            logger.info("[Report] Rendered Plotly chart: {}", key)
        except Exception as e:
            logger.exception("[Report] Failed to render Plotly chart {}: {}", key, e)
            QMessageBox.critical(self, "Render Failed", f"Failed to render chart:\n{e}")

    def _on_chart_type_changed(self, _current: str) -> None:
        key = self.chart_combo.currentData() or "species_pie"
        self._render_plotly_chart(key)

    def _on_chart_index_changed(self, _index: int) -> None:
        key = self.chart_combo.currentData() or "species_pie"
        self._render_plotly_chart(key)

    # New: allow MainWindow to trigger first render when Reports tab opens
    def show_default_chart(self) -> None:
        """Render the currently selected chart (default first item) if nothing is shown yet."""
        try:
            if getattr(self, "_webview", None) is None:
                if self.chart_combo.count() == 0:
                    return
                self._render_plotly_chart(self.chart_combo.currentData() or "species_pie")
        except Exception as e:
            logger.error("[Report] Failed to render default chart: {}", e)
