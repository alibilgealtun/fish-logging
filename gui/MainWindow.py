"""
Main Window for Fish Logging Application.

This module contains the MainWindow class, which serves as the primary GUI component
and orchestrator for the fish logging application. It implements the MVP pattern
and follows clean architecture principles.

Classes:
    MainWindow: Main application window that coordinates all GUI components

Architecture:
    - Dependency injection for testability and flexibility
    - Event-driven communication with speech recognition
    - Modular UI construction with extracted methods
    - Clean separation between presentation and business logic
"""
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
    """
    Main application window for the Fish Logging application.

    This class serves as the primary GUI orchestrator, coordinating between
    the speech recognition system, data logging, and user interface components.

    Key Responsibilities:
        - UI layout and theming management
        - Speech recognition event coordination
        - Tab-based navigation between different views
        - Real-time status and feedback display
        - User interaction handling (start/stop, species selection)

    Architecture:
        - Uses dependency injection for core components
        - Delegates business logic to use cases
        - Implements presenter pattern for status updates
        - Event-driven communication with speech system

    Attributes:
        speech_recognizer: Speech recognition service
        excel_logger: Data persistence service
        fish_parser: Text parsing service for fish data
        speech_handler: Event handler for speech recognition
        table_manager: Manages the fish entry table
        status_presenter: Handles status display logic
        settings_provider: Manages application settings
        recognizer_controller: Controls speech recognizer lifecycle
    """

    def __init__(self, speech_recognizer, excel_logger, fish_parser: FishParser = None) -> None:
        """
        Initialize the main window with required dependencies.

        Args:
            speech_recognizer: Speech recognition service instance
            excel_logger: Excel logging service for data persistence
            fish_parser: Optional fish text parser (creates default if None)

        Note:
            Uses dependency injection pattern for better testability and
            loose coupling between components.
        """
        super().__init__()

        # Core dependencies
        self.speech_recognizer = speech_recognizer
        self.excel_logger = excel_logger

        # Window setup
        self.setWindowTitle("🎣 Voice2FishLog ")
        self.resize(1100, 700)

        # Initialize domain components (with dependency injection)
        self.fish_parser = fish_parser if fish_parser is not None else FishParser()

        # Initialize use cases (Clean Architecture pattern)
        self.process_text_use_case = ProcessFinalTextUseCase(self.fish_parser)
        self.log_entry_use_case = LogFishEntryUseCase(self.excel_logger)
        self.cancel_entry_use_case = CancelLastEntryUseCase(self.excel_logger)

        # Initialize speech event handler (Event-driven architecture)
        self.speech_handler = SpeechEventHandler(
            self.process_text_use_case,
            self.log_entry_use_case,
            self.cancel_entry_use_case,
        )

        # Setup UI components in order
        self._setup_speech_handler_callbacks()
        self._setup_background()
        self._setup_modern_theme()
        self._setup_ui()

        # Initialize helper classes after UI is created (Dependency on UI components)
        self.table_manager = TableManager(self.table, self)
        self.status_presenter = StatusPresenter(self.status_panel, self.status_label, self.statusBar())
        self.settings_provider = SettingsProvider(self.settings_widget)
        self.recognizer_controller = RecognizerController(self.speech_recognizer)

        # Connect all signal handlers
        self._connect_signals()

        # Auto-start listening for immediate user experience
        self.recognizer_controller.ensure_started()

    def _setup_speech_handler_callbacks(self) -> None:
        """
        Configure callback functions for speech event handler.

        This method establishes the communication bridge between the speech
        event handler and the UI components, implementing the Observer pattern
        for loose coupling.

        Callbacks configured:
            - Partial text updates for real-time transcription
            - Entry logging success/failure notifications
            - Entry cancellation handling
            - Error display and recovery
            - Species detection feedback
        """
        # Real-time UI updates
        self.speech_handler.on_partial_update = self._update_live_text

        # Success handlers
        self.speech_handler.on_entry_logged = self._on_entry_logged_success
        self.speech_handler.on_entry_cancelled = self._on_entry_cancelled_success

        # Error handlers
        self.speech_handler.on_cancel_failed = self._on_cancel_failed
        self.speech_handler.on_error = self._show_error_message

        # Feature handlers
        self.speech_handler.on_species_detected = self._on_species_detected_internal

    def _setup_modern_theme(self) -> None:
        """
        Apply modern dark theme with glassmorphism effects.

        Implements a cohesive visual design system with:
        - Semi-transparent panels with blur effects
        - Gradient backgrounds and borders
        - Consistent color palette and typography
        - Responsive hover and selection states
        - Professional spacing and sizing

        Design Philosophy:
            - Modern glassmorphism aesthetic
            - High contrast for accessibility
            - Consistent visual hierarchy
            - Smooth transitions and animations
        """
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
        """
        Configure the blurred background image for aesthetic appeal.

        Creates a scaled and blurred background image that:
        - Maintains aspect ratio on window resize
        - Provides visual depth without distraction
        - Enhances the modern glassmorphism theme
        - Stays behind all other UI elements

        Implementation:
            - Uses QGraphicsBlurEffect for professional blur
            - Automatically scales with window resize events
            - Optimized for smooth performance
        """
        self.bg_label = QLabel(self)
        self.bg_pixmap = QPixmap("assets/bg.jpg")

        # Initial background setup with proper scaling
        self.bg_label.setPixmap(
            self.bg_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation
            )
        )
        self.bg_label.setGeometry(self.rect())
        self.bg_label.setScaledContents(True)

        # Apply blur effect for glassmorphism aesthetic
        blur = QGraphicsBlurEffect()
        blur.setBlurRadius(75)
        self.bg_label.setGraphicsEffect(blur)

        # Ensure background stays behind all other elements
        self.bg_label.lower()

    def resizeEvent(self, event):
        """
        Handle window resize events to maintain background scaling.

        This override ensures the background image maintains proper
        proportions and quality during window resize operations.

        Args:
            event: QResizeEvent containing new window dimensions

        Implementation:
            - Maintains aspect ratio during scaling
            - Uses smooth transformation for quality
            - Defensive programming with attribute checks
        """
        super().resizeEvent(event)  # Maintain Qt's resize behavior

        # Safely update background if components exist
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
        """
        Initialize and configure the main user interface layout.

        Creates a tabbed interface with three main sections:
        1. Fish Logging - Primary data entry interface
        2. Reports - Data visualization and analysis
        3. Settings - Application configuration

        Architecture:
            - Tab-based navigation for feature separation
        - Responsive layout system
        - Component-based UI construction
        - Event-driven tab switching behavior

        Post-initialization:
            - Configures signal connections
            - Sets initial state and status
            - Enables noise profile integration
        """
        central = QWidget()
        self.setCentralWidget(central)

        # Build main logging tab content (Primary interface)
        logging_tab = self._create_main_tab()
        self.logging_tab = logging_tab

        # Secondary tabs for additional functionality
        self.report_widget = ReportWidget()
        self.settings_widget = SettingsWidget()

        # Configure tabbed container with icons and labels
        self.tabs = QTabWidget()
        self.tabs.addTab(logging_tab, "🎣 Fish Logging")
        self.tabs.addTab(self.report_widget, "📊 Reports")
        self.tabs.addTab(self.settings_widget, "⚙️ Settings")

        # Smart tab switching: pause/resume recognition based on active tab
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Main layout with professional spacing
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.addWidget(self.tabs)
        central.setLayout(main_layout)

        # Set welcoming initial status
        self.statusBar().showMessage("🎧 Ready to capture your fishing stories...")

        # Initialize speech recognizer with current species context
        self._initialize_recognizer_species()

        # Connect control button signals (Logging tab specific)
        self._connect_control_buttons()

        # Integrate with settings for dynamic configuration
        self._connect_settings_integration()

    def _initialize_recognizer_species(self) -> None:
        """
        Initialize speech recognizer with current species selection.

        Provides context to the speech recognition system by setting
        the initially selected species, improving recognition accuracy
        for species-specific terminology.

        Implementation:
            - Defensive programming with exception handling
            - Graceful degradation if species selector unavailable
            - Optional recognizer feature (not all recognizers support this)
        """
        try:
            init_species = getattr(self, 'species_selector', None)
            if init_species is not None:
                cur = self.species_selector.currentSpecies()
                if cur and hasattr(self.speech_recognizer, 'set_last_species'):
                    self.speech_recognizer.set_last_species(cur)
        except Exception:
            # Graceful degradation - not critical for core functionality
            pass

    def _connect_control_buttons(self) -> None:
        """
        Connect start/stop button signals to their respective handlers.

        Establishes the user control interface for speech recognition,
        providing immediate feedback and control over the listening state.
        """
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_start.clicked.connect(self._on_start)

    def _connect_settings_integration(self) -> None:
        """
        Connect settings widget signals for dynamic configuration updates.

        Enables real-time application of noise profile changes without
        requiring application restart, improving user experience.

        Implementation:
            - Graceful handling if settings widget doesn't support signals
            - Non-blocking error handling for robustness
        """
        try:
            self.settings_widget.noiseProfileChanged.connect(self._on_noise_profile_changed)
        except Exception:
            # Not all settings widgets may support this signal
            pass

    def _on_noise_profile_changed(self, profile_key: str) -> None:
        """
        Handle noise profile changes from settings.

        Safely applies new noise profile by:
        1. Stopping current recognition
        2. Updating recognizer configuration
        3. Restarting if user is on logging tab

        Args:
            profile_key: Identifier for the noise profile to apply

        Implementation:
            - Safe state management during profile switching
            - Context-aware restart (only on logging tab)
            - User feedback through status bar
        """
        # Determine current tab context
        try:
            is_logging_tab = (self.tabs.currentWidget() is self.logging_tab)
        except Exception:
            is_logging_tab = True  # Safe default

        try:
            # Safe profile switching process
            self._on_stop()  # Stop to rebuild noise controller safely

            if hasattr(self.speech_recognizer, 'set_noise_profile'):
                self.speech_recognizer.set_noise_profile(profile_key)

            # Provide user feedback
            self.statusBar().showMessage(f"🔊 Noise profile set to {profile_key}")

        finally:
            # Context-aware restart: only if user is actively logging
            if is_logging_tab:
                self._on_start()

    def _on_tab_changed(self, index: int) -> None:
        """
        Handle tab switching with intelligent recognition management.

        Implements smart behavior based on active tab:
        - Logging tab: Start recognition, sync species context
        - Other tabs: Pause recognition, trigger tab-specific actions

        Args:
            index: Index of the newly selected tab

        Features:
            - Species synchronization on return to logging
            - Automatic chart rendering in reports
            - Resource optimization (pause when not needed)
            - Robust error handling
        """
        try:
            widget = self.tabs.widget(index)

            if widget is self.logging_tab:
                # Returning to Fish Logging tab
                self._handle_logging_tab_activation()
            else:
                # Entering non-logging tabs
                self._handle_non_logging_tab_activation(widget)

        except Exception as e:
            logger.error(f"Tab change handler failed: {e}")

    def _handle_logging_tab_activation(self) -> None:
        """
        Handle activation of the logging tab.

        Ensures optimal speech recognition context by:
        - Synchronizing current species selection
        - Starting speech recognition
        - Maintaining recognition state consistency
        """
        # Sync current species to recognizer for better accuracy
        try:
            cur = self.species_selector.currentSpecies()
            if cur and hasattr(self.speech_recognizer, 'set_last_species'):
                self.speech_recognizer.set_last_species(cur)
        except Exception:
            pass  # Non-critical feature

        # Resume speech recognition
        self._on_start()

    def _handle_non_logging_tab_activation(self, widget) -> None:
        """
        Handle activation of non-logging tabs.

        Optimizes resources and provides tab-specific functionality:
        - Pauses speech recognition to save resources
        - Triggers reports chart rendering
        - Maintains clean separation of concerns

        Args:
            widget: The activated tab widget
        """
        # Pause recognition to save resources
        self._on_stop()

        # Tab-specific activation logic
        if widget is self.report_widget:
            # Auto-render default chart for immediate visual feedback
            if hasattr(self.report_widget, "show_default_chart"):
                self.report_widget.show_default_chart()

    def _create_main_tab(self) -> QWidget:
        """
        Create and configure the main fish logging interface.

        Constructs a responsive two-panel layout:
        - Top: Live transcription display (compact)
        - Bottom: Data table and controls (expandable)

        Returns:
            QWidget: Configured main logging interface

        Design:
            - Vertical splitter for user-adjustable layout
            - Proper stretch factors for optimal space usage
            - Professional spacing and margins
        """
        main_tab = QWidget()

        # Create UI panels using extracted methods for maintainability
        transcription_panel = self._create_transcription_panel()
        table_panel = self._create_table_panel()

        # Configure responsive splitter layout
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(transcription_panel)
        splitter.addWidget(table_panel)

        # Optimize space allocation: compact transcription, expandable table
        splitter.setStretchFactor(0, 0)  # Transcription: fixed size
        splitter.setStretchFactor(1, 1)  # Table: expandable
        splitter.setSizes([150, 550])    # Initial proportions

        # Apply professional layout
        main_tab_layout = QVBoxLayout()
        main_tab_layout.setContentsMargins(20, 20, 20, 20)
        main_tab_layout.addWidget(splitter)
        main_tab.setLayout(main_tab_layout)

        return main_tab

    def _create_transcription_panel(self) -> QWidget:
        """
        Create the live transcription display panel.

        Provides real-time feedback of speech recognition with:
        - Clear visual hierarchy with header
        - Read-only text display optimized for live updates
        - Helpful placeholder text for user guidance
        - Appropriate sizing constraints

        Returns:
            QWidget: Configured transcription panel

        UX Considerations:
            - Non-wrapping text for real-time updates
            - Engaging placeholder text
            - Optimal height for single-line display
        """
        transcription_panel = GlassPanel()
        transcription_layout = QVBoxLayout(transcription_panel)
        transcription_layout.setSpacing(12)
        transcription_layout.setContentsMargins(24, 20, 24, 20)

        # Clear section header
        transcription_header = ModernLabel("🎤 Live Transcription", "header")
        transcription_layout.addWidget(transcription_header)

        # Live text display with optimal configuration
        self.live_text = ModernTextEdit()
        self.live_text.setReadOnly(True)
        self.live_text.setPlaceholderText("🎧 Listening for your voice... speak naturally about your catch!")
        self.live_text.setMinimumHeight(50)
        self.live_text.setMaximumHeight(60)

        # Configure for real-time display
        try:
            self.live_text.setLineWrapMode(self.live_text.LineWrapMode.NoWrap)
        except Exception:
            pass  # Graceful degradation

        # Enhanced typography for readability
        self.live_text.setStyleSheet(
            self.live_text.styleSheet() + "\nModernTextEdit { font-size: 16px; }\n"
        )

        transcription_layout.addWidget(self.live_text)
        return transcription_panel

    def _create_table_panel(self) -> QWidget:
        """
        Create the fish entries table panel with controls.

        Comprehensive data management interface featuring:
        - Species selector with search capability
        - Sortable table with professional styling
        - Integrated control buttons
        - Real-time status indicator

        Returns:
            QWidget: Complete table panel with all controls

        Architecture:
            - Modular construction using extracted methods
            - Clear visual hierarchy
            - Responsive layout system
        """
        table_panel = GlassPanel()
        table_layout = QVBoxLayout(table_panel)
        table_layout.setSpacing(16)
        table_layout.setContentsMargins(24, 20, 24, 20)

        # Table header with species selection
        header_row = self._create_table_header()
        table_layout.addLayout(header_row)

        # Main data table
        self._create_and_configure_table()
        table_layout.addWidget(self.table)

        # Control panel with buttons and status
        control_panel = self._create_control_panel()
        table_layout.addWidget(control_panel)

        return table_panel

    def _create_table_header(self) -> QHBoxLayout:
        """
        Create table header with species selector integration.

        Provides contextual controls at the table level:
        - Clear section identification
        - Species selection with visual feedback
        - Optimal spacing and alignment

        Returns:
            QHBoxLayout: Configured header layout

        Features:
            - Professional spacing and alignment
            - Searchable species dropdown
            - Clear visual labels
        """
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(6)

        # Section identification
        table_header = ModernLabel("📊 Logged Entries", "header")

        # Species selection context
        self.current_specie_label = ModernLabel("🐟 Current:", style="subheader")

        # Advanced species selector (searchable dropdown with codes)
        self.species_selector = SpeciesSelector()
        self.species_selector.setMinimumWidth(240)

        # Layout with proper alignment
        header_row.addWidget(table_header)
        header_row.addWidget(self.current_specie_label)
        header_row.addWidget(self.species_selector)
        header_row.addStretch()  # Push content left

        return header_row

    def _create_and_configure_table(self) -> None:
        """
        Create and configure the main fish logging table.

        Sets up a professional data table with:
        - Optimized column layout and sizing
        - User-friendly headers with icons
        - Proper selection and editing behavior
        - Visual enhancements for readability

        Table Columns:
            - Date: Fixed width, formatted display
            - Time: Fixed width, precise timing
            - Species: Expandable for long names
            - Length: Numeric with units
            - Actions: Fixed width for delete button

        UX Features:
            - Row-based selection
            - Alternating row colors
            - No editing to prevent data corruption
            - Professional header styling
        """
        # Initialize table with optimal column count
        self.table = ModernTable(0, 5)
        self.table.setHorizontalHeaderLabels([
            "📅 Date", "⏰ Time", "🐟 Species", "📏 Length (cm)", "🗑"
        ])

        # Configure table behavior for data integrity
        self.table.verticalHeader().setVisible(False)
        self.table.setSortingEnabled(False)  # Maintain chronological order
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)

        # Configure column sizing for optimal display
        self._configure_table_columns()

    def _configure_table_columns(self) -> None:
        """
        Configure table column widths and resize behavior.

        Implements responsive design principles:
        - Fixed widths for date/time consistency
        - Fixed width for action buttons
        - Expandable species and length columns
        - Graceful error handling
        """
        header = self.table.horizontalHeader()

        try:
            # Fixed columns for consistent formatting
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)  # Date
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)  # Time
            header.resizeSection(0, 200)
            header.resizeSection(1, 200)

            # Action column with minimal width
            header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
            header.resizeSection(4, 44)

        except Exception as e:
            logger.error(f"Table column configuration failed: {e}")

    def _create_control_panel(self) -> QWidget:
        """
        Create the main control panel for speech recognition.

        Provides centralized control interface with:
        - Start/Stop buttons with clear visual states
        - Real-time status indicator
        - Professional layout and spacing
        - Responsive design elements

        Returns:
            QWidget: Complete control panel

        Features:
            - Animated buttons for engaging interaction
            - Status indicator with visual feedback
            - Optimal spacing for touch interfaces
        """
        control_panel = GlassPanel()
        control_layout = QHBoxLayout(control_panel)
        control_layout.setContentsMargins(24, 16, 24, 16)
        control_layout.setSpacing(20)

        # Primary action buttons with visual hierarchy
        self.btn_start = AnimatedButton("▶️ Start Listening", primary=True)
        self.btn_start.setEnabled(False)  # Initially disabled until ready
        self.btn_stop = AnimatedButton("⏹️ Stop Listening")

        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)

        # Status indicator for real-time feedback
        status_container = self._create_status_indicator()
        control_layout.addWidget(status_container)

        # Flexible space for responsive design
        control_layout.addStretch(1)

        return control_panel

    def _create_status_indicator(self) -> QWidget:
        """
        Create visual status indicator with text label.

        Provides immediate visual feedback about system state:
        - Animated status indicator (pulsing effect)
        - Clear text description
        - Professional styling
        - Compact layout

        Returns:
            QWidget: Status indicator container

        Design:
            - Horizontal layout for icon + text
            - Consistent spacing and styling
            - High contrast for accessibility
        """
        status_container = QWidget()
        status_layout = QHBoxLayout(status_container)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(12)

        # Animated status indicator
        self.status_panel = PulsingStatusIndicator()

        # Status text with professional styling
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
        """
        Connect all Qt signals to their respective handlers.

        Establishes the complete event-driven communication system:
        - Speech recognition events (partial, final, error, status)
        - Species detection and selection
        - UI state synchronization

        Implementation:
            - Defensive disconnection to prevent duplicates
        - Graceful handling of optional signals
        - Clear separation of concerns

        Signal Categories:
            1. Core speech recognition (partial_text, final_text, error)
            2. Status updates (status_changed)
            3. Feature signals (specie_detected)
            4. User interaction (speciesChanged)
        """
        # Core speech recognition signals (Required)
        self._connect_core_speech_signals()

        # Status and feature signals (Optional)
        self._connect_optional_speech_signals()

        # User interaction signals
        self._connect_user_interaction_signals()

    def _connect_core_speech_signals(self) -> None:
        """
        Connect essential speech recognition signals.

        These signals are fundamental to the application's core functionality
        and must be handled reliably.
        """
        # Safely disconnect any existing connections
        for signal_name, handler in [
            ('partial_text', self._on_partial_text),
            ('final_text', self._on_final_text),
            ('error', self._on_error)
        ]:
            try:
                getattr(self.speech_recognizer, signal_name).disconnect(handler)
            except Exception:
                pass  # No existing connection

        # Establish fresh connections
        self.speech_recognizer.partial_text.connect(self._on_partial_text)
        self.speech_recognizer.final_text.connect(self._on_final_text)
        self.speech_recognizer.error.connect(self._on_error)

    def _connect_optional_speech_signals(self) -> None:
        """
        Connect optional speech recognition signals.

        These signals enhance functionality but are not required
        for basic operation.
        """
        # Status updates for visual feedback
        if hasattr(self.speech_recognizer, "status_changed"):
            try:
                self.speech_recognizer.status_changed.disconnect(self._on_status)
            except Exception:
                pass
            self.speech_recognizer.status_changed.connect(self._on_status)

        # Species detection for smart selection
        if hasattr(self.speech_recognizer, "specie_detected"):
            try:
                self.speech_recognizer.specie_detected.disconnect(self._on_specie_detected)
            except Exception:
                pass
            self.speech_recognizer.specie_detected.connect(self._on_specie_detected)

    def _connect_user_interaction_signals(self) -> None:
        """
        Connect user interface interaction signals.

        Handles manual user actions and maintains system synchronization.
        """
        # Species selector changes
        try:
            self.species_selector.speciesChanged.connect(self._on_species_selected)
        except Exception:
            pass  # Widget may not support this signal

    def _on_specie_detected(self, specie: str) -> None:
        """
        Handle automatic species detection from speech.

        Synchronizes the UI species selector with detected species
        for consistent user experience.

        Args:
            specie: Name of the detected species

        Implementation:
            - Best-effort synchronization
            - Graceful failure handling
            - Non-blocking operation
        """
        try:
            self.species_selector.setCurrentByName(specie)
        except Exception:
            pass  # Non-critical feature

    def _on_species_selected(self, name: str) -> None:
        """
        Handle manual species selection by user.

        Updates the speech recognizer context and provides user feedback
        when species is manually changed.

        Args:
            name: Name of the selected species

        Features:
            - Context synchronization with speech recognizer
            - Status bar feedback
            - Defensive programming
        """
        # Update speech recognizer context for better accuracy
        try:
            if hasattr(self.speech_recognizer, 'set_last_species'):
                self.speech_recognizer.set_last_species(name)
        except Exception:
            pass  # Optional feature

        # Provide user feedback
        self.statusBar().showMessage(f"Current species: {name}")

    def _on_partial_text(self, text: str) -> None:
        """
        Handle real-time partial speech recognition results.

        Delegates to the speech handler for business logic processing
        while maintaining separation of concerns.

        Args:
            text: Partial recognition text from speech engine
        """
        self.speech_handler.handle_partial_text(text)

    def _on_final_text(self, text: str, confidence: float) -> None:
        """
        Handle complete speech recognition results.

        Orchestrates the complete fish logging workflow:
        1. Retrieves current application context
        2. Delegates to speech handler for processing
        3. Maintains clean architecture separation

        Args:
            text: Final recognized speech text
            confidence: Recognition confidence score (0.0-1.0)

        Architecture:
            - Uses SettingsProvider for context data
            - Delegates business logic to speech handler
            - Maintains separation of concerns
        """
        # Gather current application context
        boat_name, station_id = self.settings_provider.get_and_save_all()

        # Delegate to speech handler with complete context
        self.speech_handler.handle_final_text(text, confidence, boat_name, station_id)

    def _prepend_table_row(self, species: str, length_cm: float, confidence: float, boat: str, station: str) -> None:
        """
        Add a new fish entry to the top of the table.

        Creates a new table row with properly formatted data and
        interactive delete functionality.

        Args:
            species: Fish species name
            length_cm: Fish length in centimeters
            confidence: Recognition confidence (for internal use)
            boat: Boat name (logged but not displayed)
            station: Station ID (logged but not displayed)

        UI Features:
            - Chronological ordering (newest first)
            - Center-aligned data for readability
            - Interactive delete button per row
            - Proper row height for touch interfaces

        Data Display:
            - Date/Time: Human-readable format
            - Species: As detected/selected
            - Length: One decimal place with units
            - Delete: Trash icon button
        """
        from datetime import datetime

        # Generate current timestamp
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        # Insert new row at top (chronological order)
        self.table.insertRow(0)

        # Create table items with proper formatting
        items = [
            QTableWidgetItem(date_str),
            QTableWidgetItem(time_str),
            QTableWidgetItem(species),
            QTableWidgetItem(f"{length_cm:.1f}")
        ]

        # Apply center alignment for professional appearance
        for item in items:
            try:
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            except Exception:
                pass  # Graceful degradation

        # Set table items
        for col, item in enumerate(items):
            self.table.setItem(0, col, item)

        # Set optimal row height
        self.table.setRowHeight(0, 46)

        # Add interactive delete button
        self._add_delete_button(0)

    def _add_delete_button(self, row: int) -> None:
        """
        Add delete button to the specified table row.

        Creates a styled delete button with proper event handling
        for safe row removal.

        Args:
            row: Row index where to add the delete button

        Features:
            - Standard trash icon for universal recognition
            - Hover effects for interactivity
            - Robust click handling with row identification
            - Compact sizing for table integration
        """
        btn_delete = QPushButton()
        btn_delete.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon))
        btn_delete.setIconSize(QSize(18, 18))
        btn_delete.setFixedSize(28, 28)

        # Style for integration with table theme
        btn_delete.setStyleSheet("""
            QPushButton { 
                border: none; 
                background: transparent; 
            }
            QPushButton:hover { 
                color: red; 
            }
        """)
        btn_delete.setFlat(True)

        # Connect to robust deletion handler
        btn_delete.clicked.connect(self._on_delete_clicked)

        # Add to table
        self.table.setCellWidget(row, 4, btn_delete)

    def _on_delete_clicked(self) -> None:
        """
        Handle delete button clicks with proper row identification.

        Safely identifies and removes the correct table row when
        a delete button is clicked, regardless of table sorting
        or dynamic content changes.

        Implementation:
            - Robust sender identification
            - Linear search for button location
            - Delegated removal with confirmation
            - Comprehensive error handling
        """
        try:
            sender = self.sender()
            if sender is None:
                return

            # Find the row containing the clicked button
            delete_column = 4
            for row in range(self.table.rowCount()):
                if self.table.cellWidget(row, delete_column) is sender:
                    self._remove_table_row(row)
                    break

        except Exception as e:
            logger.error(f"Delete click failed: {e}")

    def _remove_last_table_row(self) -> None:
        """
        Remove the most recently added table row.

        Provides a way to quickly undo the last entry,
        useful for correction workflows.

        Implementation:
            - Safe bounds checking
            - Removes from top (most recent)
        """
        if self.table.rowCount() > 0:
            self.table.removeRow(0)

    def _remove_table_row(self, row_index: int) -> None:
        """
        Remove a specific table row with user confirmation.

        Implements safe deletion workflow with user confirmation
        to prevent accidental data loss.

        Args:
            row_index: Index of the row to remove

        Safety Features:
            - Bounds checking
            - User confirmation dialog
            - Clear confirmation message
            - Graceful cancellation
        """
        if 0 <= row_index < self.table.rowCount():
            reply = QMessageBox.question(
                self,
                "Delete Entry",
                "Are you sure you want to delete this entry?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.table.removeRow(row_index)

    def _on_error(self, message: str) -> None:
        """
        Handle speech recognition errors with user feedback.

        Provides comprehensive error handling with:
        - User notification through alert dialog
        - Status display update
        - UI state recovery
        - Button state management

        Args:
            message: Error message to display to user

        Recovery Actions:
            - Re-enable start button for manual restart
            - Disable stop button (not applicable)
            - Show error in status display
            - Log error for debugging
        """
        self._alert(message)
        self.status_presenter.show_error(message)

        # Reset button states for user recovery
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _on_status(self, message: str) -> None:
        """
        Handle status updates from speech recognizer.

        Delegates status display to the dedicated presenter,
        maintaining separation of concerns.

        Args:
            message: Status message to display
        """
        self.status_presenter.show_status(message)

    def _on_stop(self) -> None:
        """
        Stop speech recognition and update UI state.

        Coordinates stopping the speech recognition with:
        - Controller delegation for clean shutdown
        - UI state updates for user feedback
        - Status display updates

        Post-stop State:
            - Start button enabled for restart
            - Stop button disabled (not applicable)
            - Status shows "stopped"
        """
        self.recognizer_controller.stop()

        # Update UI state
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_presenter.show_status("stopped")

    def _on_start(self) -> None:
        """
        Start speech recognition and update UI state.

        Coordinates starting speech recognition with:
        - Controller delegation for reliable startup
        - Success/failure handling
        - UI state synchronization

        Success State:
            - Start button disabled (already running)
            - Stop button enabled for control
            - Status shows "listening"

        Failure State:
            - Alert dialog with error message
            - UI state remains unchanged
        """
        success = self.recognizer_controller.start()

        if success:
            # Update UI for active state
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.status_presenter.show_status("listening")
        else:
            # Handle startup failure
            self._alert("Failed to start listening")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """
        Handle application shutdown with proper cleanup.

        Ensures graceful shutdown by:
        - Stopping speech recognition
        - Cleaning up resources
        - Calling parent cleanup

        Args:
            event: Qt close event (unused but required by signature)

        Implementation:
            - Defensive controller stop
            - Proper Qt cleanup chain
        """
        self.recognizer_controller.stop()
        super().closeEvent(event)

    # Missing method implementations for callbacks
    def _update_live_text(self, text: str) -> None:
        """
        Update the live transcription display.

        Args:
            text: Current transcription text to display
        """
        self.live_text.setPlainText(text)

    def _on_entry_logged_success(self, species: str, length: float, **kwargs) -> None:
        """
        Handle successful fish entry logging.

        Args:
            species: Logged fish species
            length: Logged fish length
            **kwargs: Additional context (boat, station, confidence)
        """
        # Add to table display
        confidence = kwargs.get('confidence', 0.0)
        boat = kwargs.get('boat', '')
        station = kwargs.get('station', '')
        self._prepend_table_row(species, length, confidence, boat, station)

        # Provide user feedback
        self.statusBar().showMessage(f"✅ Logged: {species} ({length:.1f}cm)")

    def _on_entry_cancelled_success(self) -> None:
        """Handle successful entry cancellation."""
        self._remove_last_table_row()
        self.statusBar().showMessage("❌ Last entry cancelled")

    def _on_cancel_failed(self, message: str) -> None:
        """
        Handle failed entry cancellation.

        Args:
            message: Error message describing the failure
        """
        self._alert(f"Failed to cancel entry: {message}")

    def _show_error_message(self, message: str) -> None:
        """
        Show error message to user.

        Args:
            message: Error message to display
        """
        self._alert(message)

    def _on_species_detected_internal(self, species: str) -> None:
        """
        Handle internal species detection.

        Args:
            species: Detected species name
        """
        self._on_specie_detected(species)

    def _alert(self, message: str) -> None:
        """
        Show alert dialog to user.

        Args:
            message: Alert message to display
        """
        QMessageBox.warning(self, "Alert", message)
