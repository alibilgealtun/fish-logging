"""
Reports Package for Fish Logging Application.

This package provides comprehensive reporting and data visualization capabilities
for the fish logging application. It generates various types of reports including
statistical analysis, length distributions, species summaries, and trend analysis.

Modules:
    length_distribution_report: Fish length distribution analysis and visualization

Features:
    - Statistical analysis of fish catch data
    - Length distribution charts and histograms
    - Species-specific reporting and analysis
    - PDF report generation with professional formatting
    - Trend analysis over time periods
    - Export capabilities for data sharing

Architecture:
    - Template pattern for different report types
    - Strategy pattern for various visualization approaches
    - Builder pattern for complex report construction
    - Factory pattern for report type creation

Use Cases:
    - Fishing trip summary reports
    - Species population analysis
    - Length distribution studies
    - Regulatory compliance reporting
    - Scientific data analysis
    - Tournament result summaries

Design Philosophy:
    - Professional report formatting
    - Configurable visualization options
    - Export-ready formats (PDF)
    - Scalable for large datasets
    - Accessible data presentation
"""

from __future__ import annotations

# Core reporting components
from .length_distribution_report import LengthDistributionReport

__all__ = [
    "LengthDistributionReport",
]

__version__ = "1.0.0"
__author__ = "Fish Logging Team"

# Supported report formats
SUPPORTED_FORMATS = ["pdf", "png", "svg", "html"]

# Default report configuration
DEFAULT_REPORT_CONFIG = {
    "page_size": "A4",
    "orientation": "portrait",
    "font_family": "Arial",
    "include_summary": True,
    "include_charts": True,
    "color_scheme": "professional"
}
