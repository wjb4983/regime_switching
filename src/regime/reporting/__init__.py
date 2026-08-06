"""Self-contained HTML reporting for regime-switching research.

The public API deliberately accepts ordinary pandas objects and Plotly/Matplotlib
figures so reporting remains independent of model implementations.
"""

from .experiment_report import DEFAULT_SECTIONS, ExperimentReportAssembler, ReportConfiguration
from .figures import (
    REPORT_VISUALIZATIONS,
    bar_chart,
    distribution_by_state,
    heatmap,
    line_chart,
    matplotlib_figure,
    probability_area_chart,
    regime_time_series,
    scatter_chart,
    table,
)
from .report import ReportBuilder, ReportFigure, VisualizationMetadata, render_report

__all__ = [
    "DEFAULT_SECTIONS",
    "REPORT_VISUALIZATIONS",
    "ExperimentReportAssembler",
    "ReportBuilder",
    "ReportConfiguration",
    "ReportFigure",
    "VisualizationMetadata",
    "bar_chart",
    "distribution_by_state",
    "heatmap",
    "line_chart",
    "matplotlib_figure",
    "probability_area_chart",
    "regime_time_series",
    "render_report",
    "scatter_chart",
    "table",
]
