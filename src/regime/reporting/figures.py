"""Reusable Plotly and Matplotlib visualization adapters."""

from __future__ import annotations

import base64
import io
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.figure import Figure

from .report import ReportFigure, VisualizationMetadata

# Supported analytical sections.  Builders below are intentionally composable: a line chart,
# heatmap, distribution, or table can represent each item without coupling reports to a model.
REPORT_VISUALIZATIONS = (
    "price and return series with regime shading",
    "filtered regime-probability area charts",
    "change probability",
    "transition matrix heatmaps",
    "duration distributions",
    "occupancy charts",
    "state-conditioned return distributions",
    "state-conditioned volatility and correlation",
    "feature distributions by state",
    "confusion or alignment matrices",
    "calibration plots",
    "reliability diagrams",
    "walk-forward metric timelines",
    "hyperparameter response surfaces",
    "stability across refits",
    "state-centroid drift",
    "model-ranking tables",
    "critical-difference plots where appropriate",
    "equity curves",
    "drawdown curves",
    "rolling Sharpe and volatility",
    "turnover and cost decomposition",
    "performance by regime",
    "option-surface visualizations by regime",
    "skew and term-structure comparisons",
    "statistical significance summaries",
    "runtime and memory comparisons",
)


def _plotly(title: str, figure: go.Figure, metadata: VisualizationMetadata) -> ReportFigure:
    figure.update_layout(
        template="plotly_white",
        title=None,
        margin={"l": 55, "r": 25, "t": 25, "b": 50},
        legend={"orientation": "h", "y": 1.03},
    )
    return ReportFigure(
        title=title,
        html=figure.to_html(full_html=False, include_plotlyjs=False),
        metadata=metadata,
    )


def line_chart(
    data: pd.DataFrame | pd.Series,
    metadata: VisualizationMetadata,
    *,
    title: str,
    y_label: str = "Value",
) -> ReportFigure:
    """Build a multi-series timeline (metrics, equity, drawdown, drift, or costs)."""
    frame = data.to_frame() if isinstance(data, pd.Series) else data
    figure = go.Figure()
    for column in frame.columns:
        figure.add_scatter(x=frame.index, y=frame[column], name=str(column), mode="lines")
    figure.update_xaxes(title="Date")
    figure.update_yaxes(title=y_label)
    return _plotly(title, figure, metadata)


def regime_time_series(
    values: pd.DataFrame | pd.Series,
    regimes: pd.Series,
    metadata: VisualizationMetadata,
    *,
    title: str = "Price and returns by regime",
) -> ReportFigure:
    """Plot price/return series with contiguous regime background shading."""
    frame = values.to_frame() if isinstance(values, pd.Series) else values
    figure = go.Figure()
    for column in frame.columns:
        figure.add_scatter(x=frame.index, y=frame[column], name=str(column), mode="lines")
    aligned = regimes.reindex(frame.index)
    palette = ("#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2")
    if len(aligned):
        groups = aligned.ne(aligned.shift()).cumsum()
        for _, run in aligned.groupby(groups):
            state = run.iloc[0]
            if pd.isna(state):
                continue
            color = palette[hash(str(state)) % len(palette)]
            figure.add_vrect(
                x0=run.index[0], x1=run.index[-1], fillcolor=color, opacity=0.14, line_width=0
            )
    figure.update_xaxes(title="Date")
    return _plotly(title, figure, metadata)


def probability_area_chart(
    probabilities: pd.DataFrame,
    metadata: VisualizationMetadata,
    *,
    title: str = "Regime probabilities",
) -> ReportFigure:
    """Build a stacked probability-area chart, including change probability."""
    figure = go.Figure()
    for column in probabilities.columns:
        figure.add_scatter(
            x=probabilities.index,
            y=probabilities[column],
            name=str(column),
            mode="lines",
            stackgroup="probability",
        )
    figure.update_yaxes(title="Probability", range=[0, 1], tickformat=".0%")
    figure.update_xaxes(title="Date")
    return _plotly(title, figure, metadata)


def heatmap(
    matrix: pd.DataFrame | np.ndarray,
    metadata: VisualizationMetadata,
    *,
    title: str,
    x_labels: Sequence[str] | None = None,
    y_labels: Sequence[str] | None = None,
) -> ReportFigure:
    """Build an annotated transition, correlation, confusion, or alignment heatmap."""
    if isinstance(matrix, pd.DataFrame):
        x_labels = [str(value) for value in matrix.columns]
        y_labels = [str(value) for value in matrix.index]
        values = matrix.to_numpy()
    else:
        values = np.asarray(matrix)
    figure = go.Figure(
        go.Heatmap(z=values, x=x_labels, y=y_labels, colorscale="Viridis", texttemplate="%{z:.3g}")
    )
    return _plotly(title, figure, metadata)


def distribution_by_state(
    values: pd.Series,
    states: pd.Series,
    metadata: VisualizationMetadata,
    *,
    title: str,
    x_label: str = "Value",
) -> ReportFigure:
    """Compare returns, durations, volatility, or feature distributions by state."""
    aligned = pd.concat([values.rename("value"), states.rename("state")], axis=1).dropna()
    figure = go.Figure()
    for state, group in aligned.groupby("state", sort=True):
        figure.add_histogram(
            x=group["value"], name=str(state), histnorm="probability density", opacity=0.6
        )
    figure.update_layout(barmode="overlay")
    figure.update_xaxes(title=x_label)
    figure.update_yaxes(title="Density")
    return _plotly(title, figure, metadata)


def bar_chart(
    values: pd.DataFrame | pd.Series,
    metadata: VisualizationMetadata,
    *,
    title: str,
    y_label: str = "Value",
) -> ReportFigure:
    """Build grouped occupancy, cost, performance, or resource-comparison bars."""
    frame = values.to_frame() if isinstance(values, pd.Series) else values
    figure = go.Figure()
    for column in frame.columns:
        figure.add_bar(x=[str(value) for value in frame.index], y=frame[column], name=str(column))
    figure.update_layout(barmode="group")
    figure.update_yaxes(title=y_label)
    return _plotly(title, figure, metadata)


def scatter_chart(
    x: Sequence[Any],
    y: Sequence[float],
    metadata: VisualizationMetadata,
    *,
    title: str,
    x_label: str,
    y_label: str,
    color: Sequence[float] | None = None,
) -> ReportFigure:
    """Build calibration, reliability, critical-difference, or response plots."""
    marker: Mapping[str, Any] = (
        {"color": color, "colorscale": "Viridis", "showscale": True} if color is not None else {}
    )
    figure = go.Figure(go.Scatter(x=x, y=y, mode="lines+markers", marker=marker))
    figure.update_xaxes(title=x_label)
    figure.update_yaxes(title=y_label)
    return _plotly(title, figure, metadata)


def pareto_tradeoff_chart(
    trials: Sequence[Mapping[str, Any]],
    metadata: VisualizationMetadata,
    *,
    objectives: Sequence[str],
) -> ReportFigure:
    """Visualize a persisted two-objective Pareto set without scalarizing it."""
    if len(objectives) != 2:
        raise ValueError("Pareto trade-off charts require exactly two objectives")
    values = [row["values"] for row in trials]
    figure = go.Figure(
        go.Scatter(
            x=[value[0] for value in values],
            y=[value[1] for value in values],
            text=[f"trial {row['trial']}" for row in trials],
            mode="markers+text",
        )
    )
    figure.update_xaxes(title=objectives[0])
    figure.update_yaxes(title=objectives[1])
    return _plotly("Pareto trade-offs", figure, metadata)


def table(
    data: pd.DataFrame,
    metadata: VisualizationMetadata,
    *,
    title: str,
) -> ReportFigure:
    """Build model-ranking or statistical-significance tables."""
    columns = [str(data.index.name or "index"), *[str(column) for column in data.columns]]
    cells = [
        [str(value) for value in data.index],
        *[data[column].tolist() for column in data.columns],
    ]
    figure = go.Figure(go.Table(header={"values": columns}, cells={"values": cells}))
    return _plotly(title, figure, metadata)


def matplotlib_figure(
    figure: Figure,
    metadata: VisualizationMetadata,
    *,
    title: str,
    dpi: int = 144,
) -> ReportFigure:
    """Embed a Matplotlib figure as a portable PNG data URI."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    html = f'<img class="matplotlib" alt="{title}" src="data:image/png;base64,{encoded}">'
    return ReportFigure(title=title, html=html, metadata=metadata, kind="matplotlib")
