"""Tests for self-contained research reporting."""

from datetime import date
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from regime.reporting import (
    ReportBuilder,
    VisualizationMetadata,
    matplotlib_figure,
    probability_area_chart,
    regime_time_series,
    table,
)

matplotlib.use("Agg")


@pytest.fixture
def metadata() -> VisualizationMetadata:
    return VisualizationMetadata.from_config(
        research_question="Do inferred states separate market behaviour?",
        interpretation="State one dominates after the change point.",
        data_period="2024-01-01 to 2024-01-03",
        model_version="hmm-1.2.0",
        config={"states": 2, "seed": 7},
        probability_kind="filtered",
    )


def test_report_renders_metadata_and_plotly_once(metadata: VisualizationMetadata) -> None:
    index = pd.date_range("2024-01-01", periods=3)
    probabilities = pd.DataFrame(
        {"state 0": [0.8, 0.3, 0.1], "state 1": [0.2, 0.7, 0.9]}, index=index
    )
    chart = probability_area_chart(probabilities, metadata)

    html = ReportBuilder("Model report").add(chart).render(generated_at=date(2024, 2, 1))

    assert "Do inferred states separate market behaviour?" in html
    assert "hmm-1.2.0" in html
    assert "filtered" in html
    assert html.count("plotly-2.35.2.min.js") == 1


def test_regime_chart_and_matplotlib_can_be_written(
    tmp_path: Path, metadata: VisualizationMetadata
) -> None:
    index = pd.date_range("2024-01-01", periods=3)
    series = pd.Series([100.0, 99.0, 102.0], index=index, name="price")
    states = pd.Series([0, 1, 1], index=index)
    regime_chart = regime_time_series(series, states, metadata)
    mpl, axis = plt.subplots()
    axis.plot(series)
    mpl_chart = matplotlib_figure(mpl, metadata, title="Static diagnostic")
    plt.close(mpl)

    output = (
        ReportBuilder("Combined").add(regime_chart).add(mpl_chart).write(tmp_path / "report.html")
    )

    assert output.exists()
    contents = output.read_text(encoding="utf-8")
    assert "add_vrect" not in contents
    assert "data:image/png;base64," in contents


def test_table_formats_numbers_and_truncates_long_cells(metadata: VisualizationMetadata) -> None:
    frame = pd.DataFrame(
        {
            "very_long_metric_name": [0.123456789, 1234567.89],
            "reason": [
                "short",
                "this is an intentionally long explanation that should be truncated",
            ],
        },
        index=pd.Index(["model-alpha", "model-with-an-excessively-long-name"], name="model"),
    )

    chart = table(frame, metadata, title="Comparison table")
    html = chart.html

    assert "0.1235" in html
    assert "1.235e+06" in html
    assert "model-with-an-excessively..." in html
    assert "this is an intentionally ..." in html
    assert '"columnwidth"' in html
