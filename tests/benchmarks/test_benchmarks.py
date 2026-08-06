"""Representative, deterministic benchmarks for performance-sensitive workflows."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from regime.data.options.features import build_delta_moneyness_tenor_grid
from regime.data.options.surface import VolSurface, interpolate_surface
from regime.features.returns import multi_horizon_returns
from regime.models.probabilistic.hmm import GaussianHMM, ProbabilisticHMMConfig
from regime.reporting.report import ReportBuilder, ReportFigure, VisualizationMetadata
from regime.validation.splitters import AnchoredWalkForwardSplitter

pytestmark = [pytest.mark.benchmark, pytest.mark.slow, pytest.mark.timeout(600)]
Recorder = Callable[[str, Callable[[], Any], int], Any]


@pytest.fixture(scope="module")
def market_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0002, 0.012, 100_000)
    return pd.DataFrame(
        {
            "close": 100.0 * np.exp(np.cumsum(returns)),
            "volume": rng.integers(1_000, 1_000_000, len(returns)),
        }
    )


@pytest.fixture(scope="module")
def parquet_file(tmp_path_factory: pytest.TempPathFactory, market_frame: pd.DataFrame) -> Path:
    path = tmp_path_factory.mktemp("benchmark-data") / "market.parquet"
    pq.write_table(pa.Table.from_pandas(market_frame, preserve_index=False), path)
    return path


def test_feature_generation(benchmark_recorder: Recorder, market_frame: pd.DataFrame) -> None:
    output = benchmark_recorder(
        "feature_generation", lambda: multi_horizon_returns(market_frame), 3
    )
    assert output.shape == (100_000, 4)


def test_duckdb_queries(benchmark_recorder: Recorder, market_frame: pd.DataFrame) -> None:
    connection = duckdb.connect()
    connection.register("market", market_frame)
    output = benchmark_recorder(
        "duckdb_queries",
        lambda: connection.sql("SELECT avg(close), sum(volume) FROM market").fetchone(),
        5,
    )
    assert output[0] > 0
    connection.close()


def test_parquet_scans(benchmark_recorder: Recorder, parquet_file: Path) -> None:
    connection = duckdb.connect()
    output = benchmark_recorder(
        "parquet_scans",
        lambda: connection.execute(
            "SELECT count(*), avg(close) FROM read_parquet(?)", [str(parquet_file)]
        ).fetchone(),
        5,
    )
    assert output[0] == 100_000
    connection.close()


def _hmm_data(size: int = 2_000) -> np.ndarray:
    rng = np.random.default_rng(7)
    states = np.repeat([0, 1], size // 2)
    return rng.normal(np.where(states == 0, -0.01, 0.01), 0.008)[:, None]


def _hmm() -> GaussianHMM:
    return GaussianHMM(
        ProbabilisticHMMConfig(model_name="benchmark-hmm", n_states=2, max_iter=8, random_seed=7)
    )


def test_hmm_fitting(benchmark_recorder: Recorder) -> None:
    model = benchmark_recorder("hmm_fitting", lambda: _hmm().fit(_hmm_data()), 2)
    assert model.metadata.training_observations == 2_000


def test_online_filtering(benchmark_recorder: Recorder) -> None:
    model = _hmm().fit(_hmm_data())
    observations = _hmm_data(1_000)
    output = benchmark_recorder(
        "online_filtering", lambda: [model.filter(row) for row in observations], 3
    )
    assert len(output) == 1_000


def test_walk_forward_evaluation(benchmark_recorder: Recorder) -> None:
    data = _hmm_data(2_000)
    splitter = AnchoredWalkForwardSplitter(
        initial_train_size=1_000, validation_size=200, test_size=200, step=200
    )

    def evaluate() -> list[float]:
        scores = []
        for split in splitter.split(data):
            model = GaussianHMM(
                ProbabilisticHMMConfig(
                    model_name="walk-forward-benchmark",
                    n_states=2,
                    max_iter=3,
                    random_seed=7,
                )
            ).fit(data[list(split.train)])
            probabilities = np.asarray(model.predict_proba(data[list(split.test)]))
            scores.append(float(np.mean(np.max(probabilities, axis=1))))
        return scores

    output = benchmark_recorder(
        "walk_forward_evaluation",
        evaluate,
        1,
    )
    assert len(output) == 4


def test_report_generation(benchmark_recorder: Recorder, tmp_path: Path) -> None:
    metadata = VisualizationMetadata.from_config(
        research_question="Benchmark report rendering",
        interpretation="Synthetic benchmark figure",
        data_period="synthetic",
        model_version="benchmark",
        config={"rows": 100_000},
    )
    figures = [
        ReportFigure(f"Figure {index}", "<div>benchmark</div>", metadata) for index in range(100)
    ]

    def render() -> Path:
        builder = ReportBuilder("Benchmark report")
        for figure in figures:
            builder.add(figure)
        return builder.write(tmp_path / "report.html", generated_at=datetime(2024, 1, 1))

    output = benchmark_recorder("report_generation", render, 3)
    assert output.stat().st_size > 0


def test_option_surface_construction(benchmark_recorder: Recorder) -> None:
    deltas = np.linspace(-0.9, 0.9, 19)
    moneyness = np.linspace(0.7, 1.3, 25)
    tenors = np.linspace(7 / 365, 2.0, 24)

    def construct() -> tuple[float, int]:
        grid = build_delta_moneyness_tenor_grid(deltas=deltas, moneyness=moneyness, tenors=tenors)
        surface = VolSurface.from_rows(
            (tenor, money, 0.2 + 0.1 * (money - 1) ** 2) for _, money, tenor in grid
        )
        return interpolate_surface(surface, tenor=0.5, moneyness=1.0), len(surface.points)

    output = benchmark_recorder("option_surface_construction", construct, 3)
    assert output[1] == 11_400


def test_large_universe_batching(benchmark_recorder: Recorder) -> None:
    rng = np.random.default_rng(9)
    returns = rng.normal(size=(2_000, 1_000))

    def batch() -> np.ndarray:
        return np.concatenate(
            [np.nanstd(block, axis=0) for block in np.array_split(returns, 20, axis=1)]
        )

    output = benchmark_recorder("large_universe_batching", batch, 3)
    assert output.shape == (1_000,)
