import pandas as pd

from regime.validation import (
    AnchoredWalkForwardSplitter,
    AssetUniverseHoldoutSplitter,
    CrisisPeriodStressTestSplitter,
    CrossSectionalSplitter,
    ExecutionDelaySensitivitySplitter,
    ExpandingWindowSplitter,
    GeographicMarketHoldoutSplitter,
    MarketPeriodHoldoutSplitter,
    PurgedTimeSeriesSplitter,
    RefitFrequencySplitter,
    RollingWindowSplitter,
)


def _frame(size: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=size),
            "asset": ["a", "b", "c"] * (size // 3) + ["a"] * (size % 3),
            "period": ["normal"] * 10 + ["crisis"] * 10 + ["recovery"] * (size - 20),
            "geography": ["us", "eu", "jp"] * (size // 3) + ["us"] * (size % 3),
            "label_start": range(size),
            "label_end": range(2, size + 2),
        }
    )


def _assert_explicit_windows(split) -> None:
    assert {window.role for window in split.windows} == {
        "train",
        "validation",
        "test",
        "embargo",
        "execution",
    }


def test_expanding_and_anchored_walk_forward_emit_ordered_windows() -> None:
    split = next(
        ExpandingWindowSplitter(initial_train_size=10, validation_size=3, test_size=2).split(
            _frame()
        )
    )

    assert split.train == tuple(range(10))
    assert split.validation == (10, 11, 12)
    assert split.test == (13, 14)
    _assert_explicit_windows(split)

    anchored = next(
        AnchoredWalkForwardSplitter(initial_train_size=10, validation_size=3, test_size=2).split(
            _frame()
        )
    )
    assert anchored.train == tuple(range(10))


def test_rolling_window_keeps_fixed_train_width() -> None:
    splits = list(
        RollingWindowSplitter(train_size=6, validation_size=2, test_size=2, step=3).split(_frame())
    )

    assert splits[1].train == tuple(range(3, 9))
    assert splits[1].validation == (9, 10)
    assert splits[1].test == (11, 12)


def test_purged_splitter_excludes_overlapping_labels_and_embargo() -> None:
    frame = _frame()
    split = next(
        PurgedTimeSeriesSplitter(
            initial_train_size=10,
            validation_size=2,
            test_size=2,
            embargo=1,
        ).split(frame)
    )

    assert 9 not in split.train
    assert 10 not in split.train
    assert 11 not in split.train
    assert 12 not in split.train
    assert 13 not in split.train
    assert all(index < min(split.validation + split.test) for index in split.train)
    assert set(split.embargo).isdisjoint(split.train)


def test_nested_model_selection_maps_inner_splits_to_outer_train() -> None:
    outer = next(
        ExpandingWindowSplitter(initial_train_size=12, validation_size=3, test_size=2).split(
            _frame()
        )
    )
    nested = outer.nested(
        RollingWindowSplitter(train_size=5, validation_size=2, test_size=1, step=2)
    )

    assert nested[0].train == tuple(range(5))
    assert nested[0].validation == (5, 6)
    assert nested[0].test == (7,)
    assert max(nested[-1].test) in outer.train


def test_cross_sectional_and_asset_universe_holdouts_exclude_assets() -> None:
    frame = _frame()
    split = next(CrossSectionalSplitter(validation_assets=("b",), test_assets=("c",)).split(frame))
    asset_split = next(AssetUniverseHoldoutSplitter(test_assets=("c",)).split(frame))

    assert set(frame.loc[list(split.train), "asset"]) == {"a"}
    assert set(frame.loc[list(split.validation), "asset"]) == {"b"}
    assert set(frame.loc[list(split.test), "asset"]) == {"c"}
    assert "c" not in set(frame.loc[list(asset_split.train), "asset"])


def test_market_crisis_and_geographic_holdouts() -> None:
    frame = _frame()
    market = next(
        MarketPeriodHoldoutSplitter(
            validation_periods=("crisis",), test_periods=("recovery",)
        ).split(frame)
    )
    crisis = next(CrisisPeriodStressTestSplitter(test_periods=("crisis",)).split(frame))
    geo = next(GeographicMarketHoldoutSplitter(test_geographies=("jp",)).split(frame))

    assert set(frame.loc[list(market.validation), "period"]) == {"crisis"}
    assert set(frame.loc[list(market.test), "period"]) == {"recovery"}
    assert set(frame.loc[list(crisis.test), "period"]) == {"crisis"}
    assert set(frame.loc[list(geo.test), "geography"]) == {"jp"}


def test_refit_frequency_and_execution_delay_sensitivity() -> None:
    frame = _frame()
    refit = list(
        RefitFrequencySplitter(
            initial_train_size=8, validation_size=2, test_size=1, refit_frequency=4
        ).split(frame)
    )
    delayed = list(
        ExecutionDelaySensitivitySplitter(
            initial_train_size=8,
            validation_size=1,
            test_size=1,
            execution_delays=(0, 2),
        ).split(frame)
    )

    assert refit[1].train == tuple(range(12))
    assert delayed[0].execution == delayed[0].validation + delayed[0].test
    delayed_by_two = next(split for split in delayed if split.metadata["execution_delay"] == 2)
    assert delayed_by_two.execution[0] == delayed_by_two.validation[0] + 2
