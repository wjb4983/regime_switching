"""Option-implied feature builders for volatility surfaces and exposures."""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.replace([np.inf, -np.inf], np.nan)


def atm_implied_volatility(
    surface: pd.DataFrame,
    moneyness_column: str = "moneyness",
    iv_column: str = "implied_volatility",
) -> pd.DataFrame:
    """Select the option implied volatility nearest at-the-money for each timestamp."""
    data = surface.assign(_atm_distance=(surface[moneyness_column].astype(float) - 1.0).abs())
    rows = data.sort_values("_atm_distance").groupby(level=0).first()
    return _clean(pd.DataFrame({"atm_implied_volatility": rows[iv_column].astype(float)}))


def skew(
    surface: pd.DataFrame,
    low_delta: float = 0.25,
    high_delta: float = 0.75,
    delta_column: str = "delta",
    iv_column: str = "implied_volatility",
) -> pd.DataFrame:
    """Compute implied-volatility skew between low- and high-delta options."""
    low = (
        surface.iloc[(surface[delta_column].astype(float) - low_delta).abs().argsort()]
        .groupby(level=0)
        .first()[iv_column]
    )
    high = (
        surface.iloc[(surface[delta_column].astype(float) - high_delta).abs().argsort()]
        .groupby(level=0)
        .first()[iv_column]
    )
    return _clean(pd.DataFrame({"implied_volatility_skew": low.astype(float) - high.astype(float)}))


def curvature(
    surface: pd.DataFrame, delta_column: str = "delta", iv_column: str = "implied_volatility"
) -> pd.DataFrame:
    """Proxy smile curvature as wing IV richness versus ATM IV."""
    low = skew(surface, 0.25, 0.50, delta_column, iv_column).iloc[:, 0]
    high = -skew(surface, 0.50, 0.75, delta_column, iv_column).iloc[:, 0]
    return _clean(pd.DataFrame({"implied_volatility_curvature": low + high}))


def term_structure_slope(
    surface: pd.DataFrame,
    near_tenor: float = 30.0,
    far_tenor: float = 90.0,
    tenor_column: str = "tenor_days",
    iv_column: str = "implied_volatility",
) -> pd.DataFrame:
    """Compute implied-volatility term-structure slope between two tenors."""
    near = (
        surface.iloc[(surface[tenor_column].astype(float) - near_tenor).abs().argsort()]
        .groupby(level=0)
        .first()[iv_column]
    )
    far = (
        surface.iloc[(surface[tenor_column].astype(float) - far_tenor).abs().argsort()]
        .groupby(level=0)
        .first()[iv_column]
    )
    return _clean(
        pd.DataFrame({"implied_volatility_term_slope": far.astype(float) - near.astype(float)})
    )


def surface_principal_components(
    surface_matrix: pd.DataFrame, n_components: int = 3
) -> pd.DataFrame:
    """Compute point-in-time scores for leading volatility-surface principal components."""
    centered = surface_matrix - surface_matrix.expanding().mean()
    records: list[dict[str, float]] = []
    for end in range(len(surface_matrix)):
        sample = centered.iloc[: end + 1].dropna(axis=1, how="any")
        if sample.shape[0] < 2 or sample.shape[1] == 0:
            records.append({f"surface_pc_{idx + 1}": np.nan for idx in range(n_components)})
            continue
        _, _, vt = np.linalg.svd(sample.to_numpy(dtype=float), full_matrices=False)
        latest = sample.iloc[-1].to_numpy(dtype=float)
        records.append(
            {
                f"surface_pc_{idx + 1}": float(np.dot(latest, vt[idx])) if idx < len(vt) else np.nan
                for idx in range(n_components)
            }
        )
    return _clean(pd.DataFrame(records, index=surface_matrix.index))


def implied_correlation(
    index_iv: pd.Series, component_ivs: pd.DataFrame, weights: pd.Series
) -> pd.DataFrame:
    """Estimate average implied correlation from index and component variances."""
    w = weights.reindex(component_ivs.columns).astype(float)
    weighted_variance = component_ivs.pow(2).mul(w.pow(2), axis=1).sum(axis=1)
    cross_weight = float(w.sum() ** 2 - w.pow(2).sum())
    correlation = (
        (index_iv.pow(2) - weighted_variance) / component_ivs.pow(2).mean(axis=1) / cross_weight
    )
    return _clean(pd.DataFrame({"implied_correlation": correlation}, index=index_iv.index))


def variance_risk_premium(
    implied_volatility: pd.Series, realized_volatility: pd.Series
) -> pd.DataFrame:
    """Compute variance-risk premium as implied variance less realized variance."""
    return _clean(
        pd.DataFrame(
            {"variance_risk_premium": implied_volatility.pow(2) - realized_volatility.pow(2)},
            index=implied_volatility.index,
        )
    )


def tail_risk_prices(
    surface: pd.DataFrame,
    put_delta: float = -0.10,
    call_delta: float = 0.10,
    delta_column: str = "delta",
    price_column: str = "mid",
) -> pd.DataFrame:
    """Compute relative deep-tail option prices from puts and calls."""
    puts = surface[surface[delta_column] < 0]
    calls = surface[surface[delta_column] > 0]
    put = (
        puts.iloc[(puts[delta_column].astype(float) - put_delta).abs().argsort()]
        .groupby(level=0)
        .first()[price_column]
    )
    call = (
        calls.iloc[(calls[delta_column].astype(float) - call_delta).abs().argsort()]
        .groupby(level=0)
        .first()[price_column]
    )
    return _clean(
        pd.DataFrame(
            {
                "put_tail_risk_price": put.astype(float),
                "call_tail_risk_price": call.astype(float),
                "put_call_tail_richness": put.astype(float) / call.astype(float),
            }
        )
    )


def put_call_relative_richness(
    surface: pd.DataFrame,
    delta_abs: float = 0.25,
    delta_column: str = "delta",
    iv_column: str = "implied_volatility",
) -> pd.DataFrame:
    """Compare put and call IV richness at matched absolute delta."""
    puts = surface[surface[delta_column] < 0]
    calls = surface[surface[delta_column] > 0]
    put = (
        puts.iloc[(puts[delta_column].abs() - delta_abs).abs().argsort()]
        .groupby(level=0)
        .first()[iv_column]
    )
    call = (
        calls.iloc[(calls[delta_column].abs() - delta_abs).abs().argsort()]
        .groupby(level=0)
        .first()[iv_column]
    )
    return _clean(
        pd.DataFrame({"put_call_relative_richness": put.astype(float) - call.astype(float)})
    )


def option_liquidity(
    options: pd.DataFrame,
    bid_column: str = "bid",
    ask_column: str = "ask",
    volume_column: str = "volume",
    open_interest_column: str = "open_interest",
) -> pd.DataFrame:
    """Summarize option spread, volume, and open-interest liquidity by timestamp."""
    midpoint = (options[bid_column].astype(float) + options[ask_column].astype(float)) / 2.0
    data = options.assign(
        _spread=(options[ask_column].astype(float) - options[bid_column].astype(float)) / midpoint
    )
    grouped = data.groupby(level=0)
    return _clean(
        pd.DataFrame(
            {
                "option_spread_mean": grouped["_spread"].mean(),
                "option_volume": grouped[volume_column].sum(),
                "option_open_interest": grouped[open_interest_column].sum(),
            }
        )
    )


def exposure_proxies(
    options: pd.DataFrame,
    underlying_price_column: str = "underlying_price",
    open_interest_column: str = "open_interest",
    delta_column: str = "delta",
    gamma_column: str = "gamma",
    vega_column: str = "vega",
    tenor_column: str = "tenor_days",
) -> pd.DataFrame:
    """Build defensible gamma, vanna, and charm exposure proxies from standard Greeks."""
    notional = (
        options[underlying_price_column].astype(float)
        * options[open_interest_column].astype(float)
        * 100.0
    )
    gamma = notional * options[gamma_column].astype(float)
    vanna = notional * options[vega_column].astype(float) * options[delta_column].astype(float)
    charm = (
        -notional
        * options[delta_column].astype(float)
        / options[tenor_column].astype(float).clip(lower=1.0)
    )
    data = options.assign(_gamma=gamma, _vanna=vanna, _charm=charm)
    grouped = data.groupby(level=0)
    return _clean(
        pd.DataFrame(
            {
                "gamma_exposure_proxy": grouped["_gamma"].sum(),
                "vanna_exposure_proxy": grouped["_vanna"].sum(),
                "charm_exposure_proxy": grouped["_charm"].sum(),
            }
        )
    )
