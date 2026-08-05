"""Cross-sectional market state feature builders."""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.replace([np.inf, -np.inf], np.nan)


def breadth(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute share of assets advancing and cross-sectional net advance score."""
    advancing = returns.gt(0.0).mean(axis=1)
    declining = returns.lt(0.0).mean(axis=1)
    return _clean(
        pd.DataFrame(
            {"breadth_advancing": advancing, "breadth_net": advancing - declining},
            index=returns.index,
        )
    )


def dispersion(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-sectional return dispersion and interquartile range."""
    return _clean(
        pd.DataFrame(
            {
                "dispersion_std": returns.std(axis=1),
                "dispersion_iqr": returns.quantile(0.75, axis=1) - returns.quantile(0.25, axis=1),
            },
            index=returns.index,
        )
    )


def concentration(weights: pd.DataFrame) -> pd.DataFrame:
    """Compute Herfindahl and top-five concentration from asset weights."""
    absolute = weights.abs()
    normalized = absolute.div(absolute.sum(axis=1), axis=0)
    return _clean(
        pd.DataFrame(
            {
                "concentration_hhi": normalized.pow(2).sum(axis=1),
                "concentration_top5": normalized.apply(
                    lambda row: float(row.nlargest(min(5, len(row))).sum()), axis=1
                ),
            },
            index=weights.index,
        )
    )


def factor_returns(asset_returns: pd.DataFrame, exposures: pd.DataFrame) -> pd.DataFrame:
    """Estimate daily factor returns with cross-sectional least squares exposures."""
    rows: list[pd.Series] = []
    factor_names = list(exposures.columns)
    for timestamp, row in asset_returns.iterrows():
        valid = row.dropna()
        aligned = exposures.reindex(valid.index).dropna()
        valid = valid.reindex(aligned.index)
        if aligned.empty:
            rows.append(pd.Series(index=factor_names, dtype=float, name=timestamp))
            continue
        coefficients = np.linalg.lstsq(
            aligned.to_numpy(dtype=float), valid.to_numpy(dtype=float), rcond=None
        )[0]
        rows.append(pd.Series(coefficients, index=factor_names, name=timestamp))
    return _clean(pd.DataFrame(rows).add_prefix("factor_return_"))


def factor_crowding(exposures: pd.DataFrame, weights: pd.Series | None = None) -> pd.DataFrame:
    """Proxy factor crowding with absolute weighted exposure concentration."""
    asset_weights = (
        pd.Series(1.0, index=exposures.index)
        if weights is None
        else weights.reindex(exposures.index).fillna(0.0)
    )
    weighted = exposures.mul(asset_weights.abs(), axis=0)
    scores = weighted.abs().sum(axis=0) / asset_weights.abs().sum()
    return _clean(pd.DataFrame([scores.add_prefix("factor_crowding_")]))


def residual_volatility(
    asset_returns: pd.DataFrame,
    factor_return_frame: pd.DataFrame,
    betas: pd.DataFrame,
    window: int = 63,
) -> pd.DataFrame:
    """Compute rolling volatility of residuals after static factor betas."""
    common_factors = [column for column in betas.columns if column in factor_return_frame.columns]
    fitted = pd.DataFrame(index=asset_returns.index, columns=asset_returns.columns, dtype=float)
    for asset in asset_returns.columns:
        fitted[asset] = factor_return_frame[common_factors].dot(betas.loc[asset, common_factors])
    residuals = asset_returns - fitted
    return _clean(residuals.rolling(window).std().add_prefix("residual_volatility_"))


def correlation_distribution(returns: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Summarize rolling pairwise correlation distributions."""
    records: list[dict[str, float]] = []
    for end in range(len(returns)):
        sample = returns.iloc[max(0, end - window + 1) : end + 1]
        corr = sample.corr().to_numpy(dtype=float)
        upper = corr[np.triu_indices_from(corr, k=1)]
        records.append(
            {
                "correlation_mean": float(np.nanmean(upper)),
                "correlation_std": float(np.nanstd(upper)),
                "correlation_p90": float(np.nanpercentile(upper, 90)),
            }
        )
    return _clean(pd.DataFrame(records, index=returns.index))


def leading_covariance_eigenvalues(
    returns: pd.DataFrame, window: int = 63, n_components: int = 3
) -> pd.DataFrame:
    """Compute leading eigenvalue shares of rolling covariance matrices."""
    records: list[dict[str, float]] = []
    for end in range(len(returns)):
        sample = returns.iloc[max(0, end - window + 1) : end + 1].dropna(axis=1, how="any")
        if sample.shape[1] == 0:
            records.append(
                {f"covariance_eigenvalue_share_{idx + 1}": np.nan for idx in range(n_components)}
            )
            continue
        eigenvalues = np.linalg.eigvalsh(sample.cov().to_numpy(dtype=float))[::-1]
        total = eigenvalues.sum()
        records.append(
            {
                f"covariance_eigenvalue_share_{idx + 1}": float(eigenvalues[idx] / total)
                if idx < len(eigenvalues) and total
                else np.nan
                for idx in range(n_components)
            }
        )
    return _clean(pd.DataFrame(records, index=returns.index))
