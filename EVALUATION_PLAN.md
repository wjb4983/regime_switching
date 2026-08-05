# Evaluation Plan

## Objectives

The evaluation framework determines whether a regime model is statistically coherent, reproducible, live-equivalent, and economically useful for a specified decision policy. These are separate claims and must be reported separately.

## Information-set rules

1. A live-equivalent evaluation can only use values with `available_ts <= decision_ts`.
2. Model parameters must be estimated using data available no later than the training-window cutoff.
3. Hyperparameter tuning must occur inside the training or validation loop, never on the final test set.
4. Smoothed probabilities, final revised data, and full-sample change points are hindsight diagnostics.
5. Any report section using hindsight inputs must be titled and tagged as diagnostic-only.

## Validation methods

- **Fixed holdout:** initial baseline with train, validation, and test periods.
- **Expanding walk-forward:** expands the training window while preserving chronology.
- **Rolling walk-forward:** keeps a fixed-length recent training window for nonstationary settings.
- **Purged k-fold:** removes overlapping label/lookback leakage around fold boundaries.
- **Embargoed split:** adds a buffer after training data before validation or test data.
- **Nested validation:** tunes model choices in inner folds and estimates performance in outer folds.

## Statistical evaluation

- Log likelihood and predictive log likelihood.
- AIC, BIC, and parameter-count-aware fit comparisons.
- Log loss, Brier score, calibration curves, and reliability tables for labeled tasks.
- Confusion matrix, precision, recall, F1, balanced accuracy, and Matthews correlation for hard labels.
- State persistence, transition matrix stability, occupancy, and state duration distributions.
- Label permutation checks for recurring latent states.
- Change-point precision, recall, false alarm rate, detection delay, and segment-length diagnostics.

## Pseudo-label replication evaluation

Pseudo-label replication is evaluated only as fidelity to a source model unless an independent target is also used.

- Agreement, adjusted Rand index, normalized mutual information, and macro F1 against pseudo-labels.
- Probability KL divergence or cross-entropy against pseudo-probabilities.
- State mapping stability across folds.
- Explicit tag for pseudo-label source: filtered, smoothed, clustered, or manually curated.

Smoothed pseudo-labels are allowed for distillation research but cannot be described as live ground truth.

## Economic evaluation

Economic usefulness is tested through a decision policy that converts probabilities into positions. Required metrics include:

- Gross and net return.
- Sharpe, Sortino, volatility, maximum drawdown, Calmar ratio.
- Turnover, transaction costs, slippage, and average holding period.
- Exposure by asset, state, sector, and calendar period.
- Hit rate, payoff ratio, downside capture, and tail losses.
- Benchmark-relative alpha, beta, tracking error, and information ratio.
- Capacity, liquidity, and sensitivity to execution delay.

A model passes economic evaluation only if it remains useful net of costs, across reasonable parameter choices, and under live-equivalent information sets.

## Evaluation gates

1. **Data gate:** schema, timestamp, and duplicate-key checks pass.
2. **Leakage gate:** no unavailable features or smoothed probabilities enter live-equivalent tests.
3. **Fit gate:** model converges and beats naive statistical baselines on validation data.
4. **Robustness gate:** performance is not concentrated in a single fold, asset, or short period.
5. **Economic gate:** strategy improves a predeclared benchmark net of costs and risk constraints.
6. **Reporting gate:** all claims are labeled as live-equivalent or hindsight diagnostic.

## Test policy

All automated checks should use explicit timeouts. Recommended commands:

- `timeout 120 uv run pytest`
- `timeout 120 uv run ruff check .`
- `timeout 120 uv run mypy`

If `uv` is unavailable, use equivalent `python -m pytest`, `ruff`, and `mypy` commands with shell `timeout`.
