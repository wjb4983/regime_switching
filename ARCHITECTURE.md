# Architecture

## System overview

The framework is organized as a reproducible research and evaluation stack for market-regime analysis. It accepts point-in-time market, macro, and engineered feature data; fits multiple model families; emits standardized regime probabilities; validates those probabilities under live-equivalent information constraints; and evaluates both statistical diagnostics and economic outcomes.

```text
raw data -> canonical datasets -> features/labels -> models -> probability outputs
        -> validation splits -> statistical evaluation -> economic backtests -> reports
```

## Package boundaries

- `regime.data`: ingestion, schema validation, calendars, point-in-time joins, and storage adapters.
- `regime.datasets`: curated dataset builders and dataset cards.
- `regime.features`: deterministic feature transforms and leakage checks.
- `regime.models`: statistical, change-point, clustering, supervised, and sequence models.
- `regime.validation`: splitters, embargo logic, walk-forward evaluation, and information-set checks.
- `regime.evaluation`: statistical metrics, replication metrics, calibration, and economic score aggregation.
- `regime.backtesting`: signal policies, portfolio construction, transaction costs, slippage, and risk overlays.
- `regime.reporting`: tables, charts, model cards, diagnostics, and final reports.
- `regime.experiments`: run manifests, config snapshots, experiment orchestration, and reproducibility hooks.
- `regime.synthetic`: controlled data generators for known-regime simulations and edge-case tests.

## Core abstractions

### Dataset contract

A dataset is a versioned table collection with entity keys, event timestamps, availability timestamps, source metadata, and validation rules. Data may be revised, delayed, or restated, so the architecture tracks both the timestamp being measured and the timestamp at which the value became usable.

### Feature contract

A feature table is indexed by `entity_id`, `event_ts`, and `feature_name`, with `value`, `as_of_ts`, `available_ts`, `lookback_start_ts`, `lookback_end_ts`, and `feature_version`. Features cannot depend on observations unavailable at `available_ts`.

### Model contract

A model must expose:

- `fit(train_data, config)`
- `predict(features, information_set)`
- `score(evaluation_data, metrics)`
- `save(path)` and `load(path)`

Every model output must include model family, run ID, probability type, state namespace, training window, inference window, and information-set eligibility.

### Probability contract

Regime probabilities are stored separately from labels and positions. Required distinctions:

- **Filtered probabilities:** `P(state_t | observations_<=t)`. Eligible for live-equivalent signals if all inputs are available at or before the decision time.
- **Smoothed probabilities:** `P(state_t | observations_<=T)` for `T > t`. Not live-equivalent. Eligible for diagnostics, visualization, and pseudo-label generation only.
- **Static classifier probabilities:** `P(label_t | features_available_at_t)`. Live eligibility depends on feature availability and label design.
- **Retrospective segment probabilities:** probabilities or confidence scores derived after a full segment is known. These are hindsight diagnostics unless adapted into an online detector.

## Change points versus recurring latent states

The architecture keeps two namespaces:

- `change_point_id` and `segment_id` for boundary-focused methods.
- `state_id` and `state_version` for recurring latent-state methods.

A change-point detector may say that a new segment began on a date, but it does not automatically prove that the segment is a recurring bull, bear, crisis, or calm state. A latent-state model may assign the same state ID to non-contiguous periods, which is the expected behavior for recurring regimes.

## Statistical fit versus economic usefulness

Model reports have two separate scorecards:

1. **Statistical fit:** log likelihood, AIC, BIC, predictive log loss, Brier score, calibration, state persistence, transition stability, and label agreement.
2. **Economic usefulness:** net returns, Sharpe, Sortino, drawdown, turnover, costs, capacity, exposure, hit rate, tail behavior, and benchmark-relative utility.

A model can have excellent fit and poor economic value if probabilities are not tradable, arrive too late, are unstable under costs, or capture regimes irrelevant to the decision rule. A model can have modest statistical fit and useful economics if it improves risk exposure timing net of costs.

## Supervised labels versus pseudo-label replication

The framework supports two target modes:

- **Supervised labels:** externally defined labels such as recession indicators, realized-volatility buckets, drawdown states, stress events, or committee-labeled regimes.
- **Pseudo-label replication:** labels produced by another model, often smoothed HMM states or clustering assignments.

Replication metrics measure how well one method imitates the pseudo-label source. They do not establish ground truth, causal validity, or live usefulness. Pseudo-labels generated from smoothed probabilities are hindsight products and must be tagged accordingly.

## Live-equivalent information sets versus hindsight diagnostics

The validation layer enforces information-set eligibility. A live-equivalent run may use only observations, features, model parameters, labels, and metadata available at the decision timestamp under the selected execution delay. Hindsight diagnostics may use full-sample smoothed states, final data revisions, and post-period segment boundaries, but reports must label them as diagnostic-only.

## Risk controls

- Prevent leakage through availability timestamps and split manifests.
- Keep smoothed probabilities out of live backtests.
- Require purging and embargo for overlapping labels or lookback windows.
- Store state mappings so state ID permutations do not corrupt comparisons.
- Report sensitivity to costs, delays, rebalance frequency, and state-count choices.
