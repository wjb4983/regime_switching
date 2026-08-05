# Roadmap

## Milestone 1: Foundation

- Keep package structure importable.
- Add project-level planning documents.
- Define schema, validation, and probability-output conventions.
- Acceptance criteria: root documentation exists, package imports pass, and tests run with timeouts.

## Milestone 2: Data layer

- Implement schema classes and table validators.
- Add local CSV, Parquet, and DuckDB readers.
- Add timestamp, duplicate-key, and availability checks.
- Acceptance criteria: invalid point-in-time data fails fast with clear errors.

## Milestone 3: Features and labels

- Add deterministic feature functions.
- Add supervised label builders.
- Add pseudo-label builders sourced from filtered, smoothed, or retrospective model outputs.
- Acceptance criteria: label metadata distinguishes supervised labels from pseudo-label replication targets.

## Milestone 4: Baseline models

- Add rule-based, mixture, Markov switching, HMM, change-point, clustering, and supervised ML baselines.
- Standardize model interfaces and probability tables.
- Acceptance criteria: filtered versus smoothed probabilities are emitted separately and tagged.

## Milestone 5: Validation and evaluation

- Implement live-equivalent splitters and embargo logic.
- Add statistical, replication, change-point, and economic metric suites.
- Acceptance criteria: statistical fit and economic usefulness appear in separate scorecards.

## Milestone 6: Backtesting

- Add probability-to-position policies.
- Add costs, slippage, risk constraints, and execution delays.
- Acceptance criteria: backtests reject hindsight-only signals by default.

## Milestone 7: Reporting and experiment management

- Add run manifests, model cards, dataset cards, and comparison reports.
- Add charts for filtered paths, smoothed hindsight paths, change points, recurring states, and economic outcomes.
- Acceptance criteria: reports label every result as live-equivalent or hindsight diagnostic.

## Key risks

- Data leakage through revised data, future features, smoothed probabilities, or unpurged overlapping labels.
- Confusing change points with recurring latent states.
- Over-valuing statistical fit while ignoring transaction costs or decision latency.
- Treating pseudo-label replication as proof of true economic regime discovery.
- State label switching across model runs.
- Overfitting through repeated model selection on the same test interval.
- Optional dependency complexity for deep learning, clustering, distributed execution, and options analytics.

## Mitigations

- Enforce availability timestamps and split manifests.
- Add report sections dedicated to concept distinctions.
- Require naive, rule-based, and statistical baselines before advanced models.
- Use walk-forward tests, purging, embargo, and nested validation.
- Run cost, delay, threshold, and state-count sensitivity analyses.
- Store state-mapping metadata and evaluate state stability.

## Quick-start tasks in recommended order

1. Run package import smoke tests.
2. Implement data schema classes.
3. Add dataset validators.
4. Add local data readers.
5. Implement feature transforms.
6. Implement supervised labels.
7. Implement pseudo-label conversion utilities.
8. Implement validation split manifests.
9. Add rule-based baseline models.
10. Add Markov switching or HMM baseline.
11. Add change-point baseline.
12. Add supervised ML baseline.
13. Standardize probability outputs.
14. Add statistical metrics.
15. Add economic backtest metrics.
16. Add report generation.
17. Add end-to-end reproducibility smoke test.
