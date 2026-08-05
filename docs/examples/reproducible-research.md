# Reproducible research example

## Protocol

1. **Predeclare** the estimand, universe, period, state interpretation, baselines, decision policy,
   metrics, tests, costs, and exclusion rules.
2. **Lock data** by source/version, availability timestamp, schema, and content hash; preserve raw
   inputs and record revisions.
3. **Freeze environments** with `uv.lock`, Python/platform details, and dependency versions.
4. **Create splits first** using chronological outer test folds, inner tuning folds, purge/embargo,
   and an explicit execution delay.
5. **Build features per fold**, fitting every transformation only on the fold's training data.
6. **Fit baselines before complex models**, using fixed seeds plus documented repeated starts.
7. **Generate point-in-time predictions** and store fit cutoff, prediction creation time, probability
   type, configuration hash, code commit, convergence, and runtime.
8. **Evaluate statistical claims**, including label alignment, uncertainty, stability, and failed runs.
9. **Evaluate the frozen policy** net of conservative costs; run predeclared delay/cost sensitivity.
10. **Render and archive** tables, figures, logs, configs, hashes, environment, and machine-readable
    predictions. Mark any smoothing/full-sample analysis diagnostic-only.
11. **Re-run from a clean checkout** and compare artifact hashes or explain deterministic numerical
    tolerances.

## Suggested experiment record

```yaml
experiment_id: hmm_synthetic_v1
code_commit: <full-git-sha>
data:
  snapshot_id: synthetic_gaussian_hmm_seed_42
  sha256: <sha256>
  availability_policy: event_time_plus_one_bar
split:
  config: configs/validation/walk_forward_daily.yaml
model:
  config: configs/models/gaussian_hmm.yaml
  probability_kind: filtered
seeds: [42]
environment:
  python: "3.11"
  lockfile: uv.lock
claims:
  primary_metric: predictive_log_score
  economic_metric: net_sharpe_ratio
diagnostic_only: [smoothed_state_plot]
```

## Reproduction commands

Run the [minimal workflow](minimal.md) in its numbered order, then run checks in this recommended
order so fast structural failures occur before the full suite:

```bash
timeout 120 uv run ruff check .
timeout 120 uv run mypy
timeout 120 uv run pytest
timeout 120 uv run mkdocs build --strict
git rev-parse HEAD
sha256sum <critical-inputs-and-artifacts>
```

Publish negative and inconclusive outcomes, all attempted model families and hyperparameter ranges,
and deviations from the protocol. A notebook alone is not a reproducible record; keep orchestration
in configurations/commands and use notebooks only for auditable exploration.
