# Regime Switching Research Framework

A typed, local-first Python framework for point-in-time financial regime research. It provides
canonical market-data schemas, leakage-aware features and validation, rule and statistical regime
models, downstream equity/options evaluation, experiment provenance, and static reports.

> **Research status:** this project is pre-alpha research software, not investment advice. Verify
> every artifact before relying on it. A statistically coherent regime is not necessarily stable,
> causal, identifiable, or economically useful after execution costs.

## Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Quick start: complete command order](#quick-start-complete-command-order)
4. [CLI command reference](#cli-command-reference)
5. [Configuration and artifacts](#configuration-and-artifacts)
6. [Training models](#training-models)
7. [Evaluation protocol](#evaluation-protocol)
8. [Equity and options backtests](#equity-and-options-backtests)
9. [Optional capabilities](#optional-capabilities)
10. [Development, tests, and quality checks](#development-tests-and-quality-checks)
11. [Cross-platform and reproducibility notes](#cross-platform-and-reproducibility-notes)
12. [Current limitations](#current-limitations)

## Requirements

- Python 3.11 or 3.12 (64-bit recommended).
- Git.
- [`uv`](https://docs.astral.sh/uv/) is recommended for reproducible environments. Standard
  `venv` plus `pip` also works.
- Windows PowerShell 7+ or a POSIX shell on Linux.
- Sufficient storage for Parquet snapshots and experiment artifacts. GPU hardware is optional.

Run every command from the repository root. Paths are handled with `pathlib`; do not embed
OS-specific path separators in YAML.

## Installation

### Recommended: uv

```bash
# Install uv if it is not already available.
python -m pip install uv

# Create/update the environment and install the package plus developer tools.
uv sync --extra dev

# Confirm the installed CLI.
uv run regime --help
```

Install only the optional groups needed by the experiment:

```bash
uv sync --extra dev --extra changepoint --extra clustering
uv sync --extra optimization --extra tracking
uv sync --extra options
uv sync --extra deep
uv sync --extra transformers
uv sync --extra app
uv sync --extra distributed
```

`gpu` installs a CUDA 12 CuPy wheel and is intentionally excluded from normal setup. Confirm the
machine's driver/toolkit compatibility before running:

```bash
uv sync --extra gpu
```

To install all extras (large download; CUDA compatibility still applies):

```bash
uv sync --all-extras
```

### Alternative: venv and pip

Linux/macOS shell:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
regime --help
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
regime --help
```

The installed `regime` entry point is the supported interface. With uv, prefix commands below with
`uv run`.

## Quick start: complete command order

The configurations are numbered by dependency. Keep this order so every stage receives the
artifacts produced upstream and simple baselines are established before complex models.

```bash
# 1. Generate reproducible observations with known latent states.
uv run regime synthetic generate --config configs/synthetic/gaussian_hmm.yaml --no-resume

# 2. Ingest through the vendor-neutral mock provider and create a mock option chain.
uv run regime data ingest --config configs/data/mock_provider.yaml --no-resume

# 3. Build causal, warm-up-aware market features.
uv run regime features build --config configs/features/core_market.yaml --no-resume

# 4. Train the transparent deterministic baseline.
uv run regime train --config configs/models/rule_volatility_threshold.yaml --no-resume

# 5. Train the clustering comparator.
uv run regime train --config configs/models/kmeans_regime.yaml --no-resume

# 6. Train the Gaussian HMM.
uv run regime train --config configs/models/gaussian_hmm.yaml --no-resume

# 7. Train the heavy-tailed HMM using the same information set.
uv run regime train --config configs/models/student_t_hmm.yaml --no-resume

# 8. Create/record the purged walk-forward evaluation schedule.
uv run regime evaluate --config configs/validation/walk_forward_daily.yaml --no-resume

# 9. Compare regime quality using filtered, never smoothed, live-equivalent probabilities.
uv run regime evaluate --config configs/evaluation/statistical_regime_quality.yaml --no-resume

# 10. Compare the fixed downstream policy with and without regime probabilities.
uv run regime evaluate --config configs/evaluation/downstream_vol_targeting.yaml --no-resume

# 11. Validate a tuning search space, then execute/resume its trials.
uv run regime tune --config configs/tuning/hmm_search.yaml --no-resume

# 12. Inspect a registered experiment group (replace the name with one in the local store).
uv run regime compare --experiment-group daily_market_regimes

# 13. Render a run report. Copy RUN_ID from a preceding command's JSON output.
uv run regime report --run-id <RUN_ID> --output artifacts/reports/<RUN_ID>.html
```

Omit `--no-resume` after validating a workflow if interrupted work should resume. The default is
`--resume`. Each command emits JSON containing `status`, `operation`, and `run_id`.

## CLI command reference

```bash
uv run regime --help
uv run regime data --help
uv run regime data ingest --help
uv run regime features --help
uv run regime features build --help
uv run regime synthetic --help
uv run regime synthetic generate --help
uv run regime train --help
uv run regime evaluate --help
uv run regime tune --help
uv run regime compare --help
uv run regime report --help
```

Available operations and required arguments:

| Operation | Command | Purpose |
|---|---|---|
| Synthetic data | `regime synthetic generate --config PATH [--resume/--no-resume]` | Register a seeded synthetic-data run. |
| Ingestion | `regime data ingest --config PATH [--resume/--no-resume]` | Register provider ingestion from canonical YAML. |
| Features | `regime features build --config PATH [--resume/--no-resume]` | Register a point-in-time feature build. |
| Training | `regime train --config PATH [--resume/--no-resume]` | Register model training. |
| Evaluation | `regime evaluate --config PATH [--resume/--no-resume]` | Register validation, statistical, or downstream evaluation. |
| Tuning | `regime tune --config PATH [--resume/--no-resume]` | Validate and register a conditional search space. |
| Comparison | `regime compare --experiment-group NAME` | Print runs/results in a registered group. |
| Reporting | `regime report --run-id ID [--output PATH]` | Build self-contained HTML for a registered run. |

Commands return exit code `2` for invalid configuration, `3` for missing files/runs, and `1` for
framework or unexpected operation failures. Output and error records are JSON and redact likely
credential values.

## Configuration and artifacts

YAML files are the source of truth. Copy an example instead of editing it in place:

```bash
cp configs/models/gaussian_hmm.yaml configs/models/my_gaussian_hmm.yaml
uv run regime train --config configs/models/my_gaussian_hmm.yaml --no-resume
```

On PowerShell use:

```powershell
Copy-Item configs/models/gaussian_hmm.yaml configs/models/my_gaussian_hmm.yaml
uv run regime train --config configs/models/my_gaussian_hmm.yaml --no-resume
```

Important configuration directories:

- `configs/data/`: providers, source snapshots, timestamp fields, and output locations.
- `configs/synthetic/`: seeded generators and known-state process parameters.
- `configs/features/`: input columns, lookbacks, warm-up handling, and transforms.
- `configs/models/`: estimator family, features, state count, initialization, regularization, seed.
- `configs/validation/`: windows, purge/embargo, execution delay, and refit cadence.
- `configs/evaluation/`: predictions, probability kind, metrics, and downstream policy.
- `configs/tuning/`: conditional search spaces.
- `configs/backtesting/`: execution, exposure, liquidity, fee, and slippage assumptions.
- `configs/report/`: report inputs and output format.

`configs/tuning.example.yaml` is retained as a legacy flat tuning example; new work should use the
structured search spaces under `configs/tuning/`, such as `configs/tuning/hmm_search.yaml`.

The current API-only backtest assumptions are recorded in
`configs/backtesting/equity_vol_targeting.yaml` and
`configs/backtesting/options_delta_hedged.yaml`. The proposed full research-report assembly is in
`configs/report/research_report.yaml`; the currently implemented `regime report` command instead
accepts `--run-id` and `--output` directly, as shown above.

Default local state is written under `experiments/`: SQLite stores run metadata and each run has
hashed configuration/manifest artifacts. Research outputs configured by examples live under
`artifacts/`. Both locations should be treated as generated data and kept out of commits.

Before using vendor data, implement a provider adapter under `regime.data.providers`, map source
fields to canonical schemas, preserve event/publication/vendor-received/effective timestamps, use a
permanent security identifier, and snapshot raw immutable Parquet before adjustment. Credentials
belong in environment variables or an ignored local secret file and must never appear in YAML,
logs, reports, notebooks, or command history.

## Training models

Start with the same feature set, training period, refit cadence, seed policy, and validation folds for
all comparators. Never compare models trained on different information sets as if they were peers.

Current runnable example configurations are:

```bash
uv run regime train --config configs/models/rule_volatility_threshold.yaml --no-resume
uv run regime train --config configs/models/kmeans_regime.yaml --no-resume
uv run regime train --config configs/models/gaussian_hmm.yaml --no-resume
uv run regime train --config configs/models/student_t_hmm.yaml --no-resume
```

The default quick start intentionally stops at these four ordered comparators. Extended
examples for every other registered implementation are grouped by family under
`configs/models/{probabilistic,clustering,jump,econometric,state_space,supervised,deep,transformers}`;
optional-backend examples should be run only after installing their declared extra.

The package also contains APIs for change-point, jump, econometric, supervised, state-space, deep,
and transformer/foundation-embedding models. These families do not all have production-ready CLI
configurations. Use them only after checking their class contract and optional backend availability.
Unsupported operations such as probabilities, smoothing, or transition matrices must raise an
explicit unsupported-operation error; do not manufacture them for model comparability.

For every fit, retain: code commit and dirty status, Python/dependency/platform/hardware versions,
random seeds, dataset/feature/config hashes, fit cutoff, training and validation periods,
optimization status, warnings, runtime, and serialized model. Align states across refits using state
statistics/distributional distance (for example Hungarian matching), never raw integer labels.

## Evaluation protocol

### Required order

1. Freeze the research question, universe, sample, labels/targets, costs, and primary metric.
2. Snapshot point-in-time raw data, including revisions, delistings, actions, and option availability.
3. Generate chronological outer folds before preprocessing.
4. Within each fold, fit scalers, imputers, PCA/surfaces, selectors, encoders, and calibration only on
   its training window.
5. Purge observations whose labels overlap the test interval and apply the configured embargo.
6. Train the no-regime and simple-rule baselines before advanced models.
7. Produce online **filtered** probabilities using information available at decision time.
8. Align state identities across initializations/refits before aggregating state metrics.
9. Evaluate statistical fit, calibration, persistence, stability, delay, and false alarms.
10. Freeze downstream policy parameters, apply execution delay and conservative costs, then compare
    `f(x_t)` against `f(x_t, p(z_t | F_t))` on identical folds.
11. Run predeclared delay, cost, refit-frequency, state-count, and crisis/holdout sensitivity tests.
12. Report negative and failed runs and correct for multiple testing across experiment grids.

Run the included evaluation configurations:

```bash
uv run regime evaluate --config configs/validation/walk_forward_daily.yaml --no-resume
uv run regime evaluate --config configs/evaluation/statistical_regime_quality.yaml --no-resume
uv run regime evaluate --config configs/evaluation/downstream_vol_targeting.yaml --no-resume
```

Statistical evaluation should include out-of-sample log/predictive scores, Brier/calibration metrics
where labels are defined, QLIKE/CRPS where applicable, occupancy, durations, entropy, switching,
separability, transition and rolling-refit stability, rare-state sample size, and synthetic recovery.
Use block-aware uncertainty and comparison tests only when their assumptions hold.

Economic evaluation must report gross and net return, volatility, Sharpe/Sortino/Calmar, drawdown,
VaR/expected shortfall, turnover, exposure, beta/factor loadings, costs, capacity/capital use, and
conditional performance. A regime model adds value only if the predeclared out-of-sample metric
improves versus no-regime and simple-rule baselines, survives uncertainty/multiplicity controls, is
stable across folds and reasonable sensitivities, and remains useful after costs. In-sample
likelihood or attractive smoothed-state charts are not sufficient.

Smoothed probabilities use future observations. They are diagnostic-only and must never feed a
live-equivalent forecast, allocation, hedge, execution decision, or simulated trade.

## Equity and options backtests

The backtest engines are currently Python APIs; the YAML files document assumptions but there is no
`regime backtest` CLI command. Do not claim the `evaluate` command executes those engines until a
worker is wired to it.

Minimal equity invocation:

```python
import pandas as pd
from regime.backtesting.equity import EquityBacktestConfig, run_equity_backtest

returns = pd.read_parquet("artifacts/features/asset_returns.parquet")
weights = pd.read_parquet("artifacts/evaluation/target_weights.parquet")
result = run_equity_backtest(
    returns,
    weights,
    config=EquityBacktestConfig(
        execution_delay=1,
        transaction_cost_bps=1.0,
        slippage_bps=1.0,
        volatility_target=0.10,
    ),
)
print(result.metrics)
```

Minimal options invocation:

```python
import pandas as pd
from regime.backtesting.options import OptionBacktestConfig, run_options_backtest

chain = pd.read_parquet("artifacts/data/mock_options.parquet")
prices = pd.read_parquet("artifacts/data/underlying_prices.parquet")["close"]
result = run_options_backtest(
    chain,
    prices,
    config=OptionBacktestConfig(
        strategy="delta_hedged",
        execution_delay=1,
        target_tenor_days=30,
        option_fee_per_contract=0.65,
        hedge_slippage_bps=1.0,
    ),
)
print(result.metrics)
```

Options studies must additionally validate quote timestamps, stale/crossed markets, liquidity,
corporate-action-adjusted contracts, rates/dividends/forwards, no-arbitrage, surface interpolation,
bid/ask execution, multiplier/margin, assignment/exercise, hedge timing, and realistic historical
chain availability. Never reconstruct unavailable chains from today's listings.

## Optional capabilities

```bash
# Hyperparameter optimization (Optuna backend).
uv sync --extra optimization
uv run regime tune --config configs/tuning/hmm_search.yaml --no-resume

# Local MLflow tracking library (no cloud service required).
uv sync --extra tracking

# Read-only artifact browser.
uv sync --extra app
uv run streamlit run src/regime/app/main.py

# Deep sequence models.
uv sync --extra deep

# Transformer encoders/foundation-model adapters.
uv sync --extra transformers

# Optional change-point and density-clustering backends.
uv sync --extra changepoint --extra clustering
```

Keep foundation models as representation generators unless an aligned out-of-sample comparison
justifies end-to-end interpretation. Cache embeddings with model revision, preprocessing, cutoff,
and data hashes. Deep/GPU results must record device, precision, deterministic settings, and backend
versions.

## Development, tests, and quality checks

Every test/check command is bounded. On Linux, `timeout` limits the complete process; on Windows,
pytest's configured `pytest-timeout` plugin limits individual tests.

Recommended order (fast feedback first):

```bash
# Linux
 timeout 120s uv run ruff format --check .
 timeout 120s uv run ruff check .
 timeout 180s uv run mypy
 timeout 300s uv run pytest
 timeout 180s uv run pytest tests/unit
 timeout 180s uv run pytest tests/integration/smoke
 timeout 300s uv run pytest tests/synthetic
 timeout 300s uv run pytest --cov=regime --cov-report=term-missing
```

PowerShell equivalents:

```powershell
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest --timeout=30
uv run pytest tests/unit --timeout=30
uv run pytest tests/integration/smoke --timeout=30
uv run pytest tests/synthetic --timeout=30
uv run pytest --cov=regime --cov-report=term-missing --timeout=30
```

The default pytest configuration excludes `slow`, `benchmark`, `gpu`, and `vendor`. Run them
explicitly and with appropriate limits:

```bash
timeout 900s uv run pytest -m slow --timeout=120
timeout 900s uv run pytest -m benchmark --timeout=120
timeout 900s uv run pytest -m gpu --timeout=120
timeout 900s uv run pytest -m vendor --timeout=120
```

Do not run vendor tests without isolated test credentials and a cost/rate-limit budget. Never use
live vendor responses as deterministic CI fixtures.

## Cross-platform and reproducibility notes

- Use spawn-safe top-level worker functions and `if __name__ == "__main__":` for multiprocessing;
  Windows uses spawn and cannot rely on fork state.
- Pass paths as CLI arguments/YAML values and resolve with `pathlib`; avoid shell glob assumptions.
- Seed Python/NumPy/scikit-learn/PyTorch components and record all seeds. Repeated initialization is
  still required because deterministic seeding does not remove estimator instability.
- Preserve immutable raw snapshots and write derived Parquet atomically. Hash dataset manifests,
  feature definitions, model configs, and artifacts.
- Store timestamps as timezone-aware UTC plus explicit market session/calendar metadata. Availability
  time, not event date alone, determines whether a row can enter a feature.
- Set credentials in provider-specific environment variables. Logs redact common secret fields, but
  redaction is defense in depth rather than permission to put secrets in configs.
- Use `--resume` for interrupted runs; inspect checkpoint and manifest hashes before accepting output.
- Keep notebooks exploratory. Reproducible claims must be executable through package APIs or the CLI
  from versioned configuration.

## Current limitations

The CLI surface and experiment registry are implemented, but the generic config commands currently
validate/load YAML and register manifests unless a workflow worker is attached. In particular, the
example commands should not yet be interpreted as proof that files named in `output:` were computed.
The equity/options engines are API-only, report generation summarizes registered artifacts, and many
advanced model families are research adapters or reference implementations rather than hardened
estimators. Verify artifact existence, schema, hashes, convergence, and numerical results after every
stage. Production ingestion, full orchestration, and vendor-specific adapters remain integration work.
