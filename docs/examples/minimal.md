# Minimal example

This synthetic workflow checks plumbing against a known two-state process. It is deliberately not a
claim about market performance. Install the project first, then execute tasks in this recommended
dependency order from the repository root:

```bash
# 0. Install the package, development tools, and documentation dependencies.
uv sync --extra dev --extra docs

# 1. Generate seeded synthetic observations.
uv run regime synthetic generate --config configs/synthetic/gaussian_hmm.yaml

# 2. Ingest them through the same data boundary used by research runs.
uv run regime data ingest --config configs/data/mock_provider.yaml

# 3. Build one-sided features.
uv run regime features build --config configs/features/core_market.yaml

# 4. Establish a transparent baseline.
uv run regime train --config configs/models/rule_volatility_threshold.yaml

# 5. Fit a retrospective clustering comparator.
uv run regime train --config configs/models/kmeans_regime.yaml

# 6. Fit the recurring-state model.
uv run regime train --config configs/models/gaussian_hmm.yaml

# 7. Evaluate filtered outputs under the declared walk-forward design.
uv run regime evaluate --config configs/evaluation/statistical_regime_quality.yaml

# 8. Evaluate a predeclared downstream policy only after statistical checks.
uv run regime evaluate --config configs/evaluation/downstream_vol_targeting.yaml
```

The numbered comments in the example YAML files document the broader artifact order. The current
CLI is config-driven: inspect each emitted run record and artifact rather than assuming a command
performed an undocumented estimator-specific action. Keep `seed: 42`, record dependency versions,
and treat synthetic true states only as generator truth.

For automated checks, always impose a timeout, for example:

```bash
timeout 120 uv run pytest
timeout 120 uv run mkdocs build --strict
```
