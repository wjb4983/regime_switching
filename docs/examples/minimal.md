# Minimal example

This synthetic workflow checks plumbing against a known two-state process. It is deliberately not a
claim about market performance. Install the project first, then execute tasks in this recommended
dependency order from the repository root:

```bash
# 0. Install the package, development tools, and documentation dependencies.
uv sync --extra dev --extra docs

# 1. Generate seeded synthetic observations.
uv run regime synthetic generate --config configs/synthetic/gaussian_hmm.yaml

# 2. Build one-sided features.
uv run regime features build --config configs/features/core_market.yaml

# 3. Establish a transparent baseline.
uv run regime train --config configs/models/rule_volatility_threshold.yaml

# 4. Fit a retrospective clustering comparator.
uv run regime train --config configs/models/kmeans_regime.yaml

# 5. Fit the recurring-state model.
uv run regime train --config configs/models/gaussian_hmm.yaml

# 6. Evaluate filtered outputs under the declared walk-forward design.
uv run regime evaluate --config configs/evaluation/statistical_regime_quality.yaml

# 7. Evaluate a predeclared downstream volatility-targeting policy.
uv run regime evaluate --config configs/evaluation/downstream_vol_targeting.yaml

# 8. Generate the report for the downstream evaluation run ID printed above.
uv run regime report --run-id <RUN_ID> --output artifacts/reports/research_report.html
```

The numbered comments in the example YAML files document the broader artifact order. The current
CLI is config-driven: inspect each emitted run record and artifact rather than assuming a command
performed an undocumented estimator-specific action. Keep `seed: 42`, record dependency versions,
and treat synthetic true states only as generator truth.

For automated checks, always impose a timeout, for example:

```bash
python -m pytest tests/integration/smoke --timeout=180
timeout 120 uv run pytest
timeout 120 uv run mkdocs build --strict
```
