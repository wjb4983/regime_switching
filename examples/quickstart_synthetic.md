# Synthetic quick start

Use this workflow to exercise the full research pipeline against the seeded Gaussian HMM fixture.
Run every command from the repository root, and keep the order below: each stage consumes artifacts
or run metadata produced by an earlier stage.

## Recommended command order

1. Generate the synthetic observations.

   ```bash
   regime synthetic generate --config configs/synthetic/gaussian_hmm.yaml
   ```

2. Build the shared market features.

   ```bash
   regime features build --config configs/features/core_market.yaml
   ```

3. Train the interpretable volatility-threshold baseline first.

   ```bash
   regime train --config configs/models/rule_volatility_threshold.yaml
   ```

4. Train the clustering comparator.

   ```bash
   regime train --config configs/models/kmeans_regime.yaml
   ```

5. Train the Gaussian HMM only after the simpler comparators.

   ```bash
   regime train --config configs/models/gaussian_hmm.yaml
   ```

6. Measure statistical regime quality.

   ```bash
   regime evaluate --config configs/evaluation/statistical_regime_quality.yaml
   ```

7. Evaluate the downstream volatility-targeting policy.

   ```bash
   regime evaluate --config configs/evaluation/downstream_vol_targeting.yaml
   ```

8. Copy the relevant `run_id` from the command output and render its research report.

   ```bash
   regime report --config configs/report/research_report.yaml --run-id <run_id>
   ```

The rule model is intentionally trained before K-means and the HMM so that added model complexity
is compared with a transparent baseline. Preserve the configured seed and retain each command's run
record when using this workflow as a reproducibility check.

## Verification

Test commands must always be bounded by a timeout. For example, after completing the workflow:

```bash
timeout 120s python -m pytest tests/cli/test_quick_start_commands.py
```
