# Equities quick start

This example prepares regime outputs for equity research, including the downstream volatility-
targeting evaluation. Run the commands from the repository root in the dependency order shown.
Do not skip the baseline models: they provide the reference needed to interpret the HMM results.

## Recommended command order

1. Generate the deterministic market fixture.

   ```bash
   regime synthetic generate --config configs/synthetic/gaussian_hmm.yaml
   ```

2. Build the core market features used by every model.

   ```bash
   regime features build --config configs/features/core_market.yaml
   ```

3. Train the volatility-threshold baseline.

   ```bash
   regime train --config configs/models/rule_volatility_threshold.yaml
   ```

4. Train the K-means comparator.

   ```bash
   regime train --config configs/models/kmeans_regime.yaml
   ```

5. Train the Gaussian HMM.

   ```bash
   regime train --config configs/models/gaussian_hmm.yaml
   ```

6. Evaluate the statistical quality and stability of the inferred regimes.

   ```bash
   regime evaluate --config configs/evaluation/statistical_regime_quality.yaml
   ```

7. Evaluate the predeclared equity volatility-targeting policy.

   ```bash
   regime evaluate --config configs/evaluation/downstream_vol_targeting.yaml
   ```

8. Use the applicable evaluation `run_id` emitted above to generate the research report.

   ```bash
   regime report --config configs/report/research_report.yaml --run-id <run_id>
   ```

Treat the downstream results as an evaluation of a fixed decision policy, not as permission to tune
on the evaluation period. Review execution delay, costs, exposure constraints, and point-in-time
feature availability before replacing the synthetic fixture with real equity data.

## Verification

Every test command must include a timeout so an unhealthy data or model job cannot block automation:

```bash
timeout 120s python -m pytest tests/cli/test_quick_start_commands.py
```
