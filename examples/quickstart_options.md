# Options quick start

Use this example to produce regime research inputs for an options study. Regime inference remains an
upstream, underlying-market task: establish its baselines and evaluate its statistical quality before
interpreting options outcomes. Run all commands from the repository root in the order below.

## Recommended command order

1. Generate the seeded underlying-market fixture.

   ```bash
   regime synthetic generate --config configs/synthetic/gaussian_hmm.yaml
   ```

2. Build core features from information available at each observation time.

   ```bash
   regime features build --config configs/features/core_market.yaml
   ```

3. Train the transparent volatility-threshold baseline.

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

6. Evaluate statistical regime quality before using regimes in an options decision rule.

   ```bash
   regime evaluate --config configs/evaluation/statistical_regime_quality.yaml
   ```

7. Evaluate the shared downstream volatility-targeting policy as the economic reference case.

   ```bash
   regime evaluate --config configs/evaluation/downstream_vol_targeting.yaml
   ```

8. Generate the research report with the relevant `run_id` printed by the workflow.

   ```bash
   regime report --config configs/report/research_report.yaml --run-id <run_id>
   ```

Before extending the results to options, validate point-in-time option-chain availability and freeze
contract selection, delta-hedging frequency, fees, slippage, and execution delay. Keep filtered
regime probabilities separate from full-sample smoothed probabilities; the latter are diagnostic and
must not drive a simulated trade.

## Verification

Keep every documented test bounded, for example:

```bash
timeout 120s python -m pytest tests/cli/test_quick_start_commands.py
```
