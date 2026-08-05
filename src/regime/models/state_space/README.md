# Experimental switching state-space models

These CPU reference implementations are **experimental** until the estimators pass
documented synthetic parameter/state recovery thresholds. They are suitable for
prototyping interfaces and inference workflows, not production decisions.

## Quick start (recommended implementation order)

1. Construct or fit `SwitchingLinearDynamicalSystem`; inspect
   `state_space_parameters` and `numerical_diagnostics`.
2. Call `infer` to obtain filtered/smoothed regime probabilities, conditional state
   means/covariances, marginal smoothed states, and log likelihood.
3. Use `SwitchingDynamicFactorModel` for wide observations after validating the
   number and interpretation of factors.
4. Prototype covariate-dependent transition logic with
   `RecurrentSwitchingLinearDynamicalSystem`.
5. Prototype non-geometric dwell times with
   `ExplicitDurationSwitchingLinearDynamicalSystem`.
6. Only after CPU recovery tests pass, install the `gpu` extra and integrate through
   `CuPyBackend`; the current estimators intentionally remain CPU implementations.

`save` writes inspectable JSON (configuration plus numerical parameters), and `load`
rejects model-class mismatches. The IMM smoother is approximate and identifies that
algorithm in its diagnostics.
