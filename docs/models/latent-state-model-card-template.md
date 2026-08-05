# Latent-state model card template

Copy this template for an HMM, Markov-switching, mixture, clustering, rule, or classifier model.
Replace every bracketed instruction; do not delete unfavorable evidence.

## Summary
[Model family, version, owner, intended decision, training period, information set, status.]

## Mathematical definition
[Variables, state space, likelihood/objective, transition/emission or prediction equations.]

## Assumptions
[Distribution, dependence, stationarity, state count, missingness, availability, label semantics.]

## Parameters
[Learned parameters and hyperparameters, constraints, priors/regularization, seed, initialization.]

## Outputs
[Schema and units; hard labels and/or probability vectors; explicitly say filtered, smoothed,
classifier-live, or retrospective; uncertainty and abstention behavior.]

## Strengths
[Evidence-backed advantages relative to named baselines.]

## Limitations
[Unsupported claims, data/domain boundaries, parameter and semantic uncertainty.]

## Failure modes
[Non-convergence, label switching, collapse, drift, rare/unseen states, monitors and fallback.]

## Computational characteristics
[Complexity in \(T,K,D\), memory, measured fit/inference time, hardware, restarts, scaling limits.]

## Suitable applications
[Supported descriptive/predictive/decision uses and explicitly unsuitable uses.]

## References
[Primary methodological sources, implementation/version, data documentation, validation report.]

## Example config
```yaml
model: gaussian_hmm
features: [return_1d, realized_volatility_21d]
n_states: 2
probability_kind: filtered
max_iter: 50
n_init: 3
random_seed: 42
```

## Training command
```bash
uv run regime train --config configs/models/gaussian_hmm.yaml
```

## Evaluation command
```bash
uv run regime evaluate --config configs/evaluation/statistical_regime_quality.yaml
```
