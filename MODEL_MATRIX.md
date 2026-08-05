# Model Matrix

| Family | Candidate models | Output concept | Live eligible? | Strengths | Primary risks | Required evaluation |
|---|---|---:|---:|---|---|---|
| Rules | Rolling volatility threshold, drawdown threshold, trend filter | Deterministic state label or score | Yes, if inputs are lagged | Transparent, fast, stable baseline | Arbitrary thresholds, regime oversimplification | Turnover, threshold sensitivity, benchmark comparison |
| Mixture | Gaussian mixture, Bayesian Gaussian mixture | Recurring latent clusters | Usually static, not sequential unless adapted | Simple distributional regimes | Ignores transition dynamics, label switching | BIC/AIC proxy, cluster stability, economic tests |
| Markov switching | Markov regression, Markov autoregression | Filtered and smoothed recurring latent states | Filtered only | Captures persistence and transitions | Local optima, distributional assumptions | Likelihood, AIC/BIC, transition stability, filtered backtest |
| HMM | Gaussian HMM, GMM-HMM, multivariate HMM | Filtered and smoothed recurring latent states | Filtered only | Flexible multivariate state dynamics | State-count sensitivity, label switching | Predictive log likelihood, calibration, state persistence |
| Change-point | PELT, binary segmentation, online CUSUM | Change points and segments | Online variants only | Finds structural breaks | Breaks are not recurring regimes | Detection delay, false alarms, segment diagnostics |
| Clustering | K-means, spectral clustering, HDBSCAN | Retrospective clusters | Usually no | Useful exploratory diagnostics | Hindsight grouping, unstable clusters | Stability, silhouette, adjusted Rand against diagnostics |
| Supervised classical ML | Logistic regression, random forest, gradient boosting, SVM | Class probabilities for supplied labels | Yes, if labels/features are live-designed | Directly optimizes target label | Label quality and imbalance | Log loss, Brier, F1, calibration, economic usefulness |
| Sequence ML | Temporal CNN, LSTM, Transformer encoder | Sequential class or state probabilities | Yes, if trained and inferred point-in-time | Nonlinear temporal features | Overfit, data hunger, opaque behavior | Walk-forward metrics, ablation, cost-aware backtest |
| Replication model | Distilled classifier from HMM or clustering labels | Pseudo-label probabilities | Depends on pseudo-label source; smoothed pseudo-labels are not live truth | Fast approximate inference | Replicates hindsight artifacts | Fidelity metrics plus independent economic validation |

## Probability handling rules

- Filtered probabilities and live classifier probabilities may feed decision policies after availability checks.
- Smoothed probabilities may be used for charts, model interpretation, and pseudo-label generation, but not for deployment claims.
- Change-point methods output event boundaries and segment IDs; they should not be merged into recurring state IDs without an explicit mapping model.
- All model comparisons must state whether they compare true supervised labels, pseudo-label replication, statistical fit, or economic usefulness.

## Minimum model metadata

Each fitted model run records:

- `run_id`, `model_name`, `model_family`, `model_version`
- `train_start_ts`, `train_end_ts`, `validation_start_ts`, `validation_end_ts`
- `state_count` or target label cardinality
- `probability_type`: `filtered`, `smoothed`, `classifier_live`, `retrospective`, or `none`
- `target_type`: `supervised`, `pseudo_label`, `unsupervised_state`, or `change_point`
- `information_set`: `live_equivalent` or `hindsight_diagnostic`
- random seeds, optimizer settings, convergence status, and dependency versions

## Acceptance criteria

- At least one transparent rule baseline and one statistical latent-state baseline are implemented before complex ML models.
- Every probability table declares its probability type.
- Every report separates change-point outputs from recurring latent-state outputs.
- Pseudo-label models are evaluated with replication metrics and separately with any economic test.
