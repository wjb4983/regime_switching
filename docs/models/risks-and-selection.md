# Model risks and selection

## Identifiability risks

- **Label switching:** permuting latent-state names leaves the likelihood unchanged. Align states by
  a declared statistic within each fold; never score raw numeric IDs across fits.
- **Overlapping emissions:** similar state distributions yield uncertain assignments and unstable
  transitions even when optimization converges.
- **Rare states:** few observations cannot identify their covariance, duration, or transition rows.
- **State splitting:** an oversized \(K\) can divide one economic condition into redundant states;
  an undersized \(K\) merges distinct behavior.
- **Scale and collinearity:** high-variance or redundant features can dominate geometry and produce
  nearly singular covariance estimates.
- **Observational equivalence:** different parameters/model families may explain the same sample.
  Economic names do not resolve statistical non-identifiability.

Use multiple seeded initializations, regularization, occupancy constraints, parameter uncertainty,
fold-wise alignment, and sensitivity analysis. AIC/BIC can inform but cannot prove semantic truth.

## Common failure modes

Non-convergence, local optima, covariance collapse, implausibly rapid switching, a dominant state,
unstable labels across folds, false change alarms under heteroskedasticity, delayed break detection,
and state proliferation are statistical warnings. Data revisions, timestamp errors, missingness
patterns, universe survivorship, and vendor changes can masquerade as regimes. In production,
distribution drift, stale parameters, unseen conditions, and latency cause additional failure.

Define monitors for likelihood/score drift, occupancy, entropy, switching rate, missing data,
feature ranges, convergence, and runtime. Define an abstain/fallback policy before deployment.

## Selecting for the question

Use a rule as the transparent baseline; a change-point method for boundary detection; an HMM or
Markov-switching model when recurring persistence is substantively justified; a mixture/clustering
model for exploratory grouping; and supervised learning only when labels have a defensible creation
and availability process. Compare against naive baselines under identical walk-forward folds.

Do not select solely on in-sample likelihood, visual neatness, or gross Sharpe. Report sensitivity
to state count, seeds, windows, features, costs, and delays, and separate descriptive fit,
out-of-sample prediction, and decision utility.
