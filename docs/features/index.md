# Causal features

Each feature needs a formula, source columns, lookback, lag, availability rule, warm-up behavior,
units, and expected missingness. At time \(t\), a one-sided rolling feature may use only observations
available by \(t\). Decide whether a close-derived feature can inform a next-bar order, then encode
that lag explicitly.

Standardizers, PCA, imputers, encoders, feature selection, and outlier thresholds are model
parameters: fit them on the training window and apply without refitting to validation/test data.
Avoid centered windows, negative shifts, full-sample normalization, backward filling, and labels
whose forward horizon overlaps a validation boundary. Purge the overlap and embargo subsequent
samples where appropriate.

Check feature distributions by fold and timestamp, compare feature and source availability, test
warm-up edges, and perturb delays by one or more bars. A feature that only works with zero latency or
final revised inputs is not a robust live feature.
