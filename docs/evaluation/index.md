# Metrics and statistical tests

No single metric establishes a “correct regime.” Match each metric to a claim and report fold-level
dispersion, sample size, uncertainty, and a baseline.

## Interpretation

| Metric | What it supports | Important limitation |
|---|---|---|
| Predictive log likelihood | Density forecast quality; higher is better | Sensitive to tail specification; compare on identical observations |
| AIC / BIC | Penalized in-sample relative fit; lower is better | BIC's regularity assumptions can be problematic in mixtures; neither proves useful states |
| Log loss | Probabilistic labeled prediction; lower is better | Undefined/large at confident zero probability without clipping policy |
| Brier score | Mean squared probability error; lower is better | Depends on prevalence; decompose or compare to climatology |
| Calibration | Whether predicted frequencies match outcomes | Needs enough independent-ish observations per bin |
| F1 / balanced accuracy | Hard-label discrimination | Ignores probability quality; F1 depends on class/threshold definition |
| ARI / NMI | Partition agreement invariant to label permutation | Agreement with pseudo-labels measures fidelity, not truth |
| Entropy | Assignment uncertainty | Low entropy can mean overconfidence, not correctness |
| Persistence / duration | Temporal coherence | High persistence can be imposed by the model and is not predictive skill |
| Change-point delay / false alarms | Online detector responsiveness | Delay–false-alarm tradeoff must use a tolerance and known/defensible events |
| Sharpe / drawdown | Risk-adjusted return / path loss | Selection, dependence, non-normality, and costs strongly affect inference |

For hard labels, align latent labels inside each training/validation exercise. For probabilistic
outputs, record whether probabilities are filtered or smoothed. Evaluate change points as events,
not as ordinary recurring-state labels.

## Statistical-test assumptions

State the null, alternative, test statistic, sampling unit, and stopping/multiple-testing plan before
examining final results. A paired test requires correctly paired observations. A t-test assumes a
meaningful mean and an approximately normal sampling distribution (or sufficient conditions for an
asymptotic approximation); the ordinary version also assumes independence, while return series are
often autocorrelated and heteroskedastic. IID bootstrap and permutation tests require exchangeability
that time series rarely possess; use justified block/resampling schemes and report block sensitivity.

Diebold–Mariano-style forecast comparisons require aligned loss differentials and appropriate
long-run variance treatment; overlapping horizons induce serial dependence. Likelihood-ratio tests
can have nonstandard null distributions for mixtures or parameters on boundaries. Chi-square
calibration/count tests need adequate expected cell counts. Sharpe comparisons need dependence- and
non-normality-aware uncertainty. These methods still do not fix adaptive model selection.

Control or disclose multiple comparisons across assets, features, state counts, seeds, thresholds,
and strategies. Prefer nested validation, a locked final test, confidence intervals, effect sizes,
and robustness checks over isolated p-values. Statistical significance is neither economic
materiality nor causal evidence.
