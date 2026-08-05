# Inference and assignments

## Filtered versus smoothed probabilities

For latent state \(S_t\), observations \(Y_{1:T}\), and parameters \(\theta\):

\[
\alpha_t(k)=P(S_t=k\mid Y_{1:t},\theta),\qquad
\gamma_t(k)=P(S_t=k\mid Y_{1:T},\theta).
\]

The **filtered** probability uses observations available through time \(t\). It can support a live
decision if parameters and features were also available then. The **smoothed** probability uses
future observations \(Y_{t+1:T}\); it generally looks cleaner and is useful for interpretation,
but is hindsight. One-sided features do not repair leakage caused by smoothed probabilities.

Parameter estimation can add another layer of hindsight: filtering 2018 with parameters fitted on
2025 data is not a 2018-live estimate. Walk-forward evaluation must refit or freeze parameters at
each declared cutoff.

## Online versus offline inference

**Online** inference updates an estimate when each observation becomes available. It should define
data-release latency, revision policy, feature warm-up, update/refit cadence, and signal-to-trade
delay. Online does not necessarily mean fast; it means respecting the contemporaneous information
set.

**Offline** inference may use the entire sample to infer states, boundaries, hyperparameters, or
representations. It is appropriate for historical description, audit, and hypothesis generation.
It must be tagged diagnostic-only if future information influences a past output. An offline model
can later be redesigned for online use, but its original results do not become live-equivalent.

## Hard versus probabilistic assignments

A hard assignment \(\hat S_t=\arg\max_k P(S_t=k\mid\mathcal I_t)\) is compact and convenient for
grouping. It discards uncertainty and can create excessive switching near a decision boundary.
A probability vector preserves ambiguity and supports calibrated policies such as gradual exposure.
Its components must sum to one and its type (`filtered`, `smoothed`, or classifier) must be recorded.

Hard decisions may still be required. Predeclare thresholds, tie handling, minimum confidence,
hysteresis, and abstention rules; apply them inside each validation fold. Do not choose them after
viewing final-period returns.
