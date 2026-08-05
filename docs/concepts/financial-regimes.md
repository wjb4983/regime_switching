# Financial regimes

## What a regime is

A financial regime is a deliberately chosen representation of a period in which selected features
of a data-generating process are treated as comparatively stable. Depending on the question, that
may mean a persistent latent state with different return means, volatilities, correlations,
liquidity, or transition behavior; a rules-based risk condition; or a segment between structural
breaks. A useful definition names the variables, horizon, market universe, and decision it serves.

Regimes are **not** directly observed natural kinds, permanent market laws, causal explanations,
guaranteed trading opportunities, or uniquely identifiable labels. “Risk-off” is an interpretation
attached after estimation, not an intrinsic meaning of state 1. Results depend on sampling,
features, scaling, state count, model family, and information available at the time.

## Change points versus recurring states

| Question | Change-point model | Recurring-state model |
|---|---|---|
| Object | Boundary where parameters change | State that may reappear |
| Typical output | Break time, segment ID, alarm score | State probability and/or label |
| Persistence | Segment lasts until another break | Governed by transitions or assignment rule |
| Reuse of labels | Segment 2 need not resemble segment 5 | The same state definition recurs |
| Live use | Only an online detector is live-equivalent | Filtering can be live-equivalent |

PELT run on a complete history answers “where does the full sample divide?” An HMM answers “which
of a fixed collection of latent distributions likely generated each observation?” Mapping several
segments to a common economic state requires an explicit second model; numbering segments does not
make them recurring regimes.

## Appropriate uses

- Compressing multivariate histories into interpretable diagnostics.
- Stress stratification, scenario design, monitoring, and conditional risk estimates.
- Testing whether a predeclared policy behaves differently under estimated conditions.
- Detecting candidate structural breaks for subsequent investigation.
- Producing probabilistic inputs to a risk policy when inference is truly point-in-time.

## Inappropriate uses

- Treating retrospective labels as ground truth or causal mechanisms.
- Feeding smoothed/full-sample states into a historical trading simulation.
- Selecting state count, features, and strategy on the final test period.
- Claiming forecastability from an attractive state-colored chart.
- Assuming the next crisis must resemble a previously learned state.
- Using short, noisy samples to make precise transition or duration claims.
