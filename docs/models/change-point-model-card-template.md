# Change-point model card template

Use this separate template so segment boundaries are not misrepresented as recurring latent states.

## Summary
[Detector, version, owner, parameter being monitored, offline/online status, intended response.]

## Mathematical definition
[Segmentation cost or sequential statistic, penalty/threshold, boundary convention, estimand.]

## Assumptions
[Noise/dependence law, abrupt or gradual change, minimum spacing, stable within-segment parameter.]

## Parameters
[Penalty/threshold, window, minimum segment, tuning process, seed if applicable.]

## Outputs
[Break timestamp/interval, segment ID, alarm score; availability and detection-delay semantics.]

## Strengths
[Types of supported change, interpretability, online behavior where applicable.]

## Limitations
[No inherent recurring-state identity, resolution limits, retrospective dependence.]

## Failure modes
[False alarms, missed/late breaks, endpoint effects, clustered alarms, variance mistaken for mean shift.]

## Computational characteristics
[Best/expected/worst time, memory, measured latency, sample length and hardware.]

## Suitable applications
[Monitoring or historical segmentation; explicitly list unsuitable live/recurring-state claims.]

## References
[Method paper, implementation/version, threshold-calibration and validation reports.]

## Example config
```yaml
model: pelt
cost: l2
penalty: 8.0
minimum_segment_length: 21
inference: offline
```

## Training command
```bash
uv run regime train --config configs/models/<change-point-config>.yaml
```

## Evaluation command
```bash
uv run regime evaluate --config configs/evaluation/<change-point-evaluation>.yaml
```

The placeholder commands become executable only after the referenced project-specific configuration
files are created; do not imply that the repository ships them.
