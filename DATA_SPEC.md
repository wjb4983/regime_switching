# Data Specification

## Design principles

- Every table is versioned and auditable.
- Event time and availability time are distinct.
- Live-equivalent evaluations use availability time, not hindsight event time.
- Labels and probabilities are stored separately.
- Change points, segments, recurring states, and supervised classes have separate identifiers.

## Common fields

| Field | Type | Required | Description |
|---|---:|---:|---|
| `dataset_id` | string | Yes | Stable dataset name. |
| `schema_version` | string | Yes | Semantic schema version. |
| `entity_id` | string | Yes | Asset, indicator, or portfolio identifier. |
| `event_ts` | timestamp | Yes | Timestamp the observation describes. |
| `as_of_ts` | timestamp | Yes | Timestamp for the data vintage. |
| `available_ts` | timestamp | Yes | Earliest timestamp the row may be used. |
| `source` | string | Yes | Provider or generation process. |
| `quality_flag` | string | No | Missing, revised, synthetic, suspect, or validated. |

## Price table

| Field | Type | Description |
|---|---:|---|
| `entity_id` | string | Instrument identifier. |
| `event_ts` | timestamp | Bar close or observation time. |
| `open`, `high`, `low`, `close` | float | OHLC prices. |
| `adjusted_close` | float | Corporate-action-adjusted close. |
| `volume` | float | Traded volume. |
| `currency` | string | Price currency. |
| `available_ts` | timestamp | Availability after vendor delay and bar close. |

## Return table

| Field | Type | Description |
|---|---:|---|
| `entity_id` | string | Instrument identifier. |
| `event_ts` | timestamp | Return endpoint. |
| `return_type` | string | Simple, log, excess, close-to-close, intraday, or overnight. |
| `horizon` | string | Return horizon, such as `1d` or `21d`. |
| `value` | float | Return value. |
| `available_ts` | timestamp | Earliest usable timestamp. |

## Feature table

| Field | Type | Description |
|---|---:|---|
| `feature_id` | string | Stable feature name. |
| `entity_id` | string | Entity being described. |
| `event_ts` | timestamp | Feature timestamp. |
| `value` | float/string/bool | Feature value. |
| `lookback_start_ts` | timestamp | First observation used. |
| `lookback_end_ts` | timestamp | Last observation used. |
| `as_of_ts` | timestamp | Data vintage timestamp. |
| `available_ts` | timestamp | Earliest decision timestamp for live use. |
| `feature_version` | string | Transform version. |

## Label table

| Field | Type | Description |
|---|---:|---|
| `label_id` | string | Label definition name. |
| `entity_id` | string | Entity or universe label applies to. |
| `event_ts` | timestamp | Label timestamp. |
| `label_value` | string/int | Class, bucket, or event marker. |
| `label_type` | string | `supervised`, `pseudo_filtered`, `pseudo_smoothed`, or `diagnostic`. |
| `source_run_id` | string | Required for pseudo-labels. |
| `available_ts` | timestamp | Availability for live-supervised labels, or diagnostic timestamp for hindsight labels. |

## Probability table

| Field | Type | Description |
|---|---:|---|
| `run_id` | string | Model run identifier. |
| `entity_id` | string | Entity or universe. |
| `event_ts` | timestamp | State timestamp. |
| `state_id` | string | Recurring latent state or class ID. |
| `probability` | float | Probability in `[0, 1]`. |
| `probability_type` | string | `filtered`, `smoothed`, `classifier_live`, or `retrospective`. |
| `information_set` | string | `live_equivalent` or `hindsight_diagnostic`. |
| `available_ts` | timestamp | Earliest timestamp the probability may be used. |

For each `run_id`, `entity_id`, and `event_ts`, probabilities over recurring states should sum to one unless the model explicitly emits independent event probabilities.

## Change-point table

| Field | Type | Description |
|---|---:|---|
| `run_id` | string | Detector run identifier. |
| `entity_id` | string | Entity or universe. |
| `change_point_ts` | timestamp | Estimated boundary time. |
| `detection_ts` | timestamp | Time when boundary became detectable. |
| `segment_id_before` | string | Previous segment identifier. |
| `segment_id_after` | string | New segment identifier. |
| `confidence` | float | Optional confidence score. |
| `information_set` | string | Online/live-equivalent or hindsight diagnostic. |

Change points are event boundaries. They are not equivalent to recurring states unless a separate mapping is learned and validated.

## Position and trade tables

Positions and trades must include signal timestamp, order timestamp, execution timestamp, costs, slippage, quantity, notional, and portfolio ID. Signal rows must link back to probability `run_id` and decision policy version.

## Validation rules

- Reject rows with `available_ts < event_ts` unless the source explicitly supports prior publication, such as scheduled forecasts.
- Reject duplicate primary keys.
- Reject live-equivalent runs that use `probability_type = smoothed`.
- Require `source_run_id` for pseudo-labels.
- Require split manifests with exact date boundaries and embargo settings.
