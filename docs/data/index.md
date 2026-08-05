# Point-in-time data

For every record, distinguish event time, publication/availability time, ingestion time, and any
revision vintage. A live-equivalent feature at decision time \(d\) may use a value only when
`available_ts <= d`. Store raw immutable snapshots, source/version identifiers, timezone, calendar,
adjustment method, units, missing-value policy, and universe membership as known then.

## Leakage risks

- Using revised macro data or finalized fundamentals before their original release.
- Survivorship-biased constituents, delisted-security omission, or present-day symbol mappings.
- Back-adjusted prices whose future corporate actions alter past values without a vintage policy.
- Same-bar close features traded at that close; require a realistic execution delay.
- Global scaling, imputation, feature selection, or winsorization before splitting.
- Forward-filled releases before publication, or interpolation that reads a future endpoint.
- Joining on event timestamps when availability timestamps differ.
- Duplicates, timezone conversion, or calendar misalignment that silently shifts observations.

Fit every learned transform within its training fold. Persist row counts, ranges, hashes, schemas,
and availability audits. Synthetic data is useful for pipeline verification and recovery of known
states, but performance on its generating assumptions is not evidence about markets.
