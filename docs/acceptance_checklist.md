# Research-completeness acceptance checklist

This checklist is the release gate for research claims made with the framework. A checked box
means that the criterion passed against the exact candidate artifacts and that the evidence is
reviewable. It does **not** mean that the capability merely exists in code.

## Evidence contract

For **every** item below, the research record must contain:

- the exact, copy-pasteable verification command and its exit code;
- an explicit timeout in the command (for example, `pytest --timeout=300` or
  `timeout 300s <command>`); a CI job timeout alone is not sufficient;
- the commit SHA, operating system, dependency lock or environment digest, UTC timestamp, and
  identifiers or hashes for the data, features, configuration, model, and result artifacts;
- the captured stdout, stderr, generated report or comparison artifact, and reviewer sign-off;
- a failure or timeout recorded as a failure, never silently retried or waived.

Commands shown below are the minimum automated gates and use a 300-second per-test timeout.
Projects may tighten that limit or add an outer process timeout. If a named test has not yet been
implemented, the corresponding item remains unchecked.

!!! danger "Research-completeness gate"
    **The framework and any study produced with it are not research-complete until every item in
    this document passes and every item has evidence satisfying the contract above.** Partial
    completion must be described as work in progress, not as validated or research-complete.

## Recommended execution order

Run these gates in order: establish reproducibility and provenance first, validate causal data
and inference next, validate the evaluation design and market simulation after that, then assess
results, portability, reporting, and security. This prevents expensive downstream checks from
running on invalid upstream artifacts.

### 1. Reproducibility and provenance

- [ ] **Reproducible seeds.** Every stochastic data split, initializer, sampler, accelerator,
  and model records a seed; two isolated runs with the same inputs and seed reproduce outputs
  within a declared tolerance, while a different seed is exercised to detect accidental
  hard-coding.
  - Command: `python -m pytest --timeout=300 tests/acceptance/test_reproducible_seeds.py`
- [ ] **Versioned data, features, configs, models, and results.** An immutable manifest links
  content hashes or version IDs for every stage; mutation changes the appropriate identity and
  stale downstream artifacts are rejected.
  - Command: `python -m pytest --timeout=300 tests/acceptance/test_artifact_lineage.py`
- [ ] **Local-first execution.** A documented clean-room workflow runs from local files and
  declared dependencies without a hosted service, cloud credential, or implicit network call;
  optional remote integrations fail closed and explain how to proceed locally.
  - Command: `python -m pytest --timeout=300 tests/acceptance/test_local_first.py`
- [ ] **Optional GPU acceleration.** CPU is the supported default; enabling a GPU is explicit,
  records device and library metadata, preserves the output contract within a declared numerical
  tolerance, and cleanly skips or falls back when no GPU is present.
  - Command: `python -m pytest --timeout=300 tests/acceptance/test_optional_gpu.py`

### 2. Point-in-time data and causal inference

- [ ] **Point-in-time data semantics.** Every observation distinguishes event, publication,
  availability, ingestion, and revision times where applicable; an as-of query returns only the
  version knowable at that instant, with boundary and revision cases tested.
  - Command: `python -m pytest --timeout=300 tests/unit/timestamps/test_point_in_time.py tests/acceptance/test_asof_semantics.py`
- [ ] **No future leakage.** Features, normalization, fitting, tuning, imputation, labels, and
  joins cannot access observations whose availability time follows the decision time; deliberate
  future-data canaries cause the test to fail.
  - Command: `python -m pytest --timeout=300 tests/acceptance/test_no_future_leakage.py`
- [ ] **Filtered versus smoothed outputs.** APIs, schemas, plots, and reports label filtered
  (information through time \(t\)) and smoothed (retrospective/full-sample) probabilities
  distinctly; backtests reject smoothed outputs as trading signals.
  - Command: `python -m pytest --timeout=300 tests/acceptance/test_inference_mode_contract.py`
- [ ] **Corporate actions.** Splits, dividends, symbol changes, mergers, delistings, and adjustment
  conventions are timestamped and traceable; prices, volumes, returns, holdings, and cash flows
  remain internally consistent across effective dates.
  - Command: `python -m pytest --timeout=300 tests/acceptance/test_corporate_actions.py`
- [ ] **Survivorship-bias controls.** Universe membership is reconstructed as of each decision
  time and includes inactive, delisted, and failed securities; current constituents cannot leak
  backward, and exclusions plus missing-history policy are reported.
  - Command: `python -m pytest --timeout=300 tests/acceptance/test_survivorship_controls.py`
- [ ] **Realistic option-chain availability.** Historical chains contain only contracts and
  quotes published and actionable at the decision time, honor listing and expiration calendars,
  quote staleness, bid/ask validity, and missing strikes/expiries, and never synthesize unavailable
  contracts without explicit labeling.
  - Command: `python -m pytest --timeout=300 tests/acceptance/test_option_chain_availability.py`

### 3. Validation and execution realism

- [ ] **Purged walk-forward validation.** Splits advance chronologically; training and tuning use
  only preceding information, and overlapping label horizons are purged at every train/validation
  and validation/test boundary. Fold timestamps and sample counts are persisted.
  - Command: `python -m pytest --timeout=300 tests/acceptance/test_purged_walk_forward.py`
- [ ] **Embargo handling.** A configured time- or sample-based embargo is applied after each test
  interval before samples can re-enter training; zero, boundary, irregular-calendar, and
  multi-asset cases match the documented policy.
  - Command: `python -m pytest --timeout=300 tests/acceptance/test_embargo.py`
- [ ] **Execution delays.** Signal observation, order submission, eligibility, and fill timestamps
  are separate; configurable latency prevents same-bar fills unless explicitly justified, and
  market-calendar boundaries are tested.
  - Command: `python -m pytest --timeout=300 tests/acceptance/test_execution_delays.py`
- [ ] **Transaction costs.** Commissions, fees, taxes, borrow costs, financing, and option contract
  multipliers are configurable, applied at the correct event and currency, reconciled to the
  ledger, and included in net performance.
  - Command: `python -m pytest --timeout=300 tests/acceptance/test_transaction_costs.py`
- [ ] **Slippage.** Fills use a documented bid/ask, spread, impact, or participation model that
  depends on side, liquidity, and size; absent or crossed markets follow an explicit policy and
  sensitivity scenarios include an adverse case.
  - Command: `python -m pytest --timeout=300 tests/acceptance/test_slippage.py`

### 4. Evaluation and robustness

- [ ] **Statistical versus economic evaluation separation.** Statistical regime quality and
  predictive evidence are computed independently of P&L, while economic results are net of the
  declared implementation assumptions; reports use separate sections and make no inference of
  tradability from statistical significance alone.
  - Command: `python -m pytest --timeout=300 tests/acceptance/test_evaluation_separation.py`
- [ ] **Simple baselines.** Every candidate is compared out of sample with predeclared trivial and
  domain baselines, including a constant/single-state or persistence baseline and an appropriate
  simple rule; identical data, folds, delays, and costs are used.
  - Command: `python -m pytest --timeout=300 tests/acceptance/test_simple_baselines.py`
- [ ] **Uncertainty-aware outputs.** Outputs expose probabilities, intervals, dispersion, or
  stability estimates rather than labels alone; calibration and coverage are measured out of
  sample, and downstream behavior under low confidence is declared and tested.
  - Command: `python -m pytest --timeout=300 tests/acceptance/test_uncertainty_outputs.py`
- [ ] **Synthetic recovery tests.** Seeded generators with known states, transitions, breaks, and
  misspecification scenarios test parameter/state recovery, false positives, calibration, and
  degradation across declared sample-size and signal-strength tolerances.
  - Command: `python -m pytest --timeout=300 tests/acceptance/test_synthetic_recovery.py`
- [ ] **State-label alignment.** Permutation-invariant metrics or a training-only alignment rule
  handles label switching; mappings are stored per fold, never fitted on held-out outcomes, and
  unmatched or changing state counts have an explicit policy.
  - Command: `python -m pytest --timeout=300 tests/unit/evaluation/test_alignment.py tests/acceptance/test_state_label_alignment.py`
- [ ] **Regime-aware versus regime-agnostic downstream comparisons.** Both strategies use the same
  information set, forecasting/trading rule capacity, folds, execution assumptions, and tuning
  budget except for regime inputs; paired out-of-sample differences and uncertainty are reported.
  - Command: `python -m pytest --timeout=300 tests/acceptance/test_downstream_comparison.py`

### 5. Portability, reporting, and governance

- [ ] **Windows and Linux CI.** The same required acceptance suite passes from clean environments
  on supported Windows and Linux versions, with OS-specific skips forbidden unless documented as
  an unsupported optional capability.
  - Linux command: `python -m pytest --timeout=300 tests/acceptance`
  - Windows command: `py -m pytest --timeout=300 tests\acceptance`
- [ ] **Report generation.** A single documented command rebuilds the report from versioned
  artifacts; tables and figures identify inference mode, sample period, folds, costs, seed, and
  provenance, and missing evidence causes generation to fail rather than produce a partial report.
  - Command: `timeout 300s python -m regime.app.main report --config configs/report/research_report.yaml`
- [ ] **Model cards.** Every released model has a versioned card covering intended use,
  non-intended use, data and feature provenance, inference mode, assumptions, validation design,
  metrics, uncertainty, limitations, ethical/market risks, owners, and update history; report
  artifacts link the exact card version.
  - Command: `python -m pytest --timeout=300 tests/acceptance/test_model_cards.py`
- [ ] **Security and credential redaction.** Tests and generated artifacts contain no live secret;
  logs, configs, exceptions, reports, manifests, caches, and subprocess output redact credential
  values and sensitive query parameters, while secret scanning covers both tracked files and the
  generated research bundle.
  - Command: `python -m pytest --timeout=300 tests/acceptance/test_credential_redaction.py`
  - Bundle scan: `timeout 300s gitleaks detect --no-banner --redact --source .`

## Completion record

The final sign-off must link an evidence index containing one row per checkbox, the exact timed
command, status, artifact hashes, and reviewer. Record the release or study identifier here:

- Candidate identifier: `____________________`
- Evidence index: `____________________`
- Reviewer and UTC sign-off: `____________________`
- All checklist items passed: [ ]

Any code, data, feature, configuration, model, dependency, or result change after sign-off
invalidates affected evidence and reopens the corresponding items.
