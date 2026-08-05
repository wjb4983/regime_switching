# Synthetic known-state tests

Run these commands in the recommended order. Every command has an explicit timeout,
both for individual tests (through `pytest-timeout`) and for the test process:

1. Fast CI suite: `timeout 180s python -m pytest tests/synthetic -m "not slow" --timeout=120`
2. Static lint: `timeout 120s python -m ruff check tests/synthetic`
3. Research suite: `timeout 600s python -m pytest tests/synthetic -m slow --timeout=300`

The fast suite covers state and change-point recovery, calibration, durations,
transitions, misspecification, nominal-label alignment, online filtering, serialization,
and numerical stability. Tests use fixed seeds and deliberately loose statistical
tolerances to distinguish regressions from ordinary finite-sample variation. The `slow`
suite uses longer samples and additional optimizer restarts; it is intended for scheduled
or pre-release research validation rather than pull-request CI.
