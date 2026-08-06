# Performance benchmarks

These tests are deliberately marked both `benchmark` and `slow`, so the default pytest
selection and PR CI do not run them. Run the complete suite explicitly, in this recommended
order:

1. Install the project and development dependencies: `python -m pip install -e '.[dev]'`.
2. Run the benchmark suite with its explicit ten-minute per-test timeout:
   `python -m pytest tests/benchmarks --timeout=600 -m benchmark`.
3. Inspect `experiments/benchmarks/experiments.sqlite3` and the run's
   `artifacts/<run-id>/benchmarks.json` artifact.

Set `REGIME_BENCHMARK_REGISTRY` to redirect the local registry and artifacts. Timings are
intended for local regression comparison on stable hardware, not as universal thresholds.
