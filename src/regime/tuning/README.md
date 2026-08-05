# Local-first tuning

Optuna is the recommended optimizer. Studies use a local SQLite database by
default, so completed trials survive interruption and the same study name can
resume later. SQLite supports local parallel workers (`n_jobs`); use a shared
Optuna-compatible database when coordinating processes on multiple machines.

## Quick start (recommended order)

1. Install the integration: `pip install -e '.[optimization]'`.
2. Copy `configs/tuning.example.yaml` and define parent parameters before their
   conditional children.
3. Load it with `SearchSpace.from_yaml(...)`.
4. Define an objective accepting `(params, trial)`. Call `trial.report(...)` and
   raise `optuna.TrialPruned` for custom pruning, or use
   `nested_validation_objective` to do both over leakage-aware inner splits.
5. Create `StudyConfig`; choose `algorithm="bayesian"` (recommended TPE) or
   `"random"`, and set `n_jobs`, `patience`, trial limit, and timeout.
6. Call `optimize(..., registry=ExperimentStore(...))`. Re-run the same study
   name and storage path after interruption to resume.
7. Repeat the complete optimization with independent seeds and pass outcomes to
   `stability_analysis`; persist its report with `save_stability`.

`MetricObjective` maps arbitrary model/backtest output to named statistical or
economic metrics. Multiple metrics return a tuple matching `directions`. Its
optional constraints store signed violations (`<= 0` is feasible) as trial
metadata, suitable for constrained sampler hooks and audit trails. Constraints
do not silently scalarize objectives: selection remains explicit and
multi-objective Pareto trials remain available from Optuna.
