# Notebooks

Notebooks are for auditable exploration and visualization after reproducible, configuration-driven
pipeline runs. They are not the source of truth for ingestion, feature construction, model fitting,
evaluation, or reporting.

## Working conventions

- Run the appropriate quick start in `examples/` before opening a notebook. Execute its tasks in the
  documented order; do not recreate or reorder pipeline stages in notebook cells.
- Read immutable artifacts and recorded run metadata. Do not silently modify source data or overwrite
  a registered artifact from a notebook.
- Record the input `run_id`, configuration paths, code commit, data snapshot, and dependency versions
  near the top of the notebook.
- Keep time ordering explicit. Use point-in-time features and filtered probabilities for decision
  analysis; label full-sample smoothing as diagnostic-only.
- Fix random seeds, restart the kernel, and run all cells before sharing an exported result.
- Do not commit large generated notebooks, embedded datasets, credentials, or local reports. Prefer a
  small notebook (or script) plus links to registered artifacts.

Start with one of these workflows:

- [`examples/quickstart_synthetic.md`](../examples/quickstart_synthetic.md) for pipeline validation.
- [`examples/quickstart_equities.md`](../examples/quickstart_equities.md) for equity research.
- [`examples/quickstart_options.md`](../examples/quickstart_options.md) for options research.

## Validation

Any test command recorded in a notebook or its documentation must have a timeout. A suitable
repository check is:

```bash
timeout 120s python -m pytest
```
