# Optional artifact browser

The Streamlit app is a read-only companion to the experiment registry. It does
not run models, mutate runs, or participate in static report generation.

## Quick start

Run these tasks in the recommended order:

1. Generate experiments and their artifacts with the normal package workflows.
2. Install the optional UI: `uv sync --extra app`.
3. Start the browser: `uv run streamlit run src/regime/app/main.py -- --root experiments`.
4. Select one or more runs in the sidebar, then inspect comparisons,
   probabilities, downstream summaries, and options surfaces before downloading
   any registered artifact.

Tabular views accept registered CSV, Parquet, or JSON artifacts. Probability
columns should include `prob` in their names. Options surfaces should provide
`tenor`, `moneyness`, and `implied_volatility` columns. Artifact metadata may set
`view` to values such as `downstream`, `performance`, `option-surface`, or
`probabilities` to make discovery explicit.
