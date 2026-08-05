# Repository Tree

## Current root-level structure

```text
.
├── ARCHITECTURE.md
├── DATA_SPEC.md
├── EVALUATION_PLAN.md
├── MODEL_MATRIX.md
├── PLAN.md
├── README.txt
├── REPOSITORY_TREE.md
├── ROADMAP.md
├── pyproject.toml
├── src/
│   └── regime/
│       ├── __init__.py
│       ├── paths.py
│       ├── py.typed
│       ├── backtesting/
│       │   └── __init__.py
│       ├── cli/
│       │   └── __init__.py
│       ├── config/
│       │   └── __init__.py
│       ├── data/
│       │   └── __init__.py
│       ├── datasets/
│       │   └── __init__.py
│       ├── errors/
│       │   └── __init__.py
│       ├── evaluation/
│       │   └── __init__.py
│       ├── experiments/
│       │   └── __init__.py
│       ├── features/
│       │   └── __init__.py
│       ├── logging/
│       │   └── __init__.py
│       ├── models/
│       │   └── __init__.py
│       ├── reporting/
│       │   └── __init__.py
│       ├── synthetic/
│       │   └── __init__.py
│       └── validation/
│           └── __init__.py
└── tests/
    └── test_package_structure.py
```

## Intended directory responsibilities

- `src/regime/data`: canonical schemas, input adapters, point-in-time joins, and data quality checks.
- `src/regime/features`: feature definitions, transform registries, feature metadata, and leakage checks.
- `src/regime/models`: model interfaces and implementations for rules, latent states, change points, clustering, and supervised ML.
- `src/regime/validation`: chronological splitters, purging, embargo, and live-equivalent information-set validation.
- `src/regime/evaluation`: statistical fit, replication fidelity, change-point quality, and economic usefulness metrics.
- `src/regime/backtesting`: strategy policies, positions, trades, costs, slippage, and risk overlays.
- `src/regime/reporting`: Markdown/HTML reports, charts, model cards, and dataset cards.
- `src/regime/experiments`: run manifests, orchestration, config snapshots, and reproducibility helpers.
- `src/regime/synthetic`: synthetic regime and change-point generators for tests.
- `tests`: unit, property, integration, and smoke tests; every test command should include a timeout.

## Planned additions

```text
configs/
├── datasets/
├── experiments/
└── models/
examples/
notebooks/
reports/
schemas/
└── README.md
```

Generated data, local artifacts, caches, and large reports should be ignored by Git unless they are small curated examples needed for tests or documentation.
