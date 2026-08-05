# Regime-switching research

This site explains how to formulate, fit, evaluate, and use financial regime models without
confusing retrospective patterns with tradable information. A regime result is a **model-based
description**, not a discovered law of markets or a promise of abnormal returns.

## Start here

1. Read [what financial regimes are](concepts/financial-regimes.md), especially the distinction
   between recurring states and change points.
2. Choose whether the question requires [online or offline inference](concepts/inference.md).
3. Write down the [model assumptions and estimand](models/formulations.md) before choosing a model.
4. Establish [point-in-time data](data/index.md) and [causal features](features/index.md).
5. Begin with the [minimal example](examples/minimal.md), in its listed execution order.
6. Interpret [metrics and statistical tests](evaluation/index.md) as separate evidence, not one
   universal score.
7. Only then run a [cost-aware, delayed backtest](backtesting/index.md).
8. For publishable work, follow the [reproducible research workflow](examples/reproducible-research.md).

## Claim discipline

Every result should identify its target (state, break, label, or decision), information set,
probability type, fitting window, random seed, and whether the evidence is statistical,
predictive, or economic. Smoothed states and full-sample change points are valuable explanatory
diagnostics, but they are not live signals.

!!! warning
    This project is research software, not investment advice. Regimes can disappear, definitions
    can drift, and a statistically coherent partition can have no economic value after costs.
