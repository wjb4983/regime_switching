"""Public, lazy registry of runnable regime models.

The registry deliberately stores import paths instead of imported classes.  Listing
models therefore remains safe in a core-only installation, while creation produces
an actionable optional-extra error.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from regime.models.base import RegimeModelConfig


class ModelRegistryError(ValueError):
    """Raised for unknown models, invalid settings, or unavailable extras."""


@dataclass(frozen=True)
class ModelSpec:
    """Descriptor for one constructible model implementation."""

    name: str
    aliases: tuple[str, ...]
    model: str
    config: str
    capabilities: frozenset[str]
    optional_dependency_group: str | None = None
    dependency_modules: tuple[str, ...] = ()
    factory: Callable[[type[Any], Any], Any] | None = None
    tunable_parameters: frozenset[str] = frozenset()

    @property
    def model_class(self) -> type[Any]:
        """Resolve the implementation lazily."""
        return _resolve(self.model)

    @property
    def config_class(self) -> type[RegimeModelConfig]:
        """Resolve the typed configuration class lazily."""
        return _resolve(self.config)


_STANDARD_ALIASES = {
    "state_count": "n_states",
    "seed": "random_seed",
    "name": "model_name",
}
_OPERATIONAL_FIELDS = {
    "model",
    "features",
    "input",
    "output",
    "fit_parameters",
    "timestamp_column",
    "missing_values",
    "missing_value_policy",
    "fit_cutoff",
    "minimum_observations",
    "min_observations",
}


def _resolve(path: str) -> Any:
    module, _, member = path.rpartition(":")
    return getattr(importlib.import_module(module), member)


def _spec(
    name: str,
    model: str,
    config: str,
    *,
    aliases: tuple[str, ...] = (),
    capabilities: tuple[str, ...] = ("fit", "predict"),
    extra: str | None = None,
    dependencies: tuple[str, ...] = (),
    factory: Callable[[type[Any], Any], Any] | None = None,
    tunable: tuple[str, ...] = ("n_states", "random_seed"),
) -> ModelSpec:
    return ModelSpec(
        name,
        aliases,
        model,
        config,
        frozenset(capabilities),
        extra,
        dependencies,
        factory,
        frozenset(tunable),
    )


def _transformer_head_factory(model_class: type[Any], config: Any) -> Any:
    """Build a local transformer encoder and a contract-compatible regime head."""
    encoder_class = _resolve("regime.models.transformers.encoders:TimeSeriesTransformerEncoder")
    encoder = encoder_class(config)
    name = model_class.__name__
    if name in {"TransformerHMMHead", "TransformerHSMMHead"}:
        head_config = _resolve(f"{P}:ProbabilisticHMMConfig")(
            model_name=config.model_name,
            n_states=config.n_states,
            random_seed=config.random_seed,
        )
    elif name == "TransformerClusteringHead":
        head_config = _resolve(f"{C}:ClusteringConfig")(
            model_name=config.model_name,
            n_states=config.n_states,
            random_seed=config.random_seed,
        )
    else:
        head_config = _resolve(f"{J}:JumpSegmentationConfig")(
            model_name=config.model_name,
            n_states=config.n_states,
            random_seed=config.random_seed,
        )
    return model_class(encoder, head_config)


P = "regime.models.probabilistic.hmm"
C = "regime.models.clustering.models"
J = "regime.models.jump.models"
E = "regime.models.econometric"
S = "regime.models.state_space.models"
U = "regime.models.supervised.models"
D = "regime.models.deep"

_SPECS = (
    _spec(
        "volatility-threshold",
        "regime.models.adapters:RuleRegimeModelAdapter",
        "regime.models.adapters:VolatilityThresholdConfig",
        aliases=("volatility_threshold", "rule_volatility_threshold"),
        capabilities=("fit", "predict", "deterministic"),
    ),
    *(
        _spec(
            name,
            f"{P}:{cls}",
            f"{P}:ProbabilisticHMMConfig",
            aliases=aliases,
            capabilities=("fit", "predict", "predict_proba", "filter", "smooth", "transitions"),
            tunable=(
                "n_states",
                "max_iter",
                "tol",
                "n_init",
                "covariance_regularization",
                "sticky_strength",
                "random_seed",
                *(
                    {
                        "student-t-hmm": ("student_t_dof",),
                        "ar-hmm": ("ar_order",),
                        "gmm-hmm": ("n_mixtures",),
                        "hsmm": ("duration_mean", "max_duration"),
                        "explicit-duration-latent-state": ("duration_mean", "max_duration"),
                    }.get(name, ())
                ),
            ),
        )
        for name, cls, aliases in (
            ("gaussian-hmm", "GaussianHMM", ("gaussian_hmm", "hmm")),
            ("sticky-hmm", "StickyHMM", ("sticky_hmm",)),
            ("student-t-hmm", "StudentTHMM", ("student_t_hmm",)),
            ("ar-hmm", "ARHMM", ("ar_hmm",)),
            ("input-output-hmm", "InputOutputHMM", ("io_hmm",)),
            ("gmm-hmm", "GMMHMM", ("gmm_hmm",)),
            ("hsmm", "HSMM", ()),
            ("explicit-duration-latent-state", "ExplicitDurationLatentStateModel", ("edlsm",)),
            ("hdp-hmm", "HDPHMMAdapter", ("hdp_hmm",)),
        )
    ),
    *(
        _spec(
            name,
            f"{C}:{cls}",
            f"{C}:ClusteringConfig",
            aliases=aliases,
            capabilities=("fit", "predict", "predict_proba"),
            extra=extra,
            dependencies=("hdbscan",) if extra else (),
        )
        for name, cls, aliases, extra in (
            ("kmeans", "KMeansRegimeModel", ("kmeans_regime",), None),
            ("gaussian-mixture", "GaussianMixtureRegimeModel", ("gmm",), None),
            (
                "hierarchical-clustering",
                "HierarchicalClusteringRegimeModel",
                ("hierarchical",),
                None,
            ),
            ("hdbscan", "HDBSCANRegimeModel", (), "clustering"),
            ("jump-penalized-kmeans", "JumpPenalizedKMeansRegimeModel", ("jump_kmeans",), None),
            ("ticc", "TICCRegimeModel", (), None),
        )
    ),
    _spec(
        "jump-segmentation",
        f"{J}:JumpSegmentationModel",
        f"{J}:JumpSegmentationConfig",
        aliases=("jump",),
        capabilities=("fit", "predict", "predict_proba", "transitions"),
    ),
    *(
        _spec(
            name,
            f"{E}:{cls}",
            f"{E}:EconometricModelConfig",
            aliases=aliases,
            capabilities=("fit", "predict", "predict_proba"),
        )
        for name, cls, aliases in (
            ("markov-switching-regression", "MarkovSwitchingRegression", ("ms_regression",)),
            ("markov-switching-ar", "MarkovSwitchingAR", ("ms_ar",)),
            ("markov-switching-var", "MarkovSwitchingVAR", ("ms_var",)),
            ("markov-switching-garch", "MarkovSwitchingGARCH", ("ms_garch",)),
            ("markov-switching-har", "MarkovSwitchingHAR", ("ms_har",)),
            ("threshold-autoregression", "ThresholdAutoregression", ("tar",)),
            ("smooth-transition-autoregression", "SmoothTransitionAutoregression", ("star",)),
            ("regime-switching-correlation", "RegimeSwitchingCorrelation", ()),
            ("regime-switching-copula", "RegimeSwitchingCopula", ()),
            ("switching-stochastic-volatility", "SwitchingStochasticVolatility", ()),
            ("regime-switching-jump-diffusion", "RegimeSwitchingJumpDiffusion", ()),
        )
    ),
    *(
        _spec(
            name,
            f"{S}:{cls}",
            f"{S}:StateSpaceConfig",
            capabilities=("fit", "predict", "predict_proba", "filter", "smooth"),
        )
        for name, cls in (
            ("switching-lds", "SwitchingLinearDynamicalSystem"),
            ("switching-dynamic-factor", "SwitchingDynamicFactorModel"),
            ("recurrent-switching-lds", "RecurrentSwitchingLinearDynamicalSystem"),
            ("explicit-duration-switching-lds", "ExplicitDurationSwitchingLinearDynamicalSystem"),
        )
    ),
    _spec(
        "supervised-classifier",
        f"{U}:SupervisedRegimeClassifier",
        f"{U}:SupervisedRegimeConfig",
        aliases=("supervised",),
        capabilities=("fit", "predict", "predict_proba"),
    ),
    _spec(
        "transition-hazard",
        f"{U}:TransitionHazardModel",
        f"{U}:TransitionHazardConfig",
        capabilities=("fit", "predict", "predict_proba"),
    ),
    *(
        _spec(
            name,
            f"{D}:{cls}",
            "regime.models.deep.config:DeepModelConfig",
            extra="deep",
            dependencies=("torch",),
            capabilities=("fit", "predict", "predict_proba"),
            factory=lambda model_cls, config: model_cls(config),
        )
        for name, cls in (
            ("lstm", "LSTM"),
            ("gru", "GRU"),
            ("tcn", "TemporalConvolutionalNetwork"),
            ("neural-hmm", "NeuralHMM"),
            ("deep-markov", "DeepMarkovModel"),
            ("variational-state-space", "VariationalStateSpaceModel"),
            ("vector-quantized-vae", "VectorQuantizedVAE"),
            ("neural-change-point", "NeuralChangePointDetector"),
            ("graph-dependency-network", "GraphDependencyNetwork"),
        )
    ),
    *(
        _spec(
            name,
            f"regime.models.transformers.heads:{cls}",
            "regime.models.transformers.config:TransformerConfig",
            extra="transformers",
            dependencies=("torch",),
            capabilities=("fit", "predict", "predict_proba"),
            factory=_transformer_head_factory,
        )
        for name, cls in (
            ("transformer-hmm", "TransformerHMMHead"),
            ("transformer-hsmm", "TransformerHSMMHead"),
            ("transformer-clustering", "TransformerClusteringHead"),
            ("transformer-jump", "TransformerJumpModelHead"),
        )
    ),
)

_BY_NAME = {key: spec for spec in _SPECS for key in (spec.name, *spec.aliases)}


def available_models() -> tuple[ModelSpec, ...]:
    """Return canonical model descriptors in stable registry order."""
    return _SPECS


def model_spec(name: str) -> ModelSpec:
    """Look up a model by canonical name or backward-compatible alias."""
    key = name.strip().lower().replace("_", "-")
    spec = _BY_NAME.get(name.strip().lower()) or _BY_NAME.get(key)
    if spec is None:
        choices = ", ".join(spec.name for spec in _SPECS)
        raise ModelRegistryError(f"Unknown model {name!r}. Available models: {choices}")
    return spec


def _configuration(spec: ModelSpec, parameters: Mapping[str, Any] | None) -> RegimeModelConfig:
    values = dict(parameters or {})
    nested = values.pop("parameters", {})
    fit = values.pop("fit_parameters", {})
    if not isinstance(nested, Mapping) or not isinstance(fit, Mapping):
        raise ModelRegistryError("parameters and fit_parameters must be mappings")
    values = {**values, **nested, **fit}
    for field in _OPERATIONAL_FIELDS - {"features"}:
        values.pop(field, None)
    for old, new in _STANDARD_ALIASES.items():
        if old in values:
            if new in values:
                raise ModelRegistryError(f"Use only one of {old!r} and {new!r}")
            values[new] = values.pop(old)
    fields = spec.config_class.model_fields
    if "features" in values:
        if "feature_names" in fields:
            values["feature_names"] = values.pop("features")
        else:
            values.pop("features")
    values.setdefault("model_name", spec.name)
    try:
        return spec.config_class.model_validate(values)
    except ValidationError as error:
        details = "; ".join(
            f"{'.'.join(map(str, item['loc']))}: {item['msg']}" for item in error.errors()
        )
        raise ModelRegistryError(f"Invalid parameters for {spec.name}: {details}") from error


def create_model(name: str, parameters: Mapping[str, Any] | None = None) -> Any:
    """Validate ``parameters`` and construct the requested model."""
    spec = model_spec(name)
    for module in spec.dependency_modules:
        try:
            importlib.import_module(module)
        except ImportError as error:
            extra = spec.optional_dependency_group
            raise ModelRegistryError(
                f"Model {spec.name!r} requires optional extra {extra!r}; install with "
                f"`pip install 'regime-switching[{extra}]'`."
            ) from error
    config = _configuration(spec, parameters)
    model_class = spec.model_class
    return spec.factory(model_class, config) if spec.factory else model_class(config)


def model_configuration(
    name: str, parameters: Mapping[str, Any] | None = None
) -> RegimeModelConfig:
    """Translate standardized workflow fields into a model's typed configuration."""
    return _configuration(model_spec(name), parameters)


__all__ = [
    "ModelRegistryError",
    "ModelSpec",
    "available_models",
    "create_model",
    "model_configuration",
    "model_spec",
]
