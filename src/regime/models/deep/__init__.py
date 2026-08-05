"""Optional deep-learning regime models.

Install with ``pip install 'regime-switching[deep]'``.  Torch is imported only when
one of the model classes is requested, keeping the base package lightweight.
"""

from __future__ import annotations

from typing import Any

from regime.models.deep.config import DeepModelConfig

_MODELS = {
    "DeepRegimeModel",
    "LSTM",
    "GRU",
    "TemporalConvolutionalNetwork",
    "NeuralHMM",
    "DeepMarkovModel",
    "VariationalStateSpaceModel",
    "VectorQuantizedVAE",
    "NeuralChangePointDetector",
    "GraphDependencyNetwork",
}


def __getattr__(name: str) -> Any:
    if name not in _MODELS:
        raise AttributeError(name)
    try:
        from regime.models.deep import models
    except ImportError as error:
        if error.name == "torch":
            raise ImportError(
                "Deep regime models require the optional dependency group; "
                "install with `pip install 'regime-switching[deep]'`."
            ) from error
        raise
    return getattr(models, name)


__all__ = ["DeepModelConfig", *_MODELS]
