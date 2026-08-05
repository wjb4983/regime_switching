"""Optional transformer representations and downstream regime heads.

Install ``regime-switching[transformers]`` before requesting neural encoders or
foundation-model adapters. Configuration and reporting safeguards remain importable
from a core installation.
"""

from __future__ import annotations

from typing import Any

from regime.models.transformers.config import TransformerConfig
from regime.models.transformers.reporting import (
    EMBEDDING_INTERPRETATION_WARNING,
    embedding_interpretation_warning,
)

_OPTIONAL = {
    "TimeSeriesTransformerEncoder": ("encoders", "TimeSeriesTransformerEncoder"),
    "PatchTransformerEncoder": ("encoders", "PatchTransformerEncoder"),
    "EmbeddingAdapter": ("adapters", "EmbeddingAdapter"),
    "HuggingFaceEmbeddingAdapter": ("adapters", "HuggingFaceEmbeddingAdapter"),
    "ChronosEmbeddingAdapter": ("adapters", "ChronosEmbeddingAdapter"),
    "MoiraiEmbeddingAdapter": ("adapters", "MoiraiEmbeddingAdapter"),
    "MomentEmbeddingAdapter": ("adapters", "MomentEmbeddingAdapter"),
    "MultimodalEmbeddingAdapter": ("adapters", "MultimodalEmbeddingAdapter"),
    "TransformerEmbeddingHead": ("heads", "TransformerEmbeddingHead"),
    "TransformerHMMHead": ("heads", "TransformerHMMHead"),
    "TransformerHSMMHead": ("heads", "TransformerHSMMHead"),
    "TransformerClusteringHead": ("heads", "TransformerClusteringHead"),
    "TransformerJumpModelHead": ("heads", "TransformerJumpModelHead"),
}


def __getattr__(name: str) -> Any:
    if name not in _OPTIONAL:
        raise AttributeError(name)
    module_name, member = _OPTIONAL[name]
    try:
        module = __import__(f"regime.models.transformers.{module_name}", fromlist=[member])
    except ImportError as error:
        if error.name in {"torch", "transformers"}:
            raise ImportError(
                "Transformer models require optional dependencies; install with "
                "`pip install 'regime-switching[transformers]'`."
            ) from error
        raise
    return getattr(module, member)


__all__ = [
    "TransformerConfig",
    "EMBEDDING_INTERPRETATION_WARNING",
    "embedding_interpretation_warning",
    *_OPTIONAL,
]
