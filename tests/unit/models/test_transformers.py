"""Tests for optional transformer representations and dependency-light safeguards."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from pydantic import ValidationError

from regime.models.transformers import TransformerConfig, embedding_interpretation_warning


@pytest.mark.unit
def test_transformer_config_validates_attention_width() -> None:
    with pytest.raises(ValidationError, match="divisible"):
        TransformerConfig(embedding_dim=10, num_heads=4)


@pytest.mark.unit
def test_report_warning_rejects_intrinsic_regime_interpretation() -> None:
    warning = embedding_interpretation_warning().lower()
    assert "not inherently interpretable regimes" in warning
    assert "independent evidence" in warning


@pytest.mark.unit
def test_encoder_import_has_actionable_error_without_extra() -> None:
    if importlib.util.find_spec("torch") is None:
        with pytest.raises(ImportError, match=r"regime-switching\[transformers\]"):
            from regime.models.transformers import TimeSeriesTransformerEncoder  # noqa: F401


@pytest.mark.unit
def test_multimodal_adapter_aligns_and_concatenates() -> None:
    pytest.importorskip("torch")
    from regime.models.transformers import MultimodalEmbeddingAdapter

    class Numeric:
        def encode(self, values: object) -> np.ndarray:
            return np.asarray(values, dtype=np.float32)

    adapter = MultimodalEmbeddingAdapter(
        lambda texts: np.asarray([[len(text)] for text in texts], dtype=np.float32), Numeric()
    )
    result = adapter.encode((["a", "abcd"], [[1, 2], [3, 4]]))
    assert result.tolist() == [[1, 1, 2], [4, 3, 4]]


@pytest.mark.unit
def test_encoder_and_clustering_head_produce_aligned_labels() -> None:
    pytest.importorskip("torch")
    from regime.models.clustering.models import ClusteringConfig
    from regime.models.transformers import TimeSeriesTransformerEncoder, TransformerClusteringHead

    config = TransformerConfig(input_dim=1, embedding_dim=8, num_heads=2, sequence_length=4)
    encoder = TimeSeriesTransformerEncoder(config).eval()
    values = np.r_[np.zeros(12), np.ones(12)]
    embeddings = encoder.encode(values)
    assert embeddings.shape == (24, 8)
    model = TransformerClusteringHead(
        encoder, ClusteringConfig(n_states=2, random_seed=7, scale=False)
    ).fit(values)
    assert len(model.predict(values)) == len(values)
