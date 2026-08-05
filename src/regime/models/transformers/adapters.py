"""Adapters for foundation-model and multimodal representation generation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class EmbeddingAdapter(Protocol):
    """Small boundary shared by local and third-party representation models."""

    def encode(self, values: Any) -> NDArray[np.float32]: ...


class HuggingFaceEmbeddingAdapter:
    """Extract hidden states from compatible Chronos, Moirai, or MOMENT checkpoints.

    Model-specific preprocessing varies, so callers may inject ``preprocess``. The
    default sends a float ``input_values`` tensor and mean-pools the last hidden state.
    Remote code is disabled unless explicitly requested by the caller.
    """

    def __init__(
        self,
        model_name: str,
        *,
        preprocess: Callable[[Any], Mapping[str, Any]] | None = None,
        trust_remote_code: bool = False,
        device: str = "cpu",
    ) -> None:
        import torch
        from transformers import AutoModel

        self._torch = torch
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model.to(device).eval()
        self.device = device
        self.preprocess = preprocess

    def encode(self, values: Any) -> NDArray[np.float32]:
        inputs = (
            self.preprocess(values)
            if self.preprocess
            else {"input_values": self._torch.as_tensor(values, dtype=self._torch.float32)}
        )
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with self._torch.inference_mode():
            output = self.model(**inputs, output_hidden_states=True)
        hidden = output.last_hidden_state
        return hidden.mean(dim=-2).detach().cpu().numpy().astype(np.float32, copy=False)


class ChronosEmbeddingAdapter(HuggingFaceEmbeddingAdapter):
    """Semantic alias for compatible Chronos checkpoints."""


class MoiraiEmbeddingAdapter(HuggingFaceEmbeddingAdapter):
    """Semantic alias for compatible Moirai checkpoints."""


class MomentEmbeddingAdapter(HuggingFaceEmbeddingAdapter):
    """Semantic alias for compatible MOMENT checkpoints."""


class MultimodalEmbeddingAdapter:
    """Concatenate text representations with aligned numerical representations."""

    def __init__(
        self, text_encoder: Callable[[Sequence[str]], Any], numerical: EmbeddingAdapter
    ) -> None:
        self.text_encoder = text_encoder
        self.numerical = numerical

    def encode(self, values: tuple[Sequence[str], Any]) -> NDArray[np.float32]:
        texts, numeric = values
        text_embeddings = np.asarray(self.text_encoder(texts), dtype=np.float32)
        numerical_embeddings = np.asarray(self.numerical.encode(numeric), dtype=np.float32)
        if text_embeddings.ndim == 1:
            text_embeddings = text_embeddings[:, None]
        if len(text_embeddings) != len(numerical_embeddings):
            raise ValueError("text and numerical embeddings must have equal row counts")
        return np.concatenate((text_embeddings, numerical_embeddings), axis=1)
