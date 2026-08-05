"""Torch transformer encoders that emit representations, not regime labels."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from regime.models.transformers.config import TransformerConfig


def _windows(values: Any, length: int) -> NDArray[np.float32]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("values must be a non-empty, finite 1D or 2D array")
    padded = np.pad(array, ((length - 1, 0), (0, 0)), mode="edge")
    return np.stack([padded[i : i + length] for i in range(len(array))])


class TimeSeriesTransformerEncoder(nn.Module):
    """Causal-window transformer returning one embedding per timestamp."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.input_dim, config.embedding_dim)
        self.position = nn.Parameter(torch.zeros(1, config.sequence_length, config.embedding_dim))
        layer = nn.TransformerEncoderLayer(
            config.embedding_dim,
            config.num_heads,
            config.feedforward_dim,
            config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, config.num_layers)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 3 or values.shape[-1] != self.config.input_dim:
            raise ValueError("values must have shape (batch, sequence, input_dim)")
        encoded = self.transformer(
            self.input_projection(values) + self.position[:, : values.shape[1]]
        )
        return encoded[:, -1] if self.config.pooling == "last" else encoded.mean(dim=1)

    def encode(self, values: Any) -> NDArray[np.float32]:
        """Create aligned embeddings without constructing gradients."""
        device = next(self.parameters()).device
        with torch.inference_mode():
            output = self(
                torch.as_tensor(_windows(values, self.config.sequence_length), device=device)
            )
        return output.detach().cpu().numpy().astype(np.float32, copy=False)


class PatchTransformerEncoder(nn.Module):
    """PatchTST-style encoder that tokenizes local flattened time-series patches."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_projection = nn.Linear(
            config.patch_length * config.input_dim, config.embedding_dim
        )
        layer = nn.TransformerEncoderLayer(
            config.embedding_dim,
            config.num_heads,
            config.feedforward_dim,
            config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, config.num_layers)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 3 or values.shape[-1] != self.config.input_dim:
            raise ValueError("values must have shape (batch, sequence, input_dim)")
        patches = values.unfold(1, self.config.patch_length, self.config.patch_stride)
        patches = patches.permute(0, 1, 3, 2).flatten(2)
        encoded = self.transformer(self.patch_projection(patches))
        return encoded[:, -1] if self.config.pooling == "last" else encoded.mean(dim=1)

    def encode(self, values: Any) -> NDArray[np.float32]:
        device = next(self.parameters()).device
        with torch.inference_mode():
            output = self(
                torch.as_tensor(_windows(values, self.config.sequence_length), device=device)
            )
        return output.detach().cpu().numpy().astype(np.float32, copy=False)
