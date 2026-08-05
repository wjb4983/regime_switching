"""PyTorch networks used by :mod:`regime.models.deep`.

This module is deliberately imported lazily by the public package, so the core
package remains usable without the ``deep`` extra.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional


class RecurrentEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden: int,
        embedding: int,
        layers: int,
        dropout: float,
        *,
        cell: str,
    ) -> None:
        super().__init__()
        recurrent = nn.LSTM if cell == "lstm" else nn.GRU
        self.recurrent = recurrent(
            input_size, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0
        )
        self.project = nn.Linear(hidden, embedding)

    def forward(self, x: Tensor) -> Tensor:
        output, _ = self.recurrent(x)
        return self.project(output[:, -1])


class TemporalConvEncoder(nn.Module):
    def __init__(
        self, input_size: int, hidden: int, embedding: int, layers: int, dropout: float
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        channels = input_size
        for index in range(layers):
            dilation = 2**index
            blocks.extend(
                (
                    nn.Conv1d(channels, hidden, 3, padding=dilation, dilation=dilation),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
            channels = hidden
        self.network = nn.Sequential(*blocks)
        self.project = nn.Linear(hidden, embedding)

    def forward(self, x: Tensor) -> Tensor:
        return self.project(self.network(x.transpose(1, 2))[:, :, -1])


class MLPEncoder(nn.Module):
    def __init__(self, input_size: int, hidden: int, embedding: int, sequence_length: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size * sequence_length, hidden),
            nn.GELU(),
            nn.Linear(hidden, embedding),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class GraphEncoder(nn.Module):
    """Dependency-aware encoder using learned asset adjacency and message passing."""

    def __init__(self, input_size: int, hidden: int, embedding: int, sequence_length: int) -> None:
        super().__init__()
        self.adjacency_logits = nn.Parameter(torch.eye(input_size) * 2)
        self.temporal = nn.Linear(sequence_length, hidden)
        self.project = nn.Linear(input_size * hidden, embedding)

    def forward(self, x: Tensor) -> Tensor:
        adjacency = torch.softmax(self.adjacency_logits, dim=-1)
        nodes = functional.gelu(self.temporal(x.transpose(1, 2)))
        messages = torch.einsum("ij,bjh->bih", adjacency, nodes)
        return self.project(messages.flatten(1))

    def adjacency(self) -> Tensor:
        return torch.softmax(self.adjacency_logits, dim=-1)


class DeepRegimeNetwork(nn.Module):
    """Encoder with reconstruction, discrete-state, and change-point heads."""

    def __init__(
        self,
        encoder: nn.Module,
        embedding: int,
        output_size: int,
        states: int,
        *,
        variational: bool = False,
        vq: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.variational = variational
        self.vq = vq
        self.log_variance = nn.Linear(embedding, embedding) if variational else None
        self.codebook = nn.Embedding(states, embedding) if vq else None
        self.state_head = nn.Linear(embedding, states)
        self.decoder = nn.Linear(embedding, output_size)
        self.change_head = nn.Linear(embedding, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        mean = self.encoder(x)
        penalty = mean.new_zeros(())
        latent = mean
        if self.variational and self.log_variance is not None:
            log_var = self.log_variance(mean).clamp(-10, 10)
            latent = (
                mean + torch.randn_like(mean) * torch.exp(0.5 * log_var) if self.training else mean
            )
            penalty = -0.5 * (1 + log_var - mean.square() - log_var.exp()).mean()
        if self.vq and self.codebook is not None:
            distance = (mean[:, None, :] - self.codebook.weight[None, :, :]).square().sum(-1)
            quantized = self.codebook(distance.argmin(-1))
            penalty = functional.mse_loss(quantized.detach(), mean) + functional.mse_loss(
                quantized, mean.detach()
            )
            latent = mean + (quantized - mean).detach()
        return self.state_head(latent), self.decoder(latent), latent, penalty
