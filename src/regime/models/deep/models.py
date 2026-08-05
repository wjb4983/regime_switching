"""Trainable deep representation models for regime discovery."""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset

from regime.models.base import ModelMetadata, RegimeModel
from regime.models.deep.config import DeepModelConfig
from regime.models.deep.networks import (
    DeepRegimeNetwork,
    GraphEncoder,
    MLPEncoder,
    RecurrentEncoder,
    TemporalConvEncoder,
)


class DeepRegimeModel(RegimeModel):
    """Common leakage-safe trainer and representation/head interoperability API."""

    architecture: ClassVar[str] = "mlp"
    variational: ClassVar[bool] = False
    vector_quantized: ClassVar[bool] = False

    def __init__(self) -> None:
        self.network: DeepRegimeNetwork | None = None
        self.config: DeepModelConfig | None = None
        self.input_size: int | None = None
        self.temperature = 1.0
        self.history: list[dict[str, float]] = []
        self._transition: np.ndarray | None = None
        self._metadata: ModelMetadata | None = None

    @property
    def metadata(self) -> ModelMetadata:
        if self._metadata is None:
            raise RuntimeError("model has not been fitted")
        return self._metadata

    @staticmethod
    def _array(dataset: Any) -> np.ndarray:
        values = dataset.to_numpy() if hasattr(dataset, "to_numpy") else np.asarray(dataset)
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("dataset must be a finite two-dimensional time series")
        return values

    @staticmethod
    def _windows(values: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
        if len(values) <= length:
            raise ValueError("dataset must contain more rows than sequence_length")
        return np.stack([values[i : i + length] for i in range(len(values) - length)]), values[
            length:
        ]

    @staticmethod
    def _device(name: str) -> torch.device:
        if name == "auto":
            name = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        if name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if name == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device(name)

    @staticmethod
    def _seed(seed: int | None, deterministic: bool) -> None:
        seed = 0 if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(deterministic, warn_only=True)

    def _build(self, input_size: int, config: DeepModelConfig) -> DeepRegimeNetwork:
        args = (input_size, config.hidden_size, config.embedding_dim)
        if self.architecture in {"lstm", "gru"}:
            encoder: nn.Module = RecurrentEncoder(
                *args, config.num_layers, config.dropout, cell=self.architecture
            )
        elif self.architecture == "tcn":
            encoder = TemporalConvEncoder(*args, config.num_layers, config.dropout)
        elif self.architecture == "graph":
            encoder = GraphEncoder(*args, config.sequence_length)
        else:
            encoder = MLPEncoder(*args, config.sequence_length)
        return DeepRegimeNetwork(
            encoder,
            config.embedding_dim,
            input_size,
            config.n_states,
            variational=self.variational,
            vq=self.vector_quantized,
        )

    def fit(self, dataset: Any, config: DeepModelConfig) -> Self:
        values = self._array(dataset)
        window = (
            math.ceil(len(values) * config.validation_window)
            if isinstance(config.validation_window, float)
            else config.validation_window
        )
        if window >= len(values) - config.sequence_length:
            raise ValueError("validation_window leaves insufficient training observations")
        train_values, validation_values = (
            values[:-window],
            values[-(window + config.sequence_length) :],
        )
        train_x, train_y = self._windows(train_values, config.sequence_length)
        validation_x, validation_y = self._windows(validation_values, config.sequence_length)
        self._seed(config.random_seed, config.deterministic)
        device = self._device(config.device)
        dtype = torch.float64 if config.precision == "float64" else torch.float32
        self.network = self._build(values.shape[1], config).to(device=device, dtype=dtype)
        self.config, self.input_size = config, values.shape[1]
        optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        generator = torch.Generator().manual_seed(config.random_seed or 0)
        loader = DataLoader(
            TensorDataset(
                torch.as_tensor(train_x, dtype=dtype), torch.as_tensor(train_y, dtype=dtype)
            ),
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
        )
        val_x = torch.as_tensor(validation_x, dtype=dtype, device=device)
        val_y = torch.as_tensor(validation_y, dtype=dtype, device=device)
        best, stale, best_state = float("inf"), 0, None
        autocast_enabled = config.precision in {"16-mixed", "bf16-mixed"} and device.type == "cuda"
        autocast_dtype = torch.float16 if config.precision == "16-mixed" else torch.bfloat16
        for epoch in range(config.max_epochs):
            self.network.train()
            total = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, enabled=autocast_enabled, dtype=autocast_dtype
                ):
                    logits, reconstruction, _, penalty = self.network(batch_x)
                    loss = functional.mse_loss(reconstruction, batch_y) + 0.01 * penalty
                    # Low-entropy assignments make the representation useful to discrete heads.
                    probabilities = logits.softmax(-1)
                    loss += (
                        1e-3 * -(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1).mean()
                    )
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip_val)
                optimizer.step()
                total += float(loss.detach()) * len(batch_x)
            self.network.eval()
            with torch.no_grad():
                logits, reconstruction, _, penalty = self.network(val_x)
                val_loss = float(functional.mse_loss(reconstruction, val_y) + 0.01 * penalty)
            self.history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": total / len(train_x),
                    "validation_loss": val_loss,
                }
            )
            if val_loss < best - config.min_delta:
                best, stale = val_loss, 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.network.state_dict().items()
                }
            else:
                stale += 1
                if stale >= config.patience:
                    break
        if best_state is not None:
            self.network.load_state_dict(best_state)
        if config.calibration == "temperature":
            self._calibrate(val_x)
        with torch.no_grad():
            probabilities = (self.network(val_x)[0] / self.temperature).softmax(-1)
            counts = probabilities[:-1].T @ probabilities[1:]
            counts += torch.finfo(counts.dtype).eps
            self._transition = (counts / counts.sum(dim=1, keepdim=True)).cpu().numpy()
        self._metadata = ModelMetadata(
            model_name=config.model_name,
            model_version=config.model_version,
            n_states=config.n_states,
            fitted_at=datetime.now(UTC),
            training_observations=len(train_values),
            config_hash=config.config_hash(),
            attributes={
                "architecture": self.architecture,
                "validation_observations": window,
                "device": str(device),
                "precision": config.precision,
            },
        )
        if config.checkpoint_path is not None:
            self.save(config.checkpoint_path)
        return self

    def _calibrate(self, validation_x: Tensor) -> None:
        """Entropy-based temperature scaling when no labels are available."""
        assert self.network is not None
        self.network.eval()
        with torch.no_grad():
            logits = self.network(validation_x)[0]
            confidence = logits.softmax(-1).amax(-1).mean().item()
        self.temperature = max(0.5, min(2.0, confidence / 0.75))

    def _prepared(self, dataset: Any) -> Tensor:
        if self.network is None or self.config is None:
            raise RuntimeError("model has not been fitted")
        windows, _ = self._windows(self._array(dataset), self.config.sequence_length)
        parameter = next(self.network.parameters())
        return torch.as_tensor(windows, dtype=parameter.dtype, device=parameter.device)

    def transform(self, dataset: Any) -> np.ndarray:
        """Export embeddings accepted by HMM, HSMM, clustering, and jump-model heads."""
        assert self.network is not None
        self.network.eval()
        with torch.no_grad():
            return self.network(self._prepared(dataset))[2].cpu().numpy()

    def predict_proba(self, dataset: Any) -> np.ndarray:
        assert self.network is not None
        self.network.eval()
        with torch.no_grad():
            logits = self.network(self._prepared(dataset))[0] / self.temperature
            return logits.softmax(-1).cpu().numpy()

    def predict(self, dataset: Any) -> np.ndarray:
        return self.predict_proba(dataset).argmax(axis=1)

    def calibrate(self, dataset: Any, labels: Any) -> Self:
        """Fit a scalar temperature against labelled calibration-window states."""
        if self.network is None:
            raise RuntimeError("model has not been fitted")
        self.network.eval()
        with torch.no_grad():
            logits = self.network(self._prepared(dataset))[0].detach()
        targets = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=logits.device)
        if targets.ndim != 1 or len(targets) != len(logits):
            raise ValueError("labels must have one state index per exported sequence window")
        log_temperature = torch.tensor(
            math.log(self.temperature), device=logits.device, requires_grad=True
        )
        optimizer = torch.optim.LBFGS([log_temperature], max_iter=50)

        def closure() -> Tensor:
            optimizer.zero_grad()
            loss = functional.cross_entropy(logits / log_temperature.exp(), targets)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(log_temperature.detach().exp().clamp(0.05, 20.0))
        return self

    def change_probabilities(self, dataset: Any) -> np.ndarray:
        probabilities = self.predict_proba(dataset)
        result = np.zeros(len(probabilities))
        result[1:] = 1.0 - np.sum(probabilities[1:] * probabilities[:-1], axis=1)
        return result

    def transition_matrix(self) -> np.ndarray:
        """Return the soft transition matrix estimated on the validation window."""
        if self._transition is None:
            raise RuntimeError("model has not been fitted")
        return self._transition.copy()

    def estimate_transition_matrix(self, dataset: Any) -> np.ndarray:
        probabilities = self.predict_proba(dataset)
        counts = probabilities[:-1].T @ probabilities[1:]
        counts += np.finfo(counts.dtype).eps
        return counts / counts.sum(axis=1, keepdims=True)

    def adjacency_matrix(self) -> np.ndarray:
        if self.network is None or not isinstance(self.network.encoder, GraphEncoder):
            raise TypeError("adjacency_matrix is only available for GraphDependencyNetwork")
        return self.network.encoder.adjacency().detach().cpu().numpy()

    def save(self, path: str | Path) -> None:
        if self.network is None or self.config is None:
            raise RuntimeError("model has not been fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "class": type(self).__name__,
                "config": self.config.model_dump(mode="json"),
                "input_size": self.input_size,
                "state_dict": self.network.state_dict(),
                "temperature": self.temperature,
                "history": self.history,
                "transition": self._transition,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model = cls()
        config = DeepModelConfig.model_validate(payload["config"])
        model.network = model._build(int(payload["input_size"]), config)
        model.network.load_state_dict(payload["state_dict"])
        model.config, model.input_size = config, int(payload["input_size"])
        model.temperature, model.history = float(payload["temperature"]), payload["history"]
        model._transition = np.asarray(payload["transition"])
        model._metadata = ModelMetadata(
            model_name=config.model_name,
            model_version=config.model_version,
            n_states=config.n_states,
            config_hash=config.config_hash(),
            attributes={"restored": True},
        )
        return model


class LSTM(DeepRegimeModel):
    architecture = "lstm"


class GRU(DeepRegimeModel):
    architecture = "gru"


class TemporalConvolutionalNetwork(DeepRegimeModel):
    architecture = "tcn"


class NeuralHMM(DeepRegimeModel):
    """Neural emission representation with an exportable discrete-state posterior."""


class DeepMarkovModel(DeepRegimeModel):
    variational = True


class VariationalStateSpaceModel(DeepRegimeModel):
    variational = True


class VectorQuantizedVAE(DeepRegimeModel):
    vector_quantized = True


class NeuralChangePointDetector(DeepRegimeModel):
    pass


class GraphDependencyNetwork(DeepRegimeModel):
    architecture = "graph"
