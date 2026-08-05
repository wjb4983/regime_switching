"""Downstream regime heads operating exclusively on transformer embeddings."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from regime.models.base import RegimeModel, RegimeModelConfig
from regime.models.clustering.models import ClusteringConfig, KMeansRegimeModel
from regime.models.jump.models import JumpSegmentationConfig, JumpSegmentationModel
from regime.models.probabilistic.hmm import HSMM, GaussianHMM, ProbabilisticHMMConfig
from regime.models.transformers.adapters import EmbeddingAdapter

HeadT = TypeVar("HeadT", bound=RegimeModel)


class TransformerEmbeddingHead(Generic[HeadT]):
    """Composition wrapper that freezes representation semantics from regime semantics."""

    def __init__(self, encoder: EmbeddingAdapter, head: HeadT) -> None:
        self.encoder = encoder
        self.head = head

    @property
    def metadata(self):  # type: ignore[no-untyped-def]
        return self.head.metadata

    def fit(
        self, dataset: Any, config: RegimeModelConfig | None = None
    ) -> TransformerEmbeddingHead[HeadT]:
        self.head.fit(self.encoder.encode(dataset), config)  # type: ignore[arg-type]
        return self

    def predict(self, dataset: Any):  # type: ignore[no-untyped-def]
        return self.head.predict(self.encoder.encode(dataset))

    def predict_proba(self, dataset: Any):  # type: ignore[no-untyped-def]
        return self.head.predict_proba(self.encoder.encode(dataset))


class TransformerHMMHead(TransformerEmbeddingHead[GaussianHMM]):
    def __init__(
        self, encoder: EmbeddingAdapter, config: ProbabilisticHMMConfig | None = None
    ) -> None:
        super().__init__(encoder, GaussianHMM(config))


class TransformerHSMMHead(TransformerEmbeddingHead[HSMM]):
    def __init__(
        self, encoder: EmbeddingAdapter, config: ProbabilisticHMMConfig | None = None
    ) -> None:
        super().__init__(encoder, HSMM(config))


class TransformerClusteringHead(TransformerEmbeddingHead[KMeansRegimeModel]):
    def __init__(self, encoder: EmbeddingAdapter, config: ClusteringConfig | None = None) -> None:
        super().__init__(encoder, KMeansRegimeModel(config))


class TransformerJumpModelHead(TransformerEmbeddingHead[JumpSegmentationModel]):
    def __init__(
        self, encoder: EmbeddingAdapter, config: JumpSegmentationConfig | None = None
    ) -> None:
        super().__init__(encoder, JumpSegmentationModel(config))
