import numpy as np
import pytest

from regime.models.base import RegimeModel
from regime.models.jump import JumpSegmentationConfig, JumpSegmentationModel, align_jump_labels


@pytest.mark.unit
def test_jump_segmentation_is_regime_model_with_recurring_states() -> None:
    data = np.array([[-2.0], [-2.1], [2.0], [2.2], [-1.9], [-2.0], [2.1], [2.0]])
    model = JumpSegmentationModel(
        JumpSegmentationConfig(n_states=2, jump_penalty=0.05, random_seed=7, scale=False)
    ).fit(data)

    assert isinstance(model, RegimeModel)
    labels = model.predict(data)
    assert len(labels) == len(data)
    assert labels[0] == labels[4]
    assert labels[2] == labels[6]
    assert model.metadata.attributes["not_change_point_segmentation"] is True


@pytest.mark.unit
def test_state_statistics_and_transition_summary_are_available() -> None:
    data = np.array([[-1.0], [-1.2], [1.0], [1.1], [-0.9]])
    model = JumpSegmentationModel(
        JumpSegmentationConfig(
            n_states=2, parameterization="gaussian_diag", random_seed=3, scale=False
        )
    ).fit(data)

    stats = model.state_statistics()
    summary = model.transition_summary()

    assert set(stats) == {"0", "1"}
    assert all("count" in values and "feature_0_mean" in values for values in stats.values())
    assert summary["labels"] == (0, 1)
    assert np.asarray(model.transition_matrix()).shape == (2, 2)
    assert model.result is not None
    assert set(model.result.distribution_parameters) == {0, 1}


@pytest.mark.unit
def test_alignment_support_relabels_candidate_path() -> None:
    reference = (0, 0, 1, 1, 0, 2)
    candidate = (5, 5, 4, 4, 5, 3)

    assert align_jump_labels(reference, candidate) == reference
