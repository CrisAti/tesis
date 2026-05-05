from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_iris

from hybrid_transfer_metric import (
    CLMConfig,
    HybridMetricConfig,
    HybridTransferMetric,
    JMDSConfig,
    compute_clm_score,
    solve_size_constrained_assignment,
)


def _metric() -> HybridTransferMetric:
    return HybridTransferMetric(
        HybridMetricConfig(
            clm=CLMConfig(n_permutations=32, random_state=7),
            jmds=JMDSConfig(n_neighbors=10, cv_folds=5, random_state=7),
            standardize=True,
        )
    )


def test_scores_are_in_range_and_natural_constraint_has_no_local_penalty() -> None:
    iris = load_iris()
    result = _metric().evaluate(
        iris.data,
        iris.target,
        target_proportions=np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
    )

    assert 0.0 <= result.transfer_score <= 1.0
    assert 0.0 <= result.global_score <= 1.0
    assert 0.0 <= result.local_factor <= 1.0
    assert result.local_factor == pytest.approx(1.0, abs=1e-8)
    assert result.local_cost == pytest.approx(0.0, abs=1e-8)
    assert result.moved_fraction == pytest.approx(0.0, abs=1e-8)


def test_original_iris_labels_score_above_randomized_labels() -> None:
    iris = load_iris()
    config = CLMConfig(n_permutations=32, random_state=11)
    original = compute_clm_score(iris.data, iris.target, config).score

    rng = np.random.default_rng(11)
    shuffled_y = rng.permutation(iris.target)
    randomized = compute_clm_score(iris.data, shuffled_y, config).score

    assert original > randomized


def test_assignment_moves_cheapest_available_source_sample() -> None:
    y = np.array([0, 0, 1, 1])
    reliability = np.array(
        [
            [0.9, 0.1],
            [0.4, 0.3],
            [0.2, 0.8],
            [0.1, 0.7],
        ]
    )

    result = solve_size_constrained_assignment(
        reliability,
        y,
        target_proportions=np.array([0.25, 0.75]),
    )

    assert result.moved_fraction == pytest.approx(0.25)
    assert result.local_cost == pytest.approx(0.025)
    assert result.local_factor == pytest.approx(0.975)
    assert result.assignment[1, 1] == pytest.approx(1.0)
    assert result.assignment[0, 0] == pytest.approx(1.0)


def test_invalid_target_proportions_are_rejected() -> None:
    y = np.array([0, 1])
    reliability = np.array([[0.8, 0.2], [0.1, 0.9]])

    with pytest.raises(ValueError):
        solve_size_constrained_assignment(
            reliability,
            y,
            target_proportions=np.array([0.6, 0.3]),
        )
