from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hybrid_transfer_metric.clm import CLMConfig, CLMResult, compute_clm_score
from hybrid_transfer_metric.constraints import AssignmentResult, solve_size_constrained_assignment
from hybrid_transfer_metric.jmds import JMDSConfig, JMDSResult, compute_jmds_reliability
from hybrid_transfer_metric.utils import (
    clipped01,
    empirical_proportions,
    encode_labels,
    standardize_features,
    to_serializable,
)


@dataclass(frozen=True)
class HybridMetricConfig:
    clm: CLMConfig = CLMConfig()
    jmds: JMDSConfig = JMDSConfig()
    standardize: bool = True


@dataclass(frozen=True)
class HybridMetricResult:
    transfer_score: float
    global_score: float
    local_factor: float
    local_cost: float
    moved_fraction: float
    observed_proportions: np.ndarray
    target_proportions: np.ndarray
    assignment_matrix: np.ndarray
    labels: np.ndarray
    clm: CLMResult
    jmds: JMDSResult
    assignment: AssignmentResult

    def to_dict(self) -> dict:
        return to_serializable(self)


class HybridTransferMetric:
    """Hybrid CLM-JMDS metric for size-constrained clustering transferability."""

    def __init__(self, config: HybridMetricConfig | None = None):
        self.config = config or HybridMetricConfig()

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_proportions: np.ndarray,
    ) -> HybridMetricResult:
        y_encoded, labels = encode_labels(y)
        n_classes = int(labels.shape[0])
        X_eval = standardize_features(X) if self.config.standardize else np.asarray(X, dtype=float)

        clm = compute_clm_score(X_eval, y_encoded, self.config.clm)
        jmds = compute_jmds_reliability(X_eval, y_encoded, n_classes, self.config.jmds)
        assignment = solve_size_constrained_assignment(
            reliability=jmds.reliability,
            y=y_encoded,
            target_proportions=target_proportions,
        )
        transfer_score = float(clipped01(np.sqrt(clm.score * assignment.local_factor)))

        return HybridMetricResult(
            transfer_score=transfer_score,
            global_score=clm.score,
            local_factor=assignment.local_factor,
            local_cost=assignment.local_cost,
            moved_fraction=assignment.moved_fraction,
            observed_proportions=empirical_proportions(y_encoded, n_classes),
            target_proportions=assignment.target_proportions,
            assignment_matrix=assignment.assignment_matrix,
            labels=labels,
            clm=clm,
            jmds=jmds,
            assignment=assignment,
        )
