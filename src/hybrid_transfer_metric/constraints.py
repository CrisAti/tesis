from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from hybrid_transfer_metric.utils import clipped01, validate_target_proportions


@dataclass(frozen=True)
class AssignmentResult:
    local_cost: float
    local_factor: float
    moved_fraction: float
    assignment: np.ndarray
    assignment_matrix: np.ndarray
    target_proportions: np.ndarray
    observed_proportions: np.ndarray


def _build_cost_matrix(reliability: np.ndarray, y: np.ndarray) -> np.ndarray:
    own = reliability[np.arange(y.shape[0]), y]
    return np.maximum(0.0, own[:, None] - reliability)


def solve_size_constrained_assignment(
    reliability: np.ndarray,
    y: np.ndarray,
    target_proportions: np.ndarray,
) -> AssignmentResult:
    """Solve the local feasibility problem as a linear assignment transport.

    Variables z[i, k] represent the fraction of sample i assigned to target class k.
    The relaxation supports non-integer target proportions while preserving a
    transparent cost in [0, 1].
    """
    reliability = np.asarray(reliability, dtype=float)
    y = np.asarray(y, dtype=int)
    if reliability.ndim != 2:
        raise ValueError("reliability must be a two-dimensional matrix.")
    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional vector.")
    n_samples, n_classes = reliability.shape
    if y.shape[0] != n_samples:
        raise ValueError("reliability and y must contain the same number of samples.")
    if np.any(y < 0) or np.any(y >= n_classes):
        raise ValueError("y contains labels outside reliability columns.")

    q = validate_target_proportions(target_proportions, n_classes)
    observed_counts = np.bincount(y, minlength=n_classes).astype(float)
    observed_proportions = observed_counts / observed_counts.sum()

    costs = _build_cost_matrix(reliability, y)
    structural_objective = (costs / n_samples).reshape(-1)

    n_variables = n_samples * n_classes
    row_constraints = np.zeros((n_samples, n_variables), dtype=float)
    for i in range(n_samples):
        row_constraints[i, i * n_classes : (i + 1) * n_classes] = 1.0

    class_constraints = np.zeros((n_classes, n_variables), dtype=float)
    for k in range(n_classes):
        class_constraints[k, k::n_classes] = 1.0

    a_eq = np.vstack([row_constraints, class_constraints])
    b_eq = np.concatenate([np.ones(n_samples), q * n_samples])

    result = linprog(
        c=structural_objective,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=(0.0, 1.0),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Constraint solver failed: {result.message}")

    move_indicator = np.ones_like(costs)
    move_indicator[np.arange(n_samples), y] = 0.0
    tie_break_result = linprog(
        c=(move_indicator / n_samples).reshape(-1),
        A_eq=a_eq,
        b_eq=b_eq,
        A_ub=structural_objective.reshape(1, -1),
        b_ub=np.array([float(result.fun) + 1e-9]),
        bounds=(0.0, 1.0),
        method="highs",
    )
    if not tie_break_result.success:
        raise RuntimeError(f"Constraint tie-break solver failed: {tie_break_result.message}")

    assignment = tie_break_result.x.reshape(n_samples, n_classes)
    local_cost = float(clipped01(np.sum(assignment * costs) / n_samples))
    local_factor = float(clipped01(1.0 - local_cost))
    own_assignment_mass = assignment[np.arange(n_samples), y].sum()
    moved_fraction = float(clipped01(1.0 - own_assignment_mass / n_samples))

    assignment_matrix = np.zeros((n_classes, n_classes), dtype=float)
    for source_class in range(n_classes):
        rows = y == source_class
        assignment_matrix[source_class] = assignment[rows].sum(axis=0) / n_samples

    return AssignmentResult(
        local_cost=local_cost,
        local_factor=local_factor,
        moved_fraction=moved_fraction,
        assignment=assignment,
        assignment_matrix=assignment_matrix,
        target_proportions=q,
        observed_proportions=observed_proportions,
    )
