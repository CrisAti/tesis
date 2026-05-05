from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import calinski_harabasz_score, silhouette_score

from hybrid_transfer_metric.utils import clipped01


@dataclass(frozen=True)
class CLMConfig:
    """Configuration for the operational CLM-inspired global score."""

    n_permutations: int = 128
    random_state: int | None = 42
    silhouette_weight: float = 0.5
    calinski_harabasz_weight: float = 0.5


@dataclass(frozen=True)
class MetricAdjustment:
    raw: float
    null_mean: float
    null_std: float
    adjusted: float


@dataclass(frozen=True)
class CLMResult:
    score: float
    metrics: dict[str, MetricAdjustment] = field(default_factory=dict)


def _validate_clustering_inputs(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional feature matrix.")
    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional label vector.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")
    n_classes = int(np.unique(y).size)
    if n_classes < 2:
        raise ValueError("At least two classes are required for CLM scoring.")
    if X.shape[0] <= n_classes:
        raise ValueError("The number of samples must exceed the number of classes.")
    return X, y, n_classes


def _safe_silhouette(X: np.ndarray, y: np.ndarray) -> float:
    return float(silhouette_score(X, y, metric="euclidean"))


def _safe_calinski_harabasz(X: np.ndarray, y: np.ndarray) -> float:
    return float(calinski_harabasz_score(X, y))


def _permuted_scores(
    X: np.ndarray,
    y: np.ndarray,
    scorer,
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    scores: list[float] = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        try:
            scores.append(float(scorer(X, y_perm)))
        except ValueError:
            continue
    if not scores:
        return np.array([0.0], dtype=float)
    return np.asarray(scores, dtype=float)


def _adjust_silhouette(raw: float, null_scores: np.ndarray) -> MetricAdjustment:
    null_mean = float(np.mean(null_scores))
    null_std = float(np.std(null_scores))
    denominator = max(1.0 - null_mean, 1e-12)
    adjusted = float(clipped01((raw - null_mean) / denominator))
    return MetricAdjustment(
        raw=raw,
        null_mean=null_mean,
        null_std=null_std,
        adjusted=adjusted,
    )


def _adjust_calinski_harabasz(raw: float, null_scores: np.ndarray) -> MetricAdjustment:
    raw_log = float(np.log1p(max(raw, 0.0)))
    null_log = np.log1p(np.maximum(null_scores, 0.0))
    null_mean = float(np.mean(null_log))
    null_std = float(np.std(null_log))
    gap = max(0.0, raw_log - null_mean)
    adjusted = gap / (gap + abs(null_mean) + 1.0)
    return MetricAdjustment(
        raw=raw,
        null_mean=float(np.expm1(null_mean)),
        null_std=null_std,
        adjusted=float(clipped01(adjusted)),
    )


def compute_clm_score(X: np.ndarray, y: np.ndarray, config: CLMConfig | None = None) -> CLMResult:
    """Compute a chance-adjusted global structure score inspired by CLM.

    The score compares internal validation measures for the given labels against
    label permutations that preserve the empirical class counts.
    """
    config = config or CLMConfig()
    X, y, _ = _validate_clustering_inputs(X, y)
    rng = np.random.default_rng(config.random_state)

    sil_raw = _safe_silhouette(X, y)
    sil_null = _permuted_scores(
        X,
        y,
        _safe_silhouette,
        config.n_permutations,
        rng,
    )
    sil = _adjust_silhouette(sil_raw, sil_null)

    ch_raw = _safe_calinski_harabasz(X, y)
    ch_null = _permuted_scores(
        X,
        y,
        _safe_calinski_harabasz,
        config.n_permutations,
        rng,
    )
    ch = _adjust_calinski_harabasz(ch_raw, ch_null)

    weight_sum = config.silhouette_weight + config.calinski_harabasz_weight
    if weight_sum <= 0:
        raise ValueError("At least one CLM metric weight must be positive.")

    score = (
        config.silhouette_weight * sil.adjusted
        + config.calinski_harabasz_weight * ch.adjusted
    ) / weight_sum

    return CLMResult(
        score=float(clipped01(score)),
        metrics={
            "silhouette": sil,
            "calinski_harabasz": ch,
        },
    )
