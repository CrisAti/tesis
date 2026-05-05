from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors

from hybrid_transfer_metric.utils import clipped01


@dataclass(frozen=True)
class JMDSConfig:
    """Configuration for the supervised JMDS adaptation."""

    n_neighbors: int = 10
    cv_folds: int = 5
    random_state: int | None = 42
    neighbor_smoothing: float = 0.0
    logistic_max_iter: int = 2000


@dataclass(frozen=True)
class JMDSResult:
    reliability: np.ndarray
    geometric_probabilities: np.ndarray
    model_probabilities: np.ndarray
    own_reliability: np.ndarray


def _knn_label_probabilities(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    n_neighbors: int,
    smoothing: float,
) -> np.ndarray:
    n_samples = X.shape[0]
    k = min(max(1, n_neighbors), max(1, n_samples - 1))
    neighbor_model = NearestNeighbors(n_neighbors=min(n_samples, k + 1))
    neighbor_model.fit(X)
    indices = neighbor_model.kneighbors(X, return_distance=False)

    probabilities = np.zeros((n_samples, n_classes), dtype=float)
    for i, row in enumerate(indices):
        neighbors = [idx for idx in row if idx != i][:k]
        counts = np.full(n_classes, smoothing, dtype=float)
        for idx in neighbors:
            counts[y[idx]] += 1.0
        total = counts.sum()
        if total <= 0:
            counts[y[i]] = 1.0
            total = 1.0
        probabilities[i] = counts / total
    return probabilities


def _cross_validated_model_probabilities(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    config: JMDSConfig,
) -> np.ndarray:
    class_counts = np.bincount(y, minlength=n_classes)
    min_class_count = int(class_counts.min())
    model = LogisticRegression(
        max_iter=config.logistic_max_iter,
        solver="lbfgs",
        random_state=config.random_state,
    )

    if min_class_count >= 2:
        n_splits = min(config.cv_folds, min_class_count)
        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=config.random_state,
        )
        return cross_val_predict(model, X, y, cv=cv, method="predict_proba")

    model.fit(X, y)
    return model.predict_proba(X)


def compute_jmds_reliability(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int | None = None,
    config: JMDSConfig | None = None,
) -> JMDSResult:
    """Compute destination-aware reliability scores R[i, k].

    R[i, k] combines a local geometric probability and a model probability,
    adapting the JMDS idea of multiplying data and model confidence.
    """
    config = config or JMDSConfig()
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional feature matrix.")
    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional label vector.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")

    inferred_classes = int(np.max(y)) + 1
    n_classes = inferred_classes if n_classes is None else int(n_classes)
    if n_classes < inferred_classes:
        raise ValueError("n_classes cannot be smaller than the labels in y.")

    geometric = _knn_label_probabilities(
        X=X,
        y=y,
        n_classes=n_classes,
        n_neighbors=config.n_neighbors,
        smoothing=config.neighbor_smoothing,
    )
    model = _cross_validated_model_probabilities(X, y, n_classes, config)
    if model.shape[1] != n_classes:
        aligned = np.zeros((X.shape[0], n_classes), dtype=float)
        present_classes = np.unique(y)
        aligned[:, present_classes] = model
        model = aligned

    reliability = clipped01(geometric * model)
    own_reliability = reliability[np.arange(X.shape[0]), y]
    return JMDSResult(
        reliability=np.asarray(reliability, dtype=float),
        geometric_probabilities=geometric,
        model_probabilities=model,
        own_reliability=np.asarray(own_reliability, dtype=float),
    )
