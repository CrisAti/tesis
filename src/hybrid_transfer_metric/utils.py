from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def encode_labels(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode arbitrary labels as stable integers in [0, K)."""
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(np.asarray(y))
    return encoded.astype(int), encoder.classes_


def standardize_features(X: np.ndarray) -> np.ndarray:
    """Standardize features for distance/model based components."""
    X = np.asarray(X, dtype=float)
    return StandardScaler().fit_transform(X)


def empirical_proportions(y_encoded: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y_encoded, minlength=n_classes).astype(float)
    return counts / counts.sum()


def validate_target_proportions(q: np.ndarray, n_classes: int, tolerance: float = 1e-8) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.ndim != 1:
        raise ValueError("target_proportions must be a one-dimensional array.")
    if q.shape[0] != n_classes:
        raise ValueError(
            f"target_proportions has length {q.shape[0]}, expected {n_classes}."
        )
    if np.any(q < -tolerance):
        raise ValueError("target_proportions cannot contain negative values.")
    total = float(q.sum())
    if not np.isclose(total, 1.0, atol=tolerance):
        raise ValueError(f"target_proportions must sum to 1.0, got {total:.12f}.")
    q = np.clip(q, 0.0, 1.0)
    return q / q.sum()


def clipped01(value: float | np.ndarray) -> float | np.ndarray:
    return np.clip(value, 0.0, 1.0)


def to_serializable(value: Any) -> Any:
    """Convert dataclasses and numpy values into JSON-friendly structures."""
    if is_dataclass(value):
        return to_serializable(asdict(value))
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
