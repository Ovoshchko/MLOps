from __future__ import annotations

import numpy as np


def fit_ridge(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 1.0,
) -> dict[str, np.ndarray | float]:
    X_work = np.hstack([np.ones((X.shape[0], 1), dtype=X.dtype), X])

    reg = np.eye(X_work.shape[1], dtype=X_work.dtype) * alpha
    reg[0, 0] = 0.0

    weights = np.linalg.solve(X_work.T @ X_work + reg, X_work.T @ y)

    intercept = float(weights[0])
    coefficients = weights[1:]

    return {
        "intercept": intercept,
        "coefficients": coefficients.astype(np.float32),
    }


def predict_ridge(model: dict[str, np.ndarray | float], X: np.ndarray) -> np.ndarray:
    coefficients = np.asarray(model["coefficients"], dtype=np.float32)
    intercept = float(model["intercept"])
    return X @ coefficients + intercept
