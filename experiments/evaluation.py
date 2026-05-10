from __future__ import annotations

import math

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total = np.sum((y_true - np.mean(y_true)) ** 2)
    if np.isclose(total, 0.0):
        return 0.0
    residual = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - residual / total)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_rmse": rmse(y_true, y_pred),
        f"{prefix}_mae": mae(y_true, y_pred),
        f"{prefix}_r2": r2(y_true, y_pred),
    }
