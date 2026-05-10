from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_true_vs_pred(y_true: np.ndarray, y_pred: np.ndarray):
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(y_true, y_pred, alpha=0.35, s=12)
    minimum = float(min(np.min(y_true), np.min(y_pred)))
    maximum = float(max(np.max(y_true), np.max(y_pred)))
    axis.plot([minimum, maximum], [minimum, maximum], linestyle="--")
    axis.set_title("True vs Predicted")
    axis.set_xlabel("True")
    axis.set_ylabel("Predicted")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    return figure


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray):
    residuals = y_true - y_pred
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(y_pred, residuals, alpha=0.35, s=12)
    axis.axhline(0.0, linestyle="--")
    axis.set_title("Residuals")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Residual")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    return figure


def plot_learning_curves(evals_result: dict[str, dict[str, list[float]]]):
    figure, axis = plt.subplots(figsize=(8, 5))
    for pool_name, pool_metrics in evals_result.items():
        for metric_name, values in pool_metrics.items():
            axis.plot(values, label=f"{pool_name}_{metric_name}".lower())
    axis.set_title("Learning Curves")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Metric value")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    return figure


def plot_feature_importance(feature_importance: list[dict[str, float]]):
    top_features = feature_importance[:20]
    names = [row["feature"] for row in reversed(top_features)]
    values = [row["importance"] for row in reversed(top_features)]

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.barh(names, values)
    axis.set_title("Top Feature Importances")
    axis.set_xlabel("Importance")
    figure.tight_layout()
    return figure
