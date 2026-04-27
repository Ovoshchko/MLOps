from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from data_pipeline.src.utils import load_yaml_config

from experiments.data.dataset import load_registered_dataset
from experiments.data.split import make_train_val_test_split
from experiments.features.build import prepare_splits
from experiments.models.catboost import build_catboost, fit_catboost
from experiments.models.ridge import fit_ridge, predict_ridge
from experiments.tracking.mlflow_tracking import (
    configure_mlflow,
    log_dataset,
    log_split,
    start_run,
)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total = np.sum((y_true - np.mean(y_true)) ** 2)
    if np.isclose(total, 0.0):
        return 0.0
    residual = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - residual / total)


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_rmse": _rmse(y_true, y_pred),
        f"{prefix}_mae": _mae(y_true, y_pred),
        f"{prefix}_r2": _r2(y_true, y_pred),
    }


def _transform_target(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "none":
        return values
    if transform == "log1p":
        if np.any(values < 0):
            raise ValueError("log1p target transform requires non-negative target values")
        return np.log1p(values).astype(np.float32)
    raise ValueError(f"unsupported target transform: {transform}")


def _inverse_target(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "none":
        return values
    if transform == "log1p":
        return np.expm1(values).astype(np.float32)
    raise ValueError(f"unsupported target transform: {transform}")


def _prepare_training_targets(prepared: dict[str, Any], transform: str) -> dict[str, Any]:
    training_prepared = dict(prepared)
    training_prepared["y_train"] = _transform_target(np.asarray(prepared["y_train"]), transform)
    training_prepared["y_val"] = _transform_target(np.asarray(prepared["y_val"]), transform)
    training_prepared["y_test"] = _transform_target(np.asarray(prepared["y_test"]), transform)
    return training_prepared


def _plot_true_vs_pred(y_true: np.ndarray, y_pred: np.ndarray):
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


def _plot_residuals(y_true: np.ndarray, y_pred: np.ndarray):
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


def _plot_learning_curves(evals_result: dict[str, dict[str, list[float]]]):
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


def _plot_feature_importance(feature_importance: list[dict[str, float]]):
    top_features = feature_importance[:20]
    names = [row["feature"] for row in reversed(top_features)]
    values = [row["importance"] for row in reversed(top_features)]

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.barh(names, values)
    axis.set_title("Top Feature Importances")
    axis.set_xlabel("Importance")
    figure.tight_layout()
    return figure


def _load_config(path: str | Path) -> dict[str, Any]:
    config = load_yaml_config(str(Path(path).resolve()))
    required = ("dataset", "split", "model", "mlflow")
    missing = [key for key in required if key not in config]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"experiment config is missing required sections: {joined}")
    if "name" not in config["model"]:
        raise ValueError("model config must contain 'name'")
    return config


def _fit_and_predict_ridge(prepared: dict[str, Any], model_params: dict[str, Any]):
    alpha = float(model_params.get("alpha", 1.0))
    model = fit_ridge(prepared["X_train"], prepared["y_train"], alpha=alpha)
    return {
        "model": model,
        "train_pred": predict_ridge(model, prepared["X_train"]),
        "val_pred": predict_ridge(model, prepared["X_val"]),
        "test_pred": predict_ridge(model, prepared["X_test"]),
    }


def _fit_and_predict_catboost(prepared: dict[str, Any], model_params: dict[str, Any]):
    model = build_catboost(model_params)
    model = fit_catboost(
        model,
        X_train=prepared["X_train"],
        y_train=prepared["y_train"],
        X_val=prepared["X_val"],
        y_val=prepared["y_val"],
        early_stopping_rounds=model_params.get("early_stopping_rounds"),
    )
    return {
        "model": model,
        "train_pred": model.predict(prepared["X_train"]),
        "val_pred": model.predict(prepared["X_val"]),
        "test_pred": model.predict(prepared["X_test"]),
    }


def _log_common_plots(prepared: dict[str, Any], test_pred: np.ndarray) -> None:
    true_vs_pred_figure = _plot_true_vs_pred(prepared["y_test"], test_pred)
    mlflow.log_figure(true_vs_pred_figure, "plots/true_vs_pred.png")
    plt.close(true_vs_pred_figure)

    residuals_figure = _plot_residuals(prepared["y_test"], test_pred)
    mlflow.log_figure(residuals_figure, "plots/residuals.png")
    plt.close(residuals_figure)


def _run_ridge(prepared: dict[str, Any], model_params: dict[str, Any], *, target_transform: str) -> None:
    training_prepared = _prepare_training_targets(prepared, target_transform)
    result = _fit_and_predict_ridge(training_prepared, model_params)
    train_pred = _inverse_target(np.asarray(result["train_pred"]), target_transform)
    val_pred = _inverse_target(np.asarray(result["val_pred"]), target_transform)
    test_pred = _inverse_target(np.asarray(result["test_pred"]), target_transform)

    metrics = {}
    metrics.update(_evaluate(prepared["y_train"], train_pred, "train"))
    metrics.update(_evaluate(prepared["y_val"], val_pred, "val"))
    metrics.update(_evaluate(prepared["y_test"], test_pred, "test"))
    mlflow.log_metrics(metrics)

    mlflow.log_dict(
        {
            "intercept": float(result["model"]["intercept"]),
            "coefficients": np.asarray(result["model"]["coefficients"]).tolist(),
        },
        "model/ridge_coefficients.json",
    )
    mlflow.log_dict(
        {
            "y_true": prepared["y_test"].tolist(),
            "y_pred": np.asarray(test_pred).tolist(),
        },
        "predictions/test_predictions.json",
    )
    _log_common_plots(prepared, test_pred)

    print(f"test_rmse={metrics['test_rmse']:.4f}")
    print(f"test_mae={metrics['test_mae']:.4f}")
    print(f"test_r2={metrics['test_r2']:.4f}")


def _run_catboost(prepared: dict[str, Any], model_params: dict[str, Any], *, target_transform: str) -> None:
    training_prepared = _prepare_training_targets(prepared, target_transform)
    result = _fit_and_predict_catboost(training_prepared, model_params)
    model = result["model"]
    train_pred = _inverse_target(np.asarray(result["train_pred"]), target_transform)
    val_pred = _inverse_target(np.asarray(result["val_pred"]), target_transform)
    test_pred = _inverse_target(np.asarray(result["test_pred"]), target_transform)

    metrics = {}
    metrics.update(_evaluate(prepared["y_train"], train_pred, "train"))
    metrics.update(_evaluate(prepared["y_val"], val_pred, "val"))
    metrics.update(_evaluate(prepared["y_test"], test_pred, "test"))
    mlflow.log_metrics(metrics)

    evals_result = model.get_evals_result()
    for pool_name, pool_metrics in evals_result.items():
        for metric_name, values in pool_metrics.items():
            metric_key = f"{pool_name}_{metric_name}".lower()
            if target_transform != "none":
                metric_key = f"{metric_key}_transformed_target"
            for step, value in enumerate(values):
                mlflow.log_metric(metric_key, float(value), step=step)
    mlflow.log_dict(evals_result, "training/learning_curves.json")

    importances = model.get_feature_importance()
    feature_importance = [
        {"feature": feature_name, "importance": float(importance)}
        for feature_name, importance in zip(prepared["feature_columns"]["feature_names"], importances)
    ]
    feature_importance.sort(key=lambda row: row["importance"], reverse=True)
    mlflow.log_dict(feature_importance, "features/feature_importance.json")
    mlflow.log_dict(
        {
            "best_iteration": int(model.get_best_iteration()),
            "best_score": model.get_best_score(),
        },
        "training/best_iteration.json",
    )
    mlflow.log_dict(
        {
            "y_true": prepared["y_test"].tolist(),
            "y_pred": np.asarray(test_pred).tolist(),
        },
        "predictions/test_predictions.json",
    )

    learning_curve_figure = _plot_learning_curves(evals_result)
    mlflow.log_figure(learning_curve_figure, "plots/learning_curves.png")
    plt.close(learning_curve_figure)

    feature_importance_figure = _plot_feature_importance(feature_importance)
    mlflow.log_figure(feature_importance_figure, "plots/feature_importance.png")
    plt.close(feature_importance_figure)

    _log_common_plots(prepared, test_pred)

    print(f"best_iteration={model.get_best_iteration()}")
    print(f"test_rmse={metrics['test_rmse']:.4f}")
    print(f"test_mae={metrics['test_mae']:.4f}")
    print(f"test_r2={metrics['test_r2']:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run experiment from config.")
    parser.add_argument("--config", default="experiments/configs/train/ridge.yaml")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = _load_config(args.config)

    manifest, df = load_registered_dataset(
        config["dataset"]["manifest_uri"],
        env_path=config["dataset"].get("env_path", "data_pipeline/.env"),
    )
    split = make_train_val_test_split(
        df,
        train_size=float(config["split"].get("train_size", 0.7)),
        val_size=float(config["split"].get("val_size", 0.15)),
        test_size=float(config["split"].get("test_size", 0.15)),
        seed=int(config["split"].get("seed", 42)),
        shuffle=bool(config["split"].get("shuffle", True)),
        strategy=str(config["split"].get("strategy", "random")),
        time_col=config["split"].get("time_col"),
    )
    feature_config = dict(config.get("features", {}))
    prepared = prepare_splits(split, target_col=manifest["target_col"], feature_config=feature_config)

    configure_mlflow(
        tracking_uri=config["mlflow"].get("tracking_uri"),
        experiment_name=str(config["mlflow"].get("experiment_name", "experiments")),
    )

    target_transform = str(config.get("target", {}).get("transform", "none")).lower()
    model_params = dict(config["model"])
    model_name = str(model_params.pop("name")).lower()
    run_name = str(config["mlflow"].get("run_name", f"{model_name}-experiment"))

    print(f"rows={len(df)}")
    print(f"train_rows={len(split['train_df'])}")
    print(f"val_rows={len(split['val_df'])}")
    print(f"test_rows={len(split['test_df'])}")

    with start_run(run_name, tags={"model": model_name, "stage": "experiment"}):
        log_dataset(manifest, df)
        log_split(split)
        mlflow.log_params(
            {
                "model_name": model_name,
                "target_transform": target_transform,
                "feature_title_stats": bool(feature_config.get("title_stats", False)),
                "feature_count": len(prepared["feature_columns"]["feature_names"]),
                **{f"model_{key}": value for key, value in model_params.items()},
            }
        )
        mlflow.log_dict(prepared["feature_columns"], "features/feature_columns.json")
        mlflow.log_dict(feature_config, "features/feature_config.json")

        if model_name == "ridge":
            _run_ridge(prepared, model_params, target_transform=target_transform)
        elif model_name == "catboost":
            _run_catboost(prepared, model_params, target_transform=target_transform)
        else:
            raise ValueError(f"unsupported model: {model_name}")


if __name__ == "__main__":
    main()
