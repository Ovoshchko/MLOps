from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
from data_pipeline.src.utils import load_yaml_config

from experiments.data.dataset import load_registered_dataset
from experiments.data.split import make_train_val_test_split
from experiments.features.build import prepare_splits
from experiments.models.catboost import build_catboost, fit_catboost
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
    required = ("dataset", "split", "model", "optuna", "search_space", "mlflow")
    missing = [key for key in required if key not in config]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"tuning config is missing required sections: {joined}")
    if config["model"].get("name") != "catboost":
        raise ValueError("tuning currently supports only model.name=catboost")
    return config


def _suggest_params(trial: optuna.Trial, search_space: dict[str, Any]) -> dict[str, Any]:
    return {
        "iterations": trial.suggest_int(
            "iterations",
            int(search_space["iterations"]["low"]),
            int(search_space["iterations"]["high"]),
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            float(search_space["learning_rate"]["low"]),
            float(search_space["learning_rate"]["high"]),
            log=bool(search_space["learning_rate"].get("log", False)),
        ),
        "depth": trial.suggest_int(
            "depth",
            int(search_space["depth"]["low"]),
            int(search_space["depth"]["high"]),
        ),
        "l2_leaf_reg": trial.suggest_float(
            "l2_leaf_reg",
            float(search_space["l2_leaf_reg"]["low"]),
            float(search_space["l2_leaf_reg"]["high"]),
            log=bool(search_space["l2_leaf_reg"].get("log", False)),
        ),
    }


def _fit_catboost(prepared: dict[str, Any], model_params: dict[str, Any]):
    model = build_catboost(model_params)
    model = fit_catboost(
        model,
        X_train=prepared["X_train"],
        y_train=prepared["y_train"],
        X_val=prepared["X_val"],
        y_val=prepared["y_val"],
        early_stopping_rounds=model_params.get("early_stopping_rounds"),
    )
    train_pred = model.predict(prepared["X_train"])
    val_pred = model.predict(prepared["X_val"])
    test_pred = model.predict(prepared["X_test"])
    return model, train_pred, val_pred, test_pred


def _log_catboost_iteration_metrics(model) -> dict[str, dict[str, list[float]]]:
    evals_result = model.get_evals_result()
    for pool_name, pool_metrics in evals_result.items():
        for metric_name, values in pool_metrics.items():
            metric_key = f"{pool_name}_{metric_name}".lower()
            for step, value in enumerate(values):
                mlflow.log_metric(metric_key, float(value), step=step)
    mlflow.log_dict(evals_result, "training/learning_curves.json")
    return evals_result


def _log_parent_artifacts(prepared: dict[str, Any], model, test_pred: np.ndarray) -> None:
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

    learning_curve_figure = _plot_learning_curves(model.get_evals_result())
    mlflow.log_figure(learning_curve_figure, "plots/learning_curves.png")
    plt.close(learning_curve_figure)

    feature_importance_figure = _plot_feature_importance(feature_importance)
    mlflow.log_figure(feature_importance_figure, "plots/feature_importance.png")
    plt.close(feature_importance_figure)

    true_vs_pred_figure = _plot_true_vs_pred(prepared["y_test"], test_pred)
    mlflow.log_figure(true_vs_pred_figure, "plots/true_vs_pred.png")
    plt.close(true_vs_pred_figure)

    residuals_figure = _plot_residuals(prepared["y_test"], test_pred)
    mlflow.log_figure(residuals_figure, "plots/residuals.png")
    plt.close(residuals_figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune experiment with Optuna.")
    parser.add_argument("--config", default="experiments/configs/tune/catboost_optuna.yaml")
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
    prepared = prepare_splits(split, target_col=manifest["target_col"])

    configure_mlflow(
        tracking_uri=config["mlflow"].get("tracking_uri"),
        experiment_name=str(config["mlflow"].get("experiment_name", "optuna-catboost")),
    )

    fixed_model_params = {
        key: value
        for key, value in config["model"].items()
        if key != "name"
    }
    optuna_config = config["optuna"]
    search_space = config["search_space"]
    run_name = str(config["mlflow"].get("run_name", "catboost-optuna"))

    sampler = optuna.samplers.TPESampler(seed=int(optuna_config.get("sampler_seed", 42)))
    storage_uri = optuna_config.get("storage_uri")
    if storage_uri in ("", None):
        storage_uri = None

    study = optuna.create_study(
        study_name=str(optuna_config.get("study_name", "catboost_tuning")),
        direction=str(optuna_config.get("direction", "minimize")),
        sampler=sampler,
        storage=storage_uri,
        load_if_exists=bool(optuna_config.get("load_if_exists", False)),
    )

    with start_run(run_name, tags={"model": "catboost", "stage": "tuning"}):
        log_dataset(manifest, df)
        log_split(split)
        mlflow.log_params(
            {
                "model_name": "catboost",
                "feature_count": len(prepared["feature_columns"]["feature_names"]),
                "optuna_study_name": str(optuna_config.get("study_name", "catboost_tuning")),
                "optuna_n_trials": int(optuna_config.get("n_trials", 30)),
                "optuna_direction": str(optuna_config.get("direction", "minimize")),
                "optuna_metric": str(optuna_config.get("metric", "val_rmse")),
                **{f"model_{key}": value for key, value in fixed_model_params.items()},
            }
        )
        mlflow.log_dict(prepared["feature_columns"], "features/feature_columns.json")
        mlflow.log_dict(search_space, "optuna/search_space.json")

        def objective(trial: optuna.Trial) -> float:
            trial_params = _suggest_params(trial, search_space)
            model_params = dict(fixed_model_params)
            model_params.update(trial_params)

            with start_run(
                f"trial-{trial.number}",
                tags={"model": "catboost", "stage": "optuna-trial"},
                nested=True,
            ):
                mlflow.log_params(
                    {
                        "trial_number": trial.number,
                        **{f"model_{key}": value for key, value in model_params.items()},
                    }
                )

                model, train_pred, val_pred, _ = _fit_catboost(prepared, model_params)
                metrics = {}
                metrics.update(_evaluate(prepared["y_train"], train_pred, "train"))
                metrics.update(_evaluate(prepared["y_val"], val_pred, "val"))
                mlflow.log_metrics(metrics)

                _log_catboost_iteration_metrics(model)
                mlflow.log_dict(
                    {
                        "best_iteration": int(model.get_best_iteration()),
                        "best_score": model.get_best_score(),
                    },
                    "training/best_iteration.json",
                )

                trial.set_user_attr("best_iteration", int(model.get_best_iteration()))
                trial.set_user_attr("val_mae", metrics["val_mae"])
                trial.set_user_attr("val_r2", metrics["val_r2"])
                return metrics[str(optuna_config.get("metric", "val_rmse"))]

        study.optimize(objective, n_trials=int(optuna_config.get("n_trials", 30)))

        best_model_params = dict(fixed_model_params)
        best_model_params.update(study.best_params)
        best_model, train_pred, val_pred, test_pred = _fit_catboost(prepared, best_model_params)

        best_metrics = {}
        best_metrics.update(_evaluate(prepared["y_train"], train_pred, "train"))
        best_metrics.update(_evaluate(prepared["y_val"], val_pred, "val"))
        best_metrics.update(_evaluate(prepared["y_test"], test_pred, "test"))
        mlflow.log_metrics(best_metrics)
        _log_catboost_iteration_metrics(best_model)

        trials_summary = [
            {
                "number": trial.number,
                "value": trial.value,
                "state": trial.state.name,
                "params": trial.params,
                "user_attrs": trial.user_attrs,
            }
            for trial in study.trials
        ]
        mlflow.log_metric("best_trial_value", float(study.best_value))
        mlflow.log_dict(study.best_params, "optuna/best_params.json")
        mlflow.log_dict(
            {
                "best_trial_number": study.best_trial.number,
                "best_value": float(study.best_value),
                "best_params": study.best_params,
            },
            "optuna/best_trial.json",
        )
        mlflow.log_dict(trials_summary, "optuna/trials.json")

        _log_parent_artifacts(prepared, best_model, test_pred)

if __name__ == "__main__":
    main()
