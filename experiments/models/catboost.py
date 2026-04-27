from __future__ import annotations

from typing import Any

import numpy as np
from catboost import CatBoostRegressor


def build_catboost(params: dict[str, Any]) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=int(params.get("iterations", 500)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        depth=int(params.get("depth", 6)),
        l2_leaf_reg=float(params.get("l2_leaf_reg", 3.0)),
        random_seed=int(params.get("seed", 42)),
        verbose=False,
    )


def fit_catboost(
    model: CatBoostRegressor,
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    early_stopping_rounds: int | None = None,
) -> CatBoostRegressor:
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        early_stopping_rounds=early_stopping_rounds,
        verbose=False,
    )
    return model
