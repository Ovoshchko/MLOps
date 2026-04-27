from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _first_non_null(series: pd.Series) -> Any | None:
    for value in series:
        if value is None:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        return value
    return None


def _is_vector_like(value: Any) -> bool:
    return isinstance(value, (list, tuple, np.ndarray))


def _add_title_features(df: pd.DataFrame) -> pd.DataFrame:
    if "title" not in df.columns:
        return df

    out = df.copy()
    title = out["title"].fillna("").astype(str)

    char_count = title.str.len().astype(np.float32)
    word_count = title.str.split().str.len().fillna(0).astype(np.float32)
    uppercase_count = title.str.count(r"[A-Z]").astype(np.float32)
    digit_count = title.str.count(r"\d").astype(np.float32)

    safe_char_count = char_count.replace(0, np.nan)

    out["title_length_chars"] = char_count
    out["title_length_words"] = word_count
    out["title_has_question"] = title.str.contains(r"\?", regex=True)
    out["title_has_exclamation"] = title.str.contains(r"!", regex=False)
    out["title_uppercase_ratio"] = (uppercase_count / safe_char_count).fillna(0.0).astype(np.float32)
    out["title_digit_ratio"] = (digit_count / safe_char_count).fillna(0.0).astype(np.float32)
    return out


def apply_feature_engineering(df: pd.DataFrame, feature_config: dict[str, object] | None = None) -> pd.DataFrame:
    config = feature_config or {}
    out = df

    if bool(config.get("title_stats", False)):
        out = _add_title_features(out)

    return out


def infer_feature_columns(df: pd.DataFrame, *, target_col: str) -> dict[str, object]:
    scalar_columns: list[str] = []
    vector_columns: dict[str, int] = {}
    feature_names: list[str] = []

    for column in df.columns:
        if column == target_col:
            continue

        sample = _first_non_null(df[column])
        if _is_vector_like(sample):
            dimension = len(sample)
            if dimension > 0:
                vector_columns[column] = dimension
                feature_names.extend(f"{column}_{idx}" for idx in range(dimension))
            continue

        series = df[column]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            scalar_columns.append(column)
            feature_names.append(column)

    return {
        "target_col": target_col,
        "scalar_columns": scalar_columns,
        "vector_columns": vector_columns,
        "feature_names": feature_names,
    }


def _build_scalar_block(df: pd.DataFrame, scalar_columns: list[str]) -> np.ndarray:
    if not scalar_columns:
        return np.empty((len(df), 0), dtype=np.float32)

    scalar_df = df[scalar_columns].copy()
    for column in scalar_columns:
        if pd.api.types.is_bool_dtype(scalar_df[column]):
            scalar_df[column] = scalar_df[column].astype(np.float32)
        else:
            scalar_df[column] = pd.to_numeric(scalar_df[column], errors="coerce").fillna(0.0)
    return scalar_df.to_numpy(dtype=np.float32)


def _build_vector_block(df: pd.DataFrame, vector_columns: dict[str, int]) -> np.ndarray:
    if not vector_columns:
        return np.empty((len(df), 0), dtype=np.float32)

    blocks: list[np.ndarray] = []
    for column, dimension in vector_columns.items():
        block = np.zeros((len(df), dimension), dtype=np.float32)
        for row_idx, value in enumerate(df[column]):
            if not _is_vector_like(value):
                continue
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size:
                block[row_idx, : min(dimension, arr.size)] = arr[:dimension]
        blocks.append(block)
    return np.concatenate(blocks, axis=1)


def build_X_y(df: pd.DataFrame, feature_columns: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    scalar_block = _build_scalar_block(df, feature_columns["scalar_columns"])
    vector_block = _build_vector_block(df, feature_columns["vector_columns"])

    if scalar_block.size == 0 and vector_block.size == 0:
        raise ValueError("no usable features were found for the current dataset")

    if scalar_block.size == 0:
        X = vector_block
    elif vector_block.size == 0:
        X = scalar_block
    else:
        X = np.concatenate([scalar_block, vector_block], axis=1)

    target_col = feature_columns["target_col"]
    y = pd.to_numeric(df[target_col], errors="coerce")
    if y.isna().any():
        raise ValueError(f"target column {target_col!r} contains NaN after numeric coercion")
    return X, y.to_numpy(dtype=np.float32)


def prepare_splits(
    split: dict[str, object],
    *,
    target_col: str,
    feature_config: dict[str, object] | None = None,
) -> dict[str, object]:
    train_df = apply_feature_engineering(split["train_df"], feature_config)
    val_df = apply_feature_engineering(split["val_df"], feature_config)
    test_df = apply_feature_engineering(split["test_df"], feature_config)

    feature_columns = infer_feature_columns(train_df, target_col=target_col)
    X_train, y_train = build_X_y(train_df, feature_columns)
    X_val, y_val = build_X_y(val_df, feature_columns)
    X_test, y_test = build_X_y(test_df, feature_columns)

    return {
        "feature_columns": feature_columns,
        "feature_config": feature_config or {},
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }
