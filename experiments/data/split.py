from __future__ import annotations

import numpy as np
import pandas as pd


def make_train_val_test_split(
    df: pd.DataFrame,
    *,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    shuffle: bool = True,
    strategy: str = "random",
    time_col: str | None = None,
) -> dict[str, object]:
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(f"split sizes must sum to 1.0, got {total}")

    indices = np.arange(len(df))
    strategy = strategy.lower()
    if strategy == "random" and shuffle:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(indices)
    elif strategy == "time":
        if not time_col:
            raise ValueError("time-based split requires split.time_col")
        if time_col not in df.columns:
            raise ValueError(f"time column {time_col!r} is missing in dataset")
        ordered = df.assign(_original_index=np.arange(len(df))).sort_values(
            by=time_col,
            kind="stable",
        )
        indices = ordered["_original_index"].to_numpy()
        shuffle = False
    elif strategy != "random":
        raise ValueError(f"unsupported split strategy: {strategy}")

    test_count = int(round(len(df) * test_size))
    val_count = int(round(len(df) * val_size))
    train_count = len(df) - test_count - val_count
    if train_count <= 0:
        raise ValueError("train split is empty; adjust train/val/test sizes")

    train_idx = indices[:train_count]
    val_idx = indices[train_count : train_count + val_count]
    test_idx = indices[train_count + val_count :]

    return {
        "train_df": df.iloc[train_idx].reset_index(drop=True),
        "val_df": df.iloc[val_idx].reset_index(drop=True),
        "test_df": df.iloc[test_idx].reset_index(drop=True),
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "seed": seed,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "shuffle": shuffle,
        "strategy": strategy,
        "time_col": time_col,
    }
