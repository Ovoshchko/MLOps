from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import mlflow
import mlflow.data
import pandas as pd


def configure_mlflow(
    *,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> None:
    mlflow.set_tracking_uri(tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    mlflow.set_experiment(experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME", "baseline"))


@contextmanager
def start_run(
    run_name: str,
    tags: dict[str, str] | None = None,
    *,
    nested: bool = False,
) -> Iterator[None]:
    with mlflow.start_run(run_name=run_name, tags=tags, nested=nested):
        yield


def log_dataset(manifest: dict, dataframe: pd.DataFrame, *, context: str = "training") -> None:
    mlflow.log_params(
        {
            "dataset_id": manifest["dataset_id"],
            "dataset_version": manifest["version"],
            "dataset_manifest_uri": manifest["manifest_uri"],
            "dataset_uri": manifest["dataset_uri"],
            "dataset_digest": manifest["digest"],
            "target_col": manifest["target_col"],
            "dataset_row_count": manifest["row_count"],
            "dataset_column_count": manifest["column_count"],
            "dataset_partition_count": manifest["partition_count"],
        }
    )
    mlflow.set_tags(
        {
            "dataset_id": manifest["dataset_id"],
            "dataset_version": manifest["version"],
            "dataset_digest": manifest["digest"],
        }
    )
    mlflow.log_dict(manifest, f"datasets/{manifest['dataset_id']}/manifest.json")
    dataset = mlflow.data.from_pandas(
        dataframe,
        source=manifest["manifest_uri"],
        name=manifest["dataset_id"],
        targets=manifest["target_col"],
    )
    mlflow.log_input(dataset, context=context)


def log_split(split: dict[str, object]) -> None:
    mlflow.log_params(
        {
            "split_seed": split["seed"],
            "split_train_size": split["train_size"],
            "split_val_size": split["val_size"],
            "split_test_size": split["test_size"],
            "split_shuffle": split["shuffle"],
            "split_strategy": split.get("strategy", "random"),
            "split_time_col": split.get("time_col"),
            "split_train_rows": len(split["train_df"]),
            "split_val_rows": len(split["val_df"]),
            "split_test_rows": len(split["test_df"]),
        }
    )
    mlflow.log_dict(
        {
            "seed": split["seed"],
            "train_size": split["train_size"],
            "val_size": split["val_size"],
            "test_size": split["test_size"],
            "shuffle": split["shuffle"],
            "strategy": split.get("strategy", "random"),
            "time_col": split.get("time_col"),
            "train_rows": len(split["train_df"]),
            "val_rows": len(split["val_df"]),
            "test_rows": len(split["test_df"]),
            "train_indices": split["train_indices"],
            "val_indices": split["val_indices"],
            "test_indices": split["test_indices"],
        },
        "splits/split.json",
    )
